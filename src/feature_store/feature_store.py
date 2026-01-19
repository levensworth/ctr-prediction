"""Parquet-based feature store implementation.

Provides persistence and retrieval of feature sets using parquet format
for efficient storage and fast loading.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.domain.entities import FeatureVector


@dataclass
class ParquetFeatureStore:
    """Feature store using parquet files for persistence.
    
    Organizes features in a directory structure:
        storage_path/
            feature_set_name/
                features.parquet
                metadata.json
    """
    
    storage_path: Path
    _feature_columns_cache: dict[str, list[str]] = field(default_factory=dict, init=False)
    
    def __post_init__(self) -> None:
        self.storage_path = Path(self.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_features(
        self,
        features_df: pl.DataFrame,
        feature_set_name: str,
        feature_columns: list[str] | None = None,
    ) -> Path:
        """Save a feature set to parquet storage.
        
        Args:
            features_df: DataFrame containing features
            feature_set_name: Identifier for this feature set
            feature_columns: Optional list of feature column names to store as metadata
        
        Returns:
            Path to the saved parquet file
        """
        feature_dir = self.storage_path / feature_set_name
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_path = feature_dir / "features.parquet"
        features_df.write_parquet(parquet_path)
        
        if feature_columns is not None:
            self._feature_columns_cache[feature_set_name] = feature_columns
            metadata_path = feature_dir / "metadata.json"
            import json
            with open(metadata_path, "w") as f:
                json.dump({
                    "feature_columns": feature_columns,
                    "n_rows": len(features_df),
                    "n_columns": len(features_df.columns),
                }, f, indent=2)
        
        return parquet_path

    def load_features(self, feature_set_name: str) -> pl.DataFrame:
        """Load a feature set from storage.
        
        Args:
            feature_set_name: Identifier for the feature set to load
        
        Returns:
            DataFrame containing the features
        
        Raises:
            FileNotFoundError: If the feature set doesn't exist
        """
        parquet_path = self.storage_path / feature_set_name / "features.parquet"
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Feature set '{feature_set_name}' not found at {parquet_path}")
        
        return pl.read_parquet(parquet_path)

    def get_feature_columns(self, feature_set_name: str) -> list[str]:
        """Get the feature column names for a feature set.
        
        Args:
            feature_set_name: Identifier for the feature set
        
        Returns:
            List of feature column names
        """
        if feature_set_name in self._feature_columns_cache:
            return self._feature_columns_cache[feature_set_name]
        
        metadata_path = self.storage_path / feature_set_name / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
                columns = metadata.get("feature_columns", [])
                self._feature_columns_cache[feature_set_name] = columns
                return columns
        
        return []

    def get_features_for_prediction(
        self,
        publication_id: str,
        campaign_id: str,
        feature_set_name: str = "prediction_features",
    ) -> FeatureVector | None:
        """Retrieve pre-computed features for a prediction request.
        
        Looks up features by publication_id and campaign_id combination.
        
        Args:
            publication_id: The publication identifier
            campaign_id: The campaign identifier
            feature_set_name: Which feature set to query
        
        Returns:
            FeatureVector if found, None otherwise
        """
        try:
            features_df = self.load_features(feature_set_name)
        except FileNotFoundError:
            return None
        
        feature_columns = self.get_feature_columns(feature_set_name)
        if not feature_columns:
            feature_columns = [
                c for c in features_df.columns 
                if c not in ["publication_id", "campaign_id", "ctr", "approved_opens", "approved_clicks"]
            ]
        
        row = features_df.filter(
            (pl.col("publication_id") == publication_id) & 
            (pl.col("campaign_id") == campaign_id)
        )
        
        if len(row) == 0:
            return None
        
        # Take the most recent if multiple rows exist
        row = row.head(1)
        
        features = row.select(feature_columns).to_numpy().flatten()
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        approved_opens = None
        if "approved_opens" in row.columns:
            approved_opens = row["approved_opens"].to_list()[0]
        
        return FeatureVector(
            features=features,
            feature_names=feature_columns,
            publication_id=publication_id,
            campaign_id=campaign_id,
            approved_opens=approved_opens,
        )

    def get_features_batch(
        self,
        requests: list[tuple[str, str]],
        feature_set_name: str = "prediction_features",
    ) -> list[FeatureVector | None]:
        """Retrieve features for multiple prediction requests.
        
        Args:
            requests: List of (publication_id, campaign_id) tuples
            feature_set_name: Which feature set to query
        
        Returns:
            List of FeatureVector or None for each request
        """
        return [
            self.get_features_for_prediction(pub_id, camp_id, feature_set_name)
            for pub_id, camp_id in requests
        ]

    def list_feature_sets(self) -> list[str]:
        """List all available feature sets.
        
        Returns:
            List of feature set names
        """
        return [
            d.name for d in self.storage_path.iterdir() 
            if d.is_dir() and (d / "features.parquet").exists()
        ]

    def feature_set_exists(self, feature_set_name: str) -> bool:
        """Check if a feature set exists.
        
        Args:
            feature_set_name: Identifier for the feature set
        
        Returns:
            True if the feature set exists
        """
        parquet_path = self.storage_path / feature_set_name / "features.parquet"
        return parquet_path.exists()

    def delete_feature_set(self, feature_set_name: str) -> bool:
        """Delete a feature set.
        
        Args:
            feature_set_name: Identifier for the feature set
        
        Returns:
            True if deleted, False if not found
        """
        feature_dir = self.storage_path / feature_set_name
        if not feature_dir.exists():
            return False
        
        import shutil
        shutil.rmtree(feature_dir)
        
        if feature_set_name in self._feature_columns_cache:
            del self._feature_columns_cache[feature_set_name]
        
        return True

    def get_feature_stats(self, feature_set_name: str) -> dict[str, Any]:
        """Get statistics about a feature set.
        
        Args:
            feature_set_name: Identifier for the feature set
        
        Returns:
            Dictionary with feature set statistics
        """
        features_df = self.load_features(feature_set_name)
        feature_columns = self.get_feature_columns(feature_set_name)
        
        return {
            "name": feature_set_name,
            "n_rows": len(features_df),
            "n_total_columns": len(features_df.columns),
            "n_feature_columns": len(feature_columns),
            "feature_columns": feature_columns,
            "columns": features_df.columns,
        }
