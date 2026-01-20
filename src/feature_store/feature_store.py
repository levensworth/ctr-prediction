"""Parquet-based feature store implementation.

Provides persistence and retrieval of feature sets using parquet format
for efficient storage and fast loading.

Supports:
- Publisher-specific features (publication_id as key)
- Campaign-specific features (campaign_id as key)
- Imputation statistics for handling unseen entities
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.domain.entities import FeatureVector, ImputationStatistics


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

    def save_publisher_features(
        self,
        features_df: pl.DataFrame,
        feature_columns: list[str],
    ) -> Path:
        """Save publisher-specific features keyed by publication_id.
        
        Args:
            features_df: DataFrame containing publication_id and feature columns
            feature_columns: List of feature column names
        
        Returns:
            Path to the saved parquet file
        """
        publisher_df = features_df.select(["publication_id"] + feature_columns)
        publisher_df = publisher_df.unique(subset=["publication_id"])
        return self.save_features(publisher_df, "publisher_features", feature_columns)

    def save_campaign_features(
        self,
        features_df: pl.DataFrame,
        feature_columns: list[str],
    ) -> Path:
        """Save campaign-specific features keyed by campaign_id.
        
        Args:
            features_df: DataFrame containing campaign_id and feature columns
            feature_columns: List of feature column names
        
        Returns:
            Path to the saved parquet file
        """
        campaign_df = features_df.select(["campaign_id"] + feature_columns)
        campaign_df = campaign_df.unique(subset=["campaign_id"])
        return self.save_features(campaign_df, "campaign_features", feature_columns)

    def compute_imputation_statistics(
        self,
        features_df: pl.DataFrame,
        numerical_columns: list[str],
        categorical_columns: list[str],
    ) -> ImputationStatistics:
        """Compute imputation statistics (mean/mode) from training features.
        
        Args:
            features_df: DataFrame containing the features
            numerical_columns: List of numerical feature column names (will use mean)
            categorical_columns: List of categorical feature column names (will use mode)
        
        Returns:
            ImputationStatistics with computed values
        """
        numerical_means = {}
        for col in numerical_columns:
            if col in features_df.columns:
                mean_val = features_df[col].mean()
                numerical_means[col] = float(mean_val) if mean_val is not None else 0.0
        
        categorical_modes = {}
        for col in categorical_columns:
            if col in features_df.columns:
                mode_val = features_df[col].mode()
                if len(mode_val) > 0:
                    categorical_modes[col] = int(mode_val[0])
                else:
                    categorical_modes[col] = 0
        
        return ImputationStatistics(
            numerical_means=numerical_means,
            categorical_modes=categorical_modes,
        )

    def save_imputation_statistics(
        self,
        stats: ImputationStatistics,
        feature_set_name: str,
    ) -> Path:
        """Save imputation statistics to JSON.
        
        Args:
            stats: ImputationStatistics object to save
            feature_set_name: Name to identify this imputation set (e.g., "publisher", "campaign")
        
        Returns:
            Path to the saved JSON file
        """
        stats_dir = self.storage_path / "imputation"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        stats_path = stats_dir / f"{feature_set_name}_imputation.json"
        with open(stats_path, "w") as f:
            json.dump({
                "numerical_means": stats.numerical_means,
                "categorical_modes": stats.categorical_modes,
            }, f, indent=2)
        
        return stats_path

    def load_imputation_statistics(self, feature_set_name: str) -> ImputationStatistics | None:
        """Load imputation statistics from JSON.
        
        Args:
            feature_set_name: Name of the imputation set to load
        
        Returns:
            ImputationStatistics if found, None otherwise
        """
        stats_path = self.storage_path / "imputation" / f"{feature_set_name}_imputation.json"
        if not stats_path.exists():
            return None
        
        with open(stats_path) as f:
            data = json.load(f)
        
        return ImputationStatistics(
            numerical_means=data["numerical_means"],
            categorical_modes=data["categorical_modes"],
        )

    def get_publisher_features(self, publication_id: str) -> dict[str, float] | None:
        """Retrieve features for a specific publication_id.
        
        Args:
            publication_id: The publication identifier
        
        Returns:
            Dictionary of feature_name -> value, or None if not found
        """
        try:
            features_df = self.load_features("publisher_features")
        except FileNotFoundError:
            return None
        
        row = features_df.filter(pl.col("publication_id") == publication_id)
        if len(row) == 0:
            return None
        
        row = row.head(1)
        feature_columns = self.get_feature_columns("publisher_features")
        
        result = {}
        for col in feature_columns:
            if col in row.columns:
                val = row[col].to_list()[0]
                result[col] = float(val) if val is not None else 0.0
        
        return result

    def get_campaign_features(self, campaign_id: str) -> dict[str, float] | None:
        """Retrieve features for a specific campaign_id.
        
        Args:
            campaign_id: The campaign identifier
        
        Returns:
            Dictionary of feature_name -> value, or None if not found
        """
        try:
            features_df = self.load_features("campaign_features")
        except FileNotFoundError:
            return None
        
        row = features_df.filter(pl.col("campaign_id") == campaign_id)
        if len(row) == 0:
            return None
        
        row = row.head(1)
        feature_columns = self.get_feature_columns("campaign_features")
        
        result = {}
        for col in feature_columns:
            if col in row.columns:
                val = row[col].to_list()[0]
                result[col] = float(val) if val is not None else 0.0
        
        return result

    def get_imputed_publisher_features(
        self,
        publication_id: str,
    ) -> tuple[dict[str, float], bool]:
        """Get publisher features, using imputation for unseen publication_id.
        
        Args:
            publication_id: The publication identifier
        
        Returns:
            Tuple of (features_dict, was_imputed)
        """
        features = self.get_publisher_features(publication_id)
        if features is not None:
            return features, False
        
        imputation_stats = self.load_imputation_statistics("publisher")
        if imputation_stats is None:
            feature_columns = self.get_feature_columns("publisher_features")
            return {col: 0.0 for col in feature_columns}, True
        
        imputed_features = {}
        for key, value in imputation_stats.numerical_means.items():
            imputed_features[key] = value
        for key, value in imputation_stats.categorical_modes.items():
            imputed_features[key] = float(value)
        
        return imputed_features, True

    def get_imputed_campaign_features(
        self,
        campaign_id: str,
    ) -> tuple[dict[str, float], bool]:
        """Get campaign features, using imputation for unseen campaign_id.
        
        Args:
            campaign_id: The campaign identifier
        
        Returns:
            Tuple of (features_dict, was_imputed)
        """
        features = self.get_campaign_features(campaign_id)
        if features is not None:
            return features, False
        
        imputation_stats = self.load_imputation_statistics("campaign")
        if imputation_stats is None:
            feature_columns = self.get_feature_columns("campaign_features")
            return {col: 0.0 for col in feature_columns}, True
        
        imputed_features = {}
        for key, value in imputation_stats.numerical_means.items():
            imputed_features[key] = value
        for key, value in imputation_stats.categorical_modes.items():
            imputed_features[key] = float(value)
        
        return imputed_features, True

    def get_features_for_prediction_with_imputation(
        self,
        publication_id: str,
        campaign_id: str,
        feature_columns: list[str],
        default_opens: int = 1000,
    ) -> tuple[FeatureVector, bool, bool]:
        """Retrieve features for prediction, imputing for unseen entities.
        
        Combines publisher and campaign features, using imputation for any
        unseen publication_id or campaign_id.
        
        Args:
            publication_id: The publication identifier
            campaign_id: The campaign identifier
            feature_columns: Ordered list of feature columns expected by the model
            default_opens: Default approved_opens value for unseen combinations
        
        Returns:
            Tuple of (FeatureVector, publisher_was_imputed, campaign_was_imputed)
        """
        pub_features, pub_imputed = self.get_imputed_publisher_features(publication_id)
        camp_features, camp_imputed = self.get_imputed_campaign_features(campaign_id)
        
        combined_features = {**pub_features, **camp_features}
        
        feature_array = np.array([
            combined_features.get(col, 0.0) for col in feature_columns
        ], dtype=np.float64)
        
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return (
            FeatureVector(
                features=feature_array,
                feature_names=feature_columns,
                publication_id=publication_id,
                campaign_id=campaign_id,
                approved_opens=default_opens,
            ),
            pub_imputed,
            camp_imputed,
        )
