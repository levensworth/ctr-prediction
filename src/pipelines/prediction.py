"""Prediction pipeline for CTR inference.

Provides a simple interface for making CTR predictions using
a trained model and pre-computed features from a feature store.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from src.domain.entities import PredictionResult
from src.feature_store.feature_store import ParquetFeatureStore
from src.features.feature_engineering import CTRFeatureEngineer
from src.models.xgboost_ensemble import XGBoostEnsembleModel
from src.pipelines.config import load_config


@dataclass
class PredictionPipeline:
    """Pipeline for making CTR predictions.
    
    Uses a trained model and feature store to generate predictions
    for publication-campaign pairs.
    """
    
    model_path: Path
    feature_store_path: Path
    feature_engineer_path: Path | None = None
    
    _model: XGBoostEnsembleModel = field(default=None, init=False)
    _feature_store: ParquetFeatureStore = field(default=None, init=False)
    _feature_engineer: CTRFeatureEngineer | None = field(default=None, init=False)
    _feature_columns: list[str] = field(default_factory=list, init=False)
    _is_loaded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.feature_store_path = Path(self.feature_store_path)
        if self.feature_engineer_path:
            self.feature_engineer_path = Path(self.feature_engineer_path)

    def load(self) -> "PredictionPipeline":
        """Load model and feature store from disk."""
        self._model = XGBoostEnsembleModel()
        self._model.load(self.model_path)
        
        self._feature_store = ParquetFeatureStore(self.feature_store_path)
        
        # Try to load feature columns from feature store
        feature_sets = self._feature_store.list_feature_sets()
        if feature_sets:
            self._feature_columns = self._feature_store.get_feature_columns(feature_sets[0])
        
        # Load feature engineer if path provided
        if self.feature_engineer_path and self.feature_engineer_path.exists():
            self._feature_engineer = CTRFeatureEngineer()
            self._feature_engineer.load(self.feature_engineer_path)
            if not self._feature_columns:
                self._feature_columns = self._feature_engineer.get_feature_columns()
        
        self._is_loaded = True
        return self

    def predict(
        self,
        publication_id: str,
        campaign_id: str,
        feature_set_name: str = "prediction_features",
    ) -> PredictionResult:
        """Generate CTR prediction for a publication-campaign pair.
        
        Looks up pre-computed features from the feature store and
        generates a prediction using the loaded model.
        
        Args:
            publication_id: Publication identifier
            campaign_id: Campaign identifier
            feature_set_name: Which feature set to query in the store
        
        Returns:
            PredictionResult with predicted CTR
        
        Raises:
            RuntimeError: If pipeline not loaded
            ValueError: If features not found for the given pair
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline must be loaded before prediction")
        
        feature_vector = self._feature_store.get_features_for_prediction(
            publication_id, campaign_id, feature_set_name
        )
        
        if feature_vector is None:
            raise ValueError(
                f"No features found for publication_id={publication_id}, "
                f"campaign_id={campaign_id}"
            )
        
        predicted_ctr = self._model.predict_single(
            feature_vector.features,
            approved_opens=feature_vector.approved_opens,
        )
        
        return PredictionResult(
            publication_id=publication_id,
            campaign_id=campaign_id,
            predicted_ctr=predicted_ctr,
            model_version=self._model.model_name,
        )

    def predict_from_features(
        self,
        features: np.ndarray,
        publication_id: str,
        campaign_id: str,
        approved_opens: int | None = None,
    ) -> PredictionResult:
        """Generate prediction from raw feature array.
        
        Use this when features are computed on-the-fly rather than
        retrieved from the feature store.
        
        Args:
            features: Feature array matching expected feature columns
            publication_id: Publication identifier
            campaign_id: Campaign identifier
            approved_opens: Optional audience size for model selection
        
        Returns:
            PredictionResult with predicted CTR
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline must be loaded before prediction")
        
        predicted_ctr = self._model.predict_single(features, approved_opens)
        
        return PredictionResult(
            publication_id=publication_id,
            campaign_id=campaign_id,
            predicted_ctr=predicted_ctr,
            model_version=self._model.model_name,
        )

    def predict_batch(
        self,
        requests: list[tuple[str, str]],
        feature_set_name: str = "prediction_features",
    ) -> list[PredictionResult | None]:
        """Generate predictions for multiple publication-campaign pairs.
        
        Args:
            requests: List of (publication_id, campaign_id) tuples
            feature_set_name: Which feature set to query
        
        Returns:
            List of PredictionResult or None for each request
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline must be loaded before prediction")
        
        results = []
        for pub_id, camp_id in requests:
            try:
                result = self.predict(pub_id, camp_id, feature_set_name)
                results.append(result)
            except ValueError:
                results.append(None)
        
        return results

    def predict_batch_from_dataframe(
        self,
        df: pl.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> np.ndarray:
        """Generate predictions for a DataFrame of features.
        
        Efficient batch prediction when features are already computed.
        
        Args:
            df: DataFrame containing features and optionally approved_opens
            feature_columns: Columns to use as features (uses stored columns if None)
        
        Returns:
            Array of predicted CTR values
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline must be loaded before prediction")
        
        columns = feature_columns or self._feature_columns
        X = df.select(columns).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        opens = None
        if "approved_opens" in df.columns:
            opens = df["approved_opens"].to_numpy()
        
        return self._model.predict(X, opens)

    def get_feature_columns(self) -> list[str]:
        """Return expected feature column names."""
        return self._feature_columns.copy()


def create_prediction_pipeline(
    artifacts_dir: Path | None = None,
    config_path: Path | str | None = None,
) -> PredictionPipeline:
    """Create and load a prediction pipeline from training artifacts.
    
    Convenience function that creates a pipeline from the standard
    directory structure created by TrainingPipeline.
    
    Args:
        artifacts_dir: Path to directory containing model and features (overrides config)
        config_path: Path to YAML configuration file
    
    Returns:
        Loaded PredictionPipeline ready for inference
    """
    if config_path is not None:
        config = load_config(config_path)
        model_path = config.paths.model_dir
        feature_store_path = config.paths.feature_store_dir
        feature_engineer_path = config.paths.feature_engineer_path
    elif artifacts_dir is not None:
        artifacts_dir = Path(artifacts_dir)
        model_path = artifacts_dir / "model"
        feature_store_path = artifacts_dir / "features"
        feature_engineer_path = artifacts_dir / "feature_engineer.pkl"
    else:
        raise ValueError("Either artifacts_dir or config_path must be provided")
    
    return PredictionPipeline(
        model_path=model_path,
        feature_store_path=feature_store_path,
        feature_engineer_path=feature_engineer_path,
    ).load()


def create_prediction_pipeline_from_config(
    config_path: Path | str = "pipeline_config.yml",
) -> PredictionPipeline:
    """Create prediction pipeline using configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Loaded PredictionPipeline ready for inference
    """
    return create_prediction_pipeline(config_path=config_path)
