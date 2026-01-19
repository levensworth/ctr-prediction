"""Protocol interfaces for CTR prediction pipeline components."""

from typing import Protocol, runtime_checkable, Any
from pathlib import Path

import numpy as np
import polars as pl

from .entities import (
    EvaluationResult,
    FeatureVector,
    ModelConfig,
    PredictionResult,
    TrainingConfig,
)


@runtime_checkable
class IModel(Protocol):
    """Interface for CTR prediction models.
    
    Implementations can include XGBoost ensembles, neural networks (two-tower), etc.
    """
    
    @property
    def model_name(self) -> str:
        """Return the model's identifier."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been trained."""
        ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """Train the model on provided data."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input features."""
        ...

    def save(self, path: Path) -> None:
        """Persist the model to disk."""
        ...

    def load(self, path: Path) -> None:
        """Load a model from disk."""
        ...

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""
        ...


@runtime_checkable
class IFeatureStore(Protocol):
    """Interface for feature storage and retrieval.
    
    Supports storing and loading features in parquet format.
    """

    def save_features(
        self,
        features_df: pl.DataFrame,
        feature_set_name: str,
    ) -> Path:
        """Save a feature set to parquet storage."""
        ...

    def load_features(self, feature_set_name: str) -> pl.DataFrame:
        """Load a feature set from storage."""
        ...

    def get_features_for_prediction(
        self,
        publication_id: str,
        campaign_id: str,
    ) -> FeatureVector | None:
        """Retrieve pre-computed features for a prediction request."""
        ...

    def list_feature_sets(self) -> list[str]:
        """List all available feature sets."""
        ...


@runtime_checkable
class IFeatureEngineer(Protocol):
    """Interface for feature engineering transformations.
    
    Handles creation and transformation of features from raw data.
    """

    def fit(self, placements_df: pl.DataFrame, campaigns_df: pl.DataFrame, tags_df: pl.DataFrame) -> None:
        """Fit any stateful transformers (e.g., TF-IDF, scalers)."""
        ...

    def transform(self, placements_df: pl.DataFrame) -> pl.DataFrame:
        """Transform raw data into features."""
        ...

    def fit_transform(
        self,
        placements_df: pl.DataFrame,
        campaigns_df: pl.DataFrame,
        tags_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Fit transformers and transform data in one step."""
        ...

    def get_feature_columns(self) -> list[str]:
        """Return the list of feature column names produced."""
        ...

    def save(self, path: Path) -> None:
        """Persist the fitted transformers."""
        ...

    def load(self, path: Path) -> None:
        """Load previously fitted transformers."""
        ...


@runtime_checkable
class IMetric(Protocol):
    """Interface for evaluation metrics."""

    @property
    def name(self) -> str:
        """Return the metric's identifier."""
        ...

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric value."""
        ...


class ITrainingPipeline(Protocol):
    """Interface for training pipelines."""

    def run(
        self,
        placements_df: pl.DataFrame,
        campaigns_df: pl.DataFrame,
        tags_df: pl.DataFrame,
        config: TrainingConfig,
    ) -> tuple[IModel, EvaluationResult]:
        """Execute the training pipeline and return trained model with evaluation."""
        ...


class IPredictionPipeline(Protocol):
    """Interface for prediction pipelines."""

    def predict(
        self,
        publication_id: str,
        campaign_id: str,
    ) -> PredictionResult:
        """Generate a CTR prediction for a publication-campaign pair."""
        ...

    def predict_batch(
        self,
        requests: list[tuple[str, str]],
    ) -> list[PredictionResult]:
        """Generate predictions for multiple publication-campaign pairs."""
        ...


class IEvaluationPipeline(Protocol):
    """Interface for evaluation pipelines."""

    def evaluate(
        self,
        model: IModel,
        X: np.ndarray,
        y_true: np.ndarray,
        metrics: list[IMetric],
    ) -> EvaluationResult:
        """Evaluate a model using specified metrics."""
        ...
