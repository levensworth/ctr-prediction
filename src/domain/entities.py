"""Domain entities for CTR prediction pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PlacementRecord:
    """Represents a single ad placement record."""
    placement_id: str
    publication_id: str
    campaign_id: str
    post_send_at: datetime
    approved_opens: int
    approved_clicks: int

    @property
    def ctr(self) -> float:
        if self.approved_opens <= 0:
            return 0.0
        return min(self.approved_clicks / self.approved_opens, 1.0)


@dataclass(frozen=True)
class CampaignRecord:
    """Represents campaign metadata."""
    campaign_id: str
    advertiser_id: str
    target_gender: str | None
    promoted_item: str | None
    target_incomes: str | None
    target_ages: str | None


@dataclass(frozen=True)
class PublicationTagRecord:
    """Represents publication tags for content classification."""
    publication_id: str
    tags: str


@dataclass
class FeatureVector:
    """Container for feature data used in model training/prediction."""
    features: np.ndarray
    feature_names: list[str]
    publication_id: str | None = None
    campaign_id: str | None = None
    approved_opens: int | None = None

    def __post_init__(self) -> None:
        if len(self.features) != len(self.feature_names):
            raise ValueError(
                f"Feature array length ({len(self.features)}) must match "
                f"feature names length ({len(self.feature_names)})"
            )


@dataclass
class PredictionResult:
    """Result of a CTR prediction."""
    publication_id: str
    campaign_id: str
    predicted_ctr: float
    confidence_interval: tuple[float, float] | None = None
    model_version: str | None = None


@dataclass
class EvaluationResult:
    """Comprehensive evaluation results for a model."""
    metrics: dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    model_name: str
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Evaluation Results for: {self.model_name}",
            f"Timestamp: {self.evaluation_timestamp}",
            "-" * 50,
        ]
        for metric_name, value in self.metrics.items():
            lines.append(f"{metric_name}: {value:.6f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model hyperparameters."""
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 200
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    objective: str = "reg:squarederror"


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training pipeline."""
    test_split_days: int = 90
    rolling_window_days: int = 90
    audience_threshold: int = 1000
    tfidf_max_features: int = 50
    tfidf_min_df: int = 5


@dataclass
class ImputationStatistics:
    """Statistics for imputing missing feature values.
    
    For unseen publication_id or campaign_id, we use:
    - mean for numerical features
    - mode for categorical (one-hot encoded) features
    """
    numerical_means: dict[str, float]
    categorical_modes: dict[str, int]
    
    def get_imputed_value(self, feature_name: str) -> float | int:
        """Get the imputed value for a feature."""
        if feature_name in self.numerical_means:
            return self.numerical_means[feature_name]
        if feature_name in self.categorical_modes:
            return self.categorical_modes[feature_name]
        return 0.0
