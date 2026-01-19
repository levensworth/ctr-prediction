"""Domain layer: entities and protocols."""

from .entities import (
    PlacementRecord,
    CampaignRecord,
    PublicationTagRecord,
    FeatureVector,
    PredictionResult,
    EvaluationResult,
    ModelConfig,
    TrainingConfig,
)

from .protocols import (
    IModel,
    IFeatureStore,
    IFeatureEngineer,
    IMetric,
)

__all__ = [
    "PlacementRecord",
    "CampaignRecord", 
    "PublicationTagRecord",
    "FeatureVector",
    "PredictionResult",
    "EvaluationResult",
    "ModelConfig",
    "TrainingConfig",
    "IModel",
    "IFeatureStore",
    "IFeatureEngineer",
    "IMetric",
]
