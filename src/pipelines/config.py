"""Configuration loader for CTR prediction pipelines.

Provides typed configuration loading from YAML files with
sensible defaults and validation using Pydantic.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from src.domain.entities import ModelConfig
from src.domain.entities import TrainingConfig


class PathsConfig(BaseModel):
    """Configuration for file system paths."""
    
    model_config = {"frozen": True}
    
    data_dir: Path = Field(default=Path("data"))
    output_dir: Path = Field(default=Path("artifacts"))
    model_dir: Path = Field(default=Path("artifacts/model"))
    feature_store_dir: Path = Field(default=Path("artifacts/features"))
    feature_engineer_path: Path = Field(default=Path("artifacts/feature_engineer.pkl"))


class ModelHyperparamsConfig(BaseModel):
    """Configuration for model hyperparameters."""
    
    model_config = {"frozen": True}
    
    max_depth: int = Field(default=6, ge=1)
    learning_rate: float = Field(default=0.1, gt=0)
    n_estimators: int = Field(default=200, ge=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    min_child_weight: int = Field(default=5, ge=1)
    reg_alpha: float = Field(default=0.1, ge=0)
    reg_lambda: float = Field(default=1.0, ge=0)
    random_state: int = Field(default=42)
    objective: str = Field(default="reg:squarederror")


class TrainingParamsConfig(BaseModel):
    """Configuration for training pipeline."""
    
    model_config = {"frozen": True}
    
    test_split_days: int = Field(default=90, ge=1)
    rolling_window_days: int = Field(default=90, ge=1)
    audience_threshold: int = Field(default=1000, ge=1)
    tfidf_max_features: int = Field(default=50, ge=1)
    tfidf_min_df: int = Field(default=5, ge=1)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation pipeline."""
    
    model_config = {"frozen": True}
    
    test_feature_set: str = Field(default="test_features")
    train_feature_set: str = Field(default="train_features")


class PredictionConfig(BaseModel):
    """Configuration for prediction pipeline."""
    
    model_config = {"frozen": True}
    
    default_feature_set: str = Field(default="prediction_features")


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    
    model_config = {"frozen": True}
    
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelHyperparamsConfig = Field(default_factory=ModelHyperparamsConfig)
    training: TrainingParamsConfig = Field(default_factory=TrainingParamsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)

    def with_base_path(self, base_path: Path) -> "PipelineConfig":
        """Return a new config with paths resolved against base_path."""
        def resolve(p: Path) -> Path:
            if not p.is_absolute():
                return base_path / p
            return p

        resolved_paths = PathsConfig(
            data_dir=resolve(self.paths.data_dir),
            output_dir=resolve(self.paths.output_dir),
            model_dir=resolve(self.paths.model_dir),
            feature_store_dir=resolve(self.paths.feature_store_dir),
            feature_engineer_path=resolve(self.paths.feature_engineer_path),
        )
        
        return PipelineConfig(
            paths=resolved_paths,
            model=self.model,
            training=self.training,
            evaluation=self.evaluation,
            prediction=self.prediction,
        )

    def to_domain_model_config(self) -> ModelConfig:
        """Convert to domain ModelConfig entity."""
        
        return ModelConfig(
            max_depth=self.model.max_depth,
            learning_rate=self.model.learning_rate,
            n_estimators=self.model.n_estimators,
            subsample=self.model.subsample,
            colsample_bytree=self.model.colsample_bytree,
            min_child_weight=self.model.min_child_weight,
            reg_alpha=self.model.reg_alpha,
            reg_lambda=self.model.reg_lambda,
            random_state=self.model.random_state,
            objective=self.model.objective,
        )

    def to_domain_training_config(self) -> TrainingConfig:
        """Convert to domain TrainingConfig entity."""
        return TrainingConfig(
            test_split_days=self.training.test_split_days,
            rolling_window_days=self.training.rolling_window_days,
            audience_threshold=self.training.audience_threshold,
            tfidf_max_features=self.training.tfidf_max_features,
            tfidf_min_df=self.training.tfidf_min_df,
        )


def load_config(config_path: Path | str, base_path: Path | None = None) -> PipelineConfig:
    """Load pipeline configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        base_path: Optional base path for resolving relative paths.
                   Defaults to the parent directory of the config file.
    
    Returns:
        PipelineConfig with all settings loaded
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        pydantic.ValidationError: If configuration validation fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if base_path is None:
        base_path = config_path.parent
    
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    
    config = PipelineConfig.model_validate(data)
    return config.with_base_path(base_path)


def get_default_config(base_path: Path | None = None) -> PipelineConfig:
    """Get default configuration without loading from file.
    
    Args:
        base_path: Optional base path for resolving relative paths.
    
    Returns:
        PipelineConfig with all default values
    """
    config = PipelineConfig()
    if base_path:
        return config.with_base_path(base_path)
    return config
