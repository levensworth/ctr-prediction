"""Training pipeline for CTR prediction models.

Orchestrates the complete training workflow:
1. Data loading and preprocessing
2. Feature engineering
3. Train/test split
4. Model training
5. Evaluation
6. Artifact persistence
"""

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from src.domain.entities import EvaluationResult, ModelConfig, TrainingConfig
from src.domain.protocols import IMetric
from src.features.feature_engineering import (
    CTRFeatureEngineer,
    create_temporal_split,
    prepare_model_data
)
from src.feature_store.feature_store import ParquetFeatureStore
from src.metrics.metrics import compute_all_metrics, create_standard_metrics, compute_baseline_metrics
from src.models.xgboost_ensemble import XGBoostEnsembleModel
from src.pipelines.config import PipelineConfig, load_config, get_default_config, PathsConfig

    

@dataclass
class TrainingPipeline:
    """Pipeline for training CTR prediction models.
    
    Handles the complete training workflow including feature engineering,
    data splitting, model training, and evaluation.
    """
    
    output_dir: Path
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    _feature_engineer: CTRFeatureEngineer | None = field(default=None, init=False)
    _feature_store: ParquetFeatureStore | None = field(default=None, init=False)
    _model: XGBoostEnsembleModel | None = field(default=None, init=False)
    
    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._feature_store = ParquetFeatureStore(self.output_dir / "features")

    def run(
        self,
        placements_df: pl.DataFrame,
        campaigns_df: pl.DataFrame,
        tags_df: pl.DataFrame,
        clusters_df: pl.DataFrame | None = None,
        metrics: list[IMetric] | None = None,
    ) -> tuple[XGBoostEnsembleModel, EvaluationResult]:
        """Execute the complete training pipeline.
        
        Args:
            placements_df: Raw placements data
            campaigns_df: Campaign metadata
            tags_df: Publication tags for TF-IDF
            clusters_df: Optional publication cluster assignments
            metrics: Optional list of metrics for evaluation
        
        Returns:
            Tuple of (trained model, evaluation results)
        """
        if metrics is None:
            metrics = create_standard_metrics()
        
        # Step 1: Feature Engineering
        self._feature_engineer = CTRFeatureEngineer(
            tfidf_max_features=self.training_config.tfidf_max_features,
            tfidf_min_df=self.training_config.tfidf_min_df,
            rolling_window_days=self.training_config.rolling_window_days,
        )
        
        features_df = self._feature_engineer.fit_transform(
            placements_df, campaigns_df, tags_df, clusters_df
        )
        
        feature_columns = self._feature_engineer.get_feature_columns()
        
        # Step 2: Temporal Train/Test Split
        train_features, test_features = create_temporal_split(
            features_df, self.training_config.test_split_days
        )
        
        # Step 3: Prepare training data
        X_train, y_train, weights_train, opens_train = prepare_model_data(
            train_features, feature_columns
        )
        
        X_test, y_test, _, opens_test = prepare_model_data(
            test_features, feature_columns
        )
        
        # Step 4: Train Model
        self._model = XGBoostEnsembleModel(
            config=self.model_config,
            audience_threshold=self.training_config.audience_threshold,
        )
        
        self._model.fit(
            X_train, y_train, opens_train,
            sample_weights=weights_train,
            feature_names=feature_columns,
        )
        
        # Step 5: Evaluate
        y_pred = self._model.predict(X_test, opens_test)
        
        metrics_results = compute_all_metrics(y_test, y_pred, metrics)
        baseline_results = compute_baseline_metrics(y_test, y_train)
        metrics_results.update(baseline_results)
        
        # Calculate improvement over baseline
        if baseline_results["baseline_mae"] > 0:
            improvement = (baseline_results["baseline_mae"] - metrics_results["mae"]) / baseline_results["baseline_mae"] * 100
            metrics_results["mae_improvement_pct"] = improvement
        
        evaluation_result = EvaluationResult(
            metrics=metrics_results,
            predictions=y_pred,
            actuals=y_test,
            model_name=self._model.model_name,
            metadata={
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_features": len(feature_columns),
                "training_config": {
                    "test_split_days": self.training_config.test_split_days,
                    "rolling_window_days": self.training_config.rolling_window_days,
                    "audience_threshold": self.training_config.audience_threshold,
                },
            }
        )
        
        # Step 6: Save Artifacts
        self._save_artifacts(train_features, test_features, feature_columns)
        
        return self._model, evaluation_result

    def _save_artifacts(
        self,
        train_features: pl.DataFrame,
        test_features: pl.DataFrame,
        feature_columns: list[str],
    ) -> None:
        """Save all training artifacts to disk."""
        # Save model
        model_path = self.output_dir / "model"
        self._model.save(model_path)
        
        # Save feature engineer
        fe_path = self.output_dir / "feature_engineer.pkl"
        self._feature_engineer.save(fe_path)
        
        # Save features to feature store
        self._feature_store.save_features(
            train_features, "train_features", feature_columns
        )
        self._feature_store.save_features(
            test_features, "test_features", feature_columns
        )
    
    def get_model(self) -> XGBoostEnsembleModel | None:
        """Return the trained model."""
        return self._model
    
    def get_feature_engineer(self) -> CTRFeatureEngineer | None:
        """Return the fitted feature engineer."""
        return self._feature_engineer


def load_training_data(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    """Load all training datasets from a data directory.
    
    Args:
        data_dir: Path to directory containing CSV files
    
    Returns:
        Tuple of (placements, campaigns, publication_tags, publication_clusters)
    """
    data_dir = Path(data_dir)
    
    placements = pl.read_csv(data_dir / "placements.csv")
    campaigns = pl.read_csv(data_dir / "campaigns.csv")
    publication_tags = pl.read_csv(data_dir / "publication_tags.csv")
    
    clusters_path = data_dir / "publication_clusters.csv"
    publication_clusters = None
    if clusters_path.exists():
        publication_clusters = pl.read_csv(clusters_path)
    
    return placements, campaigns, publication_tags, publication_clusters


def run_training(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    config_path: Path | str | None = None,
) -> tuple[XGBoostEnsembleModel, EvaluationResult]:
    """Convenience function to run the complete training pipeline.
    
    Args:
        data_dir: Path to data directory (overrides config if provided)
        output_dir: Path for saving outputs (overrides config if provided)
        model_config: Optional model configuration (overrides config if provided)
        training_config: Optional training configuration (overrides config if provided)
        config_path: Path to YAML configuration file
    
    Returns:
        Tuple of (trained model, evaluation results)
    """
    config = _resolve_config(config_path, data_dir, output_dir)
    
    final_model_config = model_config if model_config else config.to_domain_model_config()
    final_training_config = training_config if training_config else config.to_domain_training_config()
    
    placements, campaigns, tags, clusters = load_training_data(config.paths.data_dir)
    
    pipeline = TrainingPipeline(
        output_dir=config.paths.output_dir,
        model_config=final_model_config,
        training_config=final_training_config,
    )
    
    return pipeline.run(placements, campaigns, tags, clusters)


def run_training_from_config(
    config_path: Path | str = "pipeline_config.yml",
) -> tuple[XGBoostEnsembleModel, EvaluationResult]:
    """Run training pipeline using configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Tuple of (trained model, evaluation results)
    """
    return run_training(config_path=config_path)


def _resolve_config(
    config_path: Path | str | None,
    data_dir: Path | None,
    output_dir: Path | None,
) -> PipelineConfig:
    """Load configuration and apply path overrides."""
    
    
    if config_path is not None:
        config = load_config(config_path)
    else:
        config = get_default_config()
    
    if data_dir is not None or output_dir is not None:
        paths = PathsConfig(
            data_dir=data_dir if data_dir else config.paths.data_dir,
            output_dir=output_dir if output_dir else config.paths.output_dir,
            model_dir=config.paths.model_dir,
            feature_store_dir=config.paths.feature_store_dir,
            feature_engineer_path=config.paths.feature_engineer_path,
        )
        config = PipelineConfig(
            paths=paths,
            model=config.model,
            training=config.training,
            evaluation=config.evaluation,
            prediction=config.prediction,
        )
    
    return config
