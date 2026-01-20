"""Main entry point for CTR prediction pipeline.

Provides CLI interface for running training, prediction, and evaluation.

Usage:
    # Training with config file
    python -m src.main train --config pipeline_config.yml
    
    # Training with config file and overrides
    python -m src.main train --config pipeline_config.yml --max-depth 8
    
    # Training with explicit paths (no config)
    python -m src.main train --data-dir ./data --output-dir ./artifacts
    
    # Prediction
    python -m src.main predict --config pipeline_config.yml --pub-id PUB123 --camp-id CAMP456
    
    # Evaluation
    python -m src.main evaluate --config pipeline_config.yml
"""

import argparse
from pathlib import Path
import sys

from src.domain.entities import ModelConfig, TrainingConfig
from src.pipelines.config import load_config, get_default_config, PipelineConfig
from src.pipelines.training import TrainingPipeline, load_training_data
from src.pipelines.prediction import create_prediction_pipeline
from src.pipelines.evaluation import EvaluationPipeline
from src.feature_store.feature_store import ParquetFeatureStore
from src.features.feature_engineering import prepare_model_data


def _load_pipeline_config(config_path: str | None) -> PipelineConfig:
    """Load pipeline config from file or return defaults."""
    if config_path:
        return load_config(config_path)
    return get_default_config()


def train(args: argparse.Namespace) -> None:
    """Run the training pipeline."""
    config = _load_pipeline_config(args.config)
    
    data_dir = Path(args.data_dir) if args.data_dir else config.paths.data_dir
    output_dir = Path(args.output_dir) if args.output_dir else config.paths.output_dir
    
    print(f"Loading data from {data_dir}")
    placements, campaigns, tags, clusters = load_training_data(data_dir)
    
    print(f"Placements: {placements.shape}")
    print(f"Campaigns: {campaigns.shape}")
    print(f"Tags: {tags.shape}")
    if clusters is not None:
        print(f"Clusters: {clusters.shape}")
    
    # Use CLI args if provided, otherwise fall back to config
    model_config = ModelConfig(
        max_depth=args.max_depth if args.max_depth is not None else config.model.max_depth,
        learning_rate=args.learning_rate if args.learning_rate is not None else config.model.learning_rate,
        n_estimators=args.n_estimators if args.n_estimators is not None else config.model.n_estimators,
        subsample=config.model.subsample,
        colsample_bytree=config.model.colsample_bytree,
        min_child_weight=config.model.min_child_weight,
        reg_alpha=config.model.reg_alpha,
        reg_lambda=config.model.reg_lambda,
        random_state=config.model.random_state,
        objective=config.model.objective,
    )
    
    training_config = TrainingConfig(
        test_split_days=args.test_split_days if args.test_split_days is not None else config.training.test_split_days,
        rolling_window_days=args.rolling_window_days if args.rolling_window_days is not None else config.training.rolling_window_days,
        audience_threshold=args.audience_threshold if args.audience_threshold is not None else config.training.audience_threshold,
        tfidf_max_features=config.training.tfidf_max_features,
        tfidf_min_df=config.training.tfidf_min_df,
    )
    
    print("\nStarting training pipeline...")
    print(f"Model config: max_depth={model_config.max_depth}, lr={model_config.learning_rate}, n_estimators={model_config.n_estimators}")
    print(f"Training config: test_split_days={training_config.test_split_days}, audience_threshold={training_config.audience_threshold}")
    
    pipeline = TrainingPipeline(
        output_dir=output_dir,
        model_config=model_config,
        training_config=training_config,
    )
    
    model, eval_result = pipeline.run(placements, campaigns, tags, clusters)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(eval_result.summary())
    print(f"\nArtifacts saved to: {output_dir}")


def predict(args: argparse.Namespace) -> None:
    """Run prediction for a single publication-campaign pair."""
    config = _load_pipeline_config(args.config)
    
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else config.paths.output_dir
    feature_set = args.feature_set if args.feature_set else config.prediction.default_feature_set
    
    print(f"Loading model from {artifacts_dir}")
    pipeline = create_prediction_pipeline(artifacts_dir=artifacts_dir)
    
    # Use imputation mode if requested or as fallback
    if args.use_imputation:
        result, pub_imputed, camp_imputed = pipeline.predict_with_imputation(
            args.publication_id,
            args.campaign_id,
        )
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULT (with imputation)")
        print("=" * 60)
        print(f"Publication ID: {result.publication_id}")
        print(f"Campaign ID: {result.campaign_id}")
        print(f"Predicted CTR: {result.predicted_ctr:.6f}")
        print(f"Model Version: {result.model_version}")
        if pub_imputed:
            print("Note: Publisher features were imputed (unseen publication_id)")
        if camp_imputed:
            print("Note: Campaign features were imputed (unseen campaign_id)")
    else:
        try:
            result = pipeline.predict(
                args.publication_id,
                args.campaign_id,
                feature_set,
            )
            
            print("\n" + "=" * 60)
            print("PREDICTION RESULT")
            print("=" * 60)
            print(f"Publication ID: {result.publication_id}")
            print(f"Campaign ID: {result.campaign_id}")
            print(f"Predicted CTR: {result.predicted_ctr:.6f}")
            print(f"Model Version: {result.model_version}")
            
        except ValueError as e:
            print(f"Error: {e}")
            print("Tip: Use --use-imputation to make predictions for unseen combinations")
            sys.exit(1)


def evaluate(args: argparse.Namespace) -> None:
    """Run evaluation on test data."""
    config = _load_pipeline_config(args.config)
    
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else config.paths.output_dir
    feature_set = args.feature_set if args.feature_set else config.evaluation.test_feature_set
    
    print(f"Loading artifacts from {artifacts_dir}")
    
    pipeline = EvaluationPipeline(
        model_path=artifacts_dir / "model",
        feature_store_path=artifacts_dir / "features",
    ).load()
    
    feature_store = ParquetFeatureStore(artifacts_dir / "features")
    features_df = feature_store.load_features(feature_set)
    feature_columns = feature_store.get_feature_columns(feature_set)
    
    X, y_true, _, opens = prepare_model_data(features_df, feature_columns)
    
    report = pipeline.generate_report(X, y_true, opens)
    print(report)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CTR Prediction Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training subcommand
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--config", type=str, help="Path to YAML config file")
    train_parser.add_argument("--data-dir", type=str, help="Path to data directory (overrides config)")
    train_parser.add_argument("--output-dir", type=str, help="Path to save artifacts (overrides config)")
    train_parser.add_argument("--max-depth", type=int, help="XGBoost max depth (overrides config)")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate (overrides config)")
    train_parser.add_argument("--n-estimators", type=int, help="Number of boosting rounds (overrides config)")
    train_parser.add_argument("--test-split-days", type=int, help="Days for test split (overrides config)")
    train_parser.add_argument("--rolling-window-days", type=int, help="Rolling window for features (overrides config)")
    train_parser.add_argument("--audience-threshold", type=int, help="Audience split threshold (overrides config)")
    
    # Prediction subcommand
    predict_parser = subparsers.add_parser("predict", help="Make a prediction")
    predict_parser.add_argument("--config", type=str, help="Path to YAML config file")
    predict_parser.add_argument("--artifacts-dir", type=str, help="Path to artifacts (overrides config)")
    predict_parser.add_argument("--publication-id", type=str, required=True, help="Publication ID")
    predict_parser.add_argument("--campaign-id", type=str, required=True, help="Campaign ID")
    predict_parser.add_argument("--feature-set", type=str, help="Feature set name (overrides config)")
    predict_parser.add_argument("--use-imputation", action="store_true", help="Use imputation for unseen publication/campaign IDs")
    
    # Evaluation subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--config", type=str, help="Path to YAML config file")
    eval_parser.add_argument("--artifacts-dir", type=str, help="Path to artifacts (overrides config)")
    eval_parser.add_argument("--feature-set", type=str, help="Feature set to evaluate (overrides config)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
