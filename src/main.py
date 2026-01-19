"""Main entry point for CTR prediction pipeline.

Provides CLI interface for running training, prediction, and evaluation.

Usage:
    # Training
    python -m src.main train --data-dir ./data --output-dir ./artifacts
    
    # Prediction
    python -m src.main predict --artifacts-dir ./artifacts --pub-id PUB123 --camp-id CAMP456
    
    # Evaluation
    python -m src.main evaluate --artifacts-dir ./artifacts
"""

import argparse
from pathlib import Path
import sys


from src.domain.entities import ModelConfig, TrainingConfig
from src.pipelines.training import TrainingPipeline, load_training_data
from src.pipelines.prediction import create_prediction_pipeline
from src.pipelines.evaluation import EvaluationPipeline
from src.feature_store.feature_store import ParquetFeatureStore
from src.features.feature_engineering import prepare_model_data


def train(args: argparse.Namespace) -> None:
    """Run the training pipeline."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Loading data from {data_dir}")
    placements, campaigns, tags, clusters = load_training_data(data_dir)
    
    print(f"Placements: {placements.shape}")
    print(f"Campaigns: {campaigns.shape}")
    print(f"Tags: {tags.shape}")
    if clusters is not None:
        print(f"Clusters: {clusters.shape}")
    
    model_config = ModelConfig(
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
    )
    
    training_config = TrainingConfig(
        test_split_days=args.test_split_days,
        rolling_window_days=args.rolling_window_days,
        audience_threshold=args.audience_threshold,
    )
    
    print("\nStarting training pipeline...")
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
    artifacts_dir = Path(args.artifacts_dir)
    
    print(f"Loading model from {artifacts_dir}")
    pipeline = create_prediction_pipeline(artifacts_dir)
    
    try:
        result = pipeline.predict(
            args.publication_id,
            args.campaign_id,
            args.feature_set,
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
        sys.exit(1)


def evaluate(args: argparse.Namespace) -> None:
    """Run evaluation on test data."""
    artifacts_dir = Path(args.artifacts_dir)
    
    print(f"Loading artifacts from {artifacts_dir}")
    
    pipeline = EvaluationPipeline(
        model_path=artifacts_dir / "model",
        feature_store_path=artifacts_dir / "features",
    ).load()
    
    feature_store = ParquetFeatureStore(artifacts_dir / "features")
    features_df = feature_store.load_features(args.feature_set)
    feature_columns = feature_store.get_feature_columns(args.feature_set)
    
    X, y_true, _, opens = prepare_model_data(features_df, feature_columns)
    
    report = pipeline.generate_report(X, y_true, opens)
    print(report)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CTR Prediction Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training subcommand
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    train_parser.add_argument("--output-dir", type=str, required=True, help="Path to save artifacts")
    train_parser.add_argument("--max-depth", type=int, default=6, help="XGBoost max depth")
    train_parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate")
    train_parser.add_argument("--n-estimators", type=int, default=200, help="Number of boosting rounds")
    train_parser.add_argument("--test-split-days", type=int, default=90, help="Days for test split")
    train_parser.add_argument("--rolling-window-days", type=int, default=90, help="Rolling window for features")
    train_parser.add_argument("--audience-threshold", type=int, default=1000, help="Audience split threshold")
    
    # Prediction subcommand
    predict_parser = subparsers.add_parser("predict", help="Make a prediction")
    predict_parser.add_argument("--artifacts-dir", type=str, required=True, help="Path to artifacts")
    predict_parser.add_argument("--publication-id", type=str, required=True, help="Publication ID")
    predict_parser.add_argument("--campaign-id", type=str, required=True, help="Campaign ID")
    predict_parser.add_argument("--feature-set", type=str, default="test_features", help="Feature set name")
    
    # Evaluation subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--artifacts-dir", type=str, required=True, help="Path to artifacts")
    eval_parser.add_argument("--feature-set", type=str, default="test_features", help="Feature set to evaluate")
    
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
