"""Evaluation pipeline for CTR models.

Provides comprehensive model evaluation including:
- Standard metrics (MAE, RMSE, R2)
- Segmented analysis by audience size
- Comparison with baseline
- Custom metric support
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


from src.domain.entities import EvaluationResult
from src.domain.protocols import IMetric
from src.feature_store.feature_store import ParquetFeatureStore
from src.features.feature_engineering import prepare_model_data
from src.metrics.metrics import (
    compute_all_metrics,
    compute_baseline_metrics,
    analyze_mae_by_opens_bins,
    create_standard_metrics,
)
from src.models.xgboost_ensemble import XGBoostEnsembleModel
from src.pipelines.config import load_config


@dataclass
class EvaluationPipeline:
    """Pipeline for comprehensive model evaluation.
    
    Evaluates model performance using specified metrics and
    generates detailed analysis including segmentation by audience size.
    """
    
    model_path: Path | None = None
    feature_store_path: Path | None = None
    
    _model: XGBoostEnsembleModel | None = field(default=None, init=False)
    _feature_store: ParquetFeatureStore | None = field(default=None, init=False)
    _is_loaded: bool = field(default=False, init=False)

    def load(
        self,
        model: XGBoostEnsembleModel | None = None,
    ) -> "EvaluationPipeline":
        """Load or set model for evaluation.
        
        Args:
            model: Optional pre-loaded model. If None, loads from model_path.
        
        Returns:
            self for method chaining
        """
        if model is not None:
            self._model = model
        elif self.model_path is not None:
            self._model = XGBoostEnsembleModel()
            self._model.load(Path(self.model_path))
        else:
            raise ValueError("Either model or model_path must be provided")
        
        if self.feature_store_path is not None:
            self._feature_store = ParquetFeatureStore(Path(self.feature_store_path))
        
        self._is_loaded = True
        return self

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        opens: np.ndarray,
        metrics: list[IMetric] | None = None,
        y_train: np.ndarray | None = None,
    ) -> EvaluationResult:
        """Evaluate model using specified metrics.
        
        Args:
            X: Feature matrix
            y_true: Actual CTR values
            opens: Approved opens for each sample
            metrics: List of metrics to compute (uses standard if None)
            y_train: Optional training targets for baseline comparison
        
        Returns:
            EvaluationResult with all computed metrics
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline must be loaded before evaluation")
        
        if metrics is None:
            metrics = create_standard_metrics()
        
        y_pred = self._model.predict(X, opens)
        
        # Compute main metrics
        metrics_results = compute_all_metrics(y_true, y_pred, metrics)
        
        # Add baseline comparison if training data provided
        if y_train is not None:
            baseline = compute_baseline_metrics(y_true, y_train)
            metrics_results.update(baseline)
            
            if baseline["baseline_mae"] > 0:
                improvement = (baseline["baseline_mae"] - metrics_results["mae"]) / baseline["baseline_mae"] * 100
                metrics_results["mae_improvement_pct"] = improvement
        
        return EvaluationResult(
            metrics=metrics_results,
            predictions=y_pred,
            actuals=y_true,
            model_name=self._model.model_name,
            metadata={
                "n_samples": len(y_true),
                "audience_threshold": self._model.audience_threshold,
            }
        )

    def evaluate_from_feature_store(
        self,
        feature_set_name: str,
        feature_columns: list[str],
        metrics: list[IMetric] | None = None,
    ) -> EvaluationResult:
        """Evaluate using features from the feature store.
        
        Args:
            feature_set_name: Name of feature set to load
            feature_columns: List of feature column names
            metrics: Optional list of metrics
        
        Returns:
            EvaluationResult with computed metrics
        """
        if self._feature_store is None:
            raise ValueError("Feature store path must be set for this method")
        
        features_df = self._feature_store.load_features(feature_set_name)
        X, y_true, _, opens = prepare_model_data(features_df, feature_columns)
        
        return self.evaluate(X, y_true, opens, metrics)

    def evaluate_by_audience_segment(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        opens: np.ndarray,
        metrics: list[IMetric] | None = None,
    ) -> dict[str, EvaluationResult]:
        """Evaluate separately for small and large audience segments.
        
        Args:
            X: Feature matrix
            y_true: Actual CTR values
            opens: Approved opens for each sample
            metrics: Optional list of metrics
        
        Returns:
            Dictionary with "small", "large", and "combined" evaluation results
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline must be loaded before evaluation")
        
        threshold = self._model.audience_threshold
        
        small_mask = opens < threshold
        large_mask = ~small_mask
        
        results = {}
        
        # Evaluate small audience
        if np.any(small_mask):
            X_small, y_small, opens_small = X[small_mask], y_true[small_mask], opens[small_mask]
            y_pred_small = self._model.predict(X_small, opens_small)
            metrics_small = compute_all_metrics(y_small, y_pred_small, metrics)
            
            results["small"] = EvaluationResult(
                metrics=metrics_small,
                predictions=y_pred_small,
                actuals=y_small,
                model_name=f"{self._model.model_name}_small",
                metadata={"segment": "small", "n_samples": int(np.sum(small_mask))}
            )
        
        # Evaluate large audience
        if np.any(large_mask):
            X_large, y_large, opens_large = X[large_mask], y_true[large_mask], opens[large_mask]
            y_pred_large = self._model.predict(X_large, opens_large)
            metrics_large = compute_all_metrics(y_large, y_pred_large, metrics)
            
            results["large"] = EvaluationResult(
                metrics=metrics_large,
                predictions=y_pred_large,
                actuals=y_large,
                model_name=f"{self._model.model_name}_large",
                metadata={"segment": "large", "n_samples": int(np.sum(large_mask))}
            )
        
        # Combined evaluation
        results["combined"] = self.evaluate(X, y_true, opens, metrics)
        
        return results

    def analyze_by_opens_bins(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        opens: np.ndarray,
    ) -> list[dict]:
        """Analyze MAE across standard opens bins.
        
        Returns:
            List of dictionaries with bin analysis results
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline must be loaded before evaluation")
        
        y_pred = self._model.predict(X, opens)
        return analyze_mae_by_opens_bins(y_true, y_pred, opens)

    def get_feature_importance_analysis(self) -> dict[str, Any]:
        """Get feature importance analysis for both models.
        
        Returns:
            Dictionary with importance rankings for each model
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline must be loaded for importance analysis")
        
        return {
            "small_audience": self._model.get_top_features(15, "small"),
            "large_audience": self._model.get_top_features(15, "large"),
            "combined": self._model.get_top_features(15, "combined"),
        }

    def generate_report(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        opens: np.ndarray,
        y_train: np.ndarray | None = None,
    ) -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            X: Feature matrix
            y_true: Actual CTR values
            opens: Approved opens
            y_train: Optional training targets for baseline
        
        Returns:
            Formatted report string
        """
        segment_results = self.evaluate_by_audience_segment(X, y_true, opens)
        bins_analysis = self.analyze_by_opens_bins(X, y_true, opens)
        importance = self.get_feature_importance_analysis()
        
        lines = [
            "=" * 70,
            "CTR MODEL EVALUATION REPORT",
            f"Model: {self._model.model_name}",
            f"Timestamp: {datetime.now().isoformat()}",
            "=" * 70,
            "",
            "## Overall Performance",
            "-" * 50,
        ]
        
        combined = segment_results["combined"]
        for metric_name, value in combined.metrics.items():
            lines.append(f"{metric_name}: {value:.6f}")
        
        lines.extend([
            "",
            "## Performance by Audience Segment",
            "-" * 50,
        ])
        
        if "small" in segment_results:
            small = segment_results["small"]
            lines.append(f"\nSmall Audience (<{self._model.audience_threshold} opens):")
            lines.append(f"  Samples: {small.metadata['n_samples']:,}")
            lines.append(f"  MAE: {small.metrics.get('mae', 0):.6f}")
            lines.append(f"  R2: {small.metrics.get('r2', 0):.6f}")
        
        if "large" in segment_results:
            large = segment_results["large"]
            lines.append(f"\nLarge Audience (>={self._model.audience_threshold} opens):")
            lines.append(f"  Samples: {large.metadata['n_samples']:,}")
            lines.append(f"  MAE: {large.metrics.get('mae', 0):.6f}")
            lines.append(f"  R2: {large.metrics.get('r2', 0):.6f}")
        
        lines.extend([
            "",
            "## MAE by Opens Bins",
            "-" * 50,
            f"{'Bin':<20} {'Count':>12} {'MAE':>12} {'Mean CTR':>12}",
        ])
        
        for row in bins_analysis:
            lines.append(
                f"{row['label']:<20} {row['count']:>12,} {row['mae']:>12.6f} {row['mean_ctr']:>12.6f}"
            )
        
        lines.extend([
            "",
            "## Top 10 Features",
            "-" * 50,
        ])
        
        for rank, (name, imp) in enumerate(importance["combined"][:10], 1):
            lines.append(f"{rank:2d}. {name}: {imp:.4f}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def run_evaluation(
    artifacts_dir: Path | None = None,
    test_feature_set: str | None = None,
    config_path: Path | str | None = None,
) -> EvaluationResult:
    """Convenience function to run evaluation from saved artifacts.
    
    Args:
        artifacts_dir: Path to training artifacts directory (overrides config)
        test_feature_set: Name of test features in feature store (overrides config)
        config_path: Path to YAML configuration file
    
    Returns:
        EvaluationResult
    """
    if config_path is not None:
        config = load_config(config_path)
        model_path = config.paths.model_dir
        feature_store_path = config.paths.feature_store_dir
        feature_set = test_feature_set or config.evaluation.test_feature_set
    elif artifacts_dir is not None:
        artifacts_dir = Path(artifacts_dir)
        model_path = artifacts_dir / "model"
        feature_store_path = artifacts_dir / "features"
        feature_set = test_feature_set or "test_features"
    else:
        raise ValueError("Either artifacts_dir or config_path must be provided")
    
    pipeline = EvaluationPipeline(
        model_path=model_path,
        feature_store_path=feature_store_path,
    ).load()
    
    feature_store = ParquetFeatureStore(feature_store_path)
    feature_columns = feature_store.get_feature_columns(feature_set)
    
    return pipeline.evaluate_from_feature_store(feature_set, feature_columns)


def run_evaluation_from_config(
    config_path: Path | str = "pipeline_config.yml",
) -> EvaluationResult:
    """Run evaluation using configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        EvaluationResult
    """
    return run_evaluation(config_path=config_path)
