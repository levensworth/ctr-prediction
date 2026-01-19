"""Metric implementations for CTR model evaluation.

Provides standard metrics (MAE, RMSE, R2) and custom metrics for
evaluating CTR prediction models.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class MAEMetric:
    """Mean Absolute Error metric."""
    
    @property
    def name(self) -> str:
        return "mae"
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(mean_absolute_error(y_true, y_pred))


@dataclass(frozen=True)
class RMSEMetric:
    """Root Mean Squared Error metric."""
    
    @property
    def name(self) -> str:
        return "rmse"
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass(frozen=True)
class R2Metric:
    """R-squared (coefficient of determination) metric."""
    
    @property
    def name(self) -> str:
        return "r2"
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(r2_score(y_true, y_pred))


@dataclass(frozen=True)
class MAPEMetric:
    """Mean Absolute Percentage Error metric.
    
    Note: Handles zero values in y_true by excluding them from calculation.
    """
    
    @property
    def name(self) -> str:
        return "mape"
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true != 0
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


@dataclass(frozen=True)
class MedianAbsoluteErrorMetric:
    """Median Absolute Error metric.
    
    More robust to outliers than MAE.
    """
    
    @property
    def name(self) -> str:
        return "median_ae"
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.median(np.abs(y_true - y_pred)))


@dataclass
class WeightedMAEMetric:
    """Weighted Mean Absolute Error metric.
    
    Allows weighting errors by sample importance (e.g., by audience size).
    """
    
    weights: np.ndarray | None = None
    
    @property
    def name(self) -> str:
        return "weighted_mae"
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        errors = np.abs(y_true - y_pred)
        if self.weights is not None and len(self.weights) == len(errors):
            return float(np.average(errors, weights=self.weights))
        return float(np.mean(errors))


@dataclass
class CustomMetric:
    """Custom metric wrapper for user-defined metric functions.
    
    Allows users to define custom evaluation metrics.
    """
    
    metric_name: str
    compute_fn: Callable[[np.ndarray, np.ndarray], float]
    
    @property
    def name(self) -> str:
        return self.metric_name
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(self.compute_fn(y_true, y_pred))


def create_standard_metrics() -> list:
    """Create a list of standard evaluation metrics."""
    return [
        MAEMetric(),
        RMSEMetric(),
        R2Metric(),
        MAPEMetric(),
        MedianAbsoluteErrorMetric(),
    ]


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list | None = None,
) -> dict[str, float]:
    """Compute all specified metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        metrics: List of metric instances. If None, uses standard metrics.
    
    Returns:
        Dictionary mapping metric names to computed values
    """
    if metrics is None:
        metrics = create_standard_metrics()
    
    return {metric.name: metric.compute(y_true, y_pred) for metric in metrics}


def compute_metrics_by_bin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: np.ndarray,
    bin_labels: list[str],
    metrics: list | None = None,
) -> dict[str, dict[str, float]]:
    """Compute metrics for each bin separately.
    
    Useful for analyzing performance across different audience sizes or CTR ranges.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        bins: Array of bin indices for each sample
        bin_labels: Labels for each bin
        metrics: List of metric instances
    
    Returns:
        Nested dictionary: bin_label -> metric_name -> value
    """
    if metrics is None:
        metrics = create_standard_metrics()
    
    results = {}
    unique_bins = np.unique(bins)
    
    for bin_idx in unique_bins:
        mask = bins == bin_idx
        label = bin_labels[bin_idx] if bin_idx < len(bin_labels) else f"bin_{bin_idx}"
        
        if np.sum(mask) > 0:
            results[label] = compute_all_metrics(y_true[mask], y_pred[mask], metrics)
            results[label]["count"] = int(np.sum(mask))
        else:
            results[label] = {m.name: 0.0 for m in metrics}
            results[label]["count"] = 0
    
    return results


def analyze_mae_by_opens_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    opens: np.ndarray,
) -> list[dict]:
    """Analyze MAE across different opens bins.
    
    Standard bins used in CTR analysis:
    - 0-100
    - 101-500
    - 501-1,000
    - 1,001-10,000
    - 10,001-100,000
    - 100,001+
    """
    bins_def = [
        (0, 100, "0-100"),
        (101, 500, "101-500"),
        (501, 1000, "501-1,000"),
        (1001, 10000, "1,001-10,000"),
        (10001, 100000, "10,001-100,000"),
        (100001, float("inf"), "100,001+"),
    ]
    
    results = []
    for bin_min, bin_max, label in bins_def:
        if bin_max == float("inf"):
            mask = opens >= bin_min
        else:
            mask = (opens >= bin_min) & (opens <= bin_max)
        
        count = int(np.sum(mask))
        if count > 0:
            errors = np.abs(y_true[mask] - y_pred[mask])
            mae = float(np.mean(errors))
            mean_ctr = float(np.mean(y_true[mask]))
            mean_pred = float(np.mean(y_pred[mask]))
        else:
            mae = mean_ctr = mean_pred = 0.0
        
        results.append({
            "label": label,
            "count": count,
            "mae": mae,
            "mean_ctr": mean_ctr,
            "mean_pred": mean_pred,
        })
    
    return results


def compute_baseline_metrics(
    y_test: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, float]:
    """Compute baseline metrics (predict mean strategy).
    
    Useful for comparison against trained model performance.
    """
    train_mean = float(np.mean(y_train))
    baseline_pred = np.full_like(y_test, train_mean)
    
    return {
        "baseline_mae": float(mean_absolute_error(y_test, baseline_pred)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
        "baseline_r2": float(r2_score(y_test, baseline_pred)),
        "train_mean": train_mean,
    }
