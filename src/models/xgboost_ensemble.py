"""XGBoost 2-Fold Ensemble Model for CTR prediction.

Implements a two-model ensemble approach:
- Model 1: For small audiences (opens < threshold)
- Model 2: For large audiences (opens >= threshold)

This architecture captures different CTR dynamics across audience sizes.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any

import numpy as np
import xgboost as xgb

from src.domain.entities import ModelConfig


@dataclass
class XGBoostEnsembleModel:
    """XGBoost 2-Fold Ensemble for CTR prediction.
    
    Trains separate models for small and large audiences based on the
    approved_opens threshold. This captures different patterns in
    CTR behavior across audience sizes.
    """
    
    config: ModelConfig = field(default_factory=ModelConfig)
    audience_threshold: int = 1000
    
    _model_small: xgb.Booster | None = field(default=None, init=False)
    _model_large: xgb.Booster | None = field(default=None, init=False)
    _feature_names: list[str] | None = field(default=None, init=False)
    _is_fitted: bool = field(default=False, init=False)

    @property
    def model_name(self) -> str:
        return f"XGBoostEnsemble_threshold{self.audience_threshold}"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        opens: np.ndarray,
        sample_weights: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "XGBoostEnsembleModel":
        """Train the ensemble model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target CTR values (n_samples,)
            opens: Approved opens for each sample (used to split data)
            sample_weights: Optional sample weights
            feature_names: Optional list of feature names
        
        Returns:
            self for method chaining
        """
        self._feature_names = feature_names
        
        # Split data by audience size
        small_mask = opens < self.audience_threshold
        large_mask = ~small_mask
        
        X_small, y_small = X[small_mask], y[small_mask]
        X_large, y_large = X[large_mask], y[large_mask]
        
        weights_small = sample_weights[small_mask] if sample_weights is not None else None
        weights_large = sample_weights[large_mask] if sample_weights is not None else None
        
        self._model_small = self._train_single_model(
            X_small, y_small, weights_small, "small_audience"
        )
        self._model_large = self._train_single_model(
            X_large, y_large, weights_large, "large_audience"
        )
        
        self._is_fitted = True
        return self

    def _train_single_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None,
        model_name: str,
    ) -> xgb.Booster:
        """Train a single XGBoost model."""
        dtrain = xgb.DMatrix(X, label=y, weight=weights)
        
        params = {
            "objective": self.config.objective,
            "eval_metric": ["mae", "rmse"],
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "min_child_weight": self.config.min_child_weight,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "seed": self.config.random_state,
            "n_jobs": -1,
        }
        
        return xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.n_estimators,
            verbose_eval=False
        )

    def predict(
        self,
        X: np.ndarray,
        opens: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate predictions using the ensemble.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            opens: Optional approved opens for routing to appropriate model.
                   If None, uses large audience model for all samples.
        
        Returns:
            Predicted CTR values clipped to [0, 1]
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        predictions = np.zeros(X.shape[0])
        
        if opens is None:
            # Default to large audience model
            dmatrix = xgb.DMatrix(X)
            predictions = self._model_large.predict(dmatrix)
        else:
            small_mask = opens < self.audience_threshold
            large_mask = ~small_mask
            
            if np.any(small_mask):
                dmatrix_small = xgb.DMatrix(X[small_mask])
                predictions[small_mask] = self._model_small.predict(dmatrix_small)
            
            if np.any(large_mask):
                dmatrix_large = xgb.DMatrix(X[large_mask])
                predictions[large_mask] = self._model_large.predict(dmatrix_large)
        
        return np.clip(predictions, 0.0, 1.0)

    def predict_single(
        self,
        X: np.ndarray,
        approved_opens: int | None = None,
    ) -> float:
        """Predict CTR for a single sample.
        
        Args:
            X: Feature vector (n_features,) or (1, n_features)
            approved_opens: Optional opens count for model selection
        
        Returns:
            Predicted CTR value
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        opens = np.array([approved_opens]) if approved_opens is not None else None
        return float(self.predict(X, opens)[0])

    def save(self, path: Path) -> None:
        """Save the ensemble model to disk.
        
        Creates:
            - path/small_audience.json: Small audience XGBoost model
            - path/large_audience.json: Large audience XGBoost model
            - path/config.json: Model configuration and metadata
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self._model_small.save_model(str(path / "small_audience.json"))
        self._model_large.save_model(str(path / "large_audience.json"))
        
        config_data = {
            "model_name": self.model_name,
            "audience_threshold": self.audience_threshold,
            "config": {
                "max_depth": self.config.max_depth,
                "learning_rate": self.config.learning_rate,
                "n_estimators": self.config.n_estimators,
                "subsample": self.config.subsample,
                "colsample_bytree": self.config.colsample_bytree,
                "min_child_weight": self.config.min_child_weight,
                "reg_alpha": self.config.reg_alpha,
                "reg_lambda": self.config.reg_lambda,
                "random_state": self.config.random_state,
                "objective": self.config.objective,
            },
            "feature_names": self._feature_names,
        }
        
        with open(path / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

    def load(self, path: Path) -> "XGBoostEnsembleModel":
        """Load a saved ensemble model from disk."""
        path = Path(path)
        
        with open(path / "config.json") as f:
            config_data = json.load(f)
        
        self.audience_threshold = config_data["audience_threshold"]
        self._feature_names = config_data.get("feature_names")
        
        cfg = config_data["config"]
        self.config = ModelConfig(
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            min_child_weight=cfg["min_child_weight"],
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            random_state=cfg["random_state"],
            objective=cfg["objective"],
        )
        
        self._model_small = xgb.Booster()
        self._model_small.load_model(str(path / "small_audience.json"))
        
        self._model_large = xgb.Booster()
        self._model_large.load_model(str(path / "large_audience.json"))
        
        self._is_fitted = True
        return self

    def get_feature_importance(self, model_type: str = "combined") -> dict[str, float]:
        """Get feature importance scores.
        
        Args:
            model_type: One of "small", "large", or "combined" (average)
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted to get feature importance")
        
        def _get_importance(model: xgb.Booster) -> dict[str, float]:
            importance_dict = model.get_score(importance_type="gain")
            result = {}
            
            if self._feature_names:
                for idx, name in enumerate(self._feature_names):
                    feat_key = f"f{idx}"
                    result[name] = importance_dict.get(feat_key, 0.0)
            else:
                result = importance_dict
            
            return result
        
        if model_type == "small":
            return _get_importance(self._model_small)
        elif model_type == "large":
            return _get_importance(self._model_large)
        else:
            imp_small = _get_importance(self._model_small)
            imp_large = _get_importance(self._model_large)
            
            all_keys = set(imp_small.keys()) | set(imp_large.keys())
            return {
                key: (imp_small.get(key, 0) + imp_large.get(key, 0)) / 2
                for key in all_keys
            }

    def get_top_features(self, n: int = 10, model_type: str = "combined") -> list[tuple[str, float]]:
        """Get top N most important features.
        
        Args:
            n: Number of features to return
            model_type: One of "small", "large", or "combined"
        
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        importance = self.get_feature_importance(model_type)
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]
