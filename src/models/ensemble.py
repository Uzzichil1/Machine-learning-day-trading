"""Stacking ensemble — LightGBM + XGBoost + CatBoost with logistic regression meta-learner."""

import logging

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """Three-model stacking ensemble for directional classification."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.base_models = {
            "lightgbm": lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            ),
            "xgboost": xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            ),
            "catboost": CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=0,
            ),
        }
        self.meta_learner = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        self._fitted = False
        self._median_proba = 0.5  # Calibrated from validation set

    def fit(self, X: np.ndarray, y: np.ndarray, cv_splits=5):
        """Train base learners and meta-learner via stacking.

        Uses cross_val_predict to generate out-of-fold predictions for
        meta-learner training, preventing information leakage.
        """
        logger.info(f"Training stacking ensemble on {X.shape[0]} samples, {X.shape[1]} features")

        # Generate out-of-fold predictions for meta-learner
        oof_predictions = {}
        for name, model in self.base_models.items():
            logger.info(f"  Generating OOF predictions for {name}...")
            oof_preds = cross_val_predict(
                model, X, y, cv=cv_splits, method="predict_proba"
            )
            oof_predictions[name] = oof_preds[:, 1] if oof_preds.ndim > 1 else oof_preds

        # Stack OOF predictions as meta-features
        meta_X = np.column_stack(list(oof_predictions.values()))

        # Train meta-learner on OOF predictions
        logger.info("  Training meta-learner...")
        self.meta_learner.fit(meta_X, y)

        # Refit base learners on full training data
        for name, model in self.base_models.items():
            logger.info(f"  Refitting {name} on full data...")
            model.fit(X, y)

        self._fitted = True
        logger.info("Ensemble training complete")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probability of positive class (buy signal)."""
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        base_preds = []
        for name, model in self.base_models.items():
            pred = model.predict_proba(X)
            base_preds.append(pred[:, 1] if pred.ndim > 1 else pred)

        meta_X = np.column_stack(base_preds)
        return self.meta_learner.predict_proba(meta_X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self, feature_names: list) -> dict:
        """Return SHAP-style feature importance from LightGBM."""
        if not self._fitted:
            return {}
        importances = self.base_models["lightgbm"].feature_importances_
        return dict(
            sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def save(self, path: str):
        joblib.dump(
            {
                "base_models": self.base_models,
                "meta_learner": self.meta_learner,
                "median_proba": self._median_proba,
            },
            path,
        )
        logger.info(f"Ensemble saved to {path} (median_proba={self._median_proba:.4f})")

    def load(self, path: str):
        data = joblib.load(path)
        self.base_models = data["base_models"]
        self.meta_learner = data["meta_learner"]
        self._median_proba = data.get("median_proba", 0.5)
        self._fitted = True
        logger.info(f"Ensemble loaded from {path} (median_proba={self._median_proba:.4f})")
