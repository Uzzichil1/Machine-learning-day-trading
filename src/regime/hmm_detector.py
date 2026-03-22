"""GMM-based market regime detection.

Replaces hmmlearn.GaussianHMM with sklearn.mixture.GaussianMixture.
GaussianMixture clusters the return+volatility feature space into distinct
regimes without modelling temporal state transitions.  For regime-gating
purposes (trade / skip / size-scale) this is sufficient and avoids the
hmmlearn dependency that is incompatible with Python 3.14+.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detects market regimes using a Gaussian Mixture Model.

    Each mixture component is mapped to one of three regime labels based on
    its mean volatility, mirroring the original HMM state-mapping logic:

    - low_vol_trend   : lowest-volatility cluster  → TRADE (full size)
    - moderate_trend  : middle cluster(s)           → TRADE (reduced size)
    - high_vol_chaos  : highest-volatility cluster  → SKIP
    """

    def __init__(self, n_states: int = 3, lookback: int = 252):
        self.n_states = n_states
        self.lookback = lookback
        self.model = GaussianMixture(
            n_components=n_states,
            covariance_type="full",
            n_init=5,
            max_iter=200,
            random_state=42,
        )
        self._fitted = False
        self._state_mapping: dict[int, str] = {}  # GMM component index → regime label

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
        """Extract (return, volatility) features from a price DataFrame."""
        close = df["close"]
        log_ret = np.log(close / close.shift(1))
        vol = log_ret.rolling(10).std()

        features = pd.DataFrame({
            "return": log_ret,
            "volatility": vol,
        }).dropna()

        return features.values, features.index

    def _map_states(self, X: np.ndarray, states: np.ndarray) -> None:
        """Map GMM component indices to interpretable regime labels.

        Components are sorted by their mean volatility (column 1 of X).
        The lowest-volatility component becomes ``low_vol_trend``, the
        highest becomes ``high_vol_chaos``, and everything in between is
        ``moderate_trend``.
        """
        state_stats: dict[int, dict] = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() == 0:
                continue
            state_stats[s] = {
                "mean_return": float(X[mask, 0].mean()),
                "mean_vol": float(X[mask, 1].mean()),
                "count": int(mask.sum()),
            }

        # Sort observed components by ascending mean volatility
        sorted_states = sorted(state_stats.keys(), key=lambda s: state_stats[s]["mean_vol"])

        self._state_mapping = {}
        for i, state_idx in enumerate(sorted_states):
            if i == 0:
                self._state_mapping[state_idx] = "low_vol_trend"
            elif i == len(sorted_states) - 1:
                self._state_mapping[state_idx] = "high_vol_chaos"
            else:
                self._state_mapping[state_idx] = "moderate_trend"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """Fit the GMM on historical OHLCV data."""
        X, _idx = self._prepare_features(df)
        if len(X) < 100:
            logger.warning("Insufficient data for GMM fitting (%d observations)", len(X))
            return

        self.model.fit(X)
        self._fitted = True

        states = self.model.predict(X)
        self._map_states(X, states)

        logger.info("GMM fitted on %d observations, %d components", len(X), self.n_states)
        logger.info("State mapping: %s", self._state_mapping)

    def predict_regime(self, df: pd.DataFrame) -> pd.Series:
        """Return a string regime label for every bar in *df*.

        Labels: ``low_vol_trend``, ``moderate_trend``, ``high_vol_chaos``.
        """
        if not self._fitted:
            raise RuntimeError("RegimeDetector not fitted. Call fit() first.")

        X, idx = self._prepare_features(df)
        states = self.model.predict(X)
        regimes = [self._state_mapping.get(int(s), "unknown") for s in states]
        return pd.Series(regimes, index=idx, name="regime")

    def get_trading_gate(self, df: pd.DataFrame) -> pd.Series:
        """Boolean series: True = trade allowed, False = skip.

        Trading is allowed during trending regimes and skipped during
        high-volatility / chaotic regimes.
        """
        regimes = self.predict_regime(df)
        gate = regimes.isin(["low_vol_trend", "moderate_trend"])
        return gate

    def get_size_scalar(self, df: pd.DataFrame) -> pd.Series:
        """Position-size multiplier based on current regime.

        ==================  =====  ==============================
        Regime              Value  Rationale
        ==================  =====  ==============================
        low_vol_trend       1.0    Full size — clean trend
        moderate_trend      0.7    Reduced — noisier conditions
        high_vol_chaos      0.0    No trade — too dangerous
        ==================  =====  ==============================
        """
        regimes = self.predict_regime(df)
        scalars = regimes.map({
            "low_vol_trend": 1.0,
            "moderate_trend": 0.7,
            "high_vol_chaos": 0.0,
        }).fillna(0.0)
        return scalars

    def save(self, path: str) -> None:
        """Save fitted GMM model and state mapping."""
        joblib.dump(
            {"model": self.model, "state_mapping": self._state_mapping, "fitted": self._fitted},
            path,
        )
        logger.info("RegimeDetector saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously fitted GMM model."""
        data = joblib.load(path)
        self.model = data["model"]
        self._state_mapping = data["state_mapping"]
        self._fitted = data["fitted"]
        logger.info("RegimeDetector loaded from %s", path)
