"""
MORGOTH GMX - HMM Volatility Regime Detector
Phase 3 Block 1a

Detects market *volatility* regimes (Calm / Normal / Turbulent) on BTC 5-min
data using a Gaussian Hidden Markov Model. At 5-min resolution the dominant,
cleanly-separable structure is volatility magnitude, not direction -- so this
model is honest about what it learns and is used for RISK SIZING:
trade smaller (or not at all) in Turbulent regimes, size up in Calm ones.

A separate DIRECTIONAL regime model (Bull/Bear/Ranging from longer-horizon
trend features) is a planned sibling block; together they become two
independent votes in the ensemble decision engine.

Regime features are stationary (returns, volatility, normalized range) -- no
absolute price levels -- so the model generalizes across the $25k-$126k range.

NOTE ON LOOK-AHEAD: HMM training/decoding uses the full sequence (standard for
offline regime *labeling*). For LIVE inference, load the fitted model and use
incremental filtering rather than re-decoding history. The saved model + scaler
make that possible without re-fitting.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# Features used for volatility-regime detection (must exist in the features parquet).
REGIME_FEATURES = ["log_ret_1", "ret_12", "realized_vol_72", "atr_pct"]

# Column used to rank states from least to most volatile when assigning labels.
RANK_FEATURE = "realized_vol_72"


class VolatilityRegimeDetector:
    """Gaussian HMM that classifies BTC 5-min bars into volatility regimes."""

    def __init__(
        self,
        n_regimes: int = 3,
        n_iter: int = 200,
        random_state: int = 42,
        input_path: str = "data/btc_features.parquet",
    ) -> None:
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        self.input_path = input_path

        self.scaler = StandardScaler()
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )
        # Maps raw HMM state index -> human regime label, set during fit().
        self.state_to_regime: dict[int, str] = {}
        self.df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Load the engineered feature dataset and drop rolling-window warmup NaNs."""
        print(f"Loading features from {self.input_path}...")
        df = pd.read_parquet(self.input_path)
        df = df.sort_values("timestamp").reset_index(drop=True)

        before = len(df)
        df = df.dropna(subset=REGIME_FEATURES).reset_index(drop=True)
        print(f"  Rows: {len(df):,} (dropped {before - len(df)} warmup NaN rows)")
        print(f"  Regime features: {REGIME_FEATURES}")
        self.df = df
        return df

    def _feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract the regime feature columns as a float matrix."""
        return df[REGIME_FEATURES].to_numpy(dtype=float)

    def _label_states(self, df: pd.DataFrame, states: np.ndarray) -> dict[int, str]:
        """
        Assign Calm/Normal/Turbulent to raw HMM state indices.

        States are ranked by mean realized volatility: lowest -> Calm,
        highest -> Turbulent, the rest -> Normal. This reflects the structure
        the HMM actually separates at 5-min resolution.
        """
        stats = (
            pd.DataFrame({"state": states, RANK_FEATURE: df[RANK_FEATURE].to_numpy()})
            .groupby("state")[RANK_FEATURE]
            .mean()
            .sort_values()
        )
        ordered = list(stats.index)  # ascending volatility
        mapping: dict[int, str] = {}
        mapping[ordered[0]] = "Calm"
        mapping[ordered[-1]] = "Turbulent"
        for s in ordered[1:-1]:
            mapping[s] = "Normal"
        return mapping

    def fit(self) -> pd.DataFrame:
        """Fit the HMM, label regimes, and attach a 'regime' column to the data."""
        if self.df is None:
            self.load()
        df = self.df
        assert df is not None

        print("\nFitting Gaussian HMM...")
        X = self.scaler.fit_transform(self._feature_matrix(df))
        self.model.fit(X)
        log_likelihood = self.model.score(X)
        print(f"  Converged: {self.model.monitor_.converged}")
        print(f"  Iterations: {len(self.model.monitor_.history)}")
        print(f"  Log-likelihood: {log_likelihood:,.1f}")

        states = self.model.predict(X)
        self.state_to_regime = self._label_states(df, states)
        df["regime_state"] = states
        df["regime"] = df["regime_state"].map(self.state_to_regime)
        print(f"  State -> regime map: {self.state_to_regime}")
        return df

    def report(self) -> None:
        """Print regime distribution, per-regime characteristics, and transitions."""
        df = self.df
        assert df is not None and "regime" in df.columns

        print("\n" + "=" * 60)
        print("VOLATILITY REGIME ANALYSIS")
        print("=" * 60)

        order = ["Calm", "Normal", "Turbulent"]
        counts = df["regime"].value_counts()
        print("\nRegime distribution:")
        for regime in order:
            n = int(counts.get(regime, 0))
            print(f"  {regime:10s}: {n:>8,} bars ({n / len(df) * 100:5.1f}%)")

        print("\nPer-regime characteristics (means):")
        agg = df.groupby("regime").agg(
            avg_ret_12=("ret_12", "mean"),
            avg_realized_vol=("realized_vol_72", "mean"),
            avg_atr_pct=("atr_pct", "mean"),
            avg_close=("close", "mean"),
        )
        for regime in order:
            if regime not in agg.index:
                continue
            row = agg.loc[regime]
            print(
                f"  {regime:10s}: 1h-trend {row.avg_ret_12 * 100:+.3f}% | "
                f"vol {row.avg_realized_vol:.5f} | "
                f"atr {row.avg_atr_pct * 100:.3f}% | "
                f"avg close ${row.avg_close:,.0f}"
            )

        print("\nTransition matrix (rows=from, cols=to):")
        labels = [self.state_to_regime[i] for i in range(self.n_regimes)]
        print("              " + "".join(f"{l:>11s}" for l in labels))
        for i, row in enumerate(self.model.transmat_):
            print(
                f"  {self.state_to_regime[i]:10s}  "
                + "".join(f"{p:>11.3f}" for p in row)
            )

    def save(
        self,
        model_path: str = "models/hmm_volatility_regime.pkl",
        labeled_path: str = "data/btc_vol_regimes.parquet",
    ) -> None:
        """Persist the fitted model+scaler and the regime-labeled dataset."""
        df = self.df
        assert df is not None

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "state_to_regime": self.state_to_regime,
                "features": REGIME_FEATURES,
                "kind": "volatility",
            },
            model_path,
        )
        print(f"\nSaved model to: {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")

        Path(labeled_path).parent.mkdir(parents=True, exist_ok=True)
        keep = ["timestamp", "open", "high", "low", "close", "volume",
                "regime_state", "regime"]
        df[keep].to_parquet(labeled_path, compression="snappy", index=False)
        print(f"Saved regimes to: {labeled_path} ({os.path.getsize(labeled_path) / 1024**2:.1f} MB)")


def main() -> None:
    print("=" * 60)
    print("MORGOTH GMX - PHASE 3 BLOCK 1a - HMM VOLATILITY REGIMES")
    print("=" * 60)

    detector = VolatilityRegimeDetector(input_path="data/btc_features.parquet")
    detector.load()
    detector.fit()
    detector.report()
    detector.save()

    print("\n" + "=" * 60)
    print("PHASE 3 BLOCK 1a COMPLETE")
    print("Next: directional regime model (Bull/Bear/Ranging) as a sibling vote")
    print("=" * 60)


if __name__ == "__main__":
    main()
