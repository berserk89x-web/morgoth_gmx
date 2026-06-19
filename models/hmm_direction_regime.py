"""
MORGOTH GMX - HMM Directional Regime Detector
Phase 3 Block 1b

Detects DIRECTIONAL market regimes (Bull/Bear/Ranging) using longer-horizon
trend features. Complements the volatility HMM in MORGOTH's ensemble.

Uses 24h-scope features to overcome 5-min noise problem.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DirectionRegimeDetector:
    """3-state HMM for directional regime detection using longer-horizon features."""

    REGIMES = ["Bear", "Ranging", "Bull"]

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 300,
        random_state: int = 42
    ):
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=random_state,
            tol=1e-4
        )
        # TRUE long-horizon (2h-24h) features to capture DIRECTION not noise.
        # First feature is the state-sorting key, so it must be the clearest
        # directional signal -> 24h daily return.
        self.feature_cols = [
            "ret_288",                 # 24h return (daily trend) - sort key
            "ret_72",                  # 6h momentum
            "ret_24",                  # 2h momentum
            "ema_50_slope",            # 50EMA direction (1h slope)
            "ema_200_slope",           # 200EMA long-trend direction (6h slope)
            "close_ema_200_ratio",     # Price vs long trend
        ]
        self.scaler = StandardScaler()
        self.regime_mapping = None
        self.trained = False

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Extract directional features."""
        # Drop warmup (need 200 bars for EMA200)
        df = df.iloc[300:].copy()
        # Drop last bars (target leakage protection)
        df = df.iloc[:-12].copy()

        features = df[self.feature_cols].copy()

        before = len(features)
        features = features.dropna()
        after = len(features)
        if before != after:
            print(f"  Dropped {before - after} rows with NaN")

        return features.values, df.loc[features.index]

    def train(self, df: pd.DataFrame) -> None:
        """Train directional HMM."""
        print(f"\nTraining DIRECTIONAL HMM ({self.n_states} states)...")
        print(f"  Features: {self.feature_cols}")

        X_raw, _ = self.prepare_data(df)
        print(f"  Training samples: {len(X_raw):,}")

        # Standardize features (critical for HMM)
        X = self.scaler.fit_transform(X_raw)

        # Fit HMM
        self.model.fit(X)
        self.trained = True

        # Map states by DIRECTIONAL signal (24h return mean)
        # First feature is ret_288, so means_[:, 0] is the 24h return per state
        ret_288_means = self.model.means_[:, 0]
        sorted_states = np.argsort(ret_288_means)

        self.regime_mapping = {
            sorted_states[0]: "Bear",
            sorted_states[1]: "Ranging",
            sorted_states[2]: "Bull"
        }

        print(f"\nTraining complete!")
        print(f"  Converged: {self.model.monitor_.converged}")
        print(f"  Iterations: {self.model.monitor_.iter}")
        print(f"  Log-likelihood: {self.model.score(X):.2f}")

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict regimes for entire dataset."""
        X_raw, df_aligned = self.prepare_data(df)
        X = self.scaler.transform(X_raw)

        states = self.model.predict(X)
        probs = self.model.predict_proba(X)
        regimes = [self.regime_mapping[s] for s in states]

        result = df_aligned.copy()
        result["hmm_dir_state"] = states
        result["hmm_dir_regime"] = regimes

        bear_idx = [k for k, v in self.regime_mapping.items() if v == "Bear"][0]
        rang_idx = [k for k, v in self.regime_mapping.items() if v == "Ranging"][0]
        bull_idx = [k for k, v in self.regime_mapping.items() if v == "Bull"][0]

        result["hmm_dir_prob_bear"] = probs[:, bear_idx]
        result["hmm_dir_prob_ranging"] = probs[:, rang_idx]
        result["hmm_dir_prob_bull"] = probs[:, bull_idx]
        result["hmm_dir_confidence"] = probs.max(axis=1)

        return result

    def predict_current(self, df: pd.DataFrame) -> dict:
        """Predict regime for latest row (live trading)."""
        X_raw, df_aligned = self.prepare_data(df)
        X = self.scaler.transform(X_raw)

        last_X = X[-1:].reshape(1, -1)
        state = self.model.predict(last_X)[0]
        probs = self.model.predict_proba(last_X)[0]

        regime = self.regime_mapping[state]
        confidence = probs.max()

        bear_idx = [k for k, v in self.regime_mapping.items() if v == "Bear"][0]
        rang_idx = [k for k, v in self.regime_mapping.items() if v == "Ranging"][0]
        bull_idx = [k for k, v in self.regime_mapping.items() if v == "Bull"][0]

        return {
            "timestamp": df_aligned.iloc[-1]["timestamp"] if "timestamp" in df_aligned.columns else None,
            "state": int(state),
            "regime": regime,
            "confidence": float(confidence),
            "prob_bear": float(probs[bear_idx]),
            "prob_ranging": float(probs[rang_idx]),
            "prob_bull": float(probs[bull_idx])
        }

    def save(self, path: str = "models/hmm_direction_regime.pkl") -> None:
        """Save trained model + scaler."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "regime_mapping": self.regime_mapping,
                "trained": self.trained
            }, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"\nSaved to: {path}")
        print(f"  File size: {size_kb:.1f} KB")

    def evaluate(self, df_with_regimes: pd.DataFrame) -> dict:
        """Show directional regime statistics."""
        print("\n" + "=" * 60)
        print("DIRECTIONAL REGIME ANALYSIS")
        print("=" * 60)

        # Time distribution
        regime_counts = df_with_regimes["hmm_dir_regime"].value_counts()
        total = len(df_with_regimes)
        print("\nTime in each directional regime:")
        for regime in ["Bull", "Ranging", "Bear"]:
            count = regime_counts.get(regime, 0)
            pct = count / total * 100
            print(f"  {regime:10s}: {count:>7,} bars ({pct:5.1f}%)")

        # Forward returns (1 bar = 5 min)
        print("\nForward 1-bar (5min) returns by regime:")
        results = {}
        for regime in ["Bull", "Ranging", "Bear"]:
            mask = df_with_regimes["hmm_dir_regime"] == regime
            if mask.sum() > 0:
                avg_ret = df_with_regimes.loc[mask, "target_ret_1"].mean() * 100
                win_rate = (df_with_regimes.loc[mask, "target_ret_1"] > 0).mean() * 100
                print(f"  {regime:10s}: {avg_ret:+.4f}% (win rate: {win_rate:5.1f}%)")
                results[f"{regime}_1bar_ret"] = avg_ret
                results[f"{regime}_1bar_wr"] = win_rate

        # Forward returns 12-bar (1 hour)
        print("\nForward 12-bar (1h) returns by regime:")
        for regime in ["Bull", "Ranging", "Bear"]:
            mask = df_with_regimes["hmm_dir_regime"] == regime
            if mask.sum() > 0:
                avg_ret = df_with_regimes.loc[mask, "target_ret_12"].mean() * 100
                win_rate = (df_with_regimes.loc[mask, "target_ret_12"] > 0).mean() * 100
                print(f"  {regime:10s}: {avg_ret:+.4f}% (win rate: {win_rate:5.1f}%)")
                results[f"{regime}_12bar_ret"] = avg_ret
                results[f"{regime}_12bar_wr"] = win_rate

        # Persistence (transition diagonal)
        print("\nRegime persistence (diagonal of transition matrix):")
        trans = self.model.transmat_
        for state_idx, regime in self.regime_mapping.items():
            print(f"  {regime:10s}: P(stay) = {trans[state_idx, state_idx]:.3f}")

        # Average confidence
        print("\nAverage confidence per regime:")
        for regime in ["Bull", "Ranging", "Bear"]:
            mask = df_with_regimes["hmm_dir_regime"] == regime
            if mask.sum() > 0:
                avg_conf = df_with_regimes.loc[mask, "hmm_dir_confidence"].mean()
                print(f"  {regime:10s}: {avg_conf*100:.1f}%")

        return results


def main():
    print("=" * 60)
    print("MORGOTH GMX - PHASE 3 BLOCK 1b - DIRECTIONAL HMM")
    print("=" * 60)

    # Load features
    print("\nLoading features...")
    df = pd.read_parquet("data/btc_features.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Train directional HMM
    detector = DirectionRegimeDetector(n_states=3, n_iter=300)
    detector.train(df)

    # Analyze
    df_with_regimes = detector.analyze(df)
    results = detector.evaluate(df_with_regimes)

    # CRITICAL CHECK: Are regimes actually directional?
    print("\n" + "=" * 60)
    print("HONEST DIRECTIONAL TEST")
    print("=" * 60)
    bull_ret = results.get("Bull_12bar_ret", 0)
    bear_ret = results.get("Bear_12bar_ret", 0)
    spread = bull_ret - bear_ret

    print(f"\nBull 1h return: {bull_ret:+.4f}%")
    print(f"Bear 1h return: {bear_ret:+.4f}%")
    print(f"Spread:         {spread:+.4f}%")

    if spread > 0.05:
        print("\n✅ STRONG directional separation - regimes are directional!")
    elif spread > 0.02:
        print("\n⚠️ WEAK directional separation - some edge but limited")
    else:
        print("\n❌ NO directional separation - regimes still volatility-based")
        print("   Consider different feature set (daily returns, longer EMAs)")

    # Save model
    detector.save("models/hmm_direction_regime.pkl")

    # Save regime-labeled data
    output_path = "data/btc_direction_regimes.parquet"
    df_with_regimes.to_parquet(output_path, compression="snappy", index=False)
    print(f"  Regime-labeled data: {output_path}")

    # Current regime
    print("\n" + "=" * 60)
    print("CURRENT DIRECTIONAL REGIME (last bar)")
    print("=" * 60)
    current = detector.predict_current(df)
    print(f"  Timestamp: {current['timestamp']}")
    print(f"  Regime: {current['regime']}")
    print(f"  Confidence: {current['confidence']*100:.1f}%")
    print(f"  Probabilities:")
    print(f"    Bull:    {current['prob_bull']*100:.1f}%")
    print(f"    Ranging: {current['prob_ranging']*100:.1f}%")
    print(f"    Bear:    {current['prob_bear']*100:.1f}%")

    print("\n" + "=" * 60)
    print("PHASE 3 BLOCK 1b COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
