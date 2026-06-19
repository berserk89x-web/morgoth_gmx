"""
MORGOTH GMX - XGBoost Signal Classifier
Phase 3 Block 2

Multi-class signal classification: BUY / SELL / HOLD
Uses 87 features to predict 1-hour forward direction.
Third model in MORGOTH ensemble.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class SignalClassifier:
    """XGBoost classifier for BUY/SELL/HOLD signals."""

    # Class labels
    CLASS_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}

    # Threshold for signal (must beat fees + slippage)
    # GMX fees ~0.1% per side = 0.2% round-trip
    # Add slippage buffer = 0.05%
    # Total cost ~0.25%, so signal must clear this
    THRESHOLD = 0.0015  # 0.15% in 1 hour

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }
        self.model = None
        self.feature_cols = None
        self.trained = False

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create 3-class labels from forward returns."""
        ret_12 = df["target_ret_12"]

        labels = pd.Series(1, index=df.index)  # Default HOLD
        labels[ret_12 > self.THRESHOLD] = 2  # BUY
        labels[ret_12 < -self.THRESHOLD] = 0  # SELL

        return labels

    def select_features(self, df: pd.DataFrame) -> list:
        """Select feature columns (exclude targets, timestamps, raw OHLCV)."""
        exclude = [
            "timestamp",
            "open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
            # Future-leaking columns
            "target_ret_1", "target_ret_3", "target_ret_12",
            "target_direction_1", "target_direction_12",
            # HMM outputs (we'll add them as features in ensemble)
        ]

        features = [c for c in df.columns if c not in exclude]
        return features

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare X and y for training."""
        # Drop warmup (need 300 bars for all features)
        df = df.iloc[300:].copy()
        # Drop last 12 (target_ret_12 needs future)
        df = df.iloc[:-12].copy()

        # Create labels
        y = self.create_labels(df)

        # Get features
        self.feature_cols = self.select_features(df)
        X = df[self.feature_cols].copy()

        # Drop any remaining NaN
        valid = X.notna().all(axis=1) & y.notna()
        X = X[valid]
        y = y[valid]

        return X, y

    def train(self, df: pd.DataFrame) -> Dict:
        """Train with walk-forward cross-validation."""
        print("\nPreparing data...")
        X, y = self.prepare_data(df)

        print(f"  Total samples: {len(X):,}")
        print(f"  Features: {len(self.feature_cols)}")

        # Class distribution
        print("\nClass distribution:")
        for cls, name in self.CLASS_NAMES.items():
            count = (y == cls).sum()
            pct = count / len(y) * 100
            print(f"  {name:5s}: {count:>7,} ({pct:5.1f}%)")

        # Walk-forward cross-validation
        print("\nWalk-forward cross-validation (5 folds)...")
        tscv = TimeSeriesSplit(n_splits=5)

        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = xgb.XGBClassifier(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            # Overall accuracy
            acc = accuracy_score(y_test, y_pred)

            # Per-class precision (KEY METRIC for trading)
            # We care most about: when model says BUY, is it right?
            buy_mask = y_pred == 2
            sell_mask = y_pred == 0

            buy_precision = (y_test[buy_mask] == 2).mean() if buy_mask.sum() > 0 else 0
            sell_precision = (y_test[sell_mask] == 0).mean() if sell_mask.sum() > 0 else 0

            # Trade count
            buy_count = buy_mask.sum()
            sell_count = sell_mask.sum()

            print(f"  Fold {fold_idx+1}: acc={acc:.3f}, "
                  f"BUY prec={buy_precision:.3f} (n={buy_count}), "
                  f"SELL prec={sell_precision:.3f} (n={sell_count})")

            fold_results.append({
                "fold": fold_idx + 1,
                "accuracy": acc,
                "buy_precision": buy_precision,
                "sell_precision": sell_precision,
                "buy_count": int(buy_count),
                "sell_count": int(sell_count),
            })

        # Average across folds
        avg_acc = np.mean([f["accuracy"] for f in fold_results])
        avg_buy_prec = np.mean([f["buy_precision"] for f in fold_results])
        avg_sell_prec = np.mean([f["sell_precision"] for f in fold_results])

        print(f"\nCross-validation averages:")
        print(f"  Accuracy:       {avg_acc:.3f}")
        print(f"  BUY precision:  {avg_buy_prec:.3f} (need >0.50 to be useful)")
        print(f"  SELL precision: {avg_sell_prec:.3f}")

        # Train final model on ALL data
        print("\nTraining final model on full dataset...")
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, verbose=False)
        self.trained = True

        # Feature importance
        print("\nTop 15 most important features:")
        importance = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).head(15)

        for _, row in importance.iterrows():
            bar = "█" * int(row["importance"] * 300)
            print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

        return {
            "avg_accuracy": avg_acc,
            "avg_buy_precision": avg_buy_prec,
            "avg_sell_precision": avg_sell_prec,
            "fold_results": fold_results,
        }

    def predict(self, df: pd.DataFrame) -> Dict:
        """Predict signal for latest row."""
        X = df[self.feature_cols].iloc[-1:].copy()

        # Handle NaN
        if X.isna().any().any():
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "error": "NaN in features"
            }

        proba = self.model.predict_proba(X)[0]
        pred = int(np.argmax(proba))

        return {
            "signal": self.CLASS_NAMES[pred],
            "confidence": float(proba.max()),
            "prob_sell": float(proba[0]),
            "prob_hold": float(proba[1]),
            "prob_buy": float(proba[2]),
        }

    def save(self, path: str = "models/signal_classifier.pkl") -> None:
        """Save trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_cols": self.feature_cols,
                "params": self.params,
                "threshold": self.THRESHOLD,
                "trained": self.trained,
            }, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"\nSaved to: {path}")
        print(f"  File size: {size_kb:.1f} KB")


def main():
    print("=" * 60)
    print("MORGOTH GMX - PHASE 3 BLOCK 2 - XGBOOST CLASSIFIER")
    print("=" * 60)

    # Load features
    print("\nLoading features...")
    df = pd.read_parquet("data/btc_features.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Train
    classifier = SignalClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05
    )

    results = classifier.train(df)

    # Save
    classifier.save("models/signal_classifier.pkl")

    # Current signal
    print("\n" + "=" * 60)
    print("CURRENT SIGNAL (last bar)")
    print("=" * 60)

    # Need to prepare last row properly
    X_last, _ = classifier.prepare_data(df)
    last_features = df[classifier.feature_cols].iloc[-1:]

    if not last_features.isna().any().any():
        proba = classifier.model.predict_proba(last_features)[0]
        pred = int(np.argmax(proba))
        print(f"  Signal: {classifier.CLASS_NAMES[pred]}")
        print(f"  Confidence: {proba.max()*100:.1f}%")
        print(f"  Probabilities:")
        print(f"    BUY:  {proba[2]*100:.1f}%")
        print(f"    HOLD: {proba[1]*100:.1f}%")
        print(f"    SELL: {proba[0]*100:.1f}%")

    # Honest assessment
    print("\n" + "=" * 60)
    print("HONEST ASSESSMENT")
    print("=" * 60)

    buy_prec = results["avg_buy_precision"]

    if buy_prec >= 0.55:
        print(f"  ✅ STRONG: BUY precision {buy_prec*100:.1f}% > 55%")
        print(f"  This is a tradeable edge.")
    elif buy_prec >= 0.50:
        print(f"  ⚠️ WEAK: BUY precision {buy_prec*100:.1f}% > 50%")
        print(f"  Marginal edge - use as ensemble vote only.")
    else:
        print(f"  ❌ NO EDGE: BUY precision {buy_prec*100:.1f}% < 50%")
        print(f"  Below random - model needs redesign.")

    print("\n" + "=" * 60)
    print("PHASE 3 BLOCK 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
