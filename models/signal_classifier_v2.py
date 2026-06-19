"""
MORGOTH GMX - XGBoost Signal Classifier V2
Week 4 - The Moment of Truth

Retrains XGBoost on 120-col dataset (TA + funding + flow).
Compares against Phase 3 Block 2 baseline (87 col TA only).
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score
import warnings
warnings.filterwarnings('ignore')


class SignalClassifierV2:
    """XGBoost V2 with full alpha dataset."""

    CLASS_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}
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
        """3-class labels from forward returns."""
        ret_12 = df["target_ret_12"]
        labels = pd.Series(1, index=df.index)  # HOLD
        labels[ret_12 > self.THRESHOLD] = 2  # BUY
        labels[ret_12 < -self.THRESHOLD] = 0  # SELL
        return labels

    def select_features(self, df: pd.DataFrame, mode: str = "v2") -> list:
        """
        Select feature columns.
        mode='v1': only TA features (87 col baseline)
        mode='v2': TA + funding + flow (120 col full)
        """
        exclude = [
            "timestamp",
            "open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
            "target_ret_1", "target_ret_3", "target_ret_12",
            "target_direction_1", "target_direction_12",
        ]

        if mode == "v1":
            # Exclude funding and flow features (TA only baseline)
            funding_keywords = ["funding", "binance_zscore", "bybit_zscore"]
            flow_keywords = ["transactions_per_day", "tx_volume", "mempool",
                           "miners_revenue", "hash_rate", "_change_1d", "_ma_7d",
                           "_vs_30d", "_zscore_90d", "above_avg"]

            for c in df.columns:
                if any(k in c.lower() for k in funding_keywords + flow_keywords):
                    exclude.append(c)

        features = [c for c in df.columns if c not in exclude]
        return features

    def prepare_data(self, df: pd.DataFrame, mode: str = "v2") -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare X and y."""
        # Warmup: need 300 TA + 8640 30d funding zscore = ~9000
        # But also need 26000 for 90d flow zscore warmup
        df = df.iloc[26000:].copy()
        df = df.iloc[:-12].copy()

        y = self.create_labels(df)
        self.feature_cols = self.select_features(df, mode=mode)
        X = df[self.feature_cols].copy()

        valid = X.notna().all(axis=1) & y.notna()
        X = X[valid]
        y = y[valid]

        return X, y

    def train_and_compare(self, df: pd.DataFrame) -> Dict:
        """Train both v1 (TA only) and v2 (full) for head-to-head comparison."""
        results = {}

        for mode, label in [("v1", "BASELINE (87 TA only)"), ("v2", "FULL (120 with alpha)")]:
            print("\n" + "=" * 60)
            print(f"TRAINING {label}")
            print("=" * 60)

            X, y = self.prepare_data(df, mode=mode)
            print(f"  Samples: {len(X):,}")
            print(f"  Features: {len(self.feature_cols)}")

            print("\n  Class distribution:")
            for cls, name in self.CLASS_NAMES.items():
                count = (y == cls).sum()
                pct = count / len(y) * 100
                print(f"    {name:5s}: {count:>7,} ({pct:5.1f}%)")

            # Walk-forward CV
            print("\n  Walk-forward CV (5 folds):")
            tscv = TimeSeriesSplit(n_splits=5)

            fold_results = []
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model = xgb.XGBClassifier(**self.params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                buy_mask = y_pred == 2
                sell_mask = y_pred == 0

                buy_prec = (y_test[buy_mask] == 2).mean() if buy_mask.sum() > 0 else 0
                sell_prec = (y_test[sell_mask] == 0).mean() if sell_mask.sum() > 0 else 0

                buy_count = int(buy_mask.sum())
                sell_count = int(sell_mask.sum())

                print(f"    Fold {fold_idx+1}: acc={acc:.3f}, BUY prec={buy_prec:.3f} (n={buy_count}), SELL prec={sell_prec:.3f} (n={sell_count})")

                fold_results.append({
                    "fold": fold_idx + 1,
                    "accuracy": acc,
                    "buy_precision": buy_prec,
                    "sell_precision": sell_prec,
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                })

            avg_acc = np.mean([f["accuracy"] for f in fold_results])
            avg_buy_prec = np.mean([f["buy_precision"] for f in fold_results])
            avg_sell_prec = np.mean([f["sell_precision"] for f in fold_results])

            results[mode] = {
                "avg_accuracy": avg_acc,
                "avg_buy_precision": avg_buy_prec,
                "avg_sell_precision": avg_sell_prec,
                "fold_results": fold_results,
                "n_features": len(self.feature_cols),
                "n_samples": len(X),
            }

            print(f"\n  Averages:")
            print(f"    Accuracy:       {avg_acc:.3f}")
            print(f"    BUY precision:  {avg_buy_prec:.3f}")
            print(f"    SELL precision: {avg_sell_prec:.3f}")

            # For v2, train final model on all data + save
            if mode == "v2":
                print("\n  Training V2 final model on all data...")
                final_model = xgb.XGBClassifier(**self.params)
                final_model.fit(X, y, verbose=False)
                self.model = final_model
                self.trained = True

                # Top features for V2
                print("\n  Top 20 features V2 (full dataset):")
                importance_df = pd.DataFrame({
                    "feature": self.feature_cols,
                    "importance": final_model.feature_importances_
                }).sort_values("importance", ascending=False).head(20)

                # Categorize features
                for _, row in importance_df.iterrows():
                    feat = row["feature"]
                    if any(k in feat.lower() for k in ["funding", "binance_zscore", "bybit_zscore"]):
                        category = "[FUNDING]"
                    elif any(k in feat.lower() for k in ["transactions_per", "tx_volume", "mempool", "miners", "hash_rate"]):
                        category = "[FLOW]   "
                    else:
                        category = "[TA]     "
                    bar = "#" * int(row["importance"] * 200)
                    print(f"    {category} {feat:35s} {row['importance']:.4f} {bar}")

        return results

    def save(self, path: str = "models/signal_classifier_v2.pkl"):
        """Save trained V2 model."""
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
        print(f"\nSaved V2: {path} ({size_kb:.1f} KB)")


def main():
    print("=" * 60)
    print("MORGOTH GMX - WEEK 4 - XGBOOST RETRAIN (THE TRUTH TEST)")
    print("=" * 60)

    print("\nLoading features...")
    df = pd.read_parquet("data/btc_features.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    classifier = SignalClassifierV2(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05
    )

    results = classifier.train_and_compare(df)
    classifier.save("models/signal_classifier_v2.pkl")

    # FINAL HONEST COMPARISON
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 60)

    v1_buy = results["v1"]["avg_buy_precision"]
    v2_buy = results["v2"]["avg_buy_precision"]
    v1_acc = results["v1"]["avg_accuracy"]
    v2_acc = results["v2"]["avg_accuracy"]

    print(f"\n  V1 (87 TA only)     :  BUY prec={v1_buy:.3f}, accuracy={v1_acc:.3f}")
    print(f"  V2 (120 with alpha) :  BUY prec={v2_buy:.3f}, accuracy={v2_acc:.3f}")
    print(f"  DELTA               :  BUY prec={v2_buy-v1_buy:+.3f}, accuracy={v2_acc-v1_acc:+.3f}")

    # HONEST VERDICT
    print("\n" + "=" * 60)
    print("HONEST VERDICT")
    print("=" * 60)

    if v2_buy >= 0.55:
        print(f"\n  STRONG WIN: V2 BUY precision {v2_buy*100:.1f}% > 55%")
        print(f"  Alpha features rescued the classifier!")
        print(f"  MORGOTH has a tradeable signal.")
        print(f"  -> Proceed to Phase 4 (execution layer)")
    elif v2_buy >= 0.50:
        print(f"\n  PARTIAL WIN: V2 BUY precision {v2_buy*100:.1f}% (50-55%)")
        print(f"  Some edge from alpha, but marginal.")
        print(f"  Use as ensemble vote with HMMs only.")
        print(f"  -> Phase 4 cautious approach")
    elif v2_buy > v1_buy + 0.02:
        print(f"\n  IMPROVEMENT: V2 better than V1 by {(v2_buy-v1_buy)*100:+.1f}%")
        print(f"  But still below 50% - not standalone tradeable.")
        print(f"  -> Add Week 3 whale-lite or pivot strategy")
    else:
        print(f"\n  NO MEANINGFUL CHANGE: V2 ~ V1")
        print(f"  Alpha features didn't help XGBoost find direction.")
        print(f"  -> Strategic pivot needed")
        print(f"     Options: mean reversion, volatility trading, rules engine")

    print("\n" + "=" * 60)
    print("WEEK 4 COMPLETE - THE TRUTH IS NOW KNOWN")
    print("=" * 60)


if __name__ == "__main__":
    main()
