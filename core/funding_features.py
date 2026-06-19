"""
MORGOTH GMX - Funding Rate Feature Engineering
Week 1 Day 3

Merges funding rate data with BTC 5-min features.
Adds 8 funding-derived features for ML training.
Maintains causal time alignment (no look-ahead bias).
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


class FundingFeatureBuilder:
    """Builds funding-derived features from raw funding data."""

    def __init__(
        self,
        features_path: str = "data/btc_features.parquet",
        funding_path: str = "data/funding_rates.parquet"
    ):
        self.features_path = features_path
        self.funding_path = funding_path
        self.btc = None
        self.funding = None

    def load(self):
        """Load both datasets."""
        print(f"Loading {self.features_path}...")
        self.btc = pd.read_parquet(self.features_path)
        self.btc = self.btc.sort_values("timestamp").reset_index(drop=True)
        print(f"  BTC features: {len(self.btc):,} rows × {len(self.btc.columns)} cols")

        print(f"\nLoading {self.funding_path}...")
        self.funding = pd.read_parquet(self.funding_path)
        self.funding = self.funding.sort_values("timestamp").reset_index(drop=True)
        print(f"  Funding: {len(self.funding):,} rows")
        print(f"  Date range: {self.funding['timestamp'].min()} to {self.funding['timestamp'].max()}")
        print(f"  Binance non-NaN: {self.funding['funding_rate_binance'].notna().sum():,}")
        print(f"  Bybit non-NaN: {self.funding['funding_rate_bybit'].notna().sum():,}")

    def merge_forward_fill(self) -> pd.DataFrame:
        """
        Forward-fill funding rates to 5-min bars.
        Critical: only use funding that exists AT OR BEFORE each 5-min bar.
        This prevents look-ahead bias.
        """
        print("\nMerging with forward-fill (causal)...")

        # merge_asof with direction='backward' = forward fill from past values only
        # No look-ahead possible
        merged = pd.merge_asof(
            self.btc,
            self.funding,
            on="timestamp",
            direction="backward"
        )

        # Coverage check
        print(f"  Total rows: {len(merged):,}")
        print(f"  Binance fwd-filled non-NaN: {merged['funding_rate_binance'].notna().sum():,} ({merged['funding_rate_binance'].notna().mean()*100:.1f}%)")
        print(f"  Bybit fwd-filled non-NaN: {merged['funding_rate_bybit'].notna().sum():,} ({merged['funding_rate_bybit'].notna().mean()*100:.1f}%)")

        # First N rows will have NaN funding (before first funding event)
        first_funding_idx = merged['funding_rate_binance'].first_valid_index()
        if first_funding_idx:
            print(f"  First valid funding at row {first_funding_idx} ({merged.iloc[first_funding_idx]['timestamp']})")

        return merged

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add z-scores, changes, extreme flags."""
        print("\nAdding derived funding features...")

        # Spread (already exists from fetcher, but ensure it's recomputed)
        df["funding_spread"] = df["funding_rate_binance"] - df["funding_rate_bybit"]

        # Z-scores (extreme detection) - rolling windows
        # 30 days = 30 * 24 * 12 = 8640 bars of 5-min
        # 7 days = 2016 bars
        window_30d = 8640
        window_7d = 2016

        for col in ["funding_rate_binance", "funding_rate_bybit"]:
            mean_30d = df[col].rolling(window=window_30d, min_periods=window_30d // 4).mean()
            std_30d = df[col].rolling(window=window_30d, min_periods=window_30d // 4).std()
            df[f"{col.split('_')[-1]}_zscore_30d"] = (df[col] - mean_30d) / std_30d.replace(0, np.nan)

        # 7-day z-score on Binance (primary)
        mean_7d = df["funding_rate_binance"].rolling(window=window_7d, min_periods=window_7d // 4).mean()
        std_7d = df["funding_rate_binance"].rolling(window=window_7d, min_periods=window_7d // 4).std()
        df["funding_zscore_7d"] = (df["funding_rate_binance"] - mean_7d) / std_7d.replace(0, np.nan)

        # Funding change over 24h (8 funding events = 8h * 3 = 24h, but at 5min granularity = 288 bars)
        # Use change vs 288 bars ago
        df["funding_change_24h"] = df["funding_rate_binance"] - df["funding_rate_binance"].shift(288)

        # Extreme flag (z > 2 = top 2.5% of funding distribution)
        df["funding_extreme_high"] = (df["funding_zscore_7d"] > 2.0).astype(int)
        df["funding_extreme_low"] = (df["funding_zscore_7d"] < -2.0).astype(int)

        # Stats
        print(f"  funding_spread coverage: {df['funding_spread'].notna().sum():,}")
        print(f"  zscore_30d coverage: {df['binance_zscore_30d'].notna().sum():,}")
        print(f"  zscore_7d coverage: {df['funding_zscore_7d'].notna().sum():,}")
        print(f"  Extreme high events: {df['funding_extreme_high'].sum():,}")
        print(f"  Extreme low events: {df['funding_extreme_low'].sum():,}")

        return df

    def save(self, df: pd.DataFrame):
        """Save updated features file."""
        # Backup original first
        backup_path = self.features_path.replace(".parquet", "_backup.parquet")
        if not os.path.exists(backup_path):
            print(f"\nCreating backup: {backup_path}")
            import shutil
            shutil.copy(self.features_path, backup_path)

        # Save updated
        df.to_parquet(self.features_path, compression="snappy", index=False)
        size_mb = os.path.getsize(self.features_path) / 1024**2
        print(f"\nSaved: {self.features_path}")
        print(f"  Final shape: {df.shape}")
        print(f"  File size: {size_mb:.2f} MB")

    def report_quality(self, df: pd.DataFrame):
        """Final quality report."""
        print("\n" + "=" * 60)
        print("FUNDING FEATURES QUALITY REPORT")
        print("=" * 60)

        funding_cols = [
            "funding_rate_binance", "funding_rate_bybit", "funding_spread",
            "funding_rate_avg", "binance_zscore_30d", "bybit_zscore_30d",
            "funding_zscore_7d", "funding_change_24h",
            "funding_extreme_high", "funding_extreme_low"
        ]

        print(f"\n{'Feature':<30} {'Non-NaN':<12} {'Coverage':<10} {'Mean':<10} {'Std':<10}")
        print("-" * 75)
        for col in funding_cols:
            if col in df.columns:
                non_nan = df[col].notna().sum()
                coverage = df[col].notna().mean() * 100
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(f"{col:<30} {non_nan:<12,} {coverage:<10.1f} {mean_val:<10.4f} {std_val:<10.4f}")

        # Sample row from middle of dataset
        mid_idx = len(df) // 2
        print(f"\nSample row (middle of dataset, idx {mid_idx}):")
        print(f"  Timestamp: {df.iloc[mid_idx]['timestamp']}")
        for col in funding_cols:
            if col in df.columns:
                print(f"  {col}: {df.iloc[mid_idx][col]}")


def main():
    print("=" * 60)
    print("MORGOTH GMX - WEEK 1 DAY 3 - FUNDING FEATURES")
    print("=" * 60)

    builder = FundingFeatureBuilder()
    builder.load()

    df = builder.merge_forward_fill()
    df = builder.add_derived_features(df)

    builder.save(df)
    builder.report_quality(df)

    print("\n" + "=" * 60)
    print("WEEK 1 DAY 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
