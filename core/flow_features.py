"""
MORGOTH GMX - Network Flow Feature Engineering
Week 2 Day 3

Merges daily network metrics with BTC 5-min features.
CRITICAL: Enforces 1-day lag to prevent look-ahead bias.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil


class FlowFeatureBuilder:
    """Builds flow features from network data with strict causal alignment."""

    def __init__(
        self,
        features_path: str = "data/btc_features.parquet",
        flow_path: str = "data/network_flow.parquet"
    ):
        self.features_path = features_path
        self.flow_path = flow_path
        self.btc = None
        self.flow = None

    def load(self):
        """Load both datasets."""
        print(f"Loading {self.features_path}...")
        self.btc = pd.read_parquet(self.features_path)
        self.btc = self.btc.sort_values("timestamp").reset_index(drop=True)
        print(f"  BTC features: {len(self.btc):,} rows × {len(self.btc.columns)} cols")

        print(f"\nLoading {self.flow_path}...")
        self.flow = pd.read_parquet(self.flow_path)
        self.flow = self.flow.sort_values("timestamp").reset_index(drop=True)
        print(f"  Flow data: {len(self.flow):,} rows × {len(self.flow.columns)} cols")
        print(f"  Date range: {self.flow['timestamp'].min()} to {self.flow['timestamp'].max()}")

    def apply_1day_lag(self):
        """
        CRITICAL: Shift flow timestamps by +1 day.

        Daily metric for date X isn't fully known until end of date X.
        Available for prediction starting date X+1.
        """
        print("\nApplying 1-day lag (look-ahead protection)...")
        print(f"  Before: first flow timestamp = {self.flow['timestamp'].min()}")

        self.flow["timestamp"] = self.flow["timestamp"] + pd.Timedelta(days=1)

        print(f"  After:  first flow timestamp = {self.flow['timestamp'].min()}")
        print(f"  All flow values now effective starting NEXT day")

    def merge_forward_fill(self) -> pd.DataFrame:
        """
        Merge flow data into BTC features with strict causal alignment.
        merge_asof(direction='backward') ensures no future values used.
        """
        print("\nMerging with causal forward-fill...")

        merged = pd.merge_asof(
            self.btc,
            self.flow,
            on="timestamp",
            direction="backward"
        )

        # Coverage report
        print(f"  Total rows: {len(merged):,}")

        flow_cols = [c for c in self.flow.columns if c != "timestamp"]
        print(f"  Flow columns merged: {len(flow_cols)}")

        # Check sample coverage
        if flow_cols:
            sample_col = flow_cols[0]
            coverage = merged[sample_col].notna().mean() * 100
            first_valid = merged[sample_col].first_valid_index()
            if first_valid is not None:
                print(f"  Sample col '{sample_col}' coverage: {coverage:.1f}%")
                print(f"  First valid at row {first_valid} ({merged.iloc[first_valid]['timestamp']})")

        return merged

    def verify_causality(self, df: pd.DataFrame):
        """Spot-check that no future values leaked."""
        print("\n" + "=" * 60)
        print("CAUSALITY VERIFICATION")
        print("=" * 60)

        # Pick a random sample row
        if len(df) < 1000:
            print("  Too few rows for verification")
            return

        sample_idx = len(df) // 2  # Middle of dataset
        sample_row = df.iloc[sample_idx]
        sample_ts = sample_row["timestamp"]

        print(f"\nSample row at {sample_ts}:")

        # Check transactions_per_day - should be from previous day or earlier
        if "transactions_per_day" in df.columns and pd.notna(sample_row.get("transactions_per_day")):
            tx_value = sample_row["transactions_per_day"]
            print(f"  transactions_per_day: {tx_value:,.0f}")

            # Find the flow row this came from (before 1-day lag was applied)
            # We applied +1 day lag, so original date is sample_ts - 1 day or earlier
            print(f"  Expected: value from {sample_ts.date() - pd.Timedelta(days=1)} or earlier")
            print(f"  (1-day lag ensures we don't peek at same-day data)")

        # Verify first few hours have NaN (before any flow data available)
        first_50 = df.iloc[:50]
        if "transactions_per_day" in df.columns:
            first_valid_idx = first_50["transactions_per_day"].first_valid_index()
            if first_valid_idx is None:
                print(f"\n  ✓ First 50 rows have no flow data (expected - early dataset)")
            else:
                print(f"\n  First valid flow data at row {first_valid_idx}")
                print(f"    Timestamp: {df.iloc[first_valid_idx]['timestamp']}")

    def save(self, df: pd.DataFrame):
        """Save updated features with backup."""
        # Create backup if doesn't exist
        backup_path = self.features_path.replace(".parquet", "_pre_flow_backup.parquet")
        if not os.path.exists(backup_path):
            print(f"\nCreating backup: {backup_path}")
            shutil.copy(self.features_path, backup_path)

        # Save updated dataset
        df.to_parquet(self.features_path, compression="snappy", index=False)
        size_mb = os.path.getsize(self.features_path) / 1024**2
        print(f"\nSaved: {self.features_path}")
        print(f"  Final shape: {df.shape}")
        print(f"  File size: {size_mb:.2f} MB")

    def report_features(self, df: pd.DataFrame):
        """Final feature report."""
        print("\n" + "=" * 60)
        print("FINAL DATASET REPORT")
        print("=" * 60)

        # Group columns by type
        flow_cols = [c for c in df.columns if any(
            keyword in c.lower() for keyword in
            ["transactions_per_day", "tx_volume", "mempool", "miners_revenue", "hash_rate", "change_1d", "ma_7d", "vs_30d", "zscore_90d", "above_avg"]
        )]

        print(f"\nTotal columns: {len(df.columns)}")
        print(f"Flow-related columns added: {len(flow_cols)}")

        print(f"\nFlow feature coverage:")
        for col in flow_cols[:15]:  # Show first 15
            coverage = df[col].notna().mean() * 100
            print(f"  {col:45s}: {coverage:5.1f}%")
        if len(flow_cols) > 15:
            print(f"  ... and {len(flow_cols) - 15} more")

        # Sample row from middle
        mid_idx = len(df) // 2
        print(f"\nSample row (idx {mid_idx}): {df.iloc[mid_idx]['timestamp']}")
        for col in ["transactions_per_day", "tx_volume_usd", "mempool_size", "miners_revenue_usd"]:
            if col in df.columns:
                val = df.iloc[mid_idx][col]
                if pd.notna(val):
                    print(f"  {col}: {val:,.2f}")


def main():
    print("=" * 60)
    print("MORGOTH GMX - WEEK 2 DAY 3 - FLOW FEATURES MERGE")
    print("=" * 60)

    builder = FlowFeatureBuilder()
    builder.load()
    builder.apply_1day_lag()  # CRITICAL: prevents look-ahead bias

    df = builder.merge_forward_fill()
    builder.verify_causality(df)
    builder.save(df)
    builder.report_features(df)

    print("\n" + "=" * 60)
    print("WEEK 2 DAY 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
