"""
MORGOTH GMX - Network Flow Fetcher
Week 2 Day 2

Downloads BTC network activity metrics from blockchain.com Charts API.
Daily cadence, 3-year historical depth.
Critical: enforces 1-day lag at merge time to prevent look-ahead bias.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


class BlockchainComFetcher:
    """Fetches daily network metrics from blockchain.com Charts API."""

    BASE_URL = "https://api.blockchain.info/charts"

    METRICS = {
        "n-transactions": "transactions_per_day",
        "estimated-transaction-volume-usd": "tx_volume_usd",
        "mempool-count": "mempool_size",
        "miners-revenue": "miners_revenue_usd",
        "hash-rate": "hash_rate_th",
    }

    def __init__(self):
        self.session = requests.Session()

    def fetch_metric(self, metric: str, timespan: str = "3years") -> pd.DataFrame:
        """Fetch a single metric from blockchain.com Charts."""
        try:
            r = self.session.get(
                f"{self.BASE_URL}/{metric}",
                params={"timespan": timespan, "format": "json"},
                timeout=15
            )
            r.raise_for_status()
            data = r.json()

            values = data.get("values", [])
            if not values:
                print(f"  {metric}: NO DATA")
                return pd.DataFrame()

            df = pd.DataFrame(values)
            df["timestamp"] = pd.to_datetime(df["x"], unit="s")
            # Normalize to day (00:00:00) so all metrics align on the same date.
            # blockchain.com stamps some series (e.g. mempool) at odd sub-daily
            # times; without this they fail to merge with the midnight series.
            df["timestamp"] = df["timestamp"].dt.normalize()
            df["value"] = pd.to_numeric(df["y"], errors="coerce")
            df = df[["timestamp", "value"]].drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            print(f"  {metric}: {len(df):,} entries ({df['timestamp'].min()} to {df['timestamp'].max()})")
            return df
        except Exception as e:
            print(f"  {metric}: ERROR {str(e)[:100]}")
            return pd.DataFrame()

    def fetch_all_metrics(self, timespan: str = "3years") -> pd.DataFrame:
        """Fetch all metrics and merge into single DataFrame."""
        print(f"\nFetching {len(self.METRICS)} network metrics from blockchain.com")
        print(f"Timespan: {timespan}")
        print("=" * 60)

        merged = None

        for raw_name, friendly_name in self.METRICS.items():
            df = self.fetch_metric(raw_name, timespan)
            if df.empty:
                continue

            df = df.rename(columns={"value": friendly_name})

            if merged is None:
                merged = df
            else:
                merged = pd.merge(
                    merged, df,
                    on="timestamp",
                    how="outer"
                )

            time.sleep(0.3)  # Rate limit safety

        if merged is None or merged.empty:
            print("\nERROR: No metrics fetched")
            return pd.DataFrame()

        merged = merged.sort_values("timestamp").reset_index(drop=True)
        return merged

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features (changes, z-scores)."""
        print("\nAdding derived features...")

        for col in ["transactions_per_day", "tx_volume_usd", "miners_revenue_usd", "hash_rate_th"]:
            if col not in df.columns:
                continue

            # Day-over-day change (fill_method=None: don't forward-fill NaNs)
            df[f"{col}_change_1d"] = df[col].pct_change(1, fill_method=None)

            # 7-day moving average
            df[f"{col}_ma_7d"] = df[col].rolling(window=7, min_periods=3).mean()

            # Ratio vs 30d average
            ma_30d = df[col].rolling(window=30, min_periods=7).mean()
            df[f"{col}_vs_30d"] = df[col] / ma_30d.replace(0, pd.NA)

            # Z-score (rolling 90d)
            mean_90d = df[col].rolling(window=90, min_periods=30).mean()
            std_90d = df[col].rolling(window=90, min_periods=30).std()
            df[f"{col}_zscore_90d"] = (df[col] - mean_90d) / std_90d.replace(0, pd.NA)

        # Mempool congestion flag
        if "mempool_size" in df.columns:
            mempool_mean = df["mempool_size"].rolling(window=30, min_periods=7).mean()
            df["mempool_above_avg"] = (df["mempool_size"] > 2 * mempool_mean).astype(int)

        print(f"  Final shape: {df.shape}")
        return df

    def report_quality(self, df: pd.DataFrame):
        """Print quality summary."""
        print("\n" + "=" * 60)
        print("DATA QUALITY")
        print("=" * 60)

        print(f"\nTotal rows: {len(df):,}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Days covered: {(df['timestamp'].max() - df['timestamp'].min()).days}")

        print(f"\nMetric coverage:")
        for col in df.columns:
            if col == "timestamp":
                continue
            non_nan = df[col].notna().sum()
            coverage = non_nan / len(df) * 100
            print(f"  {col:40s}: {non_nan:>4,} ({coverage:.1f}%)")

        # Sample row
        if len(df) > 100:
            mid_idx = len(df) // 2
            print(f"\nSample row (idx {mid_idx}): {df.iloc[mid_idx]['timestamp']}")
            for col in ["transactions_per_day", "tx_volume_usd", "mempool_size", "miners_revenue_usd"]:
                if col in df.columns:
                    val = df.iloc[mid_idx][col]
                    if pd.notna(val):
                        print(f"  {col}: {val:,.2f}")

    def save(self, df: pd.DataFrame, path: str = "data/network_flow.parquet"):
        """Save fetched data."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression="snappy", index=False)
        size_kb = os.path.getsize(path) / 1024
        print(f"\nSaved to: {path}")
        print(f"  File size: {size_kb:.1f} KB")


def main():
    print("=" * 60)
    print("MORGOTH GMX - WEEK 2 DAY 2 - NETWORK FLOW FETCHER")
    print("=" * 60)

    fetcher = BlockchainComFetcher()

    df = fetcher.fetch_all_metrics(timespan="3years")

    if df.empty:
        print("\nFAILED: No data fetched")
        return

    df = fetcher.add_derived_features(df)
    fetcher.save(df, "data/network_flow.parquet")
    fetcher.report_quality(df)

    print("\n" + "=" * 60)
    print("WEEK 2 DAY 2 COMPLETE")
    print("=" * 60)
    print("\nIMPORTANT: When merging with btc_features tomorrow,")
    print("enforce 1-day lag to prevent look-ahead bias!")
    print("(Daily data not known until end of day)")


if __name__ == "__main__":
    main()
