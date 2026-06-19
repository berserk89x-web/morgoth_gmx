"""
MORGOTH GMX - Funding Rate Fetcher
Week 1 Day 2

Downloads historical funding rates from Binance + Bybit.
Saves to data/funding_rates.parquet for feature engineering.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List


class FundingRateFetcher:
    """Fetches historical funding rates from major exchanges."""

    BINANCE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
    BYBIT_URL = "https://api.bybit.com/v5/market/funding/history"

    def __init__(self):
        self.session = requests.Session()

    def fetch_binance_history(
        self,
        symbol: str = "BTCUSDT",
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch all historical funding from Binance with pagination.

        Args:
            symbol: Trading pair
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD" (default: now)
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        print(f"\nFetching Binance funding history")
        print(f"  Symbol: {symbol}")
        print(f"  From: {start_date}")
        print(f"  To: {end_date}")
        print("=" * 60)

        all_data = []
        current_ms = start_ms
        batch = 0

        while current_ms < end_ms:
            batch += 1
            try:
                r = self.session.get(
                    self.BINANCE_URL,
                    params={
                        "symbol": symbol,
                        "startTime": current_ms,
                        "limit": 1000
                    },
                    timeout=15
                )
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"  Batch {batch}: error {e}, retrying in 3s...")
                time.sleep(3)
                continue

            if not data:
                print(f"  Batch {batch}: empty, advancing 30 days")
                current_ms += 30 * 24 * 60 * 60 * 1000
                continue

            all_data.extend(data)
            last_time = data[-1]["fundingTime"]

            # Advance to just after last entry
            current_ms = last_time + 1

            if batch % 3 == 0 or current_ms >= end_ms:
                last_dt = datetime.fromtimestamp(last_time / 1000)
                print(f"  Batch {batch}: {last_dt} | Total: {len(all_data):,}")

            time.sleep(0.05)  # Rate limit safety

        print("=" * 60)
        print(f"Binance complete: {len(all_data):,} entries in {batch} batches")

        if not all_data:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["funding_rate_binance"] = pd.to_numeric(df["fundingRate"])
        if "markPrice" in df.columns:
            df["mark_price_binance"] = pd.to_numeric(df["markPrice"], errors="coerce")

        df = df[["timestamp", "funding_rate_binance"]].drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        return df

    def fetch_bybit_history(
        self,
        symbol: str = "BTCUSDT",
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """Fetch historical funding from Bybit."""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Bybit needs end-to-start direction with endTime
        end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)

        print(f"\nFetching Bybit funding history")
        print(f"  Symbol: {symbol}")
        print(f"  From: {start_date}")
        print(f"  To: {end_date}")
        print("=" * 60)

        all_data = []
        current_end = end_ms
        batch = 0

        while current_end > start_ms:
            batch += 1
            try:
                r = self.session.get(
                    self.BYBIT_URL,
                    params={
                        "category": "linear",
                        "symbol": symbol,
                        "endTime": current_end,
                        "limit": 200
                    },
                    timeout=15
                )
                r.raise_for_status()
                resp = r.json()
            except Exception as e:
                print(f"  Batch {batch}: error {e}, retrying in 3s...")
                time.sleep(3)
                continue

            if resp.get("retCode") != 0:
                print(f"  Batch {batch}: API error {resp.get('retMsg')}")
                break

            data = resp.get("result", {}).get("list", [])
            if not data:
                print(f"  Batch {batch}: no more data")
                break

            all_data.extend(data)
            # Bybit returns newest-first; oldest in batch is last
            oldest_time = int(data[-1]["fundingRateTimestamp"])
            current_end = oldest_time - 1

            if batch % 5 == 0:
                last_dt = datetime.fromtimestamp(oldest_time / 1000)
                print(f"  Batch {batch}: {last_dt} | Total: {len(all_data):,}")

            time.sleep(0.1)  # Rate limit safety

        print("=" * 60)
        print(f"Bybit complete: {len(all_data):,} entries in {batch} batches")

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        # NB: use int64 explicitly — on Windows .astype(int) is C long (32-bit)
        # and overflows on millisecond epochs (~1.78e12).
        df["timestamp"] = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms")
        df["funding_rate_bybit"] = pd.to_numeric(df["fundingRate"])

        df = df[["timestamp", "funding_rate_bybit"]].drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        return df

    def merge_sources(
        self,
        binance_df: pd.DataFrame,
        bybit_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge funding data from multiple exchanges."""
        print("\nMerging exchange data...")

        # Align timestamps: exchanges stamp funding with ms offsets
        # (Binance e.g. 00:00:00.005, Bybit 00:00:00.000). Floor both to the
        # hour so the same 8h funding event lines up across exchanges, giving
        # a non-NaN cross-exchange spread on every shared row.
        binance_df = binance_df.copy()
        bybit_df = bybit_df.copy()
        binance_df["timestamp"] = binance_df["timestamp"].dt.floor("h")
        bybit_df["timestamp"] = bybit_df["timestamp"].dt.floor("h")

        # Both should be 8h cadence
        merged = pd.merge(
            binance_df,
            bybit_df,
            on="timestamp",
            how="outer"
        ).sort_values("timestamp").reset_index(drop=True)

        # Cross-exchange spread feature
        merged["funding_spread_binance_bybit"] = (
            merged["funding_rate_binance"] - merged["funding_rate_bybit"]
        )

        # Average funding (cross-exchange consensus)
        merged["funding_rate_avg"] = merged[
            ["funding_rate_binance", "funding_rate_bybit"]
        ].mean(axis=1)

        # Stats
        print(f"  Total merged rows: {len(merged):,}")
        print(f"  Date range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")
        print(f"  Binance coverage: {merged['funding_rate_binance'].notna().sum():,}")
        print(f"  Bybit coverage: {merged['funding_rate_bybit'].notna().sum():,}")

        # Quality check
        print(f"\nFunding rate statistics (Binance):")
        print(f"  Mean: {merged['funding_rate_binance'].mean()*100:.4f}%")
        print(f"  Std: {merged['funding_rate_binance'].std()*100:.4f}%")
        print(f"  Min: {merged['funding_rate_binance'].min()*100:.4f}%")
        print(f"  Max: {merged['funding_rate_binance'].max()*100:.4f}%")

        print(f"\nCross-exchange spread statistics:")
        print(f"  Mean spread: {merged['funding_spread_binance_bybit'].mean()*100:.4f}%")
        print(f"  Std spread: {merged['funding_spread_binance_bybit'].std()*100:.4f}%")
        print(f"  Max abs spread: {merged['funding_spread_binance_bybit'].abs().max()*100:.4f}%")

        return merged

    def save(self, df: pd.DataFrame, path: str = "data/funding_rates.parquet"):
        """Save merged funding data."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression="snappy", index=False)
        size_mb = os.path.getsize(path) / 1024**2
        print(f"\nSaved to: {path}")
        print(f"  File size: {size_mb:.2f} MB")


def main():
    print("=" * 60)
    print("MORGOTH GMX - WEEK 1 DAY 2 - FUNDING FETCHER")
    print("=" * 60)

    fetcher = FundingRateFetcher()

    # 3 years to match our BTC dataset
    start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")

    # Fetch from both exchanges
    binance_df = fetcher.fetch_binance_history(start_date=start_date)
    bybit_df = fetcher.fetch_bybit_history(start_date=start_date)

    if binance_df.empty:
        print("ERROR: No Binance data fetched")
        return

    # Merge
    merged = fetcher.merge_sources(binance_df, bybit_df)

    # Save
    fetcher.save(merged, "data/funding_rates.parquet")

    # Preview
    print("\nFirst 3 rows:")
    print(merged.head(3))
    print("\nLast 3 rows:")
    print(merged.tail(3))

    print("\n" + "=" * 60)
    print("WEEK 1 DAY 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
