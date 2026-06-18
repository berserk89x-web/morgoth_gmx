"""
MORGOTH GMX - Data Fetcher
Downloads BTC 5-min OHLCV from Binance public API
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


class BinanceDataFetcher:
    """Fetches OHLCV data from Binance public API."""

    BASE_URL = "https://api.binance.com/api/v3/klines"
    MAX_LIMIT = 1000  # Binance max per request

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "5m"):
        self.symbol = symbol
        self.interval = interval
        self.session = requests.Session()

    def fetch_candles(self, start_ms: int, end_ms: int = None) -> list:
        """Fetch one batch of candles starting from start_ms."""
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "startTime": start_ms,
            "limit": self.MAX_LIMIT
        }
        if end_ms:
            params["endTime"] = end_ms

        try:
            r = self.session.get(self.BASE_URL, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"  Request failed: {e}")
            return []

    def download_historical(
        self,
        start_date: str,
        end_date: str = None,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        Download all candles between start_date and end_date.

        Args:
            start_date: "YYYY-MM-DD" format
            end_date: "YYYY-MM-DD" format (default: now)
            output_path: Save path for parquet file
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        print(f"\nDownloading {self.symbol} {self.interval} candles")
        print(f"From: {start_dt}")
        print(f"To:   {end_dt}")
        print(f"Expected: ~{(end_dt - start_dt).days * 24 * 12} candles")
        print("=" * 60)

        all_candles = []
        current_ms = start_ms
        batch_count = 0

        while current_ms < end_ms:
            batch_count += 1
            candles = self.fetch_candles(current_ms, end_ms)

            if not candles:
                print(f"  Batch {batch_count}: empty, retrying in 3s...")
                time.sleep(3)
                continue

            all_candles.extend(candles)

            # Move to just after the last candle's close time
            last_open = candles[-1][0]
            current_ms = last_open + (5 * 60 * 1000)  # +5 min

            # Progress reporting
            last_dt = datetime.fromtimestamp(last_open / 1000)
            if batch_count % 10 == 0 or current_ms >= end_ms:
                print(f"  Batch {batch_count:3d}: {last_dt} | Total: {len(all_candles):,} candles")

            # Rate limit safety (1200 weight/min, our calls = 1 weight each)
            time.sleep(0.05)

        print("=" * 60)
        print(f"Download complete: {len(all_candles):,} candles in {batch_count} batches")

        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        # Type conversions
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce", downcast="integer")

        # Select final columns
        df = df[[
            "timestamp", "open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"
        ]]

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # Quality report
        print(f"\nData quality:")
        print(f"  Rows: {len(df):,}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Null values: {df.isnull().sum().sum()}")
        print(f"  Price range: ${df['close'].min():,.2f} to ${df['close'].max():,.2f}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        # Save to parquet
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, compression="snappy", index=False)
            file_size_mb = os.path.getsize(output_path) / 1024**2
            print(f"\nSaved to: {output_path}")
            print(f"File size: {file_size_mb:.1f} MB")

        return df


def main():
    """Download 3 years of BTC 5-min data."""
    print("=" * 60)
    print("MORGOTH GMX - PHASE 2 DATA DOWNLOAD")
    print("=" * 60)

    # 3 years back from today
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    output = "data/btc_5min_historical.parquet"

    fetcher = BinanceDataFetcher(symbol="BTCUSDT", interval="5m")
    df = fetcher.download_historical(
        start_date=start_date,
        end_date=end_date,
        output_path=output
    )

    print("\n" + "=" * 60)
    print("PHASE 2 BLOCK 1 COMPLETE")
    print("=" * 60)
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nLast 3 rows:")
    print(df.tail(3))


if __name__ == "__main__":
    main()
