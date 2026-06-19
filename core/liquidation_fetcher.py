"""
MORGOTH GMX - Liquidation Data Fetcher
Phase 3 Block 3 - Week 1

Downloads BTC liquidation events from Hyperliquid public API.
Aggregates to 5-min buckets matching our existing OHLCV data.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class HyperliquidLiquidationFetcher:
    """Fetches liquidation events from Hyperliquid public API."""

    BASE_URL = "https://api.hyperliquid.xyz/info"

    def __init__(self, coin: str = "BTC"):
        self.coin = coin
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })

    def test_connection(self) -> bool:
        """Test API connectivity."""
        try:
            response = self.session.post(
                self.BASE_URL,
                json={"type": "meta"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            print(f"  Connected to Hyperliquid API")
            print(f"  Universe entries: {len(data.get('universe', []))}")

            # Verify BTC is available
            universe = data.get('universe', [])
            btc_found = any(asset.get('name') == self.coin for asset in universe)
            print(f"  BTC available: {btc_found}")

            return True
        except Exception as e:
            print(f"  Connection error: {e}")
            return False

    def get_recent_trades(self, n: int = 100) -> Optional[pd.DataFrame]:
        """Get recent trades to verify data structure."""
        try:
            response = self.session.post(
                self.BASE_URL,
                json={"type": "recentTrades", "coin": self.coin},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                print("  No recent trades returned")
                return None

            df = pd.DataFrame(data)
            print(f"  Recent trades retrieved: {len(df)} entries")
            print(f"  Columns: {list(df.columns)[:10]}")
            return df
        except Exception as e:
            print(f"  Trades fetch error: {e}")
            return None

    def explore_api(self) -> dict:
        """Explore available API endpoints for liquidation data."""
        results = {}

        # Test different endpoint types
        endpoints_to_try = [
            {"type": "meta"},
            {"type": "recentTrades", "coin": self.coin},
            {"type": "candleSnapshot", "req": {
                "coin": self.coin,
                "interval": "5m",
                "startTime": int((datetime.now() - timedelta(days=1)).timestamp() * 1000),
                "endTime": int(datetime.now().timestamp() * 1000)
            }},
        ]

        print("\nExploring Hyperliquid API endpoints:")
        for req_body in endpoints_to_try:
            req_type = req_body.get("type", "unknown")
            try:
                response = self.session.post(
                    self.BASE_URL,
                    json=req_body,
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list):
                    print(f"  [{req_type}]: list with {len(data)} entries")
                    if data:
                        print(f"    Sample keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'non-dict'}")
                elif isinstance(data, dict):
                    print(f"  [{req_type}]: dict with keys {list(data.keys())[:5]}")

                results[req_type] = data
            except Exception as e:
                print(f"  [{req_type}]: error {str(e)[:60]}")

        return results


def main():
    print("=" * 60)
    print("MORGOTH GMX - PHASE 3 BLOCK 3 - LIQUIDATION FETCHER")
    print("=" * 60)
    print("\nDay 1-2: Hyperliquid API research + connection test")

    fetcher = HyperliquidLiquidationFetcher(coin="BTC")

    # Test 1: Connection
    print("\n[1] Testing connection...")
    if not fetcher.test_connection():
        print("FAILED - cannot proceed")
        return

    # Test 2: Recent trades (verify data structure)
    print("\n[2] Fetching recent BTC trades...")
    trades = fetcher.get_recent_trades(n=10)

    if trades is not None and not trades.empty:
        print(f"\nSample trade data:")
        print(trades.head(3).to_string())

        # Check if there's a 'liquidation' field
        liq_columns = [col for col in trades.columns if 'liq' in col.lower()]
        if liq_columns:
            print(f"\n  Liquidation-related columns: {liq_columns}")
        else:
            print(f"\n  No liquidation flag in trade structure")
            print(f"  Note: Hyperliquid may report liquidations separately")

    # Test 3: Explore other endpoints
    print("\n[3] Exploring API for liquidation data...")
    fetcher.explore_api()

    print("\n" + "=" * 60)
    print("API EXPLORATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("- Verify if Hyperliquid exposes liquidation history endpoint")
    print("- If not: use alternative source (Coinglass, Binance liq feed)")
    print("- Build aggregation to 5-min buckets")
    print("- Add features to btc_features.parquet")


if __name__ == "__main__":
    main()
