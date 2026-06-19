"""
MORGOTH GMX - Funding Rate API Research
Week 1 Day 1

Tests free funding rate APIs from major exchanges.
Reports historical depth and data quality.
"""

import requests
import time
from datetime import datetime, timedelta
import json


class FundingAPIResearch:
    """Test funding rate APIs across exchanges."""

    def __init__(self):
        self.results = {}

    def test_binance(self):
        """Test Binance Futures funding history API."""
        print("\n[1] BINANCE FUTURES")
        print("-" * 60)

        # Test current funding rate
        try:
            r = requests.get(
                "https://fapi.binance.com/fapi/v1/premiumIndex",
                params={"symbol": "BTCUSDT"},
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
            print(f"  Current funding rate API: OK")
            print(f"  Current funding: {float(data['lastFundingRate']) * 100:.4f}%")
            print(f"  Next funding: {datetime.fromtimestamp(data['nextFundingTime']/1000)}")
        except Exception as e:
            print(f"  Current funding error: {e}")
            return

        # Test historical funding (max 1000 entries per call)
        try:
            r = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": "BTCUSDT", "limit": 1000},
                timeout=10
            )
            r.raise_for_status()
            history = r.json()
            print(f"  Historical funding API: OK")
            print(f"  Returned: {len(history)} entries")
            if history:
                oldest = datetime.fromtimestamp(history[0]['fundingTime']/1000)
                newest = datetime.fromtimestamp(history[-1]['fundingTime']/1000)
                print(f"  Date range: {oldest} to {newest}")
                print(f"  Sample: {history[0]}")

            self.results["binance"] = {
                "works": True,
                "entries_per_call": len(history),
                "has_history": True
            }
        except Exception as e:
            print(f"  Historical funding error: {e}")

    def test_hyperliquid(self):
        """Test Hyperliquid funding rate API."""
        print("\n[2] HYPERLIQUID")
        print("-" * 60)

        try:
            # Get meta first to verify BTC
            r = requests.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "meta"},
                timeout=10
            )
            r.raise_for_status()
            print(f"  API reachable: OK")

            # Try funding history endpoint
            now_ms = int(datetime.now().timestamp() * 1000)
            week_ago_ms = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

            r = requests.post(
                "https://api.hyperliquid.xyz/info",
                json={
                    "type": "fundingHistory",
                    "coin": "BTC",
                    "startTime": week_ago_ms,
                    "endTime": now_ms
                },
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
            print(f"  Historical funding API: OK")
            print(f"  Returned: {len(data)} entries (last 7 days)")
            if data:
                print(f"  Sample: {data[0]}")
                oldest = datetime.fromtimestamp(data[0]['time']/1000) if 'time' in data[0] else "unknown"
                print(f"  Oldest: {oldest}")

            self.results["hyperliquid"] = {
                "works": True,
                "entries_per_call": len(data),
                "has_history": True
            }
        except Exception as e:
            print(f"  Error: {e}")

    def test_okx(self):
        """Test OKX funding rate API."""
        print("\n[3] OKX")
        print("-" * 60)

        try:
            # Current funding rate
            r = requests.get(
                "https://www.okx.com/api/v5/public/funding-rate",
                params={"instId": "BTC-USDT-SWAP"},
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
            if data.get("code") == "0":
                rate = float(data["data"][0]["fundingRate"]) * 100
                print(f"  Current funding API: OK")
                print(f"  Current funding: {rate:.4f}%")

            # Historical funding
            r = requests.get(
                "https://www.okx.com/api/v5/public/funding-rate-history",
                params={"instId": "BTC-USDT-SWAP", "limit": 100},
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
            if data.get("code") == "0":
                history = data["data"]
                print(f"  Historical API: OK")
                print(f"  Returned: {len(history)} entries")
                if history:
                    print(f"  Sample: {history[0]}")

                self.results["okx"] = {
                    "works": True,
                    "entries_per_call": len(history),
                    "has_history": True
                }
        except Exception as e:
            print(f"  Error: {e}")

    def test_bybit(self):
        """Test Bybit funding rate API."""
        print("\n[4] BYBIT")
        print("-" * 60)

        try:
            # Current funding
            r = requests.get(
                "https://api.bybit.com/v5/market/tickers",
                params={"category": "linear", "symbol": "BTCUSDT"},
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                rate = float(data["result"]["list"][0]["fundingRate"]) * 100
                print(f"  Current funding API: OK")
                print(f"  Current funding: {rate:.4f}%")

            # Historical funding
            r = requests.get(
                "https://api.bybit.com/v5/market/funding/history",
                params={"category": "linear", "symbol": "BTCUSDT", "limit": 200},
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
            if data.get("retCode") == 0:
                history = data.get("result", {}).get("list", [])
                print(f"  Historical API: OK")
                print(f"  Returned: {len(history)} entries")
                if history:
                    print(f"  Sample: {history[0]}")

                self.results["bybit"] = {
                    "works": True,
                    "entries_per_call": len(history),
                    "has_history": True
                }
        except Exception as e:
            print(f"  Error: {e}")

    def summary(self):
        """Print decision summary."""
        print("\n" + "=" * 60)
        print("SUMMARY & RECOMMENDATIONS")
        print("=" * 60)

        for exchange, info in self.results.items():
            status = "WORKS" if info.get("works") else "FAILED"
            print(f"\n{exchange.upper()}: {status}")
            if info.get("works"):
                print(f"  Entries per call: {info.get('entries_per_call')}")
                print(f"  Historical: {info.get('has_history')}")

        print("\n" + "=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        print("""
Funding rates are typically 8-hour intervals.
3 years of data = ~3285 funding events per exchange.

Best strategy:
- PRIMARY: Pick exchange with deepest history + most reliable API
- SECONDARY: Add 1-2 more for cross-exchange spread features
- FEATURES:
  * current_funding_rate (from primary)
  * funding_z_score_30d (extreme detection)
  * cross_exchange_spread (primary - secondary)
  * funding_change_24h
  * funding_above_extreme_threshold
        """)


def main():
    print("=" * 60)
    print("MORGOTH GMX - WEEK 1 DAY 1 - FUNDING API RESEARCH")
    print("=" * 60)

    research = FundingAPIResearch()
    research.test_binance()
    research.test_hyperliquid()
    research.test_okx()
    research.test_bybit()
    research.summary()


if __name__ == "__main__":
    main()
