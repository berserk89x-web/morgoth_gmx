"""
MORGOTH GMX - Exchange Flow API Research
Week 2 Day 1

Tests free APIs for BTC network/flow data.
"""

import requests
import time
from datetime import datetime, timedelta


class FlowAPIResearch:
    """Test exchange flow / network data APIs."""

    def __init__(self):
        self.results = {}

    def test_mempool_space(self):
        """Test mempool.space free API."""
        print("\n[1] MEMPOOL.SPACE")
        print("-" * 60)

        try:
            # Test current mempool stats
            r = requests.get("https://mempool.space/api/mempool", timeout=10)
            r.raise_for_status()
            data = r.json()
            print(f"  Mempool stats: OK")
            print(f"  Sample: count={data.get('count')}, vsize={data.get('vsize')}, fees={data.get('total_fee')}")

            # Test historical mempool data
            r = requests.get("https://mempool.space/api/v1/historical-price?currency=USD&timestamp=1700000000", timeout=10)
            r.raise_for_status()
            data = r.json()
            print(f"  Historical price API: OK")
            print(f"  Sample: {data}")

            # Test recent blocks
            r = requests.get("https://mempool.space/api/v1/blocks", timeout=10)
            r.raise_for_status()
            blocks = r.json()
            print(f"  Recent blocks: {len(blocks)} returned")
            if blocks:
                print(f"  Sample block keys: {list(blocks[0].keys())[:10]}")
                # Volume / fees / tx_count are valuable
                first = blocks[0]
                print(f"  First block: height={first.get('height')}, txCount={first.get('tx_count')}, totalFees={first.get('totalFees')}")

            # Test difficulty adjustments (network health proxy)
            r = requests.get("https://mempool.space/api/v1/difficulty-adjustment", timeout=10)
            r.raise_for_status()
            print(f"  Difficulty API: OK")

            # Test mining hashrate
            r = requests.get("https://mempool.space/api/v1/mining/hashrate/1m", timeout=10)
            r.raise_for_status()
            data = r.json()
            print(f"  Hashrate API: OK")

            self.results["mempool.space"] = {
                "works": True,
                "endpoints": ["mempool", "blocks", "difficulty", "hashrate", "price"],
                "historical": True
            }
        except Exception as e:
            print(f"  Error: {e}")
            self.results["mempool.space"] = {"works": False, "error": str(e)[:100]}

    def test_blockchain_com(self):
        """Test blockchain.com free API."""
        print("\n[2] BLOCKCHAIN.COM")
        print("-" * 60)

        try:
            # Stats endpoint
            r = requests.get("https://api.blockchain.info/stats", timeout=10)
            r.raise_for_status()
            data = r.json()
            print(f"  Stats API: OK")
            print(f"  BTC price: ${data.get('market_price_usd', 0):,.2f}")
            print(f"  Total BTC: {data.get('totalbc', 0)/1e8:,.0f}")
            print(f"  Hash rate: {data.get('hash_rate', 0)/1e6:,.0f} TH/s")
            print(f"  Difficulty: {data.get('difficulty', 0):,.0f}")

            # Charts API (historical)
            metrics = [
                "n-transactions",          # Daily transactions
                "estimated-transaction-volume-usd",  # USD volume
                "n-payments",              # Daily payments
                "mempool-count",           # Mempool size
                "miners-revenue",          # Miner income
            ]

            print(f"\n  Historical charts available:")
            for metric in metrics:
                try:
                    r = requests.get(
                        f"https://api.blockchain.info/charts/{metric}?timespan=1year&format=json",
                        timeout=10
                    )
                    r.raise_for_status()
                    data = r.json()
                    values = data.get("values", [])
                    if values:
                        print(f"    {metric}: {len(values)} daily entries")
                except Exception as e:
                    print(f"    {metric}: error {str(e)[:50]}")
                time.sleep(0.1)

            self.results["blockchain.com"] = {
                "works": True,
                "endpoints": ["stats", "charts (historical)"],
                "historical": True,
                "cadence": "daily"
            }
        except Exception as e:
            print(f"  Error: {e}")
            self.results["blockchain.com"] = {"works": False, "error": str(e)[:100]}

    def test_blockcypher(self):
        """Test BlockCypher free API."""
        print("\n[3] BLOCKCYPHER")
        print("-" * 60)

        try:
            r = requests.get("https://api.blockcypher.com/v1/btc/main", timeout=10)
            r.raise_for_status()
            data = r.json()
            print(f"  Main API: OK")
            print(f"  Height: {data.get('height')}")
            print(f"  Latest hash: {data.get('hash', '')[:20]}...")
            print(f"  Unconfirmed: {data.get('unconfirmed_count')}")

            self.results["blockcypher"] = {
                "works": True,
                "note": "Rate limited 200/hour free",
                "historical": "limited"
            }
        except Exception as e:
            print(f"  Error: {e}")
            self.results["blockcypher"] = {"works": False, "error": str(e)[:100]}

    def test_bitquery_check(self):
        """Note about Bitquery (requires signup for key)."""
        print("\n[4] BITQUERY")
        print("-" * 60)
        print("  NOTE: Bitquery requires API key (free signup)")
        print("  Endpoint: https://graphql.bitquery.io")
        print("  Free tier: 1000 req/month, 1000 rows/request")
        print("  GraphQL with rich on-chain queries")
        print("  Has BTC exchange flow data with attribution")
        print("  RECOMMENDATION: Signup later if needed")

        self.results["bitquery"] = {
            "works": "requires_key",
            "note": "Free signup, 1000 req/month"
        }

    def summary(self):
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for source, info in self.results.items():
            print(f"\n{source.upper()}:")
            for key, val in info.items():
                print(f"  {key}: {val}")

        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        print("""
Based on results:

PRIMARY: blockchain.com Charts API
- Historical daily data (years back)
- No auth required
- Multiple useful metrics
- Best for backfill

SECONDARY: mempool.space
- Real-time mempool data
- Recent blocks with tx counts/fees
- Good for live signal

FEATURES TO EXTRACT (from these sources):
- Daily transaction count (network activity)
- USD transaction volume (capital flow)
- Mempool size (congestion = volatility hint)
- Miner revenue (network economics)
- Hashrate trend (security/health)

NOTE: NO exchange attribution in free tier.
These are PROXY signals, not direct exchange flow.
Still valuable for ensemble - markets respond to:
- Increased transaction activity
- Network congestion
- USD volume changes

LIMITATION:
- All data is DAILY cadence (not 5-min)
- Need to forward-fill to match BTC dataset
- Less granular than funding (8h)
- But CHEAPER (free, simple)
        """)


def main():
    print("=" * 60)
    print("MORGOTH GMX - WEEK 2 DAY 1 - FLOW API RESEARCH")
    print("=" * 60)

    research = FlowAPIResearch()
    research.test_mempool_space()
    research.test_blockchain_com()
    research.test_blockcypher()
    research.test_bitquery_check()
    research.summary()


if __name__ == "__main__":
    main()
