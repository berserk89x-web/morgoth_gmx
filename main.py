"""
MORGOTH GMX - Main Orchestrator
Phase 1 Placeholder
"""

import os
from dotenv import load_dotenv

load_dotenv()


def main():
    print("=" * 60)
    print("MORGOTH GMX - INITIALIZATION")
    print("=" * 60)
    print(f"\nMode: {'DRY_RUN' if os.getenv('DRY_RUN', 'true') == 'true' else 'LIVE'}")
    print(f"Wallet: {os.getenv('WALLET_ADDRESS', 'NOT SET')}")
    print(f"Consensus threshold: {os.getenv('CONSENSUS_THRESHOLD', '0.80')}")
    print("\nStatus: Foundation initialized")
    print("Next: Build data pipeline (Phase 2)")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
