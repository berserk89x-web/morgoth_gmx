"""
MORGOTH GMX - Flow Features Validation
Week 2 Day 4

Tests predictive power of network flow features.
Honest verdict on whether they belong in XGBoost retrain.
"""

import pandas as pd
import numpy as np


def load_data():
    """Load enriched dataset."""
    df = pd.read_parquet("data/btc_features.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Drop warmup (need 90d for z-scores = ~25k rows of 5-min)
    # Plus 300 rows for TA warmup
    df = df.iloc[26000:].copy()
    df = df.iloc[:-12].copy()

    print(f"Validation dataset: {len(df):,} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def test_mempool_signal(df):
    """Test if mempool congestion predicts returns."""
    print("\n" + "=" * 60)
    print("TEST 1: MEMPOOL CONGESTION vs FORWARD RETURNS")
    print("=" * 60)

    if "mempool_size_zscore_90d" not in df.columns:
        print("  Column not found, skipping")
        return

    valid = df[["mempool_size_zscore_90d", "target_ret_12"]].dropna()
    print(f"\n  Sample size: {len(valid):,}")

    corr = valid["mempool_size_zscore_90d"].corr(valid["target_ret_12"])
    print(f"  Correlation mempool z-score vs 1h return: {corr:+.4f}")

    # Bucket analysis
    df_v = valid.copy()
    df_v["bucket"] = pd.cut(
        df_v["mempool_size_zscore_90d"],
        bins=[-np.inf, -1, 0, 1, 2, np.inf],
        labels=["very_low", "low", "high", "very_high", "extreme"]
    )

    print(f"\n  Returns by mempool z-score bucket (1h):")
    for bucket, group in df_v.groupby("bucket", observed=True):
        if len(group) > 100:
            avg_ret = group["target_ret_12"].mean() * 100
            wr = (group["target_ret_12"] > 0).mean() * 100
            print(f"    {bucket}: n={len(group):>6,}, avg={avg_ret:+.4f}%, wr={wr:.1f}%")


def test_tx_volume_signal(df):
    """Test if transaction volume changes predict returns."""
    print("\n" + "=" * 60)
    print("TEST 2: TX VOLUME CHANGE vs FORWARD RETURNS")
    print("=" * 60)

    if "tx_volume_usd_change_1d" not in df.columns:
        print("  Column not found, skipping")
        return

    valid = df[["tx_volume_usd_change_1d", "target_ret_12"]].dropna()
    print(f"\n  Sample size: {len(valid):,}")

    corr = valid["tx_volume_usd_change_1d"].corr(valid["target_ret_12"])
    print(f"  Correlation tx volume change vs 1h return: {corr:+.4f}")


def test_multi_horizon(df):
    """Test flow features across multiple horizons."""
    print("\n" + "=" * 60)
    print("TEST 3: MULTI-HORIZON ANALYSIS")
    print("=" * 60)

    # Compute multi-day returns
    df["target_ret_288"] = df["close"].shift(-288) / df["close"] - 1   # 1 day
    df["target_ret_864"] = df["close"].shift(-864) / df["close"] - 1   # 3 days
    df["target_ret_2016"] = df["close"].shift(-2016) / df["close"] - 1 # 7 days

    flow_features = [
        ("mempool_size_zscore_90d", "mempool z-score"),
        ("tx_volume_usd_zscore_90d", "tx volume z-score"),
        ("miners_revenue_usd_zscore_90d", "miners revenue z-score"),
        ("hash_rate_th_zscore_90d", "hash rate z-score"),
    ]

    horizons = ["target_ret_12", "target_ret_288", "target_ret_864", "target_ret_2016"]
    horizon_names = ["1h", "1d", "3d", "7d"]

    print(f"\n{'Feature':<30} {'1h':<12} {'1d':<12} {'3d':<12} {'7d':<12}")
    print("-" * 78)

    for col, name in flow_features:
        if col not in df.columns:
            print(f"  {name:<30} NOT AVAILABLE")
            continue

        row = f"{name:<30}"
        for h_col, h_name in zip(horizons, horizon_names):
            valid = df[[col, h_col]].dropna()
            if len(valid) < 100:
                row += f"{'n/a':<12}"
            else:
                corr = valid[col].corr(valid[h_col])
                row += f"{corr:+.4f}      "
        print(row)


def test_extreme_events(df):
    """Test extreme network events vs returns."""
    print("\n" + "=" * 60)
    print("TEST 4: EXTREME NETWORK EVENTS")
    print("=" * 60)

    if "mempool_above_avg" not in df.columns:
        print("  Column not found, skipping")
        return

    # Mempool 2x above average (congestion event)
    congestion_mask = df["mempool_above_avg"] == 1
    congestion_count = congestion_mask.sum()

    if congestion_count > 0:
        avg_1h = df.loc[congestion_mask, "target_ret_12"].mean() * 100
        win_rate = (df.loc[congestion_mask, "target_ret_12"] > 0).mean() * 100

        base_ret = df["target_ret_12"].mean() * 100
        base_wr = (df["target_ret_12"] > 0).mean() * 100

        print(f"\nMempool congestion events (2x above 30d avg):")
        print(f"  Events: {congestion_count:,}")
        print(f"  Avg 1h return: {avg_1h:+.4f}% (baseline: {base_ret:+.4f}%)")
        print(f"  1h win rate: {win_rate:.1f}% (baseline: {base_wr:.1f}%)")

        edge = avg_1h - base_ret
        print(f"  Edge vs baseline: {edge:+.4f}%")


def summary_verdict(df):
    """Final honest verdict."""
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    signals_passing = 0

    # Check 1: Mempool z-score correlation
    if "mempool_size_zscore_90d" in df.columns:
        valid = df[["mempool_size_zscore_90d", "target_ret_12"]].dropna()
        corr_mempool = valid["mempool_size_zscore_90d"].corr(valid["target_ret_12"]) if len(valid) > 100 else 0
        if abs(corr_mempool) > 0.02:
            signals_passing += 1
        print(f"\n  Mempool correlation: {corr_mempool:+.4f} ({'PASS' if abs(corr_mempool) > 0.02 else 'FAIL'})")

    # Check 2: Tx volume correlation
    if "tx_volume_usd_zscore_90d" in df.columns:
        valid = df[["tx_volume_usd_zscore_90d", "target_ret_12"]].dropna()
        corr_volume = valid["tx_volume_usd_zscore_90d"].corr(valid["target_ret_12"]) if len(valid) > 100 else 0
        if abs(corr_volume) > 0.02:
            signals_passing += 1
        print(f"  Tx volume correlation: {corr_volume:+.4f} ({'PASS' if abs(corr_volume) > 0.02 else 'FAIL'})")

    # Check 3: Hash rate correlation
    if "hash_rate_th_zscore_90d" in df.columns:
        valid = df[["hash_rate_th_zscore_90d", "target_ret_12"]].dropna()
        corr_hash = valid["hash_rate_th_zscore_90d"].corr(valid["target_ret_12"]) if len(valid) > 100 else 0
        if abs(corr_hash) > 0.02:
            signals_passing += 1
        print(f"  Hash rate correlation: {corr_hash:+.4f} ({'PASS' if abs(corr_hash) > 0.02 else 'FAIL'})")

    print(f"\nSignals passing: {signals_passing}/3")

    if signals_passing >= 2:
        print("STRONG: Flow features show real signal - definite ensemble value")
    elif signals_passing == 1:
        print("WEAK: One signal - marginal value, keep for ensemble")
    else:
        print("FAIL: No standalone signal - tree model interactions only hope")

    print("\nIMPORTANT:")
    print("Like funding, flow is judged at Week 4 XGBoost retrain.")
    print("Tree models can extract value from features with no linear correlation.")
    print("0/3 here does NOT mean flow is useless - it means no LINEAR edge.")


def main():
    print("=" * 60)
    print("MORGOTH GMX - WEEK 2 DAY 4 - FLOW VALIDATION")
    print("=" * 60)

    df = load_data()
    test_mempool_signal(df)
    test_tx_volume_signal(df)
    test_multi_horizon(df)
    test_extreme_events(df)
    summary_verdict(df)

    print("\n" + "=" * 60)
    print("WEEK 2 DAY 4 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
