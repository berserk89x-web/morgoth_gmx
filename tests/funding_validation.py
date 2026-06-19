"""
MORGOTH GMX - Funding Features Validation
Week 1 Day 4
"""

import pandas as pd
import numpy as np


def load_data():
    df = pd.read_parquet("data/btc_features.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.iloc[9000:].copy()
    df = df.iloc[:-12].copy()
    print(f"Validation dataset: {len(df):,} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def test_extreme_events(df):
    print("\n" + "=" * 60)
    print("TEST 1: EXTREME FUNDING EVENTS vs FORWARD RETURNS")
    print("=" * 60)

    high_mask = df["funding_extreme_high"] == 1
    high_count = high_mask.sum()

    if high_count > 0:
        avg_1h = df.loc[high_mask, "target_ret_12"].mean() * 100
        win_rate_1h = (df.loc[high_mask, "target_ret_12"] > 0).mean() * 100
        print(f"\nHIGH funding extremes (crowded longs):")
        print(f"  Events: {high_count:,}")
        print(f"  Avg 1h forward return: {avg_1h:+.4f}%")
        print(f"  1h win rate: {win_rate_1h:.1f}%")
        print(f"  Hypothesis: Contrarian - should see NEGATIVE returns")

    low_mask = df["funding_extreme_low"] == 1
    low_count = low_mask.sum()

    if low_count > 0:
        avg_1h = df.loc[low_mask, "target_ret_12"].mean() * 100
        win_rate_1h = (df.loc[low_mask, "target_ret_12"] > 0).mean() * 100
        print(f"\nLOW funding extremes (capitulation):")
        print(f"  Events: {low_count:,}")
        print(f"  Avg 1h forward return: {avg_1h:+.4f}%")
        print(f"  1h win rate: {win_rate_1h:.1f}%")
        print(f"  Hypothesis: Contrarian - should see POSITIVE returns")

    base_1h = df["target_ret_12"].mean() * 100
    base_wr = (df["target_ret_12"] > 0).mean() * 100
    print(f"\nBaseline (all rows):")
    print(f"  Avg 1h return: {base_1h:+.4f}%")
    print(f"  1h win rate: {base_wr:.1f}%")


def test_zscore_correlation(df):
    print("\n" + "=" * 60)
    print("TEST 2: Z-SCORE vs FORWARD RETURNS CORRELATION")
    print("=" * 60)

    valid = df[["funding_zscore_7d", "target_ret_1", "target_ret_3", "target_ret_12"]].dropna()
    print(f"\n  Sample size: {len(valid):,}")

    for ret_col in ["target_ret_1", "target_ret_3", "target_ret_12"]:
        corr = valid["funding_zscore_7d"].corr(valid[ret_col])
        print(f"  Correlation z-score vs {ret_col}: {corr:+.4f}")

    print("\n  Returns by z-score bucket (1h forward):")
    df_v = valid.copy()
    df_v["zscore_bucket"] = pd.cut(
        df_v["funding_zscore_7d"],
        bins=[-np.inf, -2, -1, 0, 1, 2, np.inf],
        labels=["z<-2", "-2<=z<-1", "-1<=z<0", "0<=z<1", "1<=z<2", "z>=2"]
    )

    for bucket, group in df_v.groupby("zscore_bucket", observed=True):
        if len(group) > 100:
            avg_ret = group["target_ret_12"].mean() * 100
            wr = (group["target_ret_12"] > 0).mean() * 100
            print(f"    {bucket}: n={len(group):>6,}, avg={avg_ret:+.4f}%, wr={wr:.1f}%")


def test_spread_signal(df):
    print("\n" + "=" * 60)
    print("TEST 3: CROSS-EXCHANGE SPREAD vs FORWARD RETURNS")
    print("=" * 60)

    valid = df[["funding_spread", "target_ret_12"]].dropna()
    corr = valid["funding_spread"].corr(valid["target_ret_12"])
    print(f"\n  Sample size: {len(valid):,}")
    print(f"  Correlation spread vs 1h return: {corr:+.4f}")

    df_v = valid.copy()
    df_v["spread_q"] = pd.qcut(df_v["funding_spread"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    print("\n  Returns by spread quintile (1h forward):")
    for q, group in df_v.groupby("spread_q", observed=True):
        avg_ret = group["target_ret_12"].mean() * 100
        wr = (group["target_ret_12"] > 0).mean() * 100
        print(f"    {q}: n={len(group):>6,}, avg={avg_ret:+.4f}%, wr={wr:.1f}%")


def summary_verdict(df):
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    high_mask = df["funding_extreme_high"] == 1
    low_mask = df["funding_extreme_low"] == 1

    high_ret = df.loc[high_mask, "target_ret_12"].mean() if high_mask.sum() > 0 else 0
    low_ret = df.loc[low_mask, "target_ret_12"].mean() if low_mask.sum() > 0 else 0
    base_ret = df["target_ret_12"].mean()

    high_edge_pct = (high_ret - base_ret) * 100
    low_edge_pct = (low_ret - base_ret) * 100

    z_corr = df[["funding_zscore_7d", "target_ret_12"]].dropna().corr().iloc[0, 1]

    print(f"\nKey metrics:")
    print(f"  LOW funding extreme edge: {low_edge_pct:+.4f}%")
    print(f"  HIGH funding extreme edge: {high_edge_pct:+.4f}%")
    print(f"  Z-score correlation: {z_corr:+.4f}")

    contrarian_low_works = low_edge_pct > 0.05
    contrarian_high_works = high_edge_pct < -0.05
    has_correlation = abs(z_corr) > 0.02

    print(f"\nSignal quality:")
    print(f"  LOW extreme contrarian works: {'YES' if contrarian_low_works else 'NO'}")
    print(f"  HIGH extreme contrarian works: {'YES' if contrarian_high_works else 'NO'}")
    print(f"  Z-score has correlation: {'YES' if has_correlation else 'NO'}")

    signals_passing = sum([contrarian_low_works, contrarian_high_works, has_correlation])

    if signals_passing >= 2:
        print(f"\nSTRONG: {signals_passing}/3 signals show edge - PROCEED to Week 2")
    elif signals_passing == 1:
        print(f"\nWEAK: {signals_passing}/3 signals show edge - PROCEED but cautious")
    else:
        print(f"\nFAIL: {signals_passing}/3 signals show edge")


def main():
    print("=" * 60)
    print("MORGOTH GMX - WEEK 1 DAY 4 - FUNDING VALIDATION")
    print("=" * 60)

    df = load_data()
    test_extreme_events(df)
    test_zscore_correlation(df)
    test_spread_signal(df)
    summary_verdict(df)

    print("\n" + "=" * 60)
    print("WEEK 1 DAY 4 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
