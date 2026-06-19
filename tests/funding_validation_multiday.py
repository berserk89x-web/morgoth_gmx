"""
MORGOTH GMX - Funding Validation, Multi-Day Horizons
Week 1 Day 4 (follow-up)

The 1h standalone test FAILED (0/3). Funding's documented edge is multi-day,
not intraday. This re-test computes forward returns at 1d / 3d / 7d (inline,
without mutating btc_features.parquet) and re-runs the same extreme-event,
z-score, and spread tests at funding's real horizon.

Contrarian thesis -> z-score should correlate NEGATIVELY with forward return
(high funding = crowded longs = lower future returns), and high-funding
extremes should UNDERPERFORM while low-funding extremes OUTPERFORM.
"""

import pandas as pd
import numpy as np

# horizon label -> bars (5-min bars)
HORIZONS = {
    "1d (288 bars)": 288,
    "3d (864 bars)": 864,
    "7d (2016 bars)": 2016,
}
MAX_BARS = max(HORIZONS.values())


def load_data():
    df = pd.read_parquet("data/btc_features.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute multi-day forward returns INLINE (targets, not features)
    for label, bars in HORIZONS.items():
        df[f"fwd_{bars}"] = df["close"].shift(-bars) / df["close"] - 1

    # Drop warmup; drop the tail that lacks a full 7d forward window
    df = df.iloc[9000:].copy()
    df = df.iloc[:-MAX_BARS].copy()
    print(f"Validation dataset: {len(df):,} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def test_horizon(df, label, bars):
    col = f"fwd_{bars}"
    print("\n" + "=" * 60)
    print(f"HORIZON: {label}")
    print("=" * 60)

    base = df[col].mean() * 100
    base_wr = (df[col] > 0).mean() * 100
    print(f"  Baseline: avg={base:+.4f}%, wr={base_wr:.1f}%")

    high = df[df["funding_extreme_high"] == 1]
    low = df[df["funding_extreme_low"] == 1]

    high_ret = high[col].mean() * 100
    high_wr = (high[col] > 0).mean() * 100
    low_ret = low[col].mean() * 100
    low_wr = (low[col] > 0).mean() * 100

    high_edge = high_ret - base
    low_edge = low_ret - base

    print(f"  HIGH funding extreme (n={len(high):,}): avg={high_ret:+.4f}%, wr={high_wr:.1f}%, edge={high_edge:+.4f}%")
    print(f"    (contrarian wants NEGATIVE edge)")
    print(f"  LOW  funding extreme (n={len(low):,}): avg={low_ret:+.4f}%, wr={low_wr:.1f}%, edge={low_edge:+.4f}%")
    print(f"    (contrarian wants POSITIVE edge)")

    valid = df[["funding_zscore_7d", col]].dropna()
    z_corr = valid["funding_zscore_7d"].corr(valid[col])
    sp = df[["funding_spread", col]].dropna()
    spread_corr = sp["funding_spread"].corr(sp[col])
    print(f"  Z-score correlation: {z_corr:+.4f}  (contrarian wants NEGATIVE)")
    print(f"  Spread correlation:  {spread_corr:+.4f}")

    # z-score buckets
    df_v = valid.copy()
    df_v["bucket"] = pd.cut(df_v["funding_zscore_7d"],
                           bins=[-np.inf, -2, -1, 0, 1, 2, np.inf],
                           labels=["z<-2", "-2..-1", "-1..0", "0..1", "1..2", "z>=2"])
    print("  By z-score bucket:")
    for b, g in df_v.groupby("bucket", observed=True):
        if len(g) > 100:
            print(f"    {b:>7}: n={len(g):>6,}, avg={g[col].mean()*100:+.4f}%")

    # Verdict: scale the edge threshold by horizon (sqrt-time on a ~0.05%/1h base)
    # 1h base threshold 0.05%; scale by sqrt(bars/12)
    thresh = 0.05 * np.sqrt(bars / 12)
    low_works = low_edge > thresh
    high_works = high_edge < -thresh
    corr_works = z_corr < -0.02  # contrarian direction specifically

    passing = sum([low_works, high_works, corr_works])
    print(f"  Edge threshold (scaled): +/-{thresh:.4f}%")
    print(f"  Signals: LOW-contrarian={'Y' if low_works else 'N'}, "
          f"HIGH-contrarian={'Y' if high_works else 'N'}, "
          f"NEG-corr={'Y' if corr_works else 'N'}  -> {passing}/3")
    return {"label": label, "z_corr": z_corr, "high_edge": high_edge,
            "low_edge": low_edge, "passing": passing}


def main():
    print("=" * 60)
    print("MORGOTH GMX - FUNDING VALIDATION (MULTI-DAY HORIZONS)")
    print("=" * 60)

    df = load_data()
    results = [test_horizon(df, label, bars) for label, bars in HORIZONS.items()]

    print("\n" + "=" * 60)
    print("MULTI-HORIZON VERDICT")
    print("=" * 60)
    for r in results:
        verdict = "STRONG" if r["passing"] >= 2 else ("WEAK" if r["passing"] == 1 else "FAIL")
        print(f"  {r['label']:>15}: z-corr={r['z_corr']:+.4f}, "
              f"high_edge={r['high_edge']:+.4f}%, low_edge={r['low_edge']:+.4f}% "
              f"-> {r['passing']}/3 {verdict}")

    best = max(results, key=lambda r: r["passing"])
    print(f"\n  Best horizon: {best['label']} ({best['passing']}/3)")
    if best["passing"] == 0:
        print("  CONCLUSION: funding shows no standalone edge at ANY horizon tested.")
    else:
        print("  CONCLUSION: funding shows edge at longer horizon(s) - keep for ensemble.")


if __name__ == "__main__":
    main()
