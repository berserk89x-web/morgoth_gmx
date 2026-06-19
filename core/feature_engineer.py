"""
MORGOTH GMX - Feature Engineer
Computes 40+ technical indicators from OHLCV data.
NO LOOK-AHEAD BIAS - all features use only historical data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import ta


class FeatureEngineer:
    """Generates technical indicators for ML model training."""

    def __init__(self, input_path: str = "data/btc_5min_historical.parquet"):
        self.input_path = input_path
        self.df = None

    def load(self) -> pd.DataFrame:
        """Load raw OHLCV data."""
        print(f"Loading data from {self.input_path}...")
        self.df = pd.read_parquet(self.input_path)
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)
        print(f"  Rows: {len(self.df):,}")
        print(f"  Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        return self.df

    def add_price_features(self) -> None:
        """Returns, log returns, price changes."""
        df = self.df

        # Simple returns
        df["ret_1"] = df["close"].pct_change(1)
        df["ret_3"] = df["close"].pct_change(3)
        df["ret_12"] = df["close"].pct_change(12)  # 1 hour
        df["ret_36"] = df["close"].pct_change(36)  # 3 hours

        # Log returns (better for ML)
        df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
        df["log_ret_12"] = np.log(df["close"] / df["close"].shift(12))

        # High-Low spread (intra-bar volatility)
        df["hl_spread"] = (df["high"] - df["low"]) / df["close"]

        # Open-Close direction
        df["oc_change"] = (df["close"] - df["open"]) / df["open"]

        # Body and shadow ratios
        body = (df["close"] - df["open"]).abs()
        upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
        total_range = df["high"] - df["low"]

        df["body_ratio"] = body / total_range.replace(0, np.nan)
        df["upper_shadow_ratio"] = upper_shadow / total_range.replace(0, np.nan)
        df["lower_shadow_ratio"] = lower_shadow / total_range.replace(0, np.nan)

        print("  Price features: 10 added")

    def add_moving_averages(self) -> None:
        """EMAs and SMAs."""
        df = self.df

        # EMAs
        for period in [9, 21, 50, 200]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
            df[f"close_ema_{period}_ratio"] = df["close"] / df[f"ema_{period}"]

        # EMA crosses
        df["ema_9_21_cross"] = (df["ema_9"] > df["ema_21"]).astype(int)
        df["ema_21_50_cross"] = (df["ema_21"] > df["ema_50"]).astype(int)
        df["ema_50_200_cross"] = (df["ema_50"] > df["ema_200"]).astype(int)

        print("  Moving averages: 11 added")

    def add_trend_horizon_features(self) -> None:
        """
        Long-horizon directional features (2h-24h) to capture TREND DIRECTION,
        not just 5-min noise. Used by the directional regime HMM.
        Must run after add_moving_averages (depends on ema_50 / ema_200).
        """
        df = self.df

        # Longer-horizon returns (smooth out 5-min noise)
        df["ret_24"] = df["close"].pct_change(24)    # 2 hours
        df["ret_72"] = df["close"].pct_change(72)    # 6 hours
        df["ret_288"] = df["close"].pct_change(288)  # 24 hours (daily trend)

        # EMA slopes (trend direction over the EMA's own timescale)
        df["ema_50_slope"] = df["ema_50"].pct_change(12)    # 1h slope of 50EMA
        df["ema_200_slope"] = df["ema_200"].pct_change(72)  # 6h slope of 200EMA

        print("  Trend-horizon features: 5 added")

    def add_momentum_indicators(self) -> None:
        """RSI, Stochastic, CCI, Williams %R, MFI."""
        df = self.df

        # RSI
        df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["rsi_21"] = ta.momentum.RSIIndicator(df["close"], window=21).rsi()

        # MACD
        macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=df["high"], low=df["low"], close=df["close"],
            window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # CCI
        df["cci_20"] = ta.trend.CCIIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=20
        ).cci()

        # Williams %R
        df["williams_r"] = ta.momentum.WilliamsRIndicator(
            high=df["high"], low=df["low"], close=df["close"], lbp=14
        ).williams_r()

        # MFI (Money Flow Index)
        df["mfi_14"] = ta.volume.MFIIndicator(
            high=df["high"], low=df["low"], close=df["close"],
            volume=df["volume"], window=14
        ).money_flow_index()

        print("  Momentum indicators: 10 added")

    def add_volatility_indicators(self) -> None:
        """ATR, Bollinger Bands, Keltner Channels."""
        df = self.df

        # ATR
        df["atr_14"] = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14
        ).average_true_range()
        df["atr_pct"] = df["atr_14"] / df["close"]

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

        # Keltner Channels
        kc = ta.volatility.KeltnerChannel(
            high=df["high"], low=df["low"], close=df["close"], window=20, window_atr=10
        )
        df["kc_upper"] = kc.keltner_channel_hband()
        df["kc_lower"] = kc.keltner_channel_lband()
        df["kc_pct"] = (df["close"] - df["kc_lower"]) / (df["kc_upper"] - df["kc_lower"]).replace(0, np.nan)

        # Realized volatility (rolling std of returns)
        df["realized_vol_12"] = df["log_ret_1"].rolling(12).std()  # 1 hour
        df["realized_vol_72"] = df["log_ret_1"].rolling(72).std()  # 6 hours

        print("  Volatility indicators: 11 added")

    def add_volume_features(self) -> None:
        """Volume-based features."""
        df = self.df

        # Volume moving averages
        df["vol_sma_20"] = df["volume"].rolling(20).mean()
        df["vol_ratio_20"] = df["volume"] / df["vol_sma_20"]

        # OBV (On-Balance Volume)
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(
            close=df["close"], volume=df["volume"]
        ).on_balance_volume()
        df["obv_ema"] = df["obv"].ewm(span=21, adjust=False).mean()
        df["obv_signal"] = (df["obv"] > df["obv_ema"]).astype(int)

        # Taker buy ratio (aggressive buyers vs sellers)
        df["taker_buy_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
        df["taker_buy_ratio_sma"] = df["taker_buy_ratio"].rolling(20).mean()

        # Volume-weighted price (VWAP rolling)
        cum_vp = (df["close"] * df["volume"]).rolling(20).sum()
        cum_v = df["volume"].rolling(20).sum()
        df["vwap_20"] = cum_vp / cum_v.replace(0, np.nan)
        df["close_vwap_ratio"] = df["close"] / df["vwap_20"]

        print("  Volume features: 9 added")

    def add_pattern_features(self) -> None:
        """Pattern detection: higher highs, lower lows, etc."""
        df = self.df

        # Higher highs / Lower lows over windows
        df["hh_12"] = (df["high"] == df["high"].rolling(12).max()).astype(int)
        df["ll_12"] = (df["low"] == df["low"].rolling(12).min()).astype(int)
        df["hh_72"] = (df["high"] == df["high"].rolling(72).max()).astype(int)
        df["ll_72"] = (df["low"] == df["low"].rolling(72).min()).astype(int)

        # Distance from rolling max/min
        df["dist_from_high_72"] = (df["close"] - df["high"].rolling(72).max()) / df["close"]
        df["dist_from_low_72"] = (df["close"] - df["low"].rolling(72).min()) / df["close"]

        # Consecutive up/down bars
        up_streak = (df["close"] > df["close"].shift(1)).astype(int)
        df["consec_up"] = up_streak.groupby((up_streak != up_streak.shift()).cumsum()).cumsum() * up_streak
        df["consec_down"] = (1 - up_streak).groupby((up_streak != up_streak.shift()).cumsum()).cumsum() * (1 - up_streak)

        print("  Pattern features: 8 added")

    def add_time_features(self) -> None:
        """Hour of day, day of week (cyclical)."""
        df = self.df

        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.dayofweek

        # Cyclical encoding (sin/cos so model knows 23:00 close to 00:00)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

        print("  Time features: 6 added")

    def add_target_labels(self) -> None:
        """
        Future returns for training (NOT a feature - target).
        Used to train models to predict direction.
        Drop these before live trading.
        """
        df = self.df

        # Future returns (label for training)
        df["target_ret_1"] = df["close"].shift(-1) / df["close"] - 1
        df["target_ret_3"] = df["close"].shift(-3) / df["close"] - 1
        df["target_ret_12"] = df["close"].shift(-12) / df["close"] - 1

        # Direction labels (for classification)
        df["target_direction_1"] = (df["target_ret_1"] > 0).astype(int)
        df["target_direction_12"] = (df["target_ret_12"] > 0).astype(int)

        print("  Target labels: 5 added (training only)")

    def build_all(self) -> pd.DataFrame:
        """Run full feature pipeline."""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60)

        self.load()

        print("\nAdding features:")
        self.add_price_features()
        self.add_moving_averages()
        self.add_trend_horizon_features()
        self.add_momentum_indicators()
        self.add_volatility_indicators()
        self.add_volume_features()
        self.add_pattern_features()
        self.add_time_features()
        self.add_target_labels()

        total_features = len(self.df.columns) - 10  # subtract original 10
        print(f"\nTotal features added: {total_features}")
        print(f"Total columns: {len(self.df.columns)}")

        # Quality report
        print(f"\nDataset shape: {self.df.shape}")
        nan_rows = self.df.isnull().any(axis=1).sum()
        print(f"Rows with any NaN: {nan_rows:,} ({nan_rows/len(self.df)*100:.1f}%)")
        print("(NaN expected at start due to rolling windows)")

        return self.df

    def save(self, output_path: str = "data/btc_features.parquet") -> None:
        """Save engineered dataset."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(output_path, compression="snappy", index=False)
        import os
        size_mb = os.path.getsize(output_path) / 1024**2
        print(f"\nSaved to: {output_path}")
        print(f"File size: {size_mb:.1f} MB")


def main():
    print("=" * 60)
    print("MORGOTH GMX - PHASE 2 BLOCK 2 - FEATURE ENGINEERING")
    print("=" * 60)

    engineer = FeatureEngineer(input_path="data/btc_5min_historical.parquet")
    df = engineer.build_all()
    engineer.save("data/btc_features.parquet")

    print("\n" + "=" * 60)
    print("PHASE 2 BLOCK 2 COMPLETE")
    print("=" * 60)

    # Sample preview
    print("\nFeature columns (first 30):")
    for i, col in enumerate(df.columns[:30]):
        print(f"  {i+1:2d}. {col}")

    print("\nSample row (mid-dataset):")
    sample_row = df.iloc[len(df)//2]
    print(f"  Timestamp: {sample_row['timestamp']}")
    print(f"  Close: ${sample_row['close']:,.2f}")
    print(f"  RSI 14: {sample_row.get('rsi_14', 'N/A')}")
    print(f"  ATR pct: {sample_row.get('atr_pct', 'N/A')}")
    print(f"  BB pct: {sample_row.get('bb_pct', 'N/A')}")


if __name__ == "__main__":
    main()
