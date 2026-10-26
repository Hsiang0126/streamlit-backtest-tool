"""
技術指標運算模組 (Indicators Module)
支援日線 (Daily) 與週線 (Weekly) 安全重採樣，嚴格杜絕前瞻未來函數。
"""
from typing import Tuple
import pandas as pd
import numpy as np


class IndicatorEngine:
    """集中管理日線與週線之技術指標計算。"""

    @staticmethod
    def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
        """將日線 K 棒重採樣為每週五結算的標準週 K 線。"""
        weekly = df.resample("W-FRI").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        }).dropna(subset=["close"])
        return weekly

    @classmethod
    def calculate_ma(cls, df: pd.DataFrame, period: int, timeframe: str = "日線 (Daily)") -> pd.Series:
        """計算移動平均線 (MA)。若選週線，依週五收盤結算並無偷看地對齊至日線。"""
        p = max(1, int(period))
        if "週" in timeframe:
            weekly = cls._resample_weekly(df)
            weekly_ma = weekly["close"].rolling(p).mean()
            return weekly_ma.reindex(df.index, method="ffill")
        return df["close"].rolling(p).mean()

    @classmethod
    def calculate_rsi(cls, df: pd.DataFrame, period: int, timeframe: str = "日線 (Daily)") -> pd.Series:
        """計算 RSI 指標。"""
        p = max(1, int(period))
        src = cls._resample_weekly(df) if "週" in timeframe else df

        delta = src["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / p, adjust=False, min_periods=p).mean()
        avg_loss = loss.ewm(alpha=1.0 / p, adjust=False, min_periods=p).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        if "週" in timeframe:
            return rsi.reindex(df.index, method="ffill")
        return rsi

    @classmethod
    def calculate_macd(cls, df: pd.DataFrame, fast: int, slow: int, signal: int, timeframe: str = "日線 (Daily)") -> Tuple[pd.Series, pd.Series]:
        """計算 MACD 快線與訊號線。"""
        f, s, sig_p = max(1, int(fast)), max(1, int(slow)), max(1, int(signal))
        if f >= s:
            f, s = min(f, s), max(f, s)
        src = cls._resample_weekly(df) if "週" in timeframe else df

        ema_fast = src["close"].ewm(span=f, adjust=False).mean()
        ema_slow = src["close"].ewm(span=s, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=sig_p, adjust=False).mean()

        if "週" in timeframe:
            return macd.reindex(df.index, method="ffill"), sig.reindex(df.index, method="ffill")
        return macd, sig

    @classmethod
    def calculate_kd(cls, df: pd.DataFrame, period: int, timeframe: str = "日線 (Daily)") -> pd.Series:
        """計算 KD 指標的 K 值。"""
        p = max(1, int(period))
        src = cls._resample_weekly(df) if "週" in timeframe else df

        low_min = src["low"].rolling(p).min()
        high_max = src["high"].rolling(p).max()
        price_range = high_max - low_min
        rsv = (src["close"] - low_min) / price_range.replace(0, np.nan) * 100.0
        rsv = rsv.fillna(50.0)
        k = rsv.ewm(com=2, adjust=False).mean()

        if "週" in timeframe:
            return k.reindex(df.index, method="ffill")
        return k
