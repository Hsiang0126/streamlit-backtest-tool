"""
技術指標運算模組 (Indicators Module - 機構級重構版)
1. 保持在原生幣別 (Native Currency) 下計算，徹底杜絕匯率污染 K 棒。
2. 優先採用還原除權息收盤價 (adj_close)，杜絕配息與拆股跳空引發的假停損。
3. 動態每週真實結算：依該週最後一個真實開市日封閉週 K 線，消除連假延遲問題。
"""
from typing import Tuple
import pandas as pd
import numpy as np


class IndicatorEngine:
    """集中管理純淨技術指標運算。"""

    @staticmethod
    def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
        src = df.copy()
        if "is_trading_day" in src.columns:
            active_src = src[src["is_trading_day"] == True].copy()
        else:
            active_src = src.copy()

        if active_src.empty:
            active_src = src.copy()

        iso = active_src.index.isocalendar()
        active_src["_year_week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str)

        col_c = "adj_close" if "adj_close" in active_src.columns else "close"
        col_o = "open"
        col_h = "high"
        col_l = "low"

        weekly_groups = active_src.groupby("_year_week")
        last_days = weekly_groups.apply(lambda g: g.index[-1])

        weekly_df = pd.DataFrame({
            "open": weekly_groups[col_o].first().values,
            "high": weekly_groups[col_h].max().values,
            "low": weekly_groups[col_l].min().values,
            "close": weekly_groups[col_c].last().values
        }, index=last_days.values).sort_index()

        return weekly_df

    @classmethod
    def calculate_ma(cls, df: pd.DataFrame, period: int, timeframe: str = "日線 (Daily)") -> pd.Series:
        p = max(1, int(period))
        col = "adj_close" if "adj_close" in df.columns else "close"
        if "週" in timeframe:
            weekly = cls._resample_weekly(df)
            weekly_ma = weekly["close"].rolling(p).mean()
            return weekly_ma.reindex(df.index, method="ffill")
        return df[col].rolling(p).mean()

    @classmethod
    def calculate_rsi(cls, df: pd.DataFrame, period: int, timeframe: str = "日線 (Daily)") -> pd.Series:
        p = max(1, int(period))
        col = "adj_close" if "adj_close" in df.columns else "close"

        if "週" in timeframe:
            src = cls._resample_weekly(df)
            price_series = src["close"]
        else:
            price_series = df[col]

        delta = price_series.diff()
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
        f, s, sig_p = max(1, int(fast)), max(1, int(slow)), max(1, int(signal))
        if f >= s:
            f, s = min(f, s), max(f, s)
        col = "adj_close" if "adj_close" in df.columns else "close"

        if "週" in timeframe:
            src = cls._resample_weekly(df)
            price_series = src["close"]
        else:
            price_series = df[col]

        ema_fast = price_series.ewm(span=f, adjust=False).mean()
        ema_slow = price_series.ewm(span=s, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=sig_p, adjust=False).mean()

        if "週" in timeframe:
            return macd.reindex(df.index, method="ffill"), sig.reindex(df.index, method="ffill")
        return macd, sig

    @classmethod
    def calculate_kd(cls, df: pd.DataFrame, period: int, timeframe: str = "日線 (Daily)") -> pd.Series:
        p = max(1, int(period))
        if "週" in timeframe:
            src = cls._resample_weekly(df)
        else:
            src = df

        col_c = "adj_close" if "adj_close" in src.columns else "close"
        low_min = src["low"].rolling(p).min()
        high_max = src["high"].rolling(p).max()
        price_range = high_max - low_min
        rsv = (src[col_c] - low_min) / price_range.replace(0, np.nan) * 100.0
        rsv = rsv.fillna(50.0)
        k = rsv.ewm(com=2, adjust=False).mean()

        if "週" in timeframe:
            return k.reindex(df.index, method="ffill")
        return k
