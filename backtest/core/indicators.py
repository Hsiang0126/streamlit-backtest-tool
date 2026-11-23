"""
技術指標運算模組 (Indicators Module)
1. 保持在原生幣別 (Native Currency) 下計算，徹底杜絕匯率污染 K 棒。
2. 全面採用一致尺度的還原除權息 OHLC (adj_open/adj_high/adj_low/adj_close)，杜絕 KD 負值與除息假停損。
3. 動態每週真實結算：依 ISO 週碼向量化分組並於該週最後一個真實開市日封閉週 K 線。
"""
from typing import Tuple
import pandas as pd
import numpy as np


class IndicatorEngine:
    """集中管理純淨技術指標運算。"""

    @staticmethod
    def _ensure_adjusted_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """確保回傳之 DataFrame 具備一致尺度的 adj_open / adj_high / adj_low / adj_close。"""
        if {"adj_open", "adj_high", "adj_low", "adj_close"}.issubset(df.columns):
            return df

        out = df.copy()
        if "adj_close" not in out.columns:
            out["adj_close"] = out["close"]
            out["adj_open"] = out["open"]
            out["adj_high"] = out["high"]
            out["adj_low"] = out["low"]
            return out

        ratio = (out["adj_close"] / out["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        ratio = ratio.where(ratio > 0, 1.0)
        out["adj_open"] = out["open"] * ratio
        out["adj_high"] = out["high"] * ratio
        out["adj_low"] = out["low"] * ratio
        out["adj_high"] = out[["adj_high", "adj_open", "adj_close"]].max(axis=1)
        out["adj_low"] = out[["adj_low", "adj_open", "adj_close"]].min(axis=1)
        return out

    @classmethod
    def _resample_weekly(cls, df: pd.DataFrame) -> pd.DataFrame:
        src = cls._ensure_adjusted_ohlc(df)
        if "is_trading_day" in src.columns:
            active_src = src[src["is_trading_day"] == True]
            if active_src.empty:
                active_src = src
        else:
            active_src = src

        iso = active_src.index.isocalendar()
        year_week = iso["year"].astype(str).str.zfill(4) + "-W" + iso["week"].astype(str).str.zfill(2)

        tmp = pd.DataFrame(
            {
                "_yw": year_week.values,
                "_dt": active_src.index,
                "open": active_src["adj_open"].values,
                "high": active_src["adj_high"].values,
                "low": active_src["adj_low"].values,
                "close": active_src["adj_close"].values,
            }
        )

        grouped = tmp.groupby("_yw", sort=True).agg(
            _dt=("_dt", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )

        weekly_df = grouped.set_index("_dt").sort_index()
        weekly_df.index.name = df.index.name
        return weekly_df

    @classmethod
    def _get_price_series(cls, df: pd.DataFrame, timeframe: str) -> Tuple[pd.Series, bool]:
        is_weekly = "週" in timeframe
        if is_weekly:
            weekly = cls._resample_weekly(df)
            return weekly["close"], True
        col = "adj_close" if "adj_close" in df.columns else "close"
        return df[col], False

    @classmethod
    def calculate_ma(cls, df: pd.DataFrame, period: int, timeframe: str = "日線 (Daily)") -> pd.Series:
        p = max(1, int(period))
        price_series, is_weekly = cls._get_price_series(df, timeframe)
        ma = price_series.rolling(p).mean()
        if is_weekly:
            return ma.reindex(df.index, method="ffill")
        return ma

    @classmethod
    def calculate_ema(cls, df: pd.DataFrame, period: int, timeframe: str = "日線 (Daily)") -> pd.Series:
        p = max(1, int(period))
        price_series, is_weekly = cls._get_price_series(df, timeframe)
        ema = price_series.ewm(span=p, adjust=False).mean()
        if is_weekly:
            return ema.reindex(df.index, method="ffill")
        return ema

    @classmethod
    def calculate_rsi(cls, df: pd.DataFrame, period: int, timeframe: str = "日線 (Daily)") -> pd.Series:
        p = max(1, int(period))
        price_series, is_weekly = cls._get_price_series(df, timeframe)

        delta = price_series.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / p, adjust=False, min_periods=p).mean()
        avg_loss = loss.ewm(alpha=1.0 / p, adjust=False, min_periods=p).mean()

        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # 處理完全無下跌 (avg_loss == 0) 之極端邊界
        zero_loss_mask = avg_loss == 0.0
        rsi = rsi.where(~(zero_loss_mask & (avg_gain > 0.0)), 100.0)
        rsi = rsi.where(~(zero_loss_mask & (avg_gain == 0.0)), 50.0)

        if is_weekly:
            return rsi.reindex(df.index, method="ffill")
        return rsi

    @classmethod
    def calculate_macd(
        cls,
        df: pd.DataFrame,
        fast: int,
        slow: int,
        signal: int,
        timeframe: str = "日線 (Daily)",
    ) -> Tuple[pd.Series, pd.Series]:
        f, s, sig_p = max(1, int(fast)), max(1, int(slow)), max(1, int(signal))
        if f > s:
            f, s = s, f
        elif f == s:
            s = f + 1
        price_series, is_weekly = cls._get_price_series(df, timeframe)

        ema_fast = price_series.ewm(span=f, adjust=False).mean()
        ema_slow = price_series.ewm(span=s, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=sig_p, adjust=False).mean()

        if is_weekly:
            return macd.reindex(df.index, method="ffill"), sig.reindex(df.index, method="ffill")
        return macd, sig

    @classmethod
    def calculate_kd_full(
        cls,
        df: pd.DataFrame,
        period: int,
        timeframe: str = "日線 (Daily)",
    ) -> Tuple[pd.Series, pd.Series]:
        p = max(1, int(period))
        is_weekly = "週" in timeframe
        if is_weekly:
            src = cls._resample_weekly(df)
            col_h, col_l, col_c = "high", "low", "close"
        else:
            src = cls._ensure_adjusted_ohlc(df)
            col_h, col_l, col_c = "adj_high", "adj_low", "adj_close"

        low_min = src[col_l].rolling(p).min()
        high_max = src[col_h].rolling(p).max()
        price_range = high_max - low_min

        rsv = ((src[col_c] - low_min) / price_range.replace(0.0, np.nan)) * 100.0
        rsv = rsv.fillna(50.0).clip(lower=0.0, upper=100.0)
        k = rsv.ewm(com=2, adjust=False).mean().clip(lower=0.0, upper=100.0)
        d = k.ewm(com=2, adjust=False).mean().clip(lower=0.0, upper=100.0)

        if is_weekly:
            return k.reindex(df.index, method="ffill"), d.reindex(df.index, method="ffill")
        return k, d

    @classmethod
    def calculate_kd(cls, df: pd.DataFrame, period: int, timeframe: str = "日線 (Daily)") -> pd.Series:
        k, _ = cls.calculate_kd_full(df, period, timeframe)
        return k

    @classmethod
    def calculate_bollinger(
        cls,
        df: pd.DataFrame,
        period: int = 20,
        num_std: float = 2.0,
        timeframe: str = "日線 (Daily)",
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        p = max(2, int(period))
        k_std = max(0.5, float(num_std))
        price_series, is_weekly = cls._get_price_series(df, timeframe)

        mid = price_series.rolling(p).mean()
        std = price_series.rolling(p).std(ddof=0)
        upper = mid + k_std * std
        lower = mid - k_std * std

        if is_weekly:
            return (
                upper.reindex(df.index, method="ffill"),
                mid.reindex(df.index, method="ffill"),
                lower.reindex(df.index, method="ffill"),
            )
        return upper, mid, lower
