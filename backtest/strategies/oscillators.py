"""
擺盪指標擇時策略 (Oscillator Strategies)
包含 RSI、MACD、KD 隨機指標策略。
"""
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, ParamSpec
from strategies.registry import register_strategy


@register_strategy
class RSIStrategy(BaseStrategy):
    strategy_id = "rsi_strategy"
    display_name = "RSI 超買超賣"
    description = "均值回歸策略：RSI 低於超賣區買入，高於超買區賣出獲利了結。"
    is_timing = True

    @classmethod
    def get_param_specs(cls) -> List[ParamSpec]:
        return [
            ParamSpec(key="period", label="RSI 週期", param_type="int", default=14, min_value=2, max_value=50, step=1),
            ParamSpec(key="buy", label="超賣買入門檻", param_type="float", default=30.0, min_value=5.0, max_value=45.0, step=1.0),
            ParamSpec(key="sell", label="超買賣部門檻", param_type="float", default=70.0, min_value=55.0, max_value=95.0, step=1.0)
        ]

    @classmethod
    def generate_signals(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        period = int(params.get("period", 14))
        buy_thresh = float(params.get("buy", 30.0))
        sell_thresh = float(params.get("sell", 70.0))

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        sig = pd.Series(np.nan, index=df.index)
        sig[rsi < buy_thresh] = 1
        sig[rsi > sell_thresh] = 0
        return sig.ffill().fillna(0).astype(int)


@register_strategy
class MACDStrategy(BaseStrategy):
    strategy_id = "macd_strategy"
    display_name = "MACD 柱狀圖/交叉"
    description = "動能趨勢策略：MACD 快線由下往上穿越慢線/訊號線買入，跌破賣出。"
    is_timing = True

    @classmethod
    def get_param_specs(cls) -> List[ParamSpec]:
        return [
            ParamSpec(key="fast", label="快線週期 (EMA)", param_type="int", default=12, min_value=3, max_value=50, step=1),
            ParamSpec(key="slow", label="慢線週期 (EMA)", param_type="int", default=26, min_value=10, max_value=100, step=1),
            ParamSpec(key="signal", label="訊號線週期", param_type="int", default=9, min_value=2, max_value=30, step=1)
        ]

    @classmethod
    def generate_signals(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        sig_p = int(params.get("signal", 9))
        if fast >= slow:
            fast, slow = min(fast, slow), max(fast, slow)

        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=sig_p, adjust=False).mean()
        return (macd > signal).astype(int)


@register_strategy
class KDStrategy(BaseStrategy):
    strategy_id = "kd_strategy"
    display_name = "KD 隨機指標"
    description = "擺盪轉折策略：K 值低檔黃金交叉買入，高檔超買賣出。"
    is_timing = True

    @classmethod
    def get_param_specs(cls) -> List[ParamSpec]:
        return [
            ParamSpec(key="period", label="KD 計算週期", param_type="int", default=9, min_value=3, max_value=30, step=1),
            ParamSpec(key="buy_k", label="超賣買入門檻 (K)", param_type="float", default=20.0, min_value=5.0, max_value=40.0, step=1.0),
            ParamSpec(key="sell_k", label="超買賣部門檻 (K)", param_type="float", default=80.0, min_value=60.0, max_value=95.0, step=1.0)
        ]

    @classmethod
    def generate_signals(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        period = int(params.get("period", 9))
        buy_k = float(params.get("buy_k", 20.0))
        sell_k = float(params.get("sell_k", 80.0))

        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max()
        price_range = high_max - low_min
        rsv = (df["close"] - low_min) / price_range.replace(0, np.nan) * 100.0
        rsv = rsv.fillna(50.0)
        k = rsv.ewm(com=2, adjust=False).mean()

        sig = pd.Series(np.nan, index=df.index)
        sig[k < buy_k] = 1
        sig[k > sell_k] = 0
        return sig.ffill().fillna(0).astype(int)