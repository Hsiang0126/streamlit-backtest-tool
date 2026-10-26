"""
均線突破與均線交叉策略 (Moving Average Strategies)
完全杜絕未來函數，以 T 日收盤計算指標，T+1 日開盤撮合成交。
"""
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, ParamSpec
from strategies.registry import register_strategy


@register_strategy
class PriceSMABreakoutStrategy(BaseStrategy):
    strategy_id = "price_sma_breakout"
    display_name = "價格突破均線買入/跌破賣出"
    description = "趨勢追隨策略：收盤價站上均線買入持有，跌破均線全數獲利了結清倉。"
    is_timing = True

    @classmethod
    def get_param_specs(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                key="period",
                label="均線週期 (日)",
                param_type="int",
                default=20,
                min_value=3,
                max_value=250,
                step=1
            )
        ]

    @classmethod
    def generate_signals(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        period = int(params.get("period", 20))
        sma = df["close"].rolling(period).mean()
        signal = (df["close"] > sma).astype(int)
        return signal.fillna(0)


@register_strategy
class SMACrossoverStrategy(BaseStrategy):
    strategy_id = "sma_crossover"
    display_name = "SMA 雙均線交叉 (黃金交叉)"
    description = "經典雙均線策略：短期均線向上突破長期均線買入，跌破賣出避險。"
    is_timing = True

    @classmethod
    def get_param_specs(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                key="fast",
                label="快線週期 (日)",
                param_type="int",
                default=20,
                min_value=3,
                max_value=120,
                step=1
            ),
            ParamSpec(
                key="slow",
                label="慢線週期 (日)",
                param_type="int",
                default=60,
                min_value=5,
                max_value=300,
                step=5
            )
        ]

    @classmethod
    def generate_signals(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        fast = int(params.get("fast", 20))
        slow = int(params.get("slow", 60))
        if fast >= slow:
            fast, slow = min(fast, slow), max(fast, slow)

        fast_ma = df["close"].rolling(fast).mean()
        slow_ma = df["close"].rolling(slow).mean()
        signal = (fast_ma > slow_ma).astype(int)
        return signal.fillna(0)