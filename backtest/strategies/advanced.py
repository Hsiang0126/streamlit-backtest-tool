"""
進階複合與波動率策略 (Advanced Volatility & Multi-Filter Strategies)
包含：
1. 布林通道策略 (Bollinger Bands - 均值回歸 / 趨勢突破雙模式)
2. 雙重濾網動能策略 (Dual-Filter Momentum - 長期均線趨勢濾網 + RSI 回檔進場)
"""
from typing import List, Dict, Any
import pandas as pd
from strategies.base import BaseStrategy, ParamSpec
from strategies.registry import register_strategy
from core.indicators import IndicatorEngine


@register_strategy
class BollingerStrategy(BaseStrategy):
    strategy_id = "bollinger_bands"
    display_name = "布林通道策略 (Bollinger Bands)"
    short_name = "布林通道"
    description = "波動率區間策略：支援「跌破下軌超賣抄底/觸及上軌停利」之均值回歸，或「突破上軌追漲/跌破中軌出場」之趨勢突破模式。"
    is_timing = True

    @classmethod
    def get_param_specs(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                key="timeframe",
                label="計算週期",
                param_type="select",
                default="日線 (Daily)",
                options=["日線 (Daily)", "週線 (Weekly)"],
            ),
            ParamSpec(
                key="mode",
                label="交易邏輯模式",
                param_type="select",
                default="均值回歸 (破下軌買/破上軌賣)",
                options=[
                    "均值回歸 (破下軌買/破上軌賣)",
                    "趨勢突破 (破上軌買/破中軌賣)",
                ],
            ),
            ParamSpec(
                key="period",
                label="通道中軌週期 (SMA)",
                param_type="int",
                default=20,
                min_value=5,
                max_value=120,
                step=1,
            ),
            ParamSpec(
                key="num_std",
                label="標準差倍數 (σ)",
                param_type="float",
                default=2.0,
                min_value=1.0,
                max_value=3.5,
                step=0.1,
            ),
        ]

    @classmethod
    def generate_signals(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        period = int(params.get("period", 20))
        num_std = float(params.get("num_std", 2.0))
        tf = params.get("timeframe", "日線 (Daily)")
        mode = params.get("mode", "均值回歸 (破下軌買/破上軌賣)")

        upper, mid, lower = IndicatorEngine.calculate_bollinger(df, period, num_std, tf)
        col_c = "adj_close" if "adj_close" in df.columns else "close"
        price = df[col_c]

        sig = pd.Series(float("nan"), index=df.index)
        if "趨勢突破" in mode:
            sig[price > upper] = 1
            sig[price < mid] = 0
        else:
            sig[price < lower] = 1
            sig[price > upper] = 0

        return sig.ffill().fillna(0).astype(int)


@register_strategy
class DualFilterMomentumStrategy(BaseStrategy):
    strategy_id = "dual_filter_momentum"
    display_name = "雙重濾網動能 (均線趨勢 + RSI)"
    short_name = "雙重濾網"
    description = "機構常見雙重濾網：僅在價格站穩長期均線（多頭趨勢）時，搭配 RSI 相對低檔拉回進場；跌破趨勢均線或 RSI 過熱時離場。"
    is_timing = True

    @classmethod
    def get_param_specs(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                key="timeframe",
                label="計算週期",
                param_type="select",
                default="日線 (Daily)",
                options=["日線 (Daily)", "週線 (Weekly)"],
            ),
            ParamSpec(
                key="trend_ma",
                label="多頭趨勢均線週期",
                param_type="int",
                default=60,
                min_value=10,
                max_value=250,
                step=5,
            ),
            ParamSpec(
                key="rsi_period",
                label="RSI 週期",
                param_type="int",
                default=14,
                min_value=2,
                max_value=50,
                step=1,
            ),
            ParamSpec(
                key="rsi_entry",
                label="順勢拉回買入 RSI (<)",
                param_type="float",
                default=45.0,
                min_value=15.0,
                max_value=65.0,
                step=1.0,
            ),
            ParamSpec(
                key="rsi_exit",
                label="過熱停利 RSI (>)",
                param_type="float",
                default=78.0,
                min_value=60.0,
                max_value=95.0,
                step=1.0,
            ),
        ]

    @classmethod
    def generate_signals(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        trend_ma = int(params.get("trend_ma", 60))
        rsi_p = int(params.get("rsi_period", 14))
        rsi_entry = float(params.get("rsi_entry", 45.0))
        rsi_exit = float(params.get("rsi_exit", 78.0))
        tf = params.get("timeframe", "日線 (Daily)")

        ma = IndicatorEngine.calculate_ma(df, trend_ma, tf)
        rsi = IndicatorEngine.calculate_rsi(df, rsi_p, tf)
        col_c = "adj_close" if "adj_close" in df.columns else "close"
        price = df[col_c]

        n = len(df)
        sig_vals = [0] * n
        state = 0
        p_arr = price.values
        ma_arr = ma.values
        r_arr = rsi.values

        for i in range(n):
            pv, mv, rv = p_arr[i], ma_arr[i], r_arr[i]
            if pd.notna(pv) and pd.notna(mv) and pd.notna(rv):
                if state == 0:
                    if pv > mv and rv < rsi_entry:
                        state = 1
                else:
                    if pv < mv or rv > rsi_exit:
                        state = 0
            sig_vals[i] = state

        return pd.Series(sig_vals, index=df.index, dtype=int)
