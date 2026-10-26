"""
快慢線網格尋優與參數高原分析模組 (Grid Optimizer Module - 完整現金流注資版)
支援期初本金 + 每月定投注資，快速計算 Fast MA (5~30) × Slow MA (30~120) 2D 矩陣。
"""
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from core.indicators import IndicatorEngine
from core.cost_model import CostModel, AssetCostStructure


class GridOptimizer:
    """向量化矩陣尋優器。"""

    @classmethod
    def run_ma_grid(
        cls,
        df: pd.DataFrame,
        fast_range: Tuple[int, int, int] = (5, 31, 2),
        slow_range: Tuple[int, int, int] = (30, 121, 5),
        timeframe: str = "日線 (Daily)",
        initial_capital: float = 300000.0,
        monthly_inflow: float = 0.0,
        cost_struct: Optional[AssetCostStructure] = None,
        to_target_fx: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        if cost_struct is None:
            cost_struct = AssetCostStructure(0.00071, 0.003, 0.0003)

        col_c = "adj_close" if "adj_close" in df.columns else "close"
        close_p = df[col_c].values
        open_p = df["open"].values
        n_days = len(df)

        # 換月判定序列 (支援定投資金注入)
        s_months = pd.Series(df.index.to_period("M"), index=df.index)
        month_changed = (s_months != s_months.shift(1)).values

        fast_vals = list(range(fast_range[0], fast_range[1], fast_range[2]))
        slow_vals = list(range(slow_range[0], slow_range[1], slow_range[2]))

        fast_mas = {p: IndicatorEngine.calculate_ma(df, p, timeframe).values for p in fast_vals}
        slow_mas = {p: IndicatorEngine.calculate_ma(df, p, timeframe).values for p in slow_vals}

        records = []
        matrix_dict = {s: {} for s in slow_vals}

        fee = cost_struct.fee_rate
        tax = cost_struct.tax_rate
        slip = cost_struct.slippage_rate

        for slow_p in slow_vals:
            slow_arr = slow_mas[slow_p]
            for fast_p in fast_vals:
                if fast_p >= slow_p:
                    matrix_dict[slow_p][fast_p] = np.nan
                    continue

                fast_arr = fast_mas[fast_p]
                sig = (fast_arr > slow_arr).astype(int)
                pos = np.roll(sig, 1)
                pos[0] = 0

                cash = float(initial_capital)
                shares = 0.0
                trades = 0
                cum_invested = float(initial_capital)

                for i in range(n_days):
                    # 每月初注入定投資金，徹底根治期初本金為 0 時無錢下單的錯誤
                    if i > 0 and month_changed[i]:
                        cash += monthly_inflow
                        cum_invested += monthly_inflow

                    p_curr = pos[i]
                    p_prev = pos[i - 1] if i > 0 else 0
                    op = open_p[i]

                    if p_curr == 1 and (p_prev == 0 or cash > 10.0):
                        if cash > 10.0:
                            fill_p = op * (1.0 + slip) * (1.0 + fee) * to_target_fx
                            shares += cash / fill_p
                            cash = 0.0
                            trades += 1
                    elif p_curr == 0 and p_prev == 1 and shares > 0:
                        net_p = op * (1.0 - slip) * (1.0 - fee) * (1.0 - tax) * to_target_fx
                        cash += shares * net_p
                        shares = 0.0
                        trades += 1

                final_val = cash + shares * close_p[-1] * to_target_fx
                net_profit = final_val - cum_invested
                roi = (net_profit / cum_invested) if cum_invested > 0 else 0.0

                matrix_dict[slow_p][fast_p] = round(roi * 100.0, 2)
                records.append({
                    "快線": fast_p,
                    "慢線": slow_p,
                    "實賺獲利金額": round(net_profit, 0),
                    "總報酬率 (%)": round(roi * 100.0, 2),
                    "交易次數": trades
                })

        heatmap_df = pd.DataFrame(matrix_dict).T
        flat_df = pd.DataFrame(records).sort_values(by="總報酬率 (%)", ascending=False).reset_index(drop=True)
        best_params = flat_df.iloc[0].to_dict() if not flat_df.empty else {}

        return heatmap_df, flat_df, best_params
