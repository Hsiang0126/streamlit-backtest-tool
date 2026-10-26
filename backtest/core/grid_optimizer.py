"""
快慢線與通用策略 2D 網格尋優模組 (Generic Grid Optimizer Module)
支援：
1. SMA 雙均線 (快線 × 慢線)
2. MACD (EMA快線 × EMA慢線)
3. RSI (超賣買入 × 超買賣出)
4. KD (超賣買入 K × 超買賣出 K)
5. Dip DCA 逢低抄底 (均線週期 × 加碼間隔天數)
"""
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from core.indicators import IndicatorEngine
from core.cost_model import CostModel, AssetCostStructure


class GridOptimizer:
    """高速矩陣尋優器（含完整現金流注入與真實開盤撮合）。"""

    @staticmethod
    def _run_backtest_loop(
        open_p: np.ndarray,
        close_p: np.ndarray,
        pos: np.ndarray,
        month_changed: np.ndarray,
        initial_capital: float,
        monthly_inflow: float,
        fee: float,
        tax: float,
        slip: float,
        to_target_fx: float = 1.0
    ) -> Tuple[float, float, int]:
        n_days = len(open_p)
        cash = float(initial_capital)
        shares = 0.0
        trades = 0
        cum_invested = float(initial_capital)

        for i in range(n_days):
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
        return round(net_profit, 0), round(roi * 100.0, 2), trades

    # --- 1. SMA 雙均線尋優 ---
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
        cost = cost_struct or AssetCostStructure(0.00071, 0.003, 0.0003)
        col_c = "adj_close" if "adj_close" in df.columns else "close"
        close_p = df[col_c].values
        open_p = df["open"].values
        s_months = pd.Series(df.index.to_period("M"), index=df.index)
        month_changed = (s_months != s_months.shift(1)).values

        fast_vals = list(range(fast_range[0], fast_range[1], fast_range[2]))
        slow_vals = list(range(slow_range[0], slow_range[1], slow_range[2]))
        fast_mas = {p: IndicatorEngine.calculate_ma(df, p, timeframe).values for p in fast_vals}
        slow_mas = {p: IndicatorEngine.calculate_ma(df, p, timeframe).values for p in slow_vals}

        records, matrix_dict = [], {s: {} for s in slow_vals}

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

                net_profit, roi, trades = cls._run_backtest_loop(
                    open_p, close_p, pos, month_changed, initial_capital, monthly_inflow,
                    cost.fee_rate, cost.tax_rate, cost.slippage_rate, to_target_fx
                )
                matrix_dict[slow_p][fast_p] = roi
                records.append({
                    "參數X (快線)": fast_p,
                    "參數Y (慢線)": slow_p,
                    "實賺獲利金額": net_profit,
                    "總報酬率 (%)": roi,
                    "交易次數": trades
                })

        heatmap_df = pd.DataFrame(matrix_dict).T
        flat_df = pd.DataFrame(records).sort_values(by="總報酬率 (%)", ascending=False).reset_index(drop=True)
        best_params = flat_df.iloc[0].to_dict() if not flat_df.empty else {}
        return heatmap_df, flat_df, best_params

    # --- 2. MACD 尋優 (Fast EMA × Slow EMA) ---
    @classmethod
    def run_macd_grid(
        cls,
        df: pd.DataFrame,
        fast_range: Tuple[int, int, int] = (5, 21, 1),
        slow_range: Tuple[int, int, int] = (20, 61, 2),
        signal_p: int = 9,
        timeframe: str = "日線 (Daily)",
        initial_capital: float = 300000.0,
        monthly_inflow: float = 0.0,
        cost_struct: Optional[AssetCostStructure] = None,
        to_target_fx: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        cost = cost_struct or AssetCostStructure(0.00071, 0.003, 0.0003)
        col_c = "adj_close" if "adj_close" in df.columns else "close"
        close_p = df[col_c].values
        open_p = df["open"].values
        s_months = pd.Series(df.index.to_period("M"), index=df.index)
        month_changed = (s_months != s_months.shift(1)).values

        fast_vals = list(range(fast_range[0], fast_range[1], fast_range[2]))
        slow_vals = list(range(slow_range[0], slow_range[1], slow_range[2]))

        base_s = df[col_c]
        fast_emas = {f: base_s.ewm(span=f, adjust=False).mean() for f in fast_vals}
        slow_emas = {s: base_s.ewm(span=s, adjust=False).mean() for s in slow_vals}

        records, matrix_dict = [], {s: {} for s in slow_vals}

        for slow_p in slow_vals:
            for fast_p in fast_vals:
                if fast_p >= slow_p:
                    matrix_dict[slow_p][fast_p] = np.nan
                    continue
                macd = fast_emas[fast_p] - slow_emas[slow_p]
                sig_line = macd.ewm(span=signal_p, adjust=False).mean()
                sig = (macd.values > sig_line.values).astype(int)
                pos = np.roll(sig, 1)
                pos[0] = 0

                net_profit, roi, trades = cls._run_backtest_loop(
                    open_p, close_p, pos, month_changed, initial_capital, monthly_inflow,
                    cost.fee_rate, cost.tax_rate, cost.slippage_rate, to_target_fx
                )
                matrix_dict[slow_p][fast_p] = roi
                records.append({
                    "參數X (EMA快線)": fast_p,
                    "參數Y (EMA慢線)": slow_p,
                    "實賺獲利金額": net_profit,
                    "總報酬率 (%)": roi,
                    "交易次數": trades
                })

        heatmap_df = pd.DataFrame(matrix_dict).T
        flat_df = pd.DataFrame(records).sort_values(by="總報酬率 (%)", ascending=False).reset_index(drop=True)
        best_params = flat_df.iloc[0].to_dict() if not flat_df.empty else {}
        return heatmap_df, flat_df, best_params

    # --- 3. RSI 尋優 (超賣買入 × 超買賣出) ---
    @classmethod
    def run_rsi_grid(
        cls,
        df: pd.DataFrame,
        period: int = 14,
        buy_range: Tuple[int, int, int] = (15, 41, 2),
        sell_range: Tuple[int, int, int] = (60, 86, 2),
        timeframe: str = "日線 (Daily)",
        initial_capital: float = 300000.0,
        monthly_inflow: float = 0.0,
        cost_struct: Optional[AssetCostStructure] = None,
        to_target_fx: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        cost = cost_struct or AssetCostStructure(0.00071, 0.003, 0.0003)
        col_c = "adj_close" if "adj_close" in df.columns else "close"
        close_p = df[col_c].values
        open_p = df["open"].values
        s_months = pd.Series(df.index.to_period("M"), index=df.index)
        month_changed = (s_months != s_months.shift(1)).values

        rsi = IndicatorEngine.calculate_rsi(df, period, timeframe).values
        buy_vals = list(range(buy_range[0], buy_range[1], buy_range[2]))
        sell_vals = list(range(sell_range[0], sell_range[1], sell_range[2]))

        records, matrix_dict = [], {s: {} for s in sell_vals}

        for sell_th in sell_vals:
            for buy_th in buy_vals:
                n = len(rsi)
                sig = np.zeros(n, dtype=int)
                state = 0
                for i in range(n):
                    r = rsi[i]
                    if not np.isnan(r):
                        if r < buy_th:
                            state = 1
                        elif r > sell_th:
                            state = 0
                    sig[i] = state

                pos = np.roll(sig, 1)
                pos[0] = 0

                net_profit, roi, trades = cls._run_backtest_loop(
                    open_p, close_p, pos, month_changed, initial_capital, monthly_inflow,
                    cost.fee_rate, cost.tax_rate, cost.slippage_rate, to_target_fx
                )
                matrix_dict[sell_th][buy_th] = roi
                records.append({
                    "參數X (超賣買入)": buy_th,
                    "參數Y (超買賣出)": sell_th,
                    "實賺獲利金額": net_profit,
                    "總報酬率 (%)": roi,
                    "交易次數": trades
                })

        heatmap_df = pd.DataFrame(matrix_dict).T
        flat_df = pd.DataFrame(records).sort_values(by="總報酬率 (%)", ascending=False).reset_index(drop=True)
        best_params = flat_df.iloc[0].to_dict() if not flat_df.empty else {}
        return heatmap_df, flat_df, best_params

    # --- 4. KD 尋優 (超賣 K × 超買 K) ---
    @classmethod
    def run_kd_grid(
        cls,
        df: pd.DataFrame,
        period: int = 9,
        buy_range: Tuple[int, int, int] = (15, 36, 2),
        sell_range: Tuple[int, int, int] = (65, 86, 2),
        timeframe: str = "日線 (Daily)",
        initial_capital: float = 300000.0,
        monthly_inflow: float = 0.0,
        cost_struct: Optional[AssetCostStructure] = None,
        to_target_fx: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        cost = cost_struct or AssetCostStructure(0.00071, 0.003, 0.0003)
        col_c = "adj_close" if "adj_close" in df.columns else "close"
        close_p = df[col_c].values
        open_p = df["open"].values
        s_months = pd.Series(df.index.to_period("M"), index=df.index)
        month_changed = (s_months != s_months.shift(1)).values

        k_vals = IndicatorEngine.calculate_kd(df, period, timeframe).values
        buy_vals = list(range(buy_range[0], buy_range[1], buy_range[2]))
        sell_vals = list(range(sell_range[0], sell_range[1], sell_range[2]))

        records, matrix_dict = [], {s: {} for s in sell_vals}

        for sell_k in sell_vals:
            for buy_k in buy_vals:
                n = len(k_vals)
                sig = np.zeros(n, dtype=int)
                state = 0
                for i in range(n):
                    k = k_vals[i]
                    if not np.isnan(k):
                        if k < buy_k:
                            state = 1
                        elif k > sell_k:
                            state = 0
                    sig[i] = state

                pos = np.roll(sig, 1)
                pos[0] = 0

                net_profit, roi, trades = cls._run_backtest_loop(
                    open_p, close_p, pos, month_changed, initial_capital, monthly_inflow,
                    cost.fee_rate, cost.tax_rate, cost.slippage_rate, to_target_fx
                )
                matrix_dict[sell_k][buy_k] = roi
                records.append({
                    "參數X (超賣買入K)": buy_k,
                    "參數Y (超買賣出K)": sell_k,
                    "實賺獲利金額": net_profit,
                    "總報酬率 (%)": roi,
                    "交易次數": trades
                })

        heatmap_df = pd.DataFrame(matrix_dict).T
        flat_df = pd.DataFrame(records).sort_values(by="總報酬率 (%)", ascending=False).reset_index(drop=True)
        best_params = flat_df.iloc[0].to_dict() if not flat_df.empty else {}
        return heatmap_df, flat_df, best_params

    # --- 5. Dip DCA 尋優 (SMA週期 × 抄底間隔天數) ---
    @classmethod
    def run_dip_grid(
        cls,
        df: pd.DataFrame,
        sma_range: Tuple[int, int, int] = (20, 121, 10),
        interval_range: Tuple[int, int, int] = (1, 11, 1),
        buy_amount: float = 50000.0,
        timeframe: str = "日線 (Daily)",
        initial_capital: float = 300000.0,
        monthly_inflow: float = 30000.0,
        cost_struct: Optional[AssetCostStructure] = None,
        to_target_fx: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        cost = cost_struct or AssetCostStructure(0.00071, 0.003, 0.0003)
        col_c = "adj_close" if "adj_close" in df.columns else "close"
        close_p = df[col_c].values
        open_p = df["open"].values
        n_days = len(df)
        s_months = pd.Series(df.index.to_period("M"), index=df.index)
        month_changed = (s_months != s_months.shift(1)).values

        sma_vals = list(range(sma_range[0], sma_range[1], sma_range[2]))
        interval_vals = list(range(interval_range[0], interval_range[1], interval_range[2]))
        smas = {p: IndicatorEngine.calculate_ma(df, p, timeframe).values for p in sma_vals}

        records, matrix_dict = [], {iv: {} for iv in interval_vals}

        for iv in interval_vals:
            for p in sma_vals:
                sma_arr = smas[p]
                cash = float(initial_capital)
                shares = 0.0
                trades = 0
                cum_invested = float(initial_capital)
                last_buy_idx = -iv
                min_price_in_dip = float("inf")
                pending_buy = False

                for i in range(n_days):
                    if i > 0 and month_changed[i]:
                        cash += monthly_inflow
                        cum_invested += monthly_inflow

                    op = open_p[i]
                    if pending_buy and (i - last_buy_idx >= iv) and cash > 10.0:
                        actual_buy = min(cash, buy_amount)
                        fill_p = op * (1.0 + cost.slippage_rate) * (1.0 + cost.fee_rate) * to_target_fx
                        shares += actual_buy / fill_p
                        cash -= actual_buy
                        last_buy_idx = i
                        trades += 1
                    pending_buy = False

                    eval_p = close_p[i]
                    sma_v = sma_arr[i]
                    if not np.isnan(sma_v) and eval_p < sma_v:
                        if eval_p < min_price_in_dip:
                            min_price_in_dip = eval_p
                            pending_buy = True
                    else:
                        min_price_in_dip = float("inf")

                final_val = cash + shares * close_p[-1] * to_target_fx
                net_profit = final_val - cum_invested
                roi = (net_profit / cum_invested) if cum_invested > 0 else 0.0

                matrix_dict[iv][p] = round(roi * 100.0, 2)
                records.append({
                    "參數X (SMA均線週期)": p,
                    "參數Y (抄底間隔天數)": iv,
                    "實賺獲利金額": round(net_profit, 0),
                    "總報酬率 (%)": round(roi * 100.0, 2),
                    "交易次數": trades
                })

        heatmap_df = pd.DataFrame(matrix_dict).T
        flat_df = pd.DataFrame(records).sort_values(by="總報酬率 (%)", ascending=False).reset_index(drop=True)
        best_params = flat_df.iloc[0].to_dict() if not flat_df.empty else {}
        return heatmap_df, flat_df, best_params
