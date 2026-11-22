"""
快慢線與通用策略 2D 網格尋優模組 (Generic Grid Optimizer Module)
與 BacktestEngine 完全統一撮合基準：
1. 在含暖機期之完整資料計算技術指標，再切片至 user_start_idx == True，確保第 1 天指標完全成熟且無前瞻偏差。
2. 完整納入除息現金 (dividend)、逐日動態匯率 (to_target_fx_arr)、首日扣款 (invest_day_one) 與實際 TWR / MDD / Sharpe / Calmar 計算。
3. 支援多目標尋優 (roi / sharpe / calmar) 與 3x3 參數高原穩健度檢定。
"""
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from core.indicators import IndicatorEngine
from core.cost_model import AssetCostStructure
from core.portfolio_engine import BacktestEngine


class GridOptimizer:
    """高速矩陣尋優器（與主回測引擎 100% 一致的現金流、配息與開盤撮合）。"""

    METRIC_LABELS = {
        "roi": "總報酬率 (%)",
        "sharpe": "夏普比率 (Sharpe)",
        "calmar": "卡瑪比率 (Calmar)",
    }

    @classmethod
    def _prepare_sim_context(
        cls,
        df: pd.DataFrame,
        fx_series: Optional[pd.Series] = None,
        target_currency: str = "TWD",
    ) -> Dict[str, Any]:
        full_data = df.copy().sort_index()
        if "user_start_idx" in full_data.columns:
            eval_data = full_data[full_data["user_start_idx"] == True]
            if eval_data.empty:
                eval_data = full_data
            slice_mask = (full_data["user_start_idx"] == True).to_numpy(dtype=bool)
        else:
            eval_data = full_data
            slice_mask = np.ones(len(full_data), dtype=bool)

        if "quote_currency" in full_data.columns:
            quote_curr = str(full_data["quote_currency"].iloc[0]).strip().upper()
        else:
            quote_curr = "TWD" if "TWD" in target_currency.upper() else "USD"

        target_curr_code = "TWD" if "TWD" in target_currency.upper() else "USD"
        need_fx = quote_curr != target_curr_code
        if fx_series is not None and not fx_series.empty:
            aligned_fx = fx_series.reindex(eval_data.index).ffill().bfill().fillna(32.0)
        else:
            aligned_fx = pd.Series(1.0, index=eval_data.index)

        fx_mult_arr = np.array(
            [
                BacktestEngine._resolve_fx_multiplier(quote_curr, target_curr_code, float(v)) if need_fx else 1.0
                for v in aligned_fx.values
            ],
            dtype=float,
        )

        s_months = pd.Series(eval_data.index.to_period("M"), index=eval_data.index)
        month_changed = (s_months != s_months.shift(1)).fillna(False).to_numpy(dtype=bool)

        col_c = "adj_close" if "adj_close" in eval_data.columns else "close"
        elapsed_days = max(1, (pd.Timestamp(eval_data.index[-1]) - pd.Timestamp(eval_data.index[0])).days)
        years = elapsed_days / 365.25

        return {
            "full_data": full_data,
            "eval_data": eval_data,
            "slice_mask": slice_mask,
            "open_p": eval_data["open"].to_numpy(dtype=float),
            "close_p": eval_data["close"].to_numpy(dtype=float),
            "eval_close_p": eval_data[col_c].to_numpy(dtype=float),
            "div_arr": eval_data["dividend"].to_numpy(dtype=float) if "dividend" in eval_data.columns else np.zeros(len(eval_data)),
            "month_changed": month_changed,
            "fx_mult_arr": fx_mult_arr,
            "years": years,
        }

    @staticmethod
    def _compute_perf_metrics(
        portfolio_vals: np.ndarray,
        external_flows: np.ndarray,
        initial_capital: float,
        years: float,
        risk_free_rate: float = 0.015,
    ) -> Tuple[float, float, float, float, float]:
        cum_invested = float(initial_capital) + float(np.sum(external_flows))
        final_val = float(portfolio_vals[-1])
        net_profit = final_val - cum_invested
        roi = (net_profit / cum_invested) if cum_invested > 0 else 0.0

        twr = BacktestEngine._calc_twr_curve(
            portfolio_vals.tolist(),
            external_flows.tolist(),
            initial_capital,
        )
        cum_inv_series = float(initial_capital) + np.cumsum(external_flows)
        active = twr[cum_inv_series > 0]
        if len(active) < 2:
            active = twr

        start_v = float(active[0]) if active[0] > 0 else 1.0
        end_rel = max(float(active[-1]) / start_v, 0.0)
        eff_years = max(years * (len(active) / max(len(twr), 1)), 0.05)
        cagr = (end_rel ** (1.0 / eff_years) - 1.0) if end_rel > 0 else -1.0

        peaks = np.maximum.accumulate(active)
        dd = np.where(peaks > 0, (active - peaks) / peaks, 0.0)
        mdd = float(np.min(dd)) if len(dd) > 0 else 0.0

        prev_a = active[:-1]
        valid = prev_a > 0
        rets = np.where(valid, (active[1:] / np.where(valid, prev_a, 1.0)) - 1.0, 0.0)
        ann_factor = max(1.0, len(active) / max(eff_years, 1e-4))
        vol = float(np.std(rets, ddof=1) * np.sqrt(ann_factor)) if len(rets) >= 2 else 0.0
        sharpe = ((cagr - risk_free_rate) / vol) if vol > 1e-8 else 0.0
        calmar = (cagr / abs(mdd)) if abs(mdd) > 1e-6 else 0.0

        return (
            round(net_profit, 0),
            round(roi * 100.0, 2),
            round(mdd * 100.0, 2),
            round(sharpe, 3),
            round(calmar, 3),
        )

    @staticmethod
    def _run_backtest_loop(
        open_p: np.ndarray,
        close_p: np.ndarray,
        div_arr: np.ndarray,
        pos: np.ndarray,
        month_changed: np.ndarray,
        fx_mult_arr: np.ndarray,
        initial_capital: float,
        monthly_inflow: float,
        fee: float,
        tax: float,
        slip: float,
        years: float,
        invest_day_one: bool = False,
    ) -> Tuple[float, float, float, float, float, int]:
        n_days = len(open_p)
        cash = float(initial_capital)
        shares = 0.0
        trades = 0

        portfolio_vals = np.zeros(n_days, dtype=float)
        external_flows = np.zeros(n_days, dtype=float)

        for i in range(n_days):
            to_target_fx = float(fx_mult_arr[i])

            div = float(div_arr[i])
            if div > 0 and shares > 0:
                cash += shares * div * to_target_fx

            flow = 0.0
            if i == 0 and initial_capital <= 0.0 and invest_day_one and monthly_inflow > 0:
                flow = float(monthly_inflow)
                cash += flow
            elif i > 0 and month_changed[i] and monthly_inflow > 0:
                flow = float(monthly_inflow)
                cash += flow

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

            portfolio_vals[i] = cash + shares * close_p[i] * to_target_fx
            external_flows[i] = flow

        net_profit, roi_pct, mdd_pct, sharpe, calmar = GridOptimizer._compute_perf_metrics(
            portfolio_vals, external_flows, initial_capital, years
        )
        return net_profit, roi_pct, mdd_pct, sharpe, calmar, trades

    @classmethod
    def evaluate_plateau_stability(
        cls,
        heatmap_df: pd.DataFrame,
        best_x: Any,
        best_y: Any,
    ) -> Dict[str, Any]:
        """計算最佳參數周圍 3x3 鄰域的穩健度，判斷為參數高原或孤島陷阱。"""
        if heatmap_df is None or heatmap_df.empty:
            return {"status": "無資料", "ratio": 0.0, "neighbor_median": 0.0}
        try:
            y_idx = list(heatmap_df.index).index(best_y)
            x_idx = list(heatmap_df.columns).index(best_x)
        except ValueError:
            return {"status": "無資料", "ratio": 0.0, "neighbor_median": 0.0}

        y_slice = slice(max(0, y_idx - 1), min(len(heatmap_df.index), y_idx + 2))
        x_slice = slice(max(0, x_idx - 1), min(len(heatmap_df.columns), x_idx + 2))
        sub = heatmap_df.iloc[y_slice, x_slice].to_numpy(dtype=float).flatten()
        valid = sub[~np.isnan(sub)]
        best_val = float(heatmap_df.iloc[y_idx, x_idx])

        if len(valid) <= 1 or not np.isfinite(best_val):
            return {"status": "孤立點", "ratio": 0.0, "neighbor_median": best_val}

        neighbor_median = float(np.median(valid))
        if best_val <= 0:
            status = "全區虧損"
            ratio = 0.0
        else:
            ratio = neighbor_median / best_val
            if ratio >= 0.80:
                status = "🟢 穩健參數高原 (推薦)"
            elif ratio >= 0.55:
                status = "🟡 中等穩健區間"
            else:
                status = "🔴 疑似孤島過擬合 (慎用)"

        return {
            "status": status,
            "ratio": round(ratio, 2),
            "neighbor_median": round(neighbor_median, 2),
        }

    @classmethod
    def _select_score(cls, roi: float, sharpe: float, calmar: float, metric: str) -> float:
        if metric == "sharpe":
            return sharpe
        if metric == "calmar":
            return calmar
        return roi

    @classmethod
    def _finalize_grid_results(
        cls,
        matrix_dict: Dict[Any, Dict[Any, float]],
        records: list,
        metric: str,
        x_col: str,
        y_col: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        heatmap_df = pd.DataFrame(matrix_dict).T
        if not records:
            return heatmap_df, pd.DataFrame(), {}
        sort_col = cls.METRIC_LABELS.get(metric, "總報酬率 (%)")
        flat_df = (
            pd.DataFrame(records)
            .sort_values(by=[sort_col, "總報酬率 (%)"], ascending=False)
            .reset_index(drop=True)
        )
        best_params = flat_df.iloc[0].to_dict()
        best_params["plateau_info"] = cls.evaluate_plateau_stability(
            heatmap_df, best_params[x_col], best_params[y_col]
        )
        best_params["x_col"] = x_col
        best_params["y_col"] = y_col
        return heatmap_df, flat_df, best_params

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
        fx_series: Optional[pd.Series] = None,
        target_currency: str = "TWD",
        metric: str = "roi",
        invest_day_one: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        cost = cost_struct or AssetCostStructure(0.0007125, 0.003, 0.0003)
        ctx = cls._prepare_sim_context(df, fx_series=fx_series, target_currency=target_currency)
        full_df, slice_mask = ctx["full_data"], ctx["slice_mask"]

        fast_vals = list(range(fast_range[0], fast_range[1], fast_range[2]))
        slow_vals = list(range(slow_range[0], slow_range[1], slow_range[2]))
        fast_mas = {p: IndicatorEngine.calculate_ma(full_df, p, timeframe).values for p in fast_vals}
        slow_mas = {p: IndicatorEngine.calculate_ma(full_df, p, timeframe).values for p in slow_vals}

        records, matrix_dict = [], {s: {} for s in slow_vals}
        for slow_p in slow_vals:
            slow_arr = slow_mas[slow_p]
            for fast_p in fast_vals:
                if fast_p >= slow_p:
                    matrix_dict[slow_p][fast_p] = np.nan
                    continue
                fast_arr = fast_mas[fast_p]
                raw_sig = np.where(
                    np.isnan(fast_arr) | np.isnan(slow_arr), 0, (fast_arr > slow_arr).astype(int)
                )
                full_pos = np.roll(raw_sig, 1)
                full_pos[0] = 0
                pos = full_pos[slice_mask]

                net_profit, roi, mdd, sharpe, calmar, trades = cls._run_backtest_loop(
                    ctx["open_p"], ctx["close_p"], ctx["div_arr"], pos,
                    ctx["month_changed"], ctx["fx_mult_arr"],
                    initial_capital, monthly_inflow,
                    cost.fee_rate, cost.tax_rate, cost.slippage_rate,
                    ctx["years"], invest_day_one=invest_day_one,
                )
                matrix_dict[slow_p][fast_p] = cls._select_score(roi, sharpe, calmar, metric)
                records.append({
                    "參數X (快線)": fast_p,
                    "參數Y (慢線)": slow_p,
                    "實賺獲利金額": net_profit,
                    "總報酬率 (%)": roi,
                    "最大回撤 (%)": mdd,
                    "夏普比率 (Sharpe)": sharpe,
                    "卡瑪比率 (Calmar)": calmar,
                    "交易次數": trades,
                })

        return cls._finalize_grid_results(matrix_dict, records, metric, "參數X (快線)", "參數Y (慢線)")

    # --- 2. MACD 尋優 (Fast EMA × Slow EMA，支援週線與暖機期) ---
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
        fx_series: Optional[pd.Series] = None,
        target_currency: str = "TWD",
        metric: str = "roi",
        invest_day_one: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        cost = cost_struct or AssetCostStructure(0.0007125, 0.003, 0.0003)
        ctx = cls._prepare_sim_context(df, fx_series=fx_series, target_currency=target_currency)
        full_df, slice_mask = ctx["full_data"], ctx["slice_mask"]

        fast_vals = list(range(fast_range[0], fast_range[1], fast_range[2]))
        slow_vals = list(range(slow_range[0], slow_range[1], slow_range[2]))

        is_weekly = "週" in timeframe
        if is_weekly:
            weekly = IndicatorEngine._resample_weekly(full_df)
            base_s = weekly["close"]
        else:
            col_c = "adj_close" if "adj_close" in full_df.columns else "close"
            base_s = full_df[col_c]

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
                if is_weekly:
                    macd = macd.reindex(full_df.index, method="ffill")
                    sig_line = sig_line.reindex(full_df.index, method="ffill")

                raw_sig = (macd.fillna(0).values > sig_line.fillna(0).values).astype(int)
                full_pos = np.roll(raw_sig, 1)
                full_pos[0] = 0
                pos = full_pos[slice_mask]

                net_profit, roi, mdd, sharpe, calmar, trades = cls._run_backtest_loop(
                    ctx["open_p"], ctx["close_p"], ctx["div_arr"], pos,
                    ctx["month_changed"], ctx["fx_mult_arr"],
                    initial_capital, monthly_inflow,
                    cost.fee_rate, cost.tax_rate, cost.slippage_rate,
                    ctx["years"], invest_day_one=invest_day_one,
                )
                matrix_dict[slow_p][fast_p] = cls._select_score(roi, sharpe, calmar, metric)
                records.append({
                    "參數X (EMA快線)": fast_p,
                    "參數Y (EMA慢線)": slow_p,
                    "實賺獲利金額": net_profit,
                    "總報酬率 (%)": roi,
                    "最大回撤 (%)": mdd,
                    "夏普比率 (Sharpe)": sharpe,
                    "卡瑪比率 (Calmar)": calmar,
                    "交易次數": trades,
                })

        return cls._finalize_grid_results(matrix_dict, records, metric, "參數X (EMA快線)", "參數Y (EMA慢線)")

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
        fx_series: Optional[pd.Series] = None,
        target_currency: str = "TWD",
        metric: str = "roi",
        invest_day_one: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        cost = cost_struct or AssetCostStructure(0.0007125, 0.003, 0.0003)
        ctx = cls._prepare_sim_context(df, fx_series=fx_series, target_currency=target_currency)
        full_df, slice_mask = ctx["full_data"], ctx["slice_mask"]

        rsi = IndicatorEngine.calculate_rsi(full_df, period, timeframe).values
        buy_vals = list(range(buy_range[0], buy_range[1], buy_range[2]))
        sell_vals = list(range(sell_range[0], sell_range[1], sell_range[2]))

        records, matrix_dict = [], {s: {} for s in sell_vals}
        n = len(rsi)
        for sell_th in sell_vals:
            for buy_th in buy_vals:
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

                full_pos = np.roll(sig, 1)
                full_pos[0] = 0
                pos = full_pos[slice_mask]

                net_profit, roi, mdd, sharpe, calmar, trades = cls._run_backtest_loop(
                    ctx["open_p"], ctx["close_p"], ctx["div_arr"], pos,
                    ctx["month_changed"], ctx["fx_mult_arr"],
                    initial_capital, monthly_inflow,
                    cost.fee_rate, cost.tax_rate, cost.slippage_rate,
                    ctx["years"], invest_day_one=invest_day_one,
                )
                matrix_dict[sell_th][buy_th] = cls._select_score(roi, sharpe, calmar, metric)
                records.append({
                    "參數X (超賣買入)": buy_th,
                    "參數Y (超買賣出)": sell_th,
                    "實賺獲利金額": net_profit,
                    "總報酬率 (%)": roi,
                    "最大回撤 (%)": mdd,
                    "夏普比率 (Sharpe)": sharpe,
                    "卡瑪比率 (Calmar)": calmar,
                    "交易次數": trades,
                })

        return cls._finalize_grid_results(matrix_dict, records, metric, "參數X (超賣買入)", "參數Y (超買賣出)")

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
        fx_series: Optional[pd.Series] = None,
        target_currency: str = "TWD",
        metric: str = "roi",
        invest_day_one: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        cost = cost_struct or AssetCostStructure(0.0007125, 0.003, 0.0003)
        ctx = cls._prepare_sim_context(df, fx_series=fx_series, target_currency=target_currency)
        full_df, slice_mask = ctx["full_data"], ctx["slice_mask"]

        k_vals = IndicatorEngine.calculate_kd(full_df, period, timeframe).values
        buy_vals = list(range(buy_range[0], buy_range[1], buy_range[2]))
        sell_vals = list(range(sell_range[0], sell_range[1], sell_range[2]))

        records, matrix_dict = [], {s: {} for s in sell_vals}
        n = len(k_vals)
        for sell_k in sell_vals:
            for buy_k in buy_vals:
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

                full_pos = np.roll(sig, 1)
                full_pos[0] = 0
                pos = full_pos[slice_mask]

                net_profit, roi, mdd, sharpe, calmar, trades = cls._run_backtest_loop(
                    ctx["open_p"], ctx["close_p"], ctx["div_arr"], pos,
                    ctx["month_changed"], ctx["fx_mult_arr"],
                    initial_capital, monthly_inflow,
                    cost.fee_rate, cost.tax_rate, cost.slippage_rate,
                    ctx["years"], invest_day_one=invest_day_one,
                )
                matrix_dict[sell_k][buy_k] = cls._select_score(roi, sharpe, calmar, metric)
                records.append({
                    "參數X (超賣買入K)": buy_k,
                    "參數Y (超買賣出K)": sell_k,
                    "實賺獲利金額": net_profit,
                    "總報酬率 (%)": roi,
                    "最大回撤 (%)": mdd,
                    "夏普比率 (Sharpe)": sharpe,
                    "卡瑪比率 (Calmar)": calmar,
                    "交易次數": trades,
                })

        return cls._finalize_grid_results(matrix_dict, records, metric, "參數X (超賣買入K)", "參數Y (超買賣出K)")

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
        fx_series: Optional[pd.Series] = None,
        target_currency: str = "TWD",
        metric: str = "roi",
        invest_day_one: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        cost = cost_struct or AssetCostStructure(0.0007125, 0.003, 0.0003)
        ctx = cls._prepare_sim_context(df, fx_series=fx_series, target_currency=target_currency)
        full_df, slice_mask = ctx["full_data"], ctx["slice_mask"]

        open_p = ctx["open_p"]
        close_p = ctx["close_p"]
        eval_close_p = ctx["eval_close_p"]
        div_arr = ctx["div_arr"]
        month_changed = ctx["month_changed"]
        fx_mult_arr = ctx["fx_mult_arr"]
        n_days = len(open_p)

        sma_vals = list(range(sma_range[0], sma_range[1], sma_range[2]))
        interval_vals = list(range(interval_range[0], interval_range[1], interval_range[2]))
        # 先對完整含暖機期的 full_df 算均線，再切片至回測區間
        smas = {p: IndicatorEngine.calculate_ma(full_df, p, timeframe).values[slice_mask] for p in sma_vals}

        records, matrix_dict = [], {iv: {} for iv in interval_vals}
        for iv in interval_vals:
            for p in sma_vals:
                sma_arr = smas[p]
                cash = float(initial_capital)
                shares = 0.0
                trades = 0
                last_buy_idx = -iv
                min_price_in_dip = float("inf")
                pending_buy = False

                portfolio_vals = np.zeros(n_days, dtype=float)
                external_flows = np.zeros(n_days, dtype=float)

                for i in range(n_days):
                    to_target_fx = float(fx_mult_arr[i])
                    div = float(div_arr[i])
                    if div > 0 and shares > 0:
                        cash += shares * div * to_target_fx

                    flow = 0.0
                    if i == 0 and initial_capital <= 0.0 and invest_day_one and monthly_inflow > 0:
                        flow = float(monthly_inflow)
                        cash += flow
                    elif i > 0 and month_changed[i] and monthly_inflow > 0:
                        flow = float(monthly_inflow)
                        cash += flow

                    op = open_p[i]
                    if pending_buy and (i - last_buy_idx >= iv) and cash > 10.0:
                        actual_buy = min(cash, buy_amount)
                        fill_p = op * (1.0 + cost.slippage_rate) * (1.0 + cost.fee_rate) * to_target_fx
                        shares += actual_buy / fill_p
                        cash -= actual_buy
                        last_buy_idx = i
                        trades += 1
                    pending_buy = False

                    eval_p = eval_close_p[i]
                    sma_v = sma_arr[i]
                    if not np.isnan(sma_v) and eval_p < sma_v:
                        if eval_p < min_price_in_dip:
                            min_price_in_dip = eval_p
                            pending_buy = True
                    else:
                        min_price_in_dip = float("inf")

                    portfolio_vals[i] = cash + shares * close_p[i] * to_target_fx
                    external_flows[i] = flow

                net_profit, roi, mdd, sharpe, calmar = cls._compute_perf_metrics(
                    portfolio_vals, external_flows, initial_capital, ctx["years"]
                )
                matrix_dict[iv][p] = cls._select_score(roi, sharpe, calmar, metric)
                records.append({
                    "參數X (SMA均線週期)": p,
                    "參數Y (抄底間隔天數)": iv,
                    "實賺獲利金額": net_profit,
                    "總報酬率 (%)": roi,
                    "最大回撤 (%)": mdd,
                    "夏普比率 (Sharpe)": sharpe,
                    "卡瑪比率 (Calmar)": calmar,
                    "交易次數": trades,
                })

        return cls._finalize_grid_results(matrix_dict, records, metric, "參數X (SMA均線週期)", "參數Y (抄底間隔天數)")
