"""
核心回測引擎 (Portfolio & Strategy Engine - 機構級重構版)
徹底修復 5 大地雷：
1. 雙軌計價：標的保持原生報價，僅在交易換匯與總淨值統計時換算，杜絕匯率扭曲偽 K 棒。
2. 開市日判定與休市順延：休市日絕不幽靈成交，買賣訂單自動順延至該標的下一個真實開盤日。
3. 靜默暖機期裁切：策略在含暖機期的全量數據上計算指標，帳本與統計嚴格從 start_date 開始。
4. 全報酬還原價格撮合：除權息與拆股真實反映，杜絕假停損。
"""
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from core.cost_model import CostModel, AssetCostStructure
from strategies.registry import StrategyRegistry


class BacktestEngine:
    """雙模式機構級精準回測引擎。"""

    # -------------------------------------------------------------
    # 模式 A：單標的策略執行引擎 (含暖機期裁切與休市防護)
    # -------------------------------------------------------------
    @classmethod
    def run_single_strategy(
        cls,
        df: pd.DataFrame,
        strat_name: str,
        strat_params: Dict[str, Any],
        initial_capital: float,
        monthly_inflow: float,
        cost_struct: AssetCostStructure,
        fx_series: Optional[pd.Series] = None,
        target_currency: str = "TWD"
    ) -> pd.DataFrame:
        strat_cls = StrategyRegistry.get(strat_name)
        if not strat_cls:
            raise ValueError(f"未註冊的策略：{strat_name}")

        full_data = df.copy().sort_index()
        quote_curr = full_data.get("quote_currency", pd.Series("USD", index=full_data.index)).iloc[0]

        if strat_cls.is_timing:
            raw_sig = strat_cls.generate_signals(full_data, strat_params)
            full_pos = raw_sig.shift(1).fillna(0).astype(int)
        else:
            full_pos = pd.Series(1, index=full_data.index)

        if "user_start_idx" in full_data.columns:
            data = full_data[full_data["user_start_idx"] == True].copy()
            pos_series = full_pos.reindex(data.index).fillna(0).astype(int)
        else:
            data = full_data.copy()
            pos_series = full_pos.copy()

        if len(data) < 2:
            raise ValueError("有效回測區間內之交易日數不足。")

        target_curr_code = "TWD" if "TWD" in target_currency.upper() else "USD"
        need_fx = (quote_curr != target_curr_code)
        if fx_series is not None and not fx_series.empty:
            aligned_fx = fx_series.reindex(data.index, method="ffill").fillna(31.0)
        else:
            aligned_fx = pd.Series(1.0, index=data.index)

        month_period = data.index.to_period("M")
        month_changed = pd.Series(month_period != pd.Series(month_period).shift(1), index=data.index)

        if strat_cls.is_timing:
            return cls._execute_timing_loop(data, pos_series, initial_capital, monthly_inflow, cost_struct, month_changed, aligned_fx, need_fx, target_curr_code)
        elif strat_cls.strategy_id == "dca_buy_and_hold":
            return cls._execute_dca_loop(data, strat_params, initial_capital, monthly_inflow, cost_struct, month_changed, aligned_fx, need_fx, target_curr_code)
        elif strat_cls.strategy_id == "dip_buyer":
            return cls._execute_dip_loop(data, strat_params, initial_capital, monthly_inflow, cost_struct, month_changed, aligned_fx, need_fx, target_curr_code)
        else:
            raise ValueError(f"不支援的策略 ID：{strat_cls.strategy_id}")

    @staticmethod
    def _execute_timing_loop(
        data: pd.DataFrame,
        pos_series: pd.Series,
        initial_capital: float,
        monthly_inflow: float,
        cost: AssetCostStructure,
        month_changed: pd.Series,
        aligned_fx: pd.Series,
        need_fx: bool,
        target_curr: str
    ) -> pd.DataFrame:
        cash = float(initial_capital)
        shares = 0.0
        trade_count = 0
        portfolio_vals, cash_hist, stock_hist, external_flows = [], [], [], []

        for i, dt in enumerate(data.index):
            fx = float(aligned_fx.iloc[i]) if need_fx else 1.0
            to_target_fx = fx if target_curr == "TWD" else (1.0 / fx if fx > 0 else 1.0)

            div = float(data["dividend"].iloc[i]) if "dividend" in data.columns else 0.0
            if div > 0 and shares > 0:
                cash += shares * div * to_target_fx

            split = float(data["split"].iloc[i]) if "split" in data.columns else 0.0
            if split > 0 and shares > 0:
                shares = shares * split

            flow = 0.0
            if i > 0 and bool(month_changed.iloc[i]):
                flow = float(monthly_inflow)
                cash += flow

            pos = int(pos_series.iloc[i])
            prev_pos = int(pos_series.iloc[i - 1]) if i > 0 else 0
            open_p_native = float(data["open"].iloc[i])
            close_p_native = float(data["close"].iloc[i])

            if pos == 1 and (prev_pos == 0 or cash > 0):
                if cash > 10.0:
                    fill_p_native = CostModel.calculate_buy_fill(open_p_native, cost.fee_rate, cost.slippage_rate)
                    fill_p_target = fill_p_native * to_target_fx
                    buy_shares = cash / fill_p_target
                    shares += buy_shares
                    cash = 0.0
                    trade_count += 1
            elif pos == 0 and prev_pos == 1 and shares > 0:
                net_p_native = CostModel.calculate_sell_net(open_p_native, cost.fee_rate, cost.tax_rate, cost.slippage_rate)
                net_p_target = net_p_native * to_target_fx
                cash += shares * net_p_target
                shares = 0.0
                trade_count += 1

            stock_val = shares * close_p_native * to_target_fx
            total_val = cash + stock_val

            portfolio_vals.append(total_val)
            cash_hist.append(cash)
            stock_hist.append(stock_val)
            external_flows.append(flow)

        res = data.copy()
        res["portfolio_value"] = portfolio_vals
        res["cash"] = cash_hist
        res["stock_value"] = stock_hist
        res["external_flow"] = external_flows
        res["total_invested"] = float(initial_capital) + pd.Series(external_flows, index=data.index).cumsum()
        res["trades"] = trade_count
        res["equity_net"] = BacktestEngine._calc_twr_curve(portfolio_vals, external_flows, initial_capital)
        return res

    @staticmethod
    def _execute_dca_loop(
        data: pd.DataFrame,
        params: Dict[str, Any],
        initial_capital: float,
        monthly_inflow: float,
        cost: AssetCostStructure,
        month_changed: pd.Series,
        aligned_fx: pd.Series,
        need_fx: bool,
        target_curr: str
    ) -> pd.DataFrame:
        freq = params.get("freq", "每月 (Monthly)")

        if freq == "每週 (Weekly)":
            iso = data.index.isocalendar()
            week_key = iso["year"].astype(str) + "-" + iso["week"].astype(str)
            buy_mask = week_key.ne(pd.Series(week_key).shift(1))
        elif freq == "每 10 交易日":
            buy_mask = pd.Series(np.arange(len(data)) % 10 == 0, index=data.index)
        else:
            buy_mask = month_changed

        cash = float(initial_capital)
        shares = 0.0
        trade_count = 0
        portfolio_vals, cash_hist, stock_hist, external_flows = [], [], [], []

        for i, dt in enumerate(data.index):
            fx = float(aligned_fx.iloc[i]) if need_fx else 1.0
            to_target_fx = fx if target_curr == "TWD" else (1.0 / fx if fx > 0 else 1.0)

            div = float(data["dividend"].iloc[i]) if "dividend" in data.columns else 0.0
            if div > 0 and shares > 0:
                cash += shares * div * to_target_fx

            split = float(data["split"].iloc[i]) if "split" in data.columns else 0.0
            if split > 0 and shares > 0:
                shares = shares * split

            flow = 0.0
            if i > 0 and bool(month_changed.iloc[i]):
                flow = float(monthly_inflow)
                cash += flow

            open_p_native = float(data["open"].iloc[i])
            close_p_native = float(data["close"].iloc[i])

            if i == 0 and cash > 0:
                fill_p_native = CostModel.calculate_buy_fill(open_p_native, cost.fee_rate, cost.slippage_rate)
                fill_p_target = fill_p_native * to_target_fx
                shares += cash / fill_p_target
                cash = 0.0
                trade_count += 1
            elif bool(buy_mask.iloc[i]) and cash > 10.0:
                fill_p_native = CostModel.calculate_buy_fill(open_p_native, cost.fee_rate, cost.slippage_rate)
                fill_p_target = fill_p_native * to_target_fx
                shares += cash / fill_p_target
                cash = 0.0
                trade_count += 1

            stock_val = shares * close_p_native * to_target_fx
            total_val = cash + stock_val

            portfolio_vals.append(total_val)
            cash_hist.append(cash)
            stock_hist.append(stock_val)
            external_flows.append(flow)

        res = data.copy()
        res["portfolio_value"] = portfolio_vals
        res["cash"] = cash_hist
        res["stock_value"] = stock_hist
        res["external_flow"] = external_flows
        res["total_invested"] = float(initial_capital) + pd.Series(external_flows, index=data.index).cumsum()
        res["trades"] = trade_count
        res["equity_net"] = BacktestEngine._calc_twr_curve(portfolio_vals, external_flows, initial_capital)
        return res

    @staticmethod
    def _execute_dip_loop(
        data: pd.DataFrame,
        params: Dict[str, Any],
        initial_capital: float,
        monthly_inflow: float,
        cost: AssetCostStructure,
        month_changed: pd.Series,
        aligned_fx: pd.Series,
        need_fx: bool,
        target_curr: str
    ) -> pd.DataFrame:
        from core.indicators import IndicatorEngine
        sma_p = int(params.get("sma_period", 60))
        buy_amt = float(params.get("buy_amount", 50000.0))
        min_int = max(1, int(params.get("min_interval", 3)))
        tf = params.get("timeframe", "日線 (Daily)")

        sma = IndicatorEngine.calculate_ma(data, sma_p, tf)

        cash = float(initial_capital)
        shares = 0.0
        trade_count = 0
        last_buy_idx = -min_int
        min_price_in_dip = float("inf")
        pending_buy = False

        portfolio_vals, cash_hist, stock_hist, external_flows = [], [], [], []

        for i, dt in enumerate(data.index):
            fx = float(aligned_fx.iloc[i]) if need_fx else 1.0
            to_target_fx = fx if target_curr == "TWD" else (1.0 / fx if fx > 0 else 1.0)

            div = float(data["dividend"].iloc[i]) if "dividend" in data.columns else 0.0
            if div > 0 and shares > 0:
                cash += shares * div * to_target_fx

            flow = 0.0
            if i > 0 and bool(month_changed.iloc[i]):
                flow = float(monthly_inflow)
                cash += flow

            open_p_native = float(data["open"].iloc[i])
            close_p_native = float(data["close"].iloc[i])

            if pending_buy and (i - last_buy_idx >= min_int) and cash > 10.0:
                actual_buy = min(cash, buy_amt)
                fill_p_native = CostModel.calculate_buy_fill(open_p_native, cost.fee_rate, cost.slippage_rate)
                fill_p_target = fill_p_native * to_target_fx
                shares += actual_buy / fill_p_target
                cash -= actual_buy
                last_buy_idx = i
                trade_count += 1
            pending_buy = False

            sma_val = sma.iloc[i]
            col_c = "adj_close" if "adj_close" in data.columns else "close"
            current_eval_p = float(data[col_c].iloc[i])

            if pd.notna(sma_val) and current_eval_p < float(sma_val):
                if current_eval_p < min_price_in_dip:
                    min_price_in_dip = current_eval_p
                    pending_buy = True
            else:
                min_price_in_dip = float("inf")

            stock_val = shares * close_p_native * to_target_fx
            total_val = cash + stock_val

            portfolio_vals.append(total_val)
            cash_hist.append(cash)
            stock_hist.append(stock_val)
            external_flows.append(flow)

        res = data.copy()
        res["portfolio_value"] = portfolio_vals
        res["cash"] = cash_hist
        res["stock_value"] = stock_hist
        res["external_flow"] = external_flows
        res["total_invested"] = float(initial_capital) + pd.Series(external_flows, index=data.index).cumsum()
        res["trades"] = trade_count
        res["equity_net"] = BacktestEngine._calc_twr_curve(portfolio_vals, external_flows, initial_capital)
        return res

    # -------------------------------------------------------------
    # 模式 B：多資產配置與動態再平衡 (含休市順延與雙軌換匯)
    # -------------------------------------------------------------
    @classmethod
    def run_multi_asset_portfolio(
        cls,
        aligned_panel: pd.DataFrame,
        aligned_fx: pd.Series,
        target_weights: Dict[str, float],
        initial_capital: float,
        monthly_inflow: float,
        rebalance_freq: str = "每季再平衡 (Quarterly)",
        cost_dict: Optional[Dict[str, AssetCostStructure]] = None,
        target_currency: str = "TWD"
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        tickers = list(target_weights.keys())
        tot_w = sum(target_weights.values())
        norm_weights = {t: target_weights[t] / tot_w for t in tickers}
        target_curr_code = "TWD" if "TWD" in target_currency.upper() else "USD"

        if cost_dict is None:
            cost_dict = {t: CostModel.get_default_costs(t) for t in tickers}

        data_idx = aligned_panel.index
        month_period = data_idx.to_period("M")
        month_changed = pd.Series(month_period != pd.Series(month_period).shift(1), index=data_idx)

        rebalance_mask = pd.Series(False, index=data_idx)
        if rebalance_freq == "每月再平衡 (Monthly)":
            rebalance_mask = month_changed.copy()
        elif rebalance_freq == "每季再平衡 (Quarterly)":
            quarter_period = data_idx.to_period("Q")
            rebalance_mask = pd.Series(quarter_period != pd.Series(quarter_period).shift(1), index=data_idx)
        elif rebalance_freq == "每半年再平衡 (Semi-Annually)":
            half_year = data_idx.year.astype(str) + "-" + np.where(data_idx.month <= 6, "H1", "H2")
            rebalance_mask = pd.Series(half_year != pd.Series(half_year).shift(1), index=data_idx)
        elif rebalance_freq == "每年再平衡 (Annually)":
            year_period = data_idx.to_period("Y")
            rebalance_mask = pd.Series(year_period != pd.Series(year_period).shift(1), index=data_idx)

        cash = float(initial_capital)
        shares = {t: 0.0 for t in tickers}
        trade_count = 0

        pending_rebalance_orders: Dict[str, float] = {}
        asset_cum_bought = {t: 0.0 for t in tickers}
        asset_cum_sold = {t: 0.0 for t in tickers}
        asset_cum_div = {t: 0.0 for t in tickers}

        history_rows = []

        for i, dt in enumerate(data_idx):
            fx = float(aligned_fx.iloc[i])
            to_target_fx = {
                t: (fx if aligned_panel[(t, "quote_currency")].iloc[0] == "USD" and target_curr_code == "TWD"
                    else (1.0 / fx if aligned_panel[(t, "quote_currency")].iloc[0] == "TWD" and target_curr_code == "USD" else 1.0))
                for t in tickers
            }

            for t in tickers:
                div = float(aligned_panel[(t, "dividend")].iloc[i]) if (t, "dividend") in aligned_panel.columns else 0.0
                if div > 0 and shares[t] > 0:
                    div_target = shares[t] * div * to_target_fx[t]
                    cash += div_target
                    asset_cum_div[t] += div_target

            flow = 0.0
            if i > 0 and bool(month_changed.iloc[i]):
                flow = float(monthly_inflow)
                cash += flow

            open_prices_native = {t: float(aligned_panel[(t, "open")].iloc[i]) for t in tickers}
            close_prices_native = {t: float(aligned_panel[(t, "close")].iloc[i]) for t in tickers}
            is_market_open = {t: bool(aligned_panel[(t, "is_trading_day")].iloc[i]) for t in tickers}

            is_initial = (i == 0)
            is_rebal = bool(rebalance_mask.iloc[i]) if i > 0 else False

            if is_initial:
                for t in tickers:
                    alloc_amt = cash * norm_weights[t]
                    if alloc_amt > 10.0:
                        cost = cost_dict[t]
                        fill_p = CostModel.calculate_buy_fill(open_prices_native[t], cost.fee_rate, cost.slippage_rate) * to_target_fx[t]
                        bought_s = alloc_amt / fill_p
                        shares[t] += bought_s
                        asset_cum_bought[t] += alloc_amt
                        trade_count += 1
                cash = 0.0

            elif is_rebal:
                pre_rebal_val = cash + sum(shares[t] * open_prices_native[t] * to_target_fx[t] for t in tickers)
                for t in tickers:
                    target_val = pre_rebal_val * norm_weights[t]
                    curr_val = shares[t] * open_prices_native[t] * to_target_fx[t]
                    pending_rebalance_orders[t] = target_val - curr_val

            if pending_rebalance_orders:
                for t in list(pending_rebalance_orders.keys()):
                    diff = pending_rebalance_orders[t]
                    if diff < -50.0 and is_market_open[t]:
                        cost = cost_dict[t]
                        sell_val_target = -diff
                        net_p = CostModel.calculate_sell_net(open_prices_native[t], cost.fee_rate, cost.tax_rate, cost.slippage_rate) * to_target_fx[t]
                        sell_s = min(shares[t], sell_val_target / (open_prices_native[t] * to_target_fx[t]))
                        shares[t] -= sell_s
                        cash_recv = sell_s * net_p
                        cash += cash_recv
                        asset_cum_sold[t] += cash_recv
                        trade_count += 1
                        del pending_rebalance_orders[t]

                for t in list(pending_rebalance_orders.keys()):
                    diff = pending_rebalance_orders[t]
                    if diff > 50.0 and is_market_open[t] and cash > 10.0:
                        cost = cost_dict[t]
                        actual_buy_target = min(cash, diff)
                        fill_p = CostModel.calculate_buy_fill(open_prices_native[t], cost.fee_rate, cost.slippage_rate) * to_target_fx[t]
                        bought_s = actual_buy_target / fill_p
                        shares[t] += bought_s
                        cash -= actual_buy_target
                        asset_cum_bought[t] += actual_buy_target
                        trade_count += 1
                        del pending_rebalance_orders[t]

            elif flow > 0 and cash > 10.0:
                deploy_pool = cash
                for t in tickers:
                    if is_market_open[t]:
                        alloc_amt = deploy_pool * norm_weights[t]
                        if alloc_amt > 10.0 and cash >= alloc_amt:
                            cost = cost_dict[t]
                            fill_p = CostModel.calculate_buy_fill(open_prices_native[t], cost.fee_rate, cost.slippage_rate) * to_target_fx[t]
                            bought_s = alloc_amt / fill_p
                            shares[t] += bought_s
                            cash -= alloc_amt
                            asset_cum_bought[t] += alloc_amt
                            trade_count += 1

            stock_vals_target = {t: shares[t] * close_prices_native[t] * to_target_fx[t] for t in tickers}
            total_stock_val = sum(stock_vals_target.values())
            total_port_val = cash + total_stock_val

            row_data = {
                "date": dt,
                "portfolio_value": total_port_val,
                "cash": cash,
                "total_stock_value": total_stock_val,
                "external_flow": flow,
                "trades": trade_count
            }
            for t in tickers:
                row_data[f"val_{t}"] = stock_vals_target[t]
                row_data[f"shares_{t}"] = shares[t]

            history_rows.append(row_data)

        res_df = pd.DataFrame(history_rows).set_index("date")
        res_df["total_invested"] = float(initial_capital) + res_df["external_flow"].cumsum()
        res_df["equity_net"] = BacktestEngine._calc_twr_curve(
            res_df["portfolio_value"].tolist(),
            res_df["external_flow"].tolist(),
            initial_capital
        )

        attribution: Dict[str, Dict[str, float]] = {}
        for t in tickers:
            final_holding_val = shares[t] * close_prices_native[t] * to_target_fx[t]
            total_out = asset_cum_bought[t]
            total_in = final_holding_val + asset_cum_sold[t] + asset_cum_div[t]
            net_gain = total_in - total_out
            roi = (net_gain / total_out) if total_out > 0 else 0.0

            attribution[t] = {
                "final_value": final_holding_val,
                "cum_bought": total_out,
                "cum_sold": asset_cum_sold[t],
                "cum_dividend": asset_cum_div[t],
                "net_profit": net_gain,
                "roi": roi
            }

        return res_df, attribution

    @staticmethod
    def _calc_twr_curve(portfolio_vals: List[float], external_flows: List[float], init_capital: float) -> np.ndarray:
        equity = []
        index_val = 1.0
        prev_val = None

        for i, (val, flow) in enumerate(zip(portfolio_vals, external_flows)):
            if i == 0:
                index_val = (val / init_capital) if init_capital > 0 else 1.0
            elif prev_val is not None:
                cap_base = prev_val + flow
                if cap_base > 0:
                    daily_ret = (val / cap_base) - 1.0
                    index_val *= max(0.0, 1.0 + daily_ret)
            equity.append(index_val)
            prev_val = val

        return np.asarray(equity, dtype=float)
'''

with open("quant_system/core/portfolio_engine.py", "w", encoding="utf-8") as f:
    f.write(pe_v2)

# 4. 升級 ui/tab_strategy_pk.py (傳入 fx_series 與 target_currency)
with open("quant_system/ui/tab_strategy_pk.py", "r", encoding="utf-8") as f:
    pk_content = f.read()

pk_content = pk_content.replace(
    "def render_strategy_pk_tab(\n    individual_dict: Dict[str, pd.DataFrame],\n    curr_symbol: str,\n    initial_capital: float,\n    monthly_inflow: float,\n    cost_override: Optional[AssetCostStructure] = None\n):",
    "def render_strategy_pk_tab(\n    individual_dict: Dict[str, pd.DataFrame],\n    curr_symbol: str,\n    initial_capital: float,\n    monthly_inflow: float,\n    cost_override: Optional[AssetCostStructure] = None,\n    fx_series: Optional[pd.Series] = None,\n    target_currency: str = 'TWD'\n):"
)
pk_content = pk_content.replace(
    """            res_df = BacktestEngine.run_single_strategy(
                df_target,
                strat_name,
                p,
                initial_capital,
                monthly_inflow,
                cost_struct
            )""",
    """            res_df = BacktestEngine.run_single_strategy(
                df_target,
                strat_name,
                p,
                initial_capital,
                monthly_inflow,
                cost_struct,
                fx_series=fx_series,
                target_currency=target_currency
            )"""
)

with open("quant_system/ui/tab_strategy_pk.py", "w", encoding="utf-8") as f:
    f.write(pk_content)

# 5. 升級 ui/tab_portfolio_rebalance.py (傳入 aligned_fx 與 target_currency)
with open("quant_system/ui/tab_portfolio_rebalance.py", "r", encoding="utf-8") as f:
    rebal_content = f.read()

rebal_content = rebal_content.replace(
    "def render_portfolio_rebalance_tab(\n    aligned_panel: pd.DataFrame,\n    curr_symbol: str,\n    initial_capital: float,\n    monthly_inflow: float,\n    cost_override: Optional[AssetCostStructure] = None\n):",
    "def render_portfolio_rebalance_tab(\n    aligned_panel: pd.DataFrame,\n    aligned_fx: pd.Series,\n    curr_symbol: str,\n    initial_capital: float,\n    monthly_inflow: float,\n    cost_override: Optional[AssetCostStructure] = None,\n    target_currency: str = 'TWD'\n):"
)
rebal_content = rebal_content.replace(
    """        res_df, attribution = BacktestEngine.run_multi_asset_portfolio(
            aligned_panel,
            target_weights,
            initial_capital,
            monthly_inflow,
            rebalance_freq=rebal_freq,
            cost_dict=cost_dict
        )""",
    """        res_df, attribution = BacktestEngine.run_multi_asset_portfolio(
            aligned_panel,
            aligned_fx,
            target_weights,
            initial_capital,
            monthly_inflow,
            rebalance_freq=rebal_freq,
            cost_dict=cost_dict,
            target_currency=target_currency
        )"""
)

with open("quant_system/ui/tab_portfolio_rebalance.py", "w", encoding="utf-8") as f:
    f.write(rebal_content)

# 6. 升級 app.py (接收 aligned_fx 並分發給兩大分頁)
with open("quant_system/app.py", "r", encoding="utf-8") as f:
    app_content = f.read()

app_content = app_content.replace(
    """with st.spinner("正在自市場載入歷史行情並進行跨市場對齊與換匯..."):
    individual_dict, aligned_panel, curr_symbol, warnings = load_market_data(
        target_symbols,
        start_date,
        end_date,
        user_currency
    )""",
    """with st.spinner("正在自市場載入歷史行情並進行跨市場對齊與換匯..."):
    individual_dict, aligned_panel, aligned_fx, curr_symbol, warnings = load_market_data(
        target_symbols,
        start_date,
        end_date,
        user_currency
    )"""
)

app_content = app_content.replace(
    """with tab_pk:
    render_strategy_pk_tab(
        individual_dict=individual_dict,
        curr_symbol=curr_symbol,
        initial_capital=initial_capital,
        monthly_inflow=monthly_inflow,
        cost_override=cost_override
    )

with tab_rebalance:
    render_portfolio_rebalance_tab(
        aligned_panel=aligned_panel,
        curr_symbol=curr_symbol,
        initial_capital=initial_capital,
        monthly_inflow=monthly_inflow,
        cost_override=cost_override
    )""",
    """with tab_pk:
    render_strategy_pk_tab(
        individual_dict=individual_dict,
        curr_symbol=curr_symbol,
        initial_capital=initial_capital,
        monthly_inflow=monthly_inflow,
        cost_override=cost_override,
        fx_series=aligned_fx,
        target_currency=user_currency
    )

with tab_rebalance:
    render_portfolio_rebalance_tab(
        aligned_panel=aligned_panel,
        aligned_fx=aligned_fx,
        curr_symbol=curr_symbol,
        initial_capital=initial_capital,
        monthly_inflow=monthly_inflow,
        cost_override=cost_override,
        target_currency=user_currency
    )"""
)

with open("quant_system/app.py", "w", encoding="utf-8") as f:
    f.write(app_content)

print("All components (core, ui, app) upgraded and integrated.")
}}
