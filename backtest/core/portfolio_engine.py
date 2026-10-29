"""
核心回測引擎 (Portfolio & Strategy Engine - 支援台股英文字尾代碼與現金活存儲備版)
包含：
1. 單標的策略執行引擎 (Timing / DCA / Dip DCA)
2. 多資產動態再平衡引擎 (支援現金活存計息儲備、門檻再平衡 Tolerance Band、休市順延撮合)
3. 幣別與換匯智慧判定 (Native Currency 隔離)
"""
from typing import Dict, List, Any, Optional, Tuple
import re
import pandas as pd
import numpy as np
from core.cost_model import CostModel, AssetCostStructure
from strategies.registry import StrategyRegistry


class BacktestEngine:
    """雙模式機構級精準回測引擎。"""

    # -------------------------------------------------------------
    # 模式 A：單標的策略執行引擎 (真實價格撮合)
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

        # 防呆提取報價幣別
        if "quote_currency" in full_data.columns:
            quote_curr = str(full_data["quote_currency"].iloc[0]).strip().upper()
        else:
            quote_curr = "TWD" if "TWD" in target_currency.upper() else "USD"

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

        s_months = pd.Series(data.index.to_period("M"), index=data.index)
        month_changed = s_months != s_months.shift(1)

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
        portfolio_vals, cash_hist, stock_hist, external_flows, trades_hist = [], [], [], [], []

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
            trades_hist.append(trade_count)

        res = data.copy()
        res["portfolio_value"] = portfolio_vals
        res["cash"] = cash_hist
        res["stock_value"] = stock_hist
        res["external_flow"] = external_flows
        res["total_invested"] = float(initial_capital) + pd.Series(external_flows, index=data.index).cumsum()
        res["trades"] = trades_hist
        res["to_target_fx"] = [float(aligned_fx.iloc[k]) if need_fx else 1.0 for k in range(len(data))]
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
            s_weeks = pd.Series(iso["year"].astype(str) + "-" + iso["week"].astype(str), index=data.index)
            buy_mask = s_weeks != s_weeks.shift(1)
        elif freq == "每 10 交易日":
            buy_mask = pd.Series(np.arange(len(data)) % 10 == 0, index=data.index)
        else:
            buy_mask = month_changed

        cash = float(initial_capital)
        shares = 0.0
        trade_count = 0
        portfolio_vals, cash_hist, stock_hist, external_flows, trades_hist = [], [], [], [], []

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

            if i == 0 and cash > 10.0:
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
            trades_hist.append(trade_count)

        res = data.copy()
        res["portfolio_value"] = portfolio_vals
        res["cash"] = cash_hist
        res["stock_value"] = stock_hist
        res["external_flow"] = external_flows
        res["total_invested"] = float(initial_capital) + pd.Series(external_flows, index=data.index).cumsum()
        res["trades"] = trades_hist
        res["to_target_fx"] = [float(aligned_fx.iloc[k]) if need_fx else 1.0 for k in range(len(data))]
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

        portfolio_vals, cash_hist, stock_hist, external_flows, trades_hist = [], [], [], [], []

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
                bought_s = actual_buy / fill_p_target
                shares += bought_s
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
            trades_hist.append(trade_count)

        res = data.copy()
        res["portfolio_value"] = portfolio_vals
        res["cash"] = cash_hist
        res["stock_value"] = stock_hist
        res["external_flow"] = external_flows
        res["total_invested"] = float(initial_capital) + pd.Series(external_flows, index=data.index).cumsum()
        res["trades"] = trades_hist
        res["to_target_fx"] = [float(aligned_fx.iloc[k]) if need_fx else 1.0 for k in range(len(data))]
        res["equity_net"] = BacktestEngine._calc_twr_curve(portfolio_vals, external_flows, initial_capital)
        return res

    # -------------------------------------------------------------
    # 模式 B：多資產配置與動態再平衡 (含現金活存儲備方案 B)
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
        target_currency: str = "TWD",
        tolerance_band: float = 0.05,
        cash_reserve_weight: float = 0.0,
        cash_annual_rate: float = 0.018
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Any]]:
        tickers = list(target_weights.keys())
        tot_w = sum(target_weights.values()) + cash_reserve_weight
        if tot_w <= 0:
            tot_w = 1.0
        norm_weights = {t: target_weights[t] / tot_w for t in tickers}
        norm_cash_w = cash_reserve_weight / tot_w
        target_curr_code = "TWD" if "TWD" in target_currency.upper() else "USD"

        if cost_dict is None:
            cost_dict = {t: CostModel.get_default_costs(t) for t in tickers}

        # 迴圈外安全判定每檔標的的計價幣別 (支援台股英文字尾代碼如 00631L, 00981A)
        ticker_currencies = {}
        for t in tickers:
            curr = None
            if (t, "quote_currency") in aligned_panel.columns:
                curr = str(aligned_panel[(t, "quote_currency")].iloc[0]).strip().upper()
            if not curr or curr in {"NAN", "NONE", ""}:
                clean_t = str(t).strip().upper()
                if clean_t.endswith((".TW", ".TWO")) or bool(re.match(r"^\d{4,5}[A-Z0-9]?$", clean_t)):
                    curr = "TWD"
                else:
                    curr = "USD"
            ticker_currencies[t] = curr

        data_idx = aligned_panel.index
        s_months = pd.Series(data_idx.to_period("M"), index=data_idx)
        month_changed = s_months != s_months.shift(1)

        rebalance_mask = pd.Series(False, index=data_idx)
        if rebalance_freq == "每月再平衡 (Monthly)":
            rebalance_mask = month_changed.copy()
        elif rebalance_freq == "每季再平衡 (Quarterly)":
            s_quarters = pd.Series(data_idx.to_period("Q"), index=data_idx)
            rebalance_mask = s_quarters != s_quarters.shift(1)
        elif rebalance_freq == "每半年再平衡 (Semi-Annually)":
            half_arr = data_idx.year.astype(str) + "-" + np.where(data_idx.month <= 6, "H1", "H2")
            s_half = pd.Series(half_arr, index=data_idx)
            rebalance_mask = s_half != s_half.shift(1)
        elif rebalance_freq == "每年再平衡 (Annually)":
            s_years = pd.Series(data_idx.to_period("Y"), index=data_idx)
            rebalance_mask = s_years != s_years.shift(1)

        cash = float(initial_capital)
        shares = {t: 0.0 for t in tickers}
        trade_count = 0
        skipped_rebalance_count = 0
        triggered_rebalance_count = 0

        pending_rebalance_orders: Dict[str, float] = {}
        asset_cum_bought = {t: 0.0 for t in tickers}
        asset_cum_sold = {t: 0.0 for t in tickers}
        asset_cum_div = {t: 0.0 for t in tickers}
        cum_cash_interest = 0.0

        history_rows = []

        for i, dt in enumerate(data_idx):
            # 💵 每日現金活存複利計息
            daily_interest = 0.0
            if cash > 0 and cash_annual_rate > 0:
                daily_interest = cash * (cash_annual_rate / 365.0)
                cash += daily_interest
                cum_cash_interest += daily_interest

            fx = float(aligned_fx.iloc[i])
            to_target_fx = {
                t: (fx if ticker_currencies[t] == "USD" and target_curr_code == "TWD"
                    else (1.0 / fx if ticker_currencies[t] == "TWD" and target_curr_code == "USD" else 1.0))
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
                deploy_pool = cash
                for t in tickers:
                    alloc_amt = deploy_pool * norm_weights[t]
                    if alloc_amt > 10.0:
                        cost = cost_dict[t]
                        fill_p = CostModel.calculate_buy_fill(open_prices_native[t], cost.fee_rate, cost.slippage_rate) * to_target_fx[t]
                        bought_s = alloc_amt / fill_p
                        shares[t] += bought_s
                        cash -= alloc_amt
                        asset_cum_bought[t] += alloc_amt
                        trade_count += 1

            elif is_rebal:
                pre_rebal_val = cash + sum(shares[t] * open_prices_native[t] * to_target_fx[t] for t in tickers)
                curr_weights = {t: (shares[t] * open_prices_native[t] * to_target_fx[t]) / pre_rebal_val if pre_rebal_val > 0 else 0.0 for t in tickers}
                curr_cash_w = cash / pre_rebal_val if pre_rebal_val > 0 else 0.0

                deviations = [abs(curr_weights[t] - norm_weights[t]) for t in tickers]
                if norm_cash_w > 0:
                    deviations.append(abs(curr_cash_w - norm_cash_w))
                max_dev = max(deviations) if pre_rebal_val > 0 else 0.0

                if max_dev > tolerance_band:
                    for t in tickers:
                        target_val = pre_rebal_val * norm_weights[t]
                        curr_val = shares[t] * open_prices_native[t] * to_target_fx[t]
                        pending_rebalance_orders[t] = target_val - curr_val
                    triggered_rebalance_count += 1
                else:
                    skipped_rebalance_count += 1

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
                        rem_diff = diff - actual_buy_target
                        if rem_diff <= 50.0:
                            del pending_rebalance_orders[t]
                        else:
                            pending_rebalance_orders[t] = rem_diff

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

        # 💵 若有現金活存部位或產生利息，將現金活存加入利潤歸因
        if norm_cash_w > 0 or cum_cash_interest > 0:
            tot_inv_cum = float(res_df["total_invested"].iloc[-1])
            cash_base_invested = tot_inv_cum * norm_cash_w if norm_cash_w > 0 else cash
            attribution["💵 現金活存儲備"] = {
                "final_value": cash,
                "cum_bought": cash_base_invested,
                "cum_sold": 0.0,
                "cum_dividend": cum_cash_interest,
                "net_profit": cum_cash_interest,
                "roi": (cum_cash_interest / cash_base_invested) if cash_base_invested > 0 else 0.0
            }

        rebal_stats = {
            "觸發調倉次數": triggered_rebalance_count,
            "成功略過調倉次數": skipped_rebalance_count,
            "估計省下摩擦手續費": f"{round(skipped_rebalance_count * len(tickers) * 150.0, 0):,.0f}"
        }
        return res_df, attribution, rebal_stats

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
