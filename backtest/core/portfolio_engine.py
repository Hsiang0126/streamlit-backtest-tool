"""
核心回測引擎 (Portfolio & Strategy Engine)
支援 365 天暖機期平滑截斷、休市順延撮合、雙價分離與多資產利潤歸因。
"""
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from core.cost_model import CostModel, AssetCostStructure
from strategies.registry import StrategyRegistry


class BacktestEngine:
    """雙模式回測核心計算引擎。"""

    # -------------------------------------------------------------
    # 模式 A：單標的策略執行引擎 (支援 365 天暖機期傳入)
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
        eval_start_date: Optional[str] = None
    ) -> pd.DataFrame:
        strat_cls = StrategyRegistry.get(strat_name)
        if not strat_cls:
            raise ValueError(f"未註冊的策略：{strat_name}")

        data = df.copy().sort_index()
        if len(data) < 2:
            raise ValueError("回測價格數據筆數過少。")

        # 1. 策略訊號生成 (在包含暖機期的完整序列上運算，避免指標前期 NaN)
        month_period = data.index.to_period("M")
        month_changed = pd.Series(month_period != pd.Series(month_period).shift(1), index=data.index)

        if strat_cls.is_timing:
            raw_sig = strat_cls.generate_signals(data, strat_params)
            pos_series = raw_sig.shift(1).fillna(0).astype(int)
            full_res = cls._execute_timing_loop(
                data, pos_series, initial_capital, monthly_inflow, cost_struct, month_changed, eval_start_date
            )
        elif strat_cls.strategy_id == "dca_buy_and_hold":
            full_res = cls._execute_dca_loop(
                data, strat_params, initial_capital, monthly_inflow, cost_struct, month_changed, eval_start_date
            )
        elif strat_cls.strategy_id == "dip_buyer":
            full_res = cls._execute_dip_loop(
                data, strat_params, initial_capital, monthly_inflow, cost_struct, month_changed, eval_start_date
            )
        else:
            raise ValueError(f"不支援的策略 ID：{strat_cls.strategy_id}")

        # 2. 嚴格截斷：帳本統計與 TWR 僅從正式起點 eval_start_date 起算
        if eval_start_date:
            cutoff = pd.Timestamp(eval_start_date)
            return full_res[full_res.index >= cutoff].copy()
        return full_res

    @staticmethod
    def _execute_timing_loop(
        data: pd.DataFrame,
        pos_series: pd.Series,
        initial_capital: float,
        monthly_inflow: float,
        cost: AssetCostStructure,
        month_changed: pd.Series,
        eval_start_date: Optional[str] = None
    ) -> pd.DataFrame:
        eval_start_ts = pd.Timestamp(eval_start_date) if eval_start_date else data.index[0]
        cash = float(initial_capital)
        shares = 0.0
        trade_count = 0
        portfolio_vals, cash_hist, stock_hist, external_flows = [], [], [], []
        active_started = False

        for i, dt in enumerate(data.index):
            if not active_started:
                if dt >= eval_start_ts:
                    active_started = True
                else:
                    # 暖機期：資產靜置，不交易
                    portfolio_vals.append(cash)
                    cash_hist.append(cash)
                    stock_hist.append(0.0)
                    external_flows.append(0.0)
                    continue

            div = float(data["dividend"].iloc[i]) if "dividend" in data.columns else 0.0
            if div > 0 and shares > 0:
                cash += shares * div

            split = float(data["split"].iloc[i]) if "split" in data.columns else 0.0
            if split > 0 and shares > 0:
                shares = shares * split

            flow = 0.0
            if bool(month_changed.iloc[i]) and dt > eval_start_ts:
                flow = float(monthly_inflow)
                cash += flow

            pos = int(pos_series.iloc[i])
            prev_pos = int(pos_series.iloc[i - 1]) if i > 0 else 0
            open_p = float(data["open"].iloc[i])
            close_p = float(data["close"].iloc[i])

            if pos == 1 and (prev_pos == 0 or cash > 10.0):
                if cash > 10.0:
                    fill_p = CostModel.calculate_buy_fill(open_p, cost.fee_rate, cost.slippage_rate)
                    buy_shares = cash / fill_p
                    shares += buy_shares
                    cash = 0.0
                    trade_count += 1
            elif pos == 0 and prev_pos == 1 and shares > 0:
                net_p = CostModel.calculate_sell_net(open_p, cost.fee_rate, cost.tax_rate, cost.slippage_rate)
                cash += shares * net_p
                shares = 0.0
                trade_count += 1

            stock_val = shares * close_p
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
        eval_start_date: Optional[str] = None
    ) -> pd.DataFrame:
        eval_start_ts = pd.Timestamp(eval_start_date) if eval_start_date else data.index[0]
        freq = params.get("freq", "每月 (Monthly)")

        if freq == "每週 (Weekly)":
            iso = data.index.isocalendar()
            week_key = iso["year"].astype(str) + "-" + iso["week"].astype(str)
            buy_mask = pd.Series(week_key != pd.Series(week_key).shift(1), index=data.index)
        elif freq == "每 10 交易日":
            buy_mask = pd.Series(np.arange(len(data)) % 10 == 0, index=data.index)
        else:
            buy_mask = month_changed

        cash = float(initial_capital)
        shares = 0.0
        trade_count = 0
        portfolio_vals, cash_hist, stock_hist, external_flows = [], [], [], []
        active_started = False

        for i, dt in enumerate(data.index):
            if not active_started:
                if dt >= eval_start_ts:
                    active_started = True
                else:
                    portfolio_vals.append(cash)
                    cash_hist.append(cash)
                    stock_hist.append(0.0)
                    external_flows.append(0.0)
                    continue

            div = float(data["dividend"].iloc[i]) if "dividend" in data.columns else 0.0
            if div > 0 and shares > 0:
                cash += shares * div

            split = float(data["split"].iloc[i]) if "split" in data.columns else 0.0
            if split > 0 and shares > 0:
                shares = shares * split

            flow = 0.0
            is_first_active_day = (dt == eval_start_ts or len(portfolio_vals) == 0 or (len(portfolio_vals) > 0 and portfolio_vals[-1] == initial_capital and trade_count == 0))
            if bool(month_changed.iloc[i]) and dt > eval_start_ts:
                flow = float(monthly_inflow)
                cash += flow

            open_p = float(data["open"].iloc[i])
            close_p = float(data["close"].iloc[i])

            if trade_count == 0 and cash > 0:
                fill_p = CostModel.calculate_buy_fill(open_p, cost.fee_rate, cost.slippage_rate)
                shares += cash / fill_p
                cash = 0.0
                trade_count += 1
            elif bool(buy_mask.iloc[i]) and cash > 10.0:
                fill_p = CostModel.calculate_buy_fill(open_p, cost.fee_rate, cost.slippage_rate)
                shares += cash / fill_p
                cash = 0.0
                trade_count += 1

            stock_val = shares * close_p
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
        eval_start_date: Optional[str] = None
    ) -> pd.DataFrame:
        eval_start_ts = pd.Timestamp(eval_start_date) if eval_start_date else data.index[0]
        sma_p = int(params.get("sma_period", 60))
        buy_amt = float(params.get("buy_amount", 50000.0))
        min_int = max(1, int(params.get("min_interval", 3)))

        # 優先使用 Adj Close 計算參考均線
        ref_price = data["adj_close"] if "adj_close" in data.columns else data["close"]
        sma = ref_price.rolling(sma_p).mean()

        cash = float(initial_capital)
        shares = 0.0
        trade_count = 0
        last_buy_idx = -min_int
        min_price_in_dip = float("inf")
        pending_buy = False

        portfolio_vals, cash_hist, stock_hist, external_flows = [], [], [], []
        active_started = False

        for i, dt in enumerate(data.index):
            if not active_started:
                if dt >= eval_start_ts:
                    active_started = True
                else:
                    portfolio_vals.append(cash)
                    cash_hist.append(cash)
                    stock_hist.append(0.0)
                    external_flows.append(0.0)
                    continue

            div = float(data["dividend"].iloc[i]) if "dividend" in data.columns else 0.0
            if div > 0 and shares > 0:
                cash += shares * div

            flow = 0.0
            if bool(month_changed.iloc[i]) and dt > eval_start_ts:
                flow = float(monthly_inflow)
                cash += flow

            open_p = float(data["open"].iloc[i])
            close_p = float(data["close"].iloc[i])

            if pending_buy and (i - last_buy_idx >= min_int) and cash > 10.0:
                actual_buy = min(cash, buy_amt)
                fill_p = CostModel.calculate_buy_fill(open_p, cost.fee_rate, cost.slippage_rate)
                shares += actual_buy / fill_p
                cash -= actual_buy
                last_buy_idx = i
                trade_count += 1
            pending_buy = False

            sma_val = sma.iloc[i]
            if pd.notna(sma_val) and close_p < float(sma_val):
                if close_p < min_price_in_dip:
                    min_price_in_dip = close_p
                    pending_buy = True
            else:
                min_price_in_dip = float("inf")

            stock_val = shares * close_p
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
    # 模式 B：多資產配置與動態再平衡核心引擎
    # -------------------------------------------------------------
    @classmethod
    def run_multi_asset_portfolio(
        cls,
        aligned_panel: pd.DataFrame,
        target_weights: Dict[str, float],
        initial_capital: float,
        monthly_inflow: float,
        rebalance_freq: str = "每季再平衡 (Quarterly)",
        cost_dict: Optional[Dict[str, AssetCostStructure]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        tickers = list(target_weights.keys())
        tot_w = sum(target_weights.values())
        norm_weights = {t: target_weights[t] / tot_w for t in tickers}

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

        asset_cum_bought = {t: 0.0 for t in tickers}
        asset_cum_sold = {t: 0.0 for t in tickers}
        asset_cum_div = {t: 0.0 for t in tickers}

        history_rows = []

        for i, dt in enumerate(data_idx):
            for t in tickers:
                div = float(aligned_panel[(t, "dividend")].iloc[i]) if (t, "dividend") in aligned_panel.columns else 0.0
                if div > 0 and shares[t] > 0:
                    div_amt = shares[t] * div
                    cash += div_amt
                    asset_cum_div[t] += div_amt

            flow = 0.0
            if i > 0 and bool(month_changed.iloc[i]):
                flow = float(monthly_inflow)
                cash += flow

            open_prices = {t: float(aligned_panel[(t, "open")].iloc[i]) for t in tickers}
            close_prices = {t: float(aligned_panel[(t, "close")].iloc[i]) for t in tickers}

            is_initial = (i == 0)
            is_rebalance_day = bool(rebalance_mask.iloc[i]) if i > 0 else False
            should_deploy_cash = (flow > 0 and not is_rebalance_day)

            if is_initial:
                for t in tickers:
                    alloc_amt = cash * norm_weights[t]
                    if alloc_amt > 10.0:
                        cost = cost_dict[t]
                        fill_p = CostModel.calculate_buy_fill(open_prices[t], cost.fee_rate, cost.slippage_rate)
                        bought_s = alloc_amt / fill_p
                        shares[t] += bought_s
                        asset_cum_bought[t] += alloc_amt
                        trade_count += 1
                cash = 0.0

            elif is_rebalance_day:
                pre_rebal_val = cash + sum(shares[t] * open_prices[t] for t in tickers)
                target_vals = {t: pre_rebal_val * norm_weights[t] for t in tickers}
                curr_vals = {t: shares[t] * open_prices[t] for t in tickers}

                # 賣出超配標的
                for t in tickers:
                    diff = curr_vals[t] - target_vals[t]
                    if diff > 50.0:
                        cost = cost_dict[t]
                        net_p = CostModel.calculate_sell_net(open_prices[t], cost.fee_rate, cost.tax_rate, cost.slippage_rate)
                        sell_shares = diff / open_prices[t]
                        actual_sell_s = min(shares[t], sell_shares)
                        shares[t] -= actual_sell_s
                        cash_recv = actual_sell_s * net_p
                        cash += cash_recv
                        asset_cum_sold[t] += cash_recv
                        trade_count += 1

                # 買入低配標的
                for t in tickers:
                    diff = target_vals[t] - curr_vals[t]
                    if diff > 50.0 and cash > 10.0:
                        cost = cost_dict[t]
                        actual_buy_amt = min(cash, diff)
                        fill_p = CostModel.calculate_buy_fill(open_prices[t], cost.fee_rate, cost.slippage_rate)
                        bought_s = actual_buy_amt / fill_p
                        shares[t] += bought_s
                        cash -= actual_buy_amt
                        asset_cum_bought[t] += actual_buy_amt
                        trade_count += 1

            elif should_deploy_cash and cash > 10.0:
                deploy_pool = cash
                for t in tickers:
                    alloc_amt = deploy_pool * norm_weights[t]
                    if alloc_amt > 10.0:
                        cost = cost_dict[t]
                        fill_p = CostModel.calculate_buy_fill(open_prices[t], cost.fee_rate, cost.slippage_rate)
                        bought_s = alloc_amt / fill_p
                        shares[t] += bought_s
                        cash -= alloc_amt
                        asset_cum_bought[t] += alloc_amt
                        trade_count += 1

            stock_vals = {t: shares[t] * close_prices[t] for t in tickers}
            total_stock_val = sum(stock_vals.values())
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
                row_data[f"val_{t}"] = stock_vals[t]
                row_data[f"shares_{t}"] = shares[t]

            history_rows.append(row_data)

        res_df = pd.DataFrame(history_rows).set_index("date")
        res_df["total_invested"] = float(initial_capital) + res_df["external_flow"].cumsum()
        res_df["equity_net"] = BacktestEngine._calc_twr_curve(
            res_df["portfolio_value"].tolist(),
            res_df["external_flow"].tolist(),
            initial_capital
        )

        # 各資產利潤貢獻歸因
        attribution: Dict[str, Dict[str, float]] = {}
        for t in tickers:
            final_holding_val = shares[t] * close_prices[t]
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
