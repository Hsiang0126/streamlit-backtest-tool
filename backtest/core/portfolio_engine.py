"""
核心回測引擎 (Portfolio & Strategy Engine)
包含：
1. 單標的策略執行引擎 (Timing / DCA / Dip DCA)，支援固定停損與移動停利風控、首日扣款、頻率真實拆分
2. 多資產動態再平衡引擎 (支援日曆天數複利計息、現金活存儲備隔離保護、門檻再平衡 Tolerance Band、休市順延補買)
3. 幣別與換匯智慧判定 (Native Currency 隔離與精確雙向匯率乘數)

重要還原價與拆股註解：
yfinance (auto_adjust=False) 回傳的 Open/High/Low/Close 與 Dividends 皆已進行拆股 (Split) 調整，
故回測迴圈嚴禁再乘以 split 因子，避免造成 NVDA/0050 等拆股標的股數被重複放大之嚴重誤差。
"""
from typing import Dict, Any, Optional, Tuple, List
import re
import pandas as pd
import numpy as np
from core.cost_model import CostModel, AssetCostStructure
from core.indicators import IndicatorEngine
from strategies.registry import StrategyRegistry


class BacktestEngine:
    """雙模式機構級精準回測引擎。"""

    MIN_ORDER_THRESHOLD = 10.0
    MIN_REBAL_ORDER_THRESHOLD = 50.0

    @staticmethod
    def _resolve_fx_multiplier(quote_curr: str, target_curr: str, fx_rate: float) -> float:
        q = quote_curr.strip().upper()
        t = target_curr.strip().upper()
        if q == t:
            return 1.0
        if q == "USD" and t == "TWD":
            return fx_rate if fx_rate > 0 else 32.0
        if q == "TWD" and t == "USD":
            return (1.0 / fx_rate) if fx_rate > 0 else (1.0 / 32.0)
        return 1.0

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
        target_currency: str = "TWD",
        invest_day_one: bool = False,
        stop_loss_pct: float = 0.0,
        trailing_stop_pct: float = 0.0,
    ) -> pd.DataFrame:
        strat_cls = StrategyRegistry.get(strat_name)
        if not strat_cls:
            raise ValueError(f"未註冊的策略：{strat_name}")

        full_data = df.copy().sort_index()

        if "quote_currency" in full_data.columns:
            quote_curr = str(full_data["quote_currency"].iloc[0]).strip().upper()
        else:
            quote_curr = "TWD" if "TWD" in target_currency.upper() else "USD"

        # 預先在包含暖機期的 full_data 計算訊號或指標，確保回測第 1 天即成熟
        full_sma_dip = None
        if strat_cls.is_timing:
            raw_sig = strat_cls.generate_signals(full_data, strat_params)
            full_pos = raw_sig.shift(1).fillna(0).astype(int)
        else:
            full_pos = pd.Series(1, index=full_data.index)
            if strat_cls.strategy_id == "dip_buyer":
                sma_p = int(strat_params.get("sma_period", 60))
                tf = strat_params.get("timeframe", "日線 (Daily)")
                full_sma_dip = IndicatorEngine.calculate_ma(full_data, sma_p, tf)

        if "user_start_idx" in full_data.columns:
            data = full_data[full_data["user_start_idx"] == True].copy()
            pos_series = full_pos.reindex(data.index).fillna(0).astype(int)
            sma_dip = full_sma_dip.reindex(data.index) if full_sma_dip is not None else None
        else:
            data = full_data.copy()
            pos_series = full_pos.copy()
            sma_dip = full_sma_dip

        if len(data) < 2:
            raise ValueError("有效回測區間內之交易日數不足。")

        target_curr_code = "TWD" if "TWD" in target_currency.upper() else "USD"
        need_fx = quote_curr != target_curr_code
        if fx_series is not None and not fx_series.empty:
            aligned_fx = fx_series.reindex(data.index).ffill().bfill().fillna(32.0)
        else:
            aligned_fx = pd.Series(1.0, index=data.index)

        fx_mult_arr = np.array(
            [
                cls._resolve_fx_multiplier(quote_curr, target_curr_code, float(v)) if need_fx else 1.0
                for v in aligned_fx.values
            ],
            dtype=float,
        )

        s_months = pd.Series(data.index.to_period("M"), index=data.index)
        month_changed = (s_months != s_months.shift(1)).fillna(False)

        if strat_cls.is_timing:
            return cls._execute_timing_loop(
                data=data,
                pos_series=pos_series,
                initial_capital=initial_capital,
                monthly_inflow=monthly_inflow,
                cost=cost_struct,
                month_changed=month_changed,
                fx_mult_arr=fx_mult_arr,
                invest_day_one=invest_day_one,
                stop_loss_pct=max(0.0, float(stop_loss_pct)),
                trailing_stop_pct=max(0.0, float(trailing_stop_pct)),
            )
        if strat_cls.strategy_id == "dca_buy_and_hold":
            return cls._execute_dca_loop(
                data=data,
                params=strat_params,
                initial_capital=initial_capital,
                monthly_inflow=monthly_inflow,
                cost=cost_struct,
                month_changed=month_changed,
                fx_mult_arr=fx_mult_arr,
                invest_day_one=invest_day_one,
            )
        if strat_cls.strategy_id == "dip_buyer":
            return cls._execute_dip_loop(
                data=data,
                params=strat_params,
                sma_series=sma_dip,
                initial_capital=initial_capital,
                monthly_inflow=monthly_inflow,
                cost=cost_struct,
                month_changed=month_changed,
                fx_mult_arr=fx_mult_arr,
                invest_day_one=invest_day_one,
            )
        raise ValueError(f"不支援的策略 ID：{strat_cls.strategy_id}")

    @classmethod
    def _execute_timing_loop(
        cls,
        data: pd.DataFrame,
        pos_series: pd.Series,
        initial_capital: float,
        monthly_inflow: float,
        cost: AssetCostStructure,
        month_changed: pd.Series,
        fx_mult_arr: np.ndarray,
        invest_day_one: bool = False,
        stop_loss_pct: float = 0.0,
        trailing_stop_pct: float = 0.0,
    ) -> pd.DataFrame:
        n_days = len(data)
        cash = float(initial_capital)
        shares = 0.0
        trade_count = 0
        fees_paid = 0.0

        # 風控狀態追蹤（原生價格基準，杜絕匯率干擾停損判定）
        in_position = False
        stopped_out = False
        pending_risk_exit = False
        entry_ref_adj = 0.0
        peak_adj_close = 0.0

        open_arr = data["open"].values
        close_arr = data["close"].values
        eval_arr = data["adj_close"].values if "adj_close" in data.columns else close_arr
        div_arr = data["dividend"].values if "dividend" in data.columns else np.zeros(n_days)
        pos_arr = pos_series.values
        mc_arr = month_changed.values

        portfolio_vals = np.zeros(n_days, dtype=float)
        cash_hist = np.zeros(n_days, dtype=float)
        stock_hist = np.zeros(n_days, dtype=float)
        external_flows = np.zeros(n_days, dtype=float)
        trades_hist = np.zeros(n_days, dtype=int)
        shares_hist = np.zeros(n_days, dtype=float)
        actual_pos_hist = np.zeros(n_days, dtype=int)
        fees_hist = np.zeros(n_days, dtype=float)

        for i in range(n_days):
            to_target_fx = float(fx_mult_arr[i])

            # 注意：yfinance 的價格與股息均已為拆股調整後數值，此處不再重複乘以 split
            div = float(div_arr[i])
            if div > 0 and shares > 0:
                cash += shares * div * to_target_fx

            flow = 0.0
            if i == 0 and initial_capital <= 0.0 and invest_day_one and monthly_inflow > 0:
                flow = float(monthly_inflow)
                cash += flow
            elif i > 0 and bool(mc_arr[i]) and monthly_inflow > 0:
                flow = float(monthly_inflow)
                cash += flow

            raw_pos = int(pos_arr[i])
            if raw_pos == 0:
                # 原始策略轉為空倉訊號後，解除當輪停損鎖定
                stopped_out = False

            effective_buy_allowed = (raw_pos == 1) and (not stopped_out) and (not pending_risk_exit)
            open_p_native = float(open_arr[i])
            close_p_native = float(close_arr[i])
            eval_p_adj = float(eval_arr[i])

            # 1. 優先執行昨日收盤觸發之風控出場或原始賣出訊號
            if in_position and (pending_risk_exit or raw_pos == 0) and shares > 0:
                gross_val = shares * open_p_native * to_target_fx
                net_p_native = CostModel.calculate_sell_net(
                    open_p_native, cost.fee_rate, cost.tax_rate, cost.slippage_rate
                )
                net_val = shares * net_p_native * to_target_fx
                fees_paid += max(0.0, gross_val - net_val)
                cash += net_val
                shares = 0.0
                in_position = False
                if pending_risk_exit:
                    stopped_out = True
                pending_risk_exit = False
                trade_count += 1

            # 2. 若允許持倉且有足夠現金，於今日開盤買入或加碼投入
            elif effective_buy_allowed and cash > cls.MIN_ORDER_THRESHOLD:
                fill_p_native = CostModel.calculate_buy_fill(
                    open_p_native, cost.fee_rate, cost.slippage_rate
                )
                fill_p_target = fill_p_native * to_target_fx
                bought_shares = cash / fill_p_target
                gross_stock_val = bought_shares * open_p_native * to_target_fx
                fees_paid += max(0.0, cash - gross_stock_val)
                shares += bought_shares
                cash = 0.0
                if not in_position:
                    in_position = True
                    # 記錄當日基準調整價作為風控起點
                    adj_ratio = (eval_p_adj / close_p_native) if close_p_native > 0 else 1.0
                    entry_ref_adj = open_p_native * adj_ratio * (1.0 + cost.slippage_rate) * (1.0 + cost.fee_rate)
                    peak_adj_close = max(entry_ref_adj, eval_p_adj)
                trade_count += 1

            # 3. T 日收盤評估停損 / 移動停利，若有觸發則預約於 T+1 日開盤賣出（零前瞻偏差）
            if in_position and shares > 0:
                peak_adj_close = max(peak_adj_close, eval_p_adj)
                if stop_loss_pct > 0 and entry_ref_adj > 0:
                    if eval_p_adj <= entry_ref_adj * (1.0 - stop_loss_pct):
                        pending_risk_exit = True
                if trailing_stop_pct > 0 and peak_adj_close > 0:
                    if eval_p_adj <= peak_adj_close * (1.0 - trailing_stop_pct):
                        pending_risk_exit = True

            stock_val = shares * close_p_native * to_target_fx
            total_val = cash + stock_val

            portfolio_vals[i] = total_val
            cash_hist[i] = cash
            stock_hist[i] = stock_val
            external_flows[i] = flow
            trades_hist[i] = trade_count
            shares_hist[i] = shares
            actual_pos_hist[i] = 1 if in_position else 0
            fees_hist[i] = fees_paid

        res = data.copy()
        res["portfolio_value"] = portfolio_vals
        res["cash"] = cash_hist
        res["stock_value"] = stock_hist
        res["shares"] = shares_hist
        res["position"] = actual_pos_hist
        res["fees_paid"] = fees_hist
        res["external_flow"] = external_flows
        res["total_invested"] = float(initial_capital) + np.cumsum(external_flows)
        res["trades"] = trades_hist
        res["to_target_fx"] = fx_mult_arr
        res["equity_net"] = cls._calc_twr_curve(portfolio_vals.tolist(), external_flows.tolist(), initial_capital)
        return res

    @classmethod
    def _execute_dca_loop(
        cls,
        data: pd.DataFrame,
        params: Dict[str, Any],
        initial_capital: float,
        monthly_inflow: float,
        cost: AssetCostStructure,
        month_changed: pd.Series,
        fx_mult_arr: np.ndarray,
        invest_day_one: bool = False,
    ) -> pd.DataFrame:
        freq = params.get("freq", "每月 (Monthly)")
        reinvest_idle = bool(params.get("reinvest_idle", True))
        n_days = len(data)

        if freq == "每週 (Weekly)":
            iso = data.index.isocalendar()
            s_weeks = pd.Series(
                iso["year"].astype(str).str.zfill(4) + "-" + iso["week"].astype(str).str.zfill(2),
                index=data.index,
            )
            buy_mask = (s_weeks != s_weeks.shift(1)).fillna(False)
        elif freq == "每 10 交易日":
            buy_mask = pd.Series(np.arange(n_days) % 10 == 0, index=data.index)
        else:
            buy_mask = month_changed

        # 將每月固定新注資按該月份扣款次數等比拆分為每次注資，維持每月總自付本金一致
        periods = data.index.to_period("M")
        first_period = periods[0]
        mc_arr = month_changed.values
        bm_arr = buy_mask.values

        # 計算每個月份有幾次扣款日
        counts_per_month: Dict[Any, int] = {}
        for i in range(n_days):
            p = periods[i]
            if i == 0:
                if initial_capital <= 0.0 and invest_day_one:
                    counts_per_month[p] = counts_per_month.get(p, 0) + 1
                continue
            if p == first_period and not invest_day_one:
                # 若第 1 個月不扣月定投（等次月扣款），則此月不計定期注資
                continue
            if bool(bm_arr[i]):
                counts_per_month[p] = counts_per_month.get(p, 0) + 1

        cash = float(initial_capital)
        shares = 0.0
        trade_count = 0
        fees_paid = 0.0

        open_arr = data["open"].values
        close_arr = data["close"].values
        div_arr = data["dividend"].values if "dividend" in data.columns else np.zeros(n_days)

        portfolio_vals = np.zeros(n_days, dtype=float)
        cash_hist = np.zeros(n_days, dtype=float)
        stock_hist = np.zeros(n_days, dtype=float)
        external_flows = np.zeros(n_days, dtype=float)
        trades_hist = np.zeros(n_days, dtype=int)
        shares_hist = np.zeros(n_days, dtype=float)
        actual_pos_hist = np.zeros(n_days, dtype=int)
        fees_hist = np.zeros(n_days, dtype=float)

        for i in range(n_days):
            to_target_fx = float(fx_mult_arr[i])
            p = periods[i]

            div = float(div_arr[i])
            if div > 0 and shares > 0:
                cash += shares * div * to_target_fx

            flow = 0.0
            if freq == "每月 (Monthly)":
                if i == 0 and initial_capital <= 0.0 and invest_day_one and monthly_inflow > 0:
                    flow = float(monthly_inflow)
                elif i > 0 and bool(mc_arr[i]) and monthly_inflow > 0:
                    flow = float(monthly_inflow)
            else:
                # 週定投 / 每10交易日：把該月 monthly_inflow 平分到該月各扣款日
                n_slots = counts_per_month.get(p, 0)
                if n_slots > 0 and monthly_inflow > 0:
                    if i == 0 and initial_capital <= 0.0 and invest_day_one:
                        flow = float(monthly_inflow) / n_slots
                    elif i > 0 and (p != first_period or invest_day_one) and bool(bm_arr[i]):
                        flow = float(monthly_inflow) / n_slots

            if flow > 0:
                cash += flow

            open_p_native = float(open_arr[i])
            close_p_native = float(close_arr[i])

            is_buy_day = (i == 0 and cash > cls.MIN_ORDER_THRESHOLD) or (
                i > 0 and bool(bm_arr[i]) and cash > cls.MIN_ORDER_THRESHOLD
            )
            if is_buy_day:
                if i == 0 and initial_capital > 0:
                    # 第 1 天若有期初本金一律全額建倉
                    buy_amt = cash
                elif reinvest_idle:
                    buy_amt = cash
                else:
                    # 不將股利或先前未投入現金併入，僅用當期新注入之本金買入
                    buy_amt = min(cash, flow) if flow > 0 else 0.0

                if buy_amt > cls.MIN_ORDER_THRESHOLD:
                    fill_p_native = CostModel.calculate_buy_fill(
                        open_p_native, cost.fee_rate, cost.slippage_rate
                    )
                    fill_p_target = fill_p_native * to_target_fx
                    bought_shares = buy_amt / fill_p_target
                    gross_stock_val = bought_shares * open_p_native * to_target_fx
                    fees_paid += max(0.0, buy_amt - gross_stock_val)
                    shares += bought_shares
                    cash -= buy_amt
                    trade_count += 1

            stock_val = shares * close_p_native * to_target_fx
            total_val = cash + stock_val

            portfolio_vals[i] = total_val
            cash_hist[i] = cash
            stock_hist[i] = stock_val
            external_flows[i] = flow
            trades_hist[i] = trade_count
            shares_hist[i] = shares
            actual_pos_hist[i] = 1 if shares > 0 else 0
            fees_hist[i] = fees_paid

        res = data.copy()
        res["portfolio_value"] = portfolio_vals
        res["cash"] = cash_hist
        res["stock_value"] = stock_hist
        res["shares"] = shares_hist
        res["position"] = actual_pos_hist
        res["fees_paid"] = fees_hist
        res["external_flow"] = external_flows
        res["total_invested"] = float(initial_capital) + np.cumsum(external_flows)
        res["trades"] = trades_hist
        res["to_target_fx"] = fx_mult_arr
        res["equity_net"] = cls._calc_twr_curve(portfolio_vals.tolist(), external_flows.tolist(), initial_capital)
        return res

    @classmethod
    def _execute_dip_loop(
        cls,
        data: pd.DataFrame,
        params: Dict[str, Any],
        sma_series: Optional[pd.Series],
        initial_capital: float,
        monthly_inflow: float,
        cost: AssetCostStructure,
        month_changed: pd.Series,
        fx_mult_arr: np.ndarray,
        invest_day_one: bool = False,
    ) -> pd.DataFrame:
        n_days = len(data)
        buy_amt = float(params.get("buy_amount", 50000.0))
        min_int = max(1, int(params.get("min_interval", 3)))
        if sma_series is None:
            sma_p = int(params.get("sma_period", 60))
            tf = params.get("timeframe", "日線 (Daily)")
            sma_series = IndicatorEngine.calculate_ma(data, sma_p, tf)

        sma_arr = sma_series.values
        open_arr = data["open"].values
        close_arr = data["close"].values
        eval_arr = data["adj_close"].values if "adj_close" in data.columns else close_arr
        div_arr = data["dividend"].values if "dividend" in data.columns else np.zeros(n_days)
        mc_arr = month_changed.values

        cash = float(initial_capital)
        shares = 0.0
        trade_count = 0
        fees_paid = 0.0
        last_buy_idx = -min_int
        min_price_in_dip = float("inf")
        pending_buy = False

        portfolio_vals = np.zeros(n_days, dtype=float)
        cash_hist = np.zeros(n_days, dtype=float)
        stock_hist = np.zeros(n_days, dtype=float)
        external_flows = np.zeros(n_days, dtype=float)
        trades_hist = np.zeros(n_days, dtype=int)
        shares_hist = np.zeros(n_days, dtype=float)
        actual_pos_hist = np.zeros(n_days, dtype=int)
        fees_hist = np.zeros(n_days, dtype=float)

        for i in range(n_days):
            to_target_fx = float(fx_mult_arr[i])

            div = float(div_arr[i])
            if div > 0 and shares > 0:
                cash += shares * div * to_target_fx

            flow = 0.0
            if i == 0 and initial_capital <= 0.0 and invest_day_one and monthly_inflow > 0:
                flow = float(monthly_inflow)
                cash += flow
            elif i > 0 and bool(mc_arr[i]) and monthly_inflow > 0:
                flow = float(monthly_inflow)
                cash += flow

            open_p_native = float(open_arr[i])
            close_p_native = float(close_arr[i])

            if pending_buy and (i - last_buy_idx >= min_int) and cash > cls.MIN_ORDER_THRESHOLD:
                actual_buy = min(cash, buy_amt)
                fill_p_native = CostModel.calculate_buy_fill(
                    open_p_native, cost.fee_rate, cost.slippage_rate
                )
                fill_p_target = fill_p_native * to_target_fx
                bought_s = actual_buy / fill_p_target
                gross_stock_val = bought_s * open_p_native * to_target_fx
                fees_paid += max(0.0, actual_buy - gross_stock_val)
                shares += bought_s
                cash -= actual_buy
                last_buy_idx = i
                trade_count += 1
            pending_buy = False

            sma_val = sma_arr[i]
            current_eval_p = float(eval_arr[i])
            if pd.notna(sma_val) and current_eval_p < float(sma_val):
                if current_eval_p < min_price_in_dip:
                    min_price_in_dip = current_eval_p
                    pending_buy = True
            else:
                min_price_in_dip = float("inf")

            stock_val = shares * close_p_native * to_target_fx
            total_val = cash + stock_val

            portfolio_vals[i] = total_val
            cash_hist[i] = cash
            stock_hist[i] = stock_val
            external_flows[i] = flow
            trades_hist[i] = trade_count
            shares_hist[i] = shares
            actual_pos_hist[i] = 1 if shares > 0 else 0
            fees_hist[i] = fees_paid

        res = data.copy()
        res["portfolio_value"] = portfolio_vals
        res["cash"] = cash_hist
        res["stock_value"] = stock_hist
        res["shares"] = shares_hist
        res["position"] = actual_pos_hist
        res["fees_paid"] = fees_hist
        res["external_flow"] = external_flows
        res["total_invested"] = float(initial_capital) + np.cumsum(external_flows)
        res["trades"] = trades_hist
        res["to_target_fx"] = fx_mult_arr
        res["equity_net"] = cls._calc_twr_curve(portfolio_vals.tolist(), external_flows.tolist(), initial_capital)
        return res

    # -------------------------------------------------------------
    # 模式 B：多資產配置與動態再平衡 (含現金活存儲備與休市順延)
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
        cash_annual_rate: float = 0.018,
        invest_day_one: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Any]]:
        tickers = list(target_weights.keys())
        risky_sum = sum(max(0.0, float(target_weights[t])) for t in tickers)
        tot_w = risky_sum + max(0.0, float(cash_reserve_weight))
        if tot_w <= 0:
            tot_w = 1.0
        norm_weights = {t: max(0.0, float(target_weights[t])) / tot_w for t in tickers}
        norm_cash_w = max(0.0, float(cash_reserve_weight)) / tot_w
        norm_risky_sum = sum(norm_weights.values())
        risky_rel_weights = {
            t: (norm_weights[t] / norm_risky_sum) if norm_risky_sum > 0 else 0.0
            for t in tickers
        }
        target_curr_code = "TWD" if "TWD" in target_currency.upper() else "USD"

        if cost_dict is None:
            cost_dict = {t: CostModel.get_default_costs(t) for t in tickers}

        ticker_currencies: Dict[str, str] = {}
        for t in tickers:
            curr = None
            if (t, "quote_currency") in aligned_panel.columns:
                curr = str(aligned_panel[(t, "quote_currency")].iloc[0]).strip().upper()
            if not curr or curr in {"NAN", "NONE", ""}:
                clean_t = str(t).strip().upper()
                if clean_t.endswith((".TW", ".TWO")) or bool(re.match(r"^\d{4,6}[A-Z]?$", clean_t)):
                    curr = "TWD"
                else:
                    curr = "USD"
            ticker_currencies[t] = curr

        data_idx = aligned_panel.index
        fx_series = (
            aligned_fx.reindex(data_idx).ffill().bfill().fillna(32.0)
            if aligned_fx is not None and not aligned_fx.empty
            else pd.Series(32.0, index=data_idx)
        )

        s_months = pd.Series(data_idx.to_period("M"), index=data_idx)
        month_changed = (s_months != s_months.shift(1)).fillna(False)

        rebalance_mask = pd.Series(False, index=data_idx)
        if rebalance_freq == "每月再平衡 (Monthly)":
            rebalance_mask = month_changed.copy()
        elif rebalance_freq == "每季再平衡 (Quarterly)":
            s_quarters = pd.Series(data_idx.to_period("Q"), index=data_idx)
            rebalance_mask = (s_quarters != s_quarters.shift(1)).fillna(False)
        elif rebalance_freq == "每半年再平衡 (Semi-Annually)":
            half_arr = data_idx.year.astype(str) + "-" + np.where(data_idx.month <= 6, "H1", "H2")
            s_half = pd.Series(half_arr, index=data_idx)
            rebalance_mask = (s_half != s_half.shift(1)).fillna(False)
        elif rebalance_freq == "每年再平衡 (Annually)":
            s_years = pd.Series(data_idx.to_period("Y"), index=data_idx)
            rebalance_mask = (s_years != s_years.shift(1)).fillna(False)

        # 預先提取 NumPy 陣列提升多資產日迴圈速度
        open_mat = {t: aligned_panel[(t, "open")].to_numpy(dtype=float) for t in tickers}
        close_mat = {t: aligned_panel[(t, "close")].to_numpy(dtype=float) for t in tickers}
        open_flag_mat = {t: aligned_panel[(t, "is_trading_day")].to_numpy(dtype=bool) for t in tickers}
        div_mat = {
            t: (
                aligned_panel[(t, "dividend")].to_numpy(dtype=float)
                if (t, "dividend") in aligned_panel.columns
                else np.zeros(len(data_idx))
            )
            for t in tickers
        }
        fx_arr = fx_series.to_numpy(dtype=float)
        mc_arr = month_changed.to_numpy(dtype=bool)
        rebal_arr = rebalance_mask.to_numpy(dtype=bool)

        cash = float(initial_capital)
        shares = {t: 0.0 for t in tickers}
        trade_count = 0
        skipped_rebalance_count = 0
        triggered_rebalance_count = 0
        estimated_saved_fees = 0.0
        total_fees_paid = 0.0
        rebalance_dates: List[pd.Timestamp] = []

        pending_rebalance_orders: Dict[str, float] = {}
        asset_cum_bought = {t: 0.0 for t in tickers}
        asset_cum_sold = {t: 0.0 for t in tickers}
        asset_cum_div = {t: 0.0 for t in tickers}
        cum_cash_interest = 0.0

        history_rows = []
        prev_dt: Optional[pd.Timestamp] = None

        for i, dt in enumerate(data_idx):
            # 1. 依真實日曆天數計算現金活存複利利息
            if prev_dt is not None and cash > 0 and cash_annual_rate > 0:
                delta_days = max(1, (pd.Timestamp(dt) - pd.Timestamp(prev_dt)).days)
                daily_interest = cash * (cash_annual_rate * (delta_days / 365.0))
                cash += daily_interest
                cum_cash_interest += daily_interest
            prev_dt = pd.Timestamp(dt)

            fx = float(fx_arr[i])
            to_target_fx = {
                t: cls._resolve_fx_multiplier(ticker_currencies[t], target_curr_code, fx)
                for t in tickers
            }

            # 2. 收取現金股利（皆先入現金池）
            for t in tickers:
                div = float(div_mat[t][i])
                if div > 0 and shares[t] > 0:
                    div_target = shares[t] * div * to_target_fx[t]
                    cash += div_target
                    asset_cum_div[t] += div_target

            # 3. 處理期初首日注資 / 每月固定新注資
            flow = 0.0
            if i == 0 and initial_capital <= 0.0 and invest_day_one and monthly_inflow > 0:
                flow = float(monthly_inflow)
                cash += flow
            elif i > 0 and bool(mc_arr[i]) and monthly_inflow > 0:
                flow = float(monthly_inflow)
                cash += flow

            open_prices_native = {t: float(open_mat[t][i]) for t in tickers}
            close_prices_native = {t: float(close_mat[t][i]) for t in tickers}
            is_market_open = {t: bool(open_flag_mat[t][i]) for t in tickers}

            is_initial = i == 0
            is_rebal = bool(rebal_arr[i]) if i > 0 else False

            if is_initial:
                deploy_pool = cash
                for t in tickers:
                    alloc_amt = deploy_pool * norm_weights[t]
                    if alloc_amt > cls.MIN_ORDER_THRESHOLD:
                        if is_market_open[t]:
                            cost = cost_dict[t]
                            fill_p = CostModel.calculate_buy_fill(
                                open_prices_native[t], cost.fee_rate, cost.slippage_rate
                            ) * to_target_fx[t]
                            bought_s = alloc_amt / fill_p
                            gross_val = bought_s * open_prices_native[t] * to_target_fx[t]
                            total_fees_paid += max(0.0, alloc_amt - gross_val)
                            shares[t] += bought_s
                            cash -= alloc_amt
                            asset_cum_bought[t] += alloc_amt
                            trade_count += 1
                        else:
                            # 若第 1 天該市場休市，先掛單等待開市日買入，杜絕休市日成交
                            pending_rebalance_orders[t] = pending_rebalance_orders.get(t, 0.0) + alloc_amt

            elif is_rebal:
                pre_rebal_val = cash + sum(
                    shares[t] * open_prices_native[t] * to_target_fx[t] for t in tickers
                )
                curr_weights = {
                    t: (
                        (shares[t] * open_prices_native[t] * to_target_fx[t]) / pre_rebal_val
                        if pre_rebal_val > 0
                        else 0.0
                    )
                    for t in tickers
                }
                curr_cash_w = (cash / pre_rebal_val) if pre_rebal_val > 0 else 0.0

                deviations = [abs(curr_weights[t] - norm_weights[t]) for t in tickers]
                if norm_cash_w > 0:
                    deviations.append(abs(curr_cash_w - norm_cash_w))
                max_dev = max(deviations) if pre_rebal_val > 0 else 0.0

                if max_dev > tolerance_band:
                    pending_rebalance_orders.clear()
                    for t in tickers:
                        target_val = pre_rebal_val * norm_weights[t]
                        curr_val = shares[t] * open_prices_native[t] * to_target_fx[t]
                        pending_rebalance_orders[t] = target_val - curr_val
                    triggered_rebalance_count += 1
                    rebalance_dates.append(pd.Timestamp(dt))
                else:
                    skipped_rebalance_count += 1
                    # 依實際偏離金額與實際標的費率估算省下之雙邊調倉手續費/稅金/滑價
                    for t in tickers:
                        diff_amt = abs(pre_rebal_val * norm_weights[t] - shares[t] * open_prices_native[t] * to_target_fx[t])
                        if diff_amt > cls.MIN_REBAL_ORDER_THRESHOLD:
                            c = cost_dict[t]
                            rate = c.fee_rate + c.slippage_rate + (
                                c.tax_rate if shares[t] * open_prices_native[t] * to_target_fx[t] > pre_rebal_val * norm_weights[t] else 0.0
                            )
                            estimated_saved_fees += diff_amt * rate

            elif flow > 0 and cash > cls.MIN_ORDER_THRESHOLD:
                # 計算可供投資之額度：保留目標現金儲備 (pre_val * norm_cash_w)，僅將超額現金投入風險資產
                pre_open_val = cash + sum(
                    shares[t] * open_prices_native[t] * to_target_fx[t] for t in tickers
                )
                target_cash_floor = pre_open_val * norm_cash_w
                investable = max(0.0, cash - target_cash_floor) if norm_cash_w > 0 else cash
                if investable > cls.MIN_ORDER_THRESHOLD and norm_risky_sum > 0:
                    for t in tickers:
                        alloc_amt = investable * risky_rel_weights[t]
                        if alloc_amt > cls.MIN_ORDER_THRESHOLD:
                            pending_rebalance_orders[t] = pending_rebalance_orders.get(t, 0.0) + alloc_amt

            # 4. 統一撮合所有掛單（先賣出釋放現金，再執行買入；休市標的自留待隔日開市成交）
            if pending_rebalance_orders:
                for t in list(pending_rebalance_orders.keys()):
                    diff = pending_rebalance_orders[t]
                    if diff < -cls.MIN_REBAL_ORDER_THRESHOLD and is_market_open[t] and shares[t] > 0:
                        cost = cost_dict[t]
                        sell_val_target = -diff
                        open_target = open_prices_native[t] * to_target_fx[t]
                        sell_s = min(shares[t], sell_val_target / open_target) if open_target > 0 else 0.0
                        if sell_s > 0:
                            net_p = CostModel.calculate_sell_net(
                                open_prices_native[t], cost.fee_rate, cost.tax_rate, cost.slippage_rate
                            ) * to_target_fx[t]
                            gross_recv = sell_s * open_target
                            cash_recv = sell_s * net_p
                            total_fees_paid += max(0.0, gross_recv - cash_recv)
                            shares[t] -= sell_s
                            cash += cash_recv
                            asset_cum_sold[t] += cash_recv
                            trade_count += 1
                        del pending_rebalance_orders[t]
                    elif abs(diff) <= cls.MIN_REBAL_ORDER_THRESHOLD:
                        del pending_rebalance_orders[t]

                for t in list(pending_rebalance_orders.keys()):
                    diff = pending_rebalance_orders[t]
                    if diff > cls.MIN_ORDER_THRESHOLD and is_market_open[t] and cash > cls.MIN_ORDER_THRESHOLD:
                        cost = cost_dict[t]
                        actual_buy_target = min(cash, diff)
                        fill_p = CostModel.calculate_buy_fill(
                            open_prices_native[t], cost.fee_rate, cost.slippage_rate
                        ) * to_target_fx[t]
                        bought_s = actual_buy_target / fill_p
                        gross_stock_val = bought_s * open_prices_native[t] * to_target_fx[t]
                        total_fees_paid += max(0.0, actual_buy_target - gross_stock_val)
                        shares[t] += bought_s
                        cash -= actual_buy_target
                        asset_cum_bought[t] += actual_buy_target
                        trade_count += 1
                        rem_diff = diff - actual_buy_target
                        if rem_diff <= cls.MIN_ORDER_THRESHOLD:
                            del pending_rebalance_orders[t]
                        else:
                            pending_rebalance_orders[t] = rem_diff

            stock_vals_target = {t: shares[t] * close_prices_native[t] * to_target_fx[t] for t in tickers}
            total_stock_val = sum(stock_vals_target.values())
            total_port_val = cash + total_stock_val

            row_data = {
                "date": dt,
                "portfolio_value": total_port_val,
                "cash": cash,
                "total_stock_value": total_stock_val,
                "external_flow": flow,
                "trades": trade_count,
                "fees_paid": total_fees_paid,
            }
            for t in tickers:
                row_data[f"val_{t}"] = stock_vals_target[t]
                row_data[f"shares_{t}"] = shares[t]
                row_data[f"weight_{t}"] = (stock_vals_target[t] / total_port_val) if total_port_val > 0 else 0.0
            row_data["weight_CASH"] = (cash / total_port_val) if total_port_val > 0 else 0.0

            history_rows.append(row_data)

        res_df = pd.DataFrame(history_rows).set_index("date")
        res_df["total_invested"] = float(initial_capital) + res_df["external_flow"].cumsum()
        res_df["equity_net"] = cls._calc_twr_curve(
            res_df["portfolio_value"].tolist(),
            res_df["external_flow"].tolist(),
            initial_capital,
        )

        last_fx = float(fx_arr[-1])
        last_close = {t: float(close_mat[t][-1]) for t in tickers}
        last_to_target = {
            t: cls._resolve_fx_multiplier(ticker_currencies[t], target_curr_code, last_fx)
            for t in tickers
        }

        attribution: Dict[str, Dict[str, float]] = {}
        for t in tickers:
            final_holding_val = shares[t] * last_close[t] * last_to_target[t]
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
                "roi": roi,
            }

        if norm_cash_w > 0 or cum_cash_interest > 0:
            tot_inv_cum = float(res_df["total_invested"].iloc[-1])
            cash_base_invested = tot_inv_cum * norm_cash_w if norm_cash_w > 0 else max(cash - cum_cash_interest, 0.0)
            attribution["💵 現金活存儲備"] = {
                "final_value": cash,
                "cum_bought": cash_base_invested,
                "cum_sold": 0.0,
                "cum_dividend": cum_cash_interest,
                "net_profit": cum_cash_interest,
                "roi": (cum_cash_interest / cash_base_invested) if cash_base_invested > 0 else 0.0,
            }

        rebal_stats = {
            "觸發調倉次數": triggered_rebalance_count,
            "成功略過調倉次數": skipped_rebalance_count,
            "估計省下摩擦手續費": f"{round(estimated_saved_fees, 0):,.0f}",
            "_raw_saved_fees": estimated_saved_fees,
            "累計實付摩擦成本": total_fees_paid,
            "累積現金利息": cum_cash_interest,
            "rebalance_dates": rebalance_dates,
        }
        return res_df, attribution, rebal_stats

    @staticmethod
    def _calc_twr_curve(
        portfolio_vals: List[float],
        external_flows: List[float],
        init_capital: float,
    ) -> np.ndarray:
        equity = []
        index_val = 1.0
        prev_val: Optional[float] = None

        for i, (val, flow) in enumerate(zip(portfolio_vals, external_flows)):
            if i == 0:
                base0 = init_capital if init_capital > 0 else flow
                index_val = (val / base0) if base0 > 0 else 1.0
            elif prev_val is not None:
                cap_base = prev_val + flow
                if cap_base > 0:
                    daily_ret = (val / cap_base) - 1.0
                    index_val *= max(0.0, 1.0 + daily_ret)
            equity.append(index_val)
            prev_val = val

        return np.asarray(equity, dtype=float)
