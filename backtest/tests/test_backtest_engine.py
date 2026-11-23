"""
自動化測試套件：涵蓋拆股除權息還原、KD 指標尺度、暖機期、DCA 頻率拆分、
雙向匯率與帳本一致性、GridOptimizer vs BacktestEngine 一致性、
多資產日曆天數計息與無前瞻對齊、全部 10 種策略與停損風控、以及 UI 靜態檢查。
"""
import ast
import os
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.cost_model import AssetCostStructure, CostModel
from core.data_engine import DataEngine
from core.grid_optimizer import GridOptimizer
from core.indicators import IndicatorEngine
from core.metrics import PerformanceMetrics
from core.portfolio_engine import BacktestEngine
from core.portfolio_optimizer import PortfolioOptimizer
from core.trade_ledger import TradeLedger
import strategies.dca
import strategies.ma_cross
import strategies.oscillators
import strategies.advanced
from strategies.registry import StrategyRegistry


def _make_synthetic_df(
    start: str = "2023-01-02",
    periods: int = 320,
    user_start: str = "2023-09-01",
    quote_currency: str = "TWD",
    base_price: float = 100.0,
    daily_growth: float = 0.0008,
    amplitude: float = 0.03,
    dividend_idx: int = -1,
    dividend_amt: float = 0.0,
    split_idx: int = -1,
    split_ratio: float = 0.0,
    adj_scale: float = 1.0,
) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=periods)
    t = np.arange(periods, dtype=float)
    trend = base_price * np.power(1.0 + daily_growth, t)
    wave = 1.0 + amplitude * np.sin(t / 7.0)
    close = trend * wave
    open_p = close * 0.998
    high_p = np.maximum(open_p, close) * 1.006
    low_p = np.minimum(open_p, close) * 0.994

    adj_close = close * adj_scale
    adj_open = open_p * adj_scale
    adj_high = high_p * adj_scale
    adj_low = low_p * adj_scale

    div = np.zeros(periods)
    if 0 <= dividend_idx < periods:
        div[dividend_idx] = dividend_amt

    split = np.zeros(periods)
    if 0 <= split_idx < periods:
        split[split_idx] = split_ratio

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close,
            "adj_open": adj_open,
            "adj_high": adj_high,
            "adj_low": adj_low,
            "adj_close": adj_close,
            "dividend": div,
            "split": split,
            "is_trading_day": True,
            "quote_currency": quote_currency,
            "user_start_idx": dates >= pd.Timestamp(user_start),
        },
        index=dates,
    )
    return df


class TestBacktestSystem(unittest.TestCase):
    def test_cost_model_bond_etf_and_discount(self):
        bond_cost = CostModel.get_default_costs("00679B", tw_fee_discount=0.5)
        self.assertEqual(bond_cost.tax_rate, 0.0)
        self.assertAlmostEqual(bond_cost.fee_rate, 0.001425 * 0.5, places=8)

        etf_cost = CostModel.get_default_costs("0050", tw_fee_discount=0.28)
        self.assertEqual(etf_cost.tax_rate, 0.001)
        self.assertAlmostEqual(etf_cost.fee_rate, 0.001425 * 0.28, places=8)

        stock_cost = CostModel.get_default_costs("2330", tw_fee_discount=1.0)
        self.assertEqual(stock_cost.tax_rate, 0.003)
        self.assertAlmostEqual(stock_cost.fee_rate, 0.001425, places=8)

        self.assertTrue(DataEngine.is_taiwan_ticker("00679B"))
        self.assertTrue(DataEngine.is_taiwan_ticker("00981A"))
        self.assertTrue(DataEngine.is_taiwan_ticker("2881A"))

    def test_no_double_split_multiplication(self):
        # 模擬 10:1 拆股 (yfinance 價格已是拆股調整後均勻價格，split=10 僅作事件註記)
        dates = pd.bdate_range("2024-01-02", periods=60)
        prices = np.linspace(100.0, 120.0, 60)
        splits = np.zeros(60)
        splits[30] = 10.0
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "adj_open": prices,
                "adj_high": prices * 1.01,
                "adj_low": prices * 0.99,
                "adj_close": prices,
                "dividend": 0.0,
                "split": splits,
                "is_trading_day": True,
                "quote_currency": "USD",
                "user_start_idx": True,
            },
            index=dates,
        )
        zero_cost = AssetCostStructure(0.0, 0.0, 0.0)
        res = BacktestEngine.run_single_strategy(
            df,
            "定期定額 / 買入持有 (DCA / Buy & Hold)",
            {"freq": "每月 (Monthly)", "reinvest_idle": True},
            initial_capital=100000.0,
            monthly_inflow=0.0,
            cost_struct=zero_cost,
            fx_series=pd.Series(1.0, index=dates),
            target_currency="USD",
        )
        # 若無重複拆股錯誤，最終總資產必為 100000 * (120 / 100) = 120000
        self.assertAlmostEqual(float(res["portfolio_value"].iloc[-1]), 120000.0, delta=1.0)

        # 同樣檢查擇時策略不乘 split
        res_timing = BacktestEngine.run_single_strategy(
            df,
            "價格突破均線買入/跌破賣出",
            {"timeframe": "日線 (Daily)", "period": 3},
            initial_capital=100000.0,
            monthly_inflow=0.0,
            cost_struct=zero_cost,
            fx_series=pd.Series(1.0, index=dates),
            target_currency="USD",
        )
        self.assertLess(float(res_timing["portfolio_value"].iloc[-1]), 130000.0)

    def test_dividend_cash_and_reinvest_flag(self):
        dates = pd.bdate_range("2024-01-02", periods=60)
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "adj_open": 100.0,
                "adj_high": 100.0,
                "adj_low": 100.0,
                "adj_close": 100.0,
                "dividend": 0.0,
                "split": 0.0,
                "is_trading_day": True,
                "quote_currency": "TWD",
                "user_start_idx": True,
            },
            index=dates,
        )
        df.iloc[10, df.columns.get_loc("dividend")] = 5.0  # 每股發 5 元
        zero_cost = AssetCostStructure(0.0, 0.0, 0.0)

        res_reinvest = BacktestEngine.run_single_strategy(
            df,
            "定期定額 / 買入持有 (DCA / Buy & Hold)",
            {"freq": "每月 (Monthly)", "reinvest_idle": True},
            initial_capital=100000.0,
            monthly_inflow=0.0,
            cost_struct=zero_cost,
            target_currency="TWD",
        )
        # 第 2 月首個交易日會自動把股利 5,000 再買入成 50 股，總股數應變為 1050
        self.assertAlmostEqual(float(res_reinvest["shares"].iloc[-1]), 1050.0, delta=0.01)
        self.assertAlmostEqual(float(res_reinvest["cash"].iloc[-1]), 0.0, delta=0.01)

        res_no_reinvest = BacktestEngine.run_single_strategy(
            df,
            "定期定額 / 買入持有 (DCA / Buy & Hold)",
            {"freq": "每月 (Monthly)", "reinvest_idle": False},
            initial_capital=100000.0,
            monthly_inflow=10000.0,
            cost_struct=zero_cost,
            target_currency="TWD",
        )
        # 關閉閒置資金再投資時，股利 5,000 應留在 cash 內不被花掉
        self.assertGreaterEqual(float(res_no_reinvest["cash"].iloc[-1]), 4999.0)

    def test_kd_scale_and_no_pandas_warnings(self):
        # 模擬極大除權息落差 (adj_close = 0.6 * close)，舊版會產出負數 KD
        df = _make_synthetic_df(adj_scale=0.6)
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            k_d, d_d = IndicatorEngine.calculate_kd_full(df, period=9, timeframe="日線 (Daily)")
            k_w, d_w = IndicatorEngine.calculate_kd_full(df, period=9, timeframe="週線 (Weekly)")

        future_warns = [w for w in w_list if issubclass(w.category, FutureWarning)]
        self.assertEqual(len(future_warns), 0, f"Unexpected FutureWarning: {future_warns}")

        for s in (k_d, d_d, k_w, d_w):
            valid = s.dropna()
            self.assertGreater(len(valid), 0)
            self.assertGreaterEqual(float(valid.min()), 0.0)
            self.assertLessEqual(float(valid.max()), 100.0)

    def test_dip_dca_warmup_availability(self):
        # 製造暖機期後於 user_start_idx 開始下跌之資料，確保第 1 週即具備成熟 SMA 並可買入
        dates = pd.bdate_range("2024-01-02", periods=100)
        prices = np.concatenate([np.full(80, 120.0), np.linspace(110.0, 90.0, 20)])
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "adj_open": prices,
                "adj_high": prices * 1.01,
                "adj_low": prices * 0.99,
                "adj_close": prices,
                "dividend": 0.0,
                "split": 0.0,
                "is_trading_day": True,
                "quote_currency": "TWD",
                "user_start_idx": dates >= dates[80],
            },
            index=dates,
        )
        res = BacktestEngine.run_single_strategy(
            df,
            "跌破均線且創新低買入 (Dip DCA)",
            {"timeframe": "日線 (Daily)", "sma_period": 60, "buy_amount": 50000.0, "min_interval": 1},
            initial_capital=200000.0,
            monthly_inflow=0.0,
            cost_struct=AssetCostStructure(0.0, 0.0, 0.0),
            target_currency="TWD",
        )
        # 第 2 天（T+1 開盤）即應完成第一筆抄底買入
        self.assertGreater(float(res["shares"].iloc[1]), 0.0)

    def test_dca_weekly_budget_consistency(self):
        df = _make_synthetic_df(periods=260, user_start="2023-03-01", quote_currency="TWD")
        cost = AssetCostStructure(0.0, 0.0, 0.0)
        res_m = BacktestEngine.run_single_strategy(
            df,
            "定期定額 / 買入持有 (DCA / Buy & Hold)",
            {"freq": "每月 (Monthly)", "reinvest_idle": True},
            initial_capital=0.0,
            monthly_inflow=30000.0,
            cost_struct=cost,
            target_currency="TWD",
            invest_day_one=True,
        )
        res_w = BacktestEngine.run_single_strategy(
            df,
            "定期定額 / 買入持有 (DCA / Buy & Hold)",
            {"freq": "每週 (Weekly)", "reinvest_idle": True},
            initial_capital=0.0,
            monthly_inflow=30000.0,
            cost_struct=cost,
            target_currency="TWD",
            invest_day_one=True,
        )
        # 週定投將每月固定 30,000 平均拆分成各週買入，總累積投入本金應與月定投相同（誤差 < 1 元）
        self.assertAlmostEqual(
            float(res_m["total_invested"].iloc[-1]),
            float(res_w["total_invested"].iloc[-1]),
            delta=1.0,
        )
        self.assertGreater(int(res_w["trades"].iloc[-1]), int(res_m["trades"].iloc[-1]) * 3)

    def test_bidirectional_fx_and_ledger_accuracy(self):
        # 1) TWD 標的以 USD 計價 (to_target_fx 必須為 1/32，不能出現雙重乘法)
        df_tw = _make_synthetic_df(periods=120, user_start="2023-03-01", quote_currency="TWD", base_price=320.0, daily_growth=0.0, amplitude=0.0)
        fx_s = pd.Series(32.0, index=df_tw.index)
        cost = AssetCostStructure(0.0, 0.0, 0.0)
        res_usd_view = BacktestEngine.run_single_strategy(
            df_tw,
            "定期定額 / 買入持有 (DCA / Buy & Hold)",
            {"freq": "每月 (Monthly)", "reinvest_idle": True},
            initial_capital=1000.0,
            monthly_inflow=0.0,
            cost_struct=cost,
            fx_series=fx_s,
            target_currency="USD",
        )
        # 1000 USD = 32,000 TWD，以每股約 320 TWD 買入約 100 股
        self.assertAlmostEqual(float(res_usd_view["shares"].iloc[0]), 100.0, delta=1.0)
        buys_df, buy_sum = TradeLedger.extract_buy_records(res_usd_view, "$")
        self.assertFalse(buys_df.empty)
        self.assertIn("100.", buy_sum["累積持有股數"])

        # 2) USD 標的以 TWD 計價，確認 TradeLedger 不把股數乘以匯率
        df_us = _make_synthetic_df(periods=120, user_start="2023-03-01", quote_currency="USD", base_price=100.0, daily_growth=0.0, amplitude=0.0)
        res_twd_view = BacktestEngine.run_single_strategy(
            df_us,
            "定期定額 / 買入持有 (DCA / Buy & Hold)",
            {"freq": "每月 (Monthly)", "reinvest_idle": True},
            initial_capital=32000.0,
            monthly_inflow=0.0,
            cost_struct=cost,
            fx_series=fx_s,
            target_currency="TWD",
        )
        self.assertAlmostEqual(float(res_twd_view["shares"].iloc[0]), 10.0, delta=0.2)
        _, buy_sum_us = TradeLedger.extract_buy_records(res_twd_view, "NT$")
        self.assertIn("10.", buy_sum_us["累積持有股數"])

    def test_grid_optimizer_matches_backtest_engine(self):
        df = _make_synthetic_df(
            periods=350,
            user_start="2023-08-01",
            quote_currency="USD",
            base_price=100.0,
            dividend_idx=220,
            dividend_amt=1.5,
        )
        fx_s = pd.Series(31.5, index=df.index)
        cost = AssetCostStructure(0.0005, 0.001, 0.0002)

        # 測試週線 MACD 尋優與 BacktestEngine 一致性
        _, flat_macd, _ = GridOptimizer.run_macd_grid(
            df,
            fast_range=(12, 13, 1),
            slow_range=(26, 27, 1),
            signal_p=9,
            timeframe="週線 (Weekly)",
            initial_capital=300000.0,
            monthly_inflow=10000.0,
            cost_struct=cost,
            fx_series=fx_s,
            target_currency="TWD",
        )
        res_macd = BacktestEngine.run_single_strategy(
            df,
            "MACD 柱狀圖/交叉",
            {"timeframe": "週線 (Weekly)", "fast": 12, "slow": 26, "signal": 9},
            initial_capital=300000.0,
            monthly_inflow=10000.0,
            cost_struct=cost,
            fx_series=fx_s,
            target_currency="TWD",
        )
        m_macd = PerformanceMetrics.evaluate(res_macd, curr_symbol="NT$", initial_capital=300000.0)
        self.assertAlmostEqual(
            float(flat_macd.iloc[0]["總報酬率 (%)"]),
            float(m_macd["_raw_roi"]) * 100.0,
            delta=0.05,
        )

        # 測試 Dip DCA 尋優與 BacktestEngine 一致性
        _, flat_dip, _ = GridOptimizer.run_dip_grid(
            df,
            sma_range=(60, 61, 10),
            interval_range=(3, 4, 1),
            buy_amount=40000.0,
            timeframe="日線 (Daily)",
            initial_capital=300000.0,
            monthly_inflow=10000.0,
            cost_struct=cost,
            fx_series=fx_s,
            target_currency="TWD",
        )
        res_dip = BacktestEngine.run_single_strategy(
            df,
            "跌破均線且創新低買入 (Dip DCA)",
            {"timeframe": "日線 (Daily)", "sma_period": 60, "buy_amount": 40000.0, "min_interval": 3},
            initial_capital=300000.0,
            monthly_inflow=10000.0,
            cost_struct=cost,
            fx_series=fx_s,
            target_currency="TWD",
        )
        m_dip = PerformanceMetrics.evaluate(res_dip, curr_symbol="NT$", initial_capital=300000.0)
        self.assertAlmostEqual(
            float(flat_dip.iloc[0]["總報酬率 (%)"]),
            float(m_dip["_raw_roi"]) * 100.0,
            delta=0.05,
        )

    def test_multi_asset_cash_interest_isolation_and_holiday_order(self):
        # 2024 完整一年工作日 (~261 個工作日)，100% 現金或 50% 現金活存測試真實日曆天數計息
        dates = pd.bdate_range("2024-01-02", "2024-12-31")
        n = len(dates)
        asset_a = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "adj_open": 100.0,
                "adj_high": 100.0,
                "adj_low": 100.0,
                "adj_close": 100.0,
                "dividend": 0.0,
                "split": 0.0,
                "is_trading_day": True,
                "quote_currency": "TWD",
            },
            index=dates,
        )
        asset_b = asset_a.copy()
        # 模擬第 0 天資產 B 休市 (不得在休市日成交，應順延到第 1 天成交)
        asset_b.iloc[0, asset_b.columns.get_loc("is_trading_day")] = False
        panel = pd.concat({"A": asset_a, "B": asset_b}, axis=1)
        fx = pd.Series(32.0, index=dates)
        zero_cost = {"A": AssetCostStructure(0.0, 0.0, 0.0), "B": AssetCostStructure(0.0, 0.0, 0.0)}

        res_df, attrib, stats = BacktestEngine.run_multi_asset_portfolio(
            aligned_panel=panel,
            aligned_fx=fx,
            target_weights={"A": 0.3, "B": 0.3},
            initial_capital=100000.0,
            monthly_inflow=10000.0,
            rebalance_freq="每季再平衡 (Quarterly)",
            cost_dict=zero_cost,
            target_currency="TWD",
            tolerance_band=0.05,
            cash_reserve_weight=0.40,
            cash_annual_rate=0.05,
        )
        # 驗證休市順延：第 0 天 B 休市 -> shares_B[0] == 0，第 1 天開市後 -> shares_B[1] > 0
        self.assertEqual(float(res_df["shares_B"].iloc[0]), 0.0)
        self.assertGreater(float(res_df["shares_B"].iloc[1]), 0.0)
        # 驗證現金儲備隔離：期初留 40% 現金，每月注入也保持活存佔比附近，不被清空
        self.assertGreater(float(res_df["weight_CASH"].iloc[-1]), 0.30)
        # 驗證真實日曆天數複利利息：若以 5% 年利率計息，40,000+ 平均現金池一年利息應明顯超過 2,000
        self.assertGreater(float(stats["累積現金利息"]), 2000.0)
        self.assertIn("💵 現金活存儲備", attrib)

    def test_no_bfill_lookahead_in_data_engine(self):
        # 模擬標的 A 上市早，標的 B 晚 90 天才上市；mock yf.download 驗證共同起始日自動切齊無 bfill 假平盤
        dates_all = pd.bdate_range("2023-01-02", "2024-06-30")
        dates_late = dates_all[dates_all >= pd.Timestamp("2024-01-02")]

        def make_yf_frame(idx, start_p):
            p = np.linspace(start_p, start_p * 1.2, len(idx))
            return pd.DataFrame(
                {
                    "Open": p,
                    "High": p * 1.01,
                    "Low": p * 0.99,
                    "Close": p,
                    "Adj Close": p,
                    "Dividends": 0.0,
                    "Stock Splits": 0.0,
                },
                index=idx,
            )

        df_a = make_yf_frame(dates_all, 100.0)
        df_b = make_yf_frame(dates_late, 50.0)
        df_fx = make_yf_frame(dates_all, 32.0)

        combined = pd.concat({"A": df_a, "B": df_b, "USDTWD=X": df_fx}, axis=1)
        with patch("core.data_engine.yf.download", return_value=combined):
            indiv, panel, fx_out, _, warns = DataEngine.load_aligned_market_data(
                ["A", "B"], "2023-06-01", "2024-06-30", "TWD"
            )
        self.assertIn("A", indiv)
        self.assertIn("B", indiv)
        # aligned_panel 起始日必須切齊至晚上市標的 B 的 2024-01-02，而非回填 2023-06-01
        self.assertGreaterEqual(pd.Timestamp(panel.index.min()), pd.Timestamp("2024-01-02"))
        self.assertTrue(any("共同起始日" in w for w in warns))
        self.assertFalse(fx_out.isna().any())

    def test_all_registered_strategies_and_risk_controls(self):
        df = _make_synthetic_df(periods=360, user_start="2023-09-01", quote_currency="TWD")
        cost = AssetCostStructure(0.0007125, 0.001, 0.0002)
        all_strats = StrategyRegistry.list_names()
        self.assertGreaterEqual(len(all_strats), 10)

        for name in all_strats:
            strat_cls = StrategyRegistry.get(name)
            params = {spec.key: spec.default for spec in strat_cls.get_param_specs()}
            res = BacktestEngine.run_single_strategy(
                df,
                name,
                params,
                initial_capital=300000.0,
                monthly_inflow=10000.0,
                cost_struct=cost,
                target_currency="TWD",
                stop_loss_pct=0.08 if strat_cls.is_timing else 0.0,
                trailing_stop_pct=0.12 if strat_cls.is_timing else 0.0,
            )
            m = PerformanceMetrics.evaluate(res, "NT$", 300000.0)
            self.assertIn("夏普比率 (Sharpe)", m)
            self.assertIn("最長套牢天數", m)
            monthly = PerformanceMetrics.calculate_monthly_returns(res)
            self.assertFalse(monthly.empty)

            if strat_cls.is_timing:
                trades_df, summary = TradeLedger.extract_trades(res, curr_symbol="NT$")
                self.assertIn("單筆期望值", summary)
            else:
                buys_df, buy_sum = TradeLedger.extract_buy_records(res, "NT$")
                self.assertIn("累積持有股數", buy_sum)

    def test_portfolio_optimizer_weekly_and_custom_point(self):
        dates = pd.bdate_range("2023-01-02", periods=300)
        p1 = 100.0 * np.cumprod(1.0 + np.random.default_rng(1).normal(0.0005, 0.01, len(dates)))
        p2 = 50.0 * np.cumprod(1.0 + np.random.default_rng(2).normal(0.0003, 0.008, len(dates)))
        f1 = pd.DataFrame(
            {"open": p1, "high": p1, "low": p1, "close": p1, "adj_close": p1, "quote_currency": "TWD"},
            index=dates,
        )
        f2 = pd.DataFrame(
            {"open": p2, "high": p2, "low": p2, "close": p2, "adj_close": p2, "quote_currency": "USD"},
            index=dates,
        )
        panel = pd.concat({"0050": f1, "QQQ": f2}, axis=1)
        fx = pd.Series(32.0, index=dates)
        ms, mv, rp, eq, frontier, stats = PortfolioOptimizer.optimize_portfolio(panel, fx, "TWD")
        self.assertAlmostEqual(sum(ms.values()), 100.0, delta=0.2)
        self.assertIn("corr_matrix", stats)
        pt = PortfolioOptimizer.evaluate_custom_weights(stats, {"0050": 0.5, "QQQ": 0.5})
        self.assertIsNotNone(pt)

    def test_long_20_year_horizon_and_weekly_candlestick(self):
        from ui.components_chart import render_candlestick_with_markers

        df_long = _make_synthetic_df(
            start="2004-01-02",
            periods=5500,
            user_start="2005-01-03",
            quote_currency="USD",
            daily_growth=0.0003,
            amplitude=0.04,
        )
        cost = CostModel.get_default_costs("SPY")
        res = BacktestEngine.run_single_strategy(
            df_long,
            "SMA 雙均線交叉 (黃金交叉)",
            {"timeframe": "日線 (Daily)", "fast": 20, "slow": 60},
            initial_capital=10000.0,
            monthly_inflow=500.0,
            cost_struct=cost,
            target_currency="USD",
        )
        m = PerformanceMetrics.evaluate(res, "$", 10000.0)
        self.assertTrue(np.isfinite(m["_raw_cagr"]))
        self.assertTrue(np.isfinite(m["_raw_sharpe"]))
        self.assertTrue(np.isfinite(m["_raw_xirr"]))
        monthly = PerformanceMetrics.calculate_monthly_returns(res)
        self.assertGreaterEqual(len(monthly), 20)

        captured_figs = []
        with patch("ui.components_chart.st.plotly_chart", side_effect=lambda fig, **kw: captured_figs.append(fig)):
            render_candlestick_with_markers(res, res["position"], "SPY", curr_symbol="$", resample_weekly=True)

        self.assertEqual(len(captured_figs), 1)
        fig = captured_figs[0]
        candle_trace = fig.data[0]
        # 21+ years of weekly bars (~1100 weeks), far lighter than ~5240 daily bars
        self.assertLess(len(candle_trace.x), 1400)
        self.assertGreater(len(candle_trace.x), 900)

    def test_monthly_rebalance_skip_still_invests_inflow_and_small_orders(self):
        dates = pd.bdate_range("2024-01-02", periods=250)
        df_a = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "adj_open": 100.0,
                "adj_high": 101.0,
                "adj_low": 99.0,
                "adj_close": 100.0,
                "dividend": 0.0,
                "split": 0.0,
                "is_trading_day": True,
                "quote_currency": "USD",
            },
            index=dates,
        )
        panel = pd.concat({"A": df_a, "B": df_a.copy(), "C": df_a.copy()}, axis=1)
        fx = pd.Series(32.0, index=dates)
        costs = {k: AssetCostStructure(0.0, 0.0, 0.0) for k in ("A", "B", "C")}

        # Bug #2: 每月再平衡遇門檻略過時，當月定投仍須投入，不可卡在現金
        res1, _, stats1 = BacktestEngine.run_multi_asset_portfolio(
            panel,
            fx,
            {"A": 0.34, "B": 0.33, "C": 0.33},
            1000000.0,
            30000.0,
            "每月再平衡 (Monthly)",
            costs,
            "TWD",
            tolerance_band=0.05,
        )
        self.assertGreater(stats1["成功略過調倉次數"], 0)
        self.assertLess(float(res1["cash"].iloc[-1]), 50.0)

        # Bug #3: 每月定投 $100 平分至 3 檔 ($33.3/檔, 介於 $10..$50) 不得被賣單迴圈誤刪
        res2, _, _ = BacktestEngine.run_multi_asset_portfolio(
            panel,
            fx,
            {"A": 0.34, "B": 0.33, "C": 0.33},
            0.0,
            100.0,
            "不進行再平衡 (Buy & Drift)",
            costs,
            "USD",
            tolerance_band=0.0,
            invest_day_one=True,
        )
        self.assertLess(float(res2["cash"].iloc[-1]), 20.0)
        self.assertGreater(float(res2["shares_A"].iloc[-1]), 4.0)

    def test_stoploss_overlap_with_bear_exit_and_equal_ma_guard(self):
        dates = pd.bdate_range("2024-01-02", periods=10)
        df_sl = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
                "high": [100.0] * 10,
                "low": [80.0] * 10,
                "close": [100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
                "adj_close": [100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
                "dividend": [0.0] * 10,
                "quote_currency": "USD",
            },
            index=dates,
        )
        pos = pd.Series([1, 1, 1, 0, 1, 1, 1, 1, 1, 1], index=dates)
        mc = pd.Series([False] * 10, index=dates)
        res_sl = BacktestEngine._execute_timing_loop(
            df_sl,
            pos,
            10000.0,
            0.0,
            AssetCostStructure(0.0, 0.0, 0.0),
            mc,
            np.ones(10),
            stop_loss_pct=0.10,
        )
        # Bug #4: 第 4 天停損出場恰遇 raw_pos == 0，後續第 5 天再轉多時必須正常買入
        self.assertEqual(int(res_sl["position"].iloc[3]), 0)
        self.assertEqual(int(res_sl["position"].iloc[-1]), 1)

        # #15 & zero capital safety: fast == slow 自動保護且 0 資金不崩潰
        df_syn = _make_synthetic_df(periods=120, user_start="2023-01-02", daily_growth=0.002, amplitude=0.05)
        sma_cls = StrategyRegistry.get("SMA 雙均線交叉 (黃金交叉)")
        sig_eq = sma_cls.generate_signals(df_syn, {"timeframe": "日線 (Daily)", "fast": 20, "slow": 20})
        self.assertGreater(int(sig_eq.sum()), 0)
        macd_line, sig_line = IndicatorEngine.calculate_macd(df_syn, 12, 12, 9)
        self.assertGreater(float(macd_line.abs().sum()), 0.0)

        res_zero = BacktestEngine.run_single_strategy(
            df_syn,
            "定期定額 / 買入持有 (DCA / Buy & Hold)",
            {"freq": "每月 (Monthly)", "reinvest_idle": True},
            initial_capital=0.0,
            monthly_inflow=0.0,
            cost_struct=AssetCostStructure(0.0, 0.0, 0.0),
            target_currency="TWD",
        )
        m_zero = PerformanceMetrics.evaluate(res_zero, "NT$", 0.0)
        self.assertEqual(m_zero["_raw_final"], 0.0)

    def test_first_month_return_cagr_and_vectorized_oscillator(self):
        # Bug #6: 首日即漲 +5%，次月收 +10% -> 1月應為 +5%，全年 +10%
        d_m = pd.to_datetime(["2024-01-31", "2024-02-29"])
        df_m = pd.DataFrame({"equity_net": [1.05, 1.10], "total_invested": [100000.0, 100000.0]}, index=d_m)
        monthly = PerformanceMetrics.calculate_monthly_returns(df_m)
        self.assertAlmostEqual(float(monthly.loc["2024", "1月"]), 0.05, places=6)
        self.assertAlmostEqual(float(monthly.loc["2024", "全年"]), 0.10, places=6)

        # Bug #9: 向量化 _oscillator_signals 須與序列 RSI / KD 邏輯完全一致
        df_osc = _make_synthetic_df(periods=200, user_start="2023-01-02", amplitude=0.12)
        rsi_vals = IndicatorEngine.calculate_rsi(df_osc, 14).values
        vec_sig = GridOptimizer._oscillator_signals(rsi_vals, 30.0, 70.0)
        rsi_cls = StrategyRegistry.get("RSI 超買超賣")
        seq_sig = rsi_cls.generate_signals(df_osc, {"timeframe": "日線 (Daily)", "period": 14, "buy": 30.0, "sell": 70.0}).to_numpy(dtype=int)
        np.testing.assert_array_equal(vec_sig, seq_sig)

    def test_ui_files_syntax_ast(self):
        py_files = [
            ROOT_DIR / "app.py",
            ROOT_DIR / "ui" / "theme.py",
            ROOT_DIR / "ui" / "components_chart.py",
            ROOT_DIR / "ui" / "tab_strategy_pk.py",
            ROOT_DIR / "ui" / "tab_portfolio_rebalance.py",
        ]
        for path in py_files:
            src = path.read_text(encoding="utf-8")
            ast.parse(src, filename=str(path))


if __name__ == "__main__":
    unittest.main()
