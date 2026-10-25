"""回測系統 v32 核心單元測試。執行：python test_backtest_system_v32.py"""
import ast
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path(__file__).with_name('backtest_system_v32_optimized.py')
source_text = SOURCE.read_text(encoding='utf-8')
module_ast = ast.parse(source_text)
selected = [n for n in module_ast.body if isinstance(n, ast.ClassDef) and n.name in {
    'DataFetcher', 'IndicatorCalculator', 'BacktestEngine', 'PerformanceEvaluator'
}]
class _FakeStreamlit:
    @staticmethod
    def cache_data(**kwargs):
        return lambda fn: fn
namespace = {'np': np, 'pd': pd, 'st': _FakeStreamlit()}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), namespace)
DataFetcher = namespace['DataFetcher']
IndicatorCalculator = namespace['IndicatorCalculator']
BacktestEngine = namespace['BacktestEngine']
PerformanceEvaluator = namespace['PerformanceEvaluator']


class TestBacktestCore(unittest.TestCase):
    def test_buy_cost_and_sell_cost_are_applied(self):
        buy = BacktestEngine._buy_fill_price(100, fee=0.01, slip=0.02)
        sell = BacktestEngine._sell_net_price(100, fee=0.01, tax=0.03, slip=0.02)
        self.assertAlmostEqual(buy, 103.02, places=8)
        self.assertAlmostEqual(sell, 100 * .98 * .99 * .97, places=8)

    def test_contribution_does_not_create_twr_return(self):
        idx = pd.to_datetime(['2024-01-02', '2024-02-01'])
        frame = pd.DataFrame(index=idx)
        result = BacktestEngine._attach_portfolio_history(
            frame, [100.0, 150.0], [100.0, 150.0], [0.0, 0.0],
            [0.0, 50.0], 100.0, 0
        )
        self.assertAlmostEqual(result['equity_net'].iloc[0], 1.0)
        self.assertAlmostEqual(result['equity_net'].iloc[1], 1.0)

    def test_flat_market_monthly_inflows_do_not_inflate_dca_twr(self):
        idx = pd.date_range('2024-01-02', periods=65, freq='B')
        data = pd.DataFrame({
            'open': 100.0, 'close': 100.0, 'high': 100.0, 'low': 100.0,
            'dividend': 0.0, 'split': 0.0
        }, index=idx)
        result = BacktestEngine.run_dca_strategy(
            data, 10000, 500, '每月 (Monthly)', 500, 20, 0.0, 0.0
        )
        self.assertTrue(np.allclose(result['equity_net'], 1.0, atol=1e-10))
        self.assertAlmostEqual(result['total_invested'].iloc[-1], 10000.0 + 500.0 * (idx.to_period('M').nunique() - 1))

    def test_mdd_includes_first_day_transaction_cost(self):
        idx = pd.date_range('2024-01-02', periods=3, freq='B')
        data = pd.DataFrame({
            'open': [100.0] * 3, 'close': [100.0] * 3,
            'high': [100.0] * 3, 'low': [100.0] * 3,
            'dividend': [0.0] * 3, 'split': [0.0] * 3
        }, index=idx)
        result = BacktestEngine.run_dca_strategy(
            data, 10000, 0, '每月 (Monthly)', 0, 20, .01, 0.0
        )
        metrics = PerformanceEvaluator.evaluate(result, '$', initial_capital=10000)
        self.assertTrue(metrics)
        self.assertAlmostEqual(metrics['_raw_mdd'], result['equity_net'].iloc[0] - 1.0, places=8)

    def test_dip_strategy_executes_next_day_open_not_signal_close(self):
        idx = pd.date_range('2024-01-02', periods=4, freq='B')
        data = pd.DataFrame({
            'open': [100.0, 100.0, 100.0, 100.0],
            'close': [100.0, 90.0, 80.0, 70.0],
            'high': [100.0, 100.0, 100.0, 100.0],
            'low': [100.0, 90.0, 80.0, 70.0],
            'dividend': [0.0] * 4, 'split': [0.0] * 4
        }, index=idx)
        result = BacktestEngine.run_dip_dca(
            data, 1000, 0,
            {'sma_period': 2, 'timeframe': '日線 (Daily)', 'buy_amount': 100, 'min_interval': 1},
            fee=0.0, slip=0.0
        )
        # 第 2 日收盤才知道 close < SMA，不能在第 2 日收盤價成交；第 3 日開盤才買進。
        self.assertAlmostEqual(result['stock_value'].iloc[1], 0.0)
        self.assertAlmostEqual(result['stock_value'].iloc[2], 80.0)
        self.assertEqual(int(result['trades'].iloc[-1]), 2)

    def test_timing_signal_executes_next_day_open_not_same_day(self):
        idx = pd.date_range('2024-01-02', periods=3, freq='B')
        data = pd.DataFrame({
            'open': [10.0, 20.0, 30.0],
            'close': [10.0, 20.0, 30.0],
            'high': [10.0, 20.0, 30.0],
            'low': [10.0, 20.0, 30.0],
        }, index=idx)
        signal = pd.Series([1, 0, 0], index=idx)
        result = BacktestEngine.run_timing_strategy(
            data, signal, fee=0, tax=0, slip=0,
            init_capital=1000, monthly_inflow=0
        )
        # 第一天不能使用當天收盤後才形成的訊號；第二天開盤買進，第三天開盤賣出。
        self.assertAlmostEqual(result['portfolio_value'].iloc[0], 1000.0)
        self.assertAlmostEqual(result['portfolio_value'].iloc[-1], 1500.0)
        self.assertEqual(int(result['trades'].iloc[-1]), 2)

    def test_flat_market_rsi_is_not_oversold(self):
        idx = pd.date_range('2024-01-01', periods=40, freq='B')
        data = pd.DataFrame({'close': np.full(40, 100.0)}, index=idx)
        signal = IndicatorCalculator.generate_signal(
            data, 'RSI 超買超賣', {'period': 14, 'buy': 30, 'sell': 70}
        )
        self.assertTrue((signal == 0).all())

    def test_invalid_ohlc_is_rejected(self):
        idx = pd.date_range('2024-01-01', periods=2, freq='B')
        data = pd.DataFrame({
            'open': [10.0, 10.0], 'close': [10.0, 10.0],
            'high': [9.0, 11.0], 'low': [8.0, 9.0],
        }, index=idx)
        with self.assertRaises(ValueError):
            BacktestEngine._validate_price_columns(data, require_open=True)

    def test_xirr_known_result(self):
        dates = pd.to_datetime(['2023-01-01', '2024-01-01'])
        # 365 天期間，投入 100、收回 110，XIRR 應約為 10%。
        value = PerformanceEvaluator._calculate_xirr([-100.0, 110.0], dates)
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(value, 0.10, delta=0.002)

    def test_fx_open_uses_prior_known_rate(self):
        idx = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04'])
        prices = pd.DataFrame({
            'open': [10.0, 10.0, 10.0], 'close': [10.0, 10.0, 10.0],
            'high': [10.0, 10.0, 10.0], 'low': [10.0, 10.0, 10.0],
            'dividend': [0.0, 0.0, 0.0], 'split': [0.0, 0.0, 0.0],
        }, index=idx)
        fx_idx = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'])
        fx = pd.Series([30.0, 31.0, 32.0, 33.0], index=fx_idx)
        converted = DataFetcher._convert_prices_to_target_currency(prices, fx, 'TWD')
        self.assertAlmostEqual(converted.loc[idx[0], 'open'], 300.0)
        self.assertAlmostEqual(converted.loc[idx[0], 'close'], 310.0)
        self.assertAlmostEqual(converted.loc[idx[1], 'open'], 310.0)
        self.assertAlmostEqual(converted.loc[idx[1], 'close'], 320.0)

    def test_fx_conversion_drops_rows_without_prior_rate(self):
        idx = pd.to_datetime(['2024-01-02', '2024-01-03'])
        prices = pd.DataFrame({
            'open': [10.0, 10.0], 'close': [10.0, 10.0],
            'high': [10.0, 10.0], 'low': [10.0, 10.0], 'dividend': [0.0, 0.0]
        }, index=idx)
        fx = pd.Series([30.0, 31.0], index=pd.to_datetime(['2024-01-02', '2024-01-03']))
        converted = DataFetcher._convert_prices_to_target_currency(prices, fx, 'TWD')
        self.assertEqual(list(converted.index), [idx[1]])

    def test_mocked_data_pipeline_uses_safe_fx_timing(self):
        stock_idx = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04'])
        fx_idx = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'])
        stock = pd.DataFrame({
            'Open': [10.0, 10.0, 10.0], 'High': [10.5, 10.5, 10.5],
            'Low': [9.5, 9.5, 9.5], 'Close': [10.0, 10.0, 10.0],
            'Dividends': [0.0, 0.0, 0.0], 'Stock Splits': [0.0, 0.0, 0.0]
        }, index=stock_idx)
        fx_frame = pd.DataFrame({'Close': [30.0, 31.0, 32.0, 33.0]}, index=fx_idx)
        class FakeYahoo:
            @staticmethod
            def download(tickers, **kwargs):
                return fx_frame.copy() if tickers == 'USDTWD=X' else stock.copy()
        namespace['yf'] = FakeYahoo
        result, symbol, warnings = DataFetcher.load_data_and_fx(
            ['QQQ'], pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-05'),
            auto_adjust=True, user_curr='新台幣 (TWD)'
        )
        self.assertEqual(symbol, 'NT$')
        self.assertIn('QQQ', result)
        self.assertAlmostEqual(result['QQQ'].loc[stock_idx[0], 'open'], 300.0)
        self.assertAlmostEqual(result['QQQ'].loc[stock_idx[0], 'close'], 310.0)
        self.assertTrue(any('前一筆可用匯率' in w for w in warnings))

    def test_dividend_is_paid_on_existing_shares_only(self):
        idx = pd.date_range('2024-01-02', periods=2, freq='B')
        data = pd.DataFrame({'close': [10.0, 10.0], 'dividend': [1.0, 1.0], 'split': [0.0, 0.0]}, index=idx)
        shares, cash = BacktestEngine._apply_corporate_actions(data, 0, 2.0, 5.0)
        self.assertEqual(shares, 2.0)
        self.assertEqual(cash, 7.0)
        shares, cash = BacktestEngine._apply_corporate_actions(data, 1, 0.0, 5.0)
        self.assertEqual(shares, 0.0)
        self.assertEqual(cash, 5.0)

    def test_performance_rejects_unreconciled_ledger(self):
        idx = pd.date_range('2024-01-02', periods=2, freq='B')
        frame = pd.DataFrame({
            'portfolio_value': [100.0, 110.0], 'stock_value': [100.0, 110.0],
            'cash': [0.0, 0.0], 'total_invested': [100.0, 100.0],
            'equity_net': [1.0, 1.1], 'external_flow': [0.0, 0.0], 'trades': [1, 1]
        }, index=idx)
        frame.loc[idx[1], 'portfolio_value'] = 120.0
        self.assertEqual(PerformanceEvaluator.evaluate(frame, '$', initial_capital=100.0), {})

    def test_xirr_rejects_flows_without_sign_change(self):
        dates = pd.to_datetime(['2023-01-01', '2024-01-01'])
        value = PerformanceEvaluator._calculate_xirr([-100.0, -10.0], dates)
        self.assertTrue(np.isnan(value))


    def test_xirr_rejects_multiple_roots_cashflows(self):
        # -100, +230, -132 對應常見多重 IRR 情境，不應任意回傳其中一個解。
        dates = pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01'])
        value = PerformanceEvaluator._calculate_xirr([-100.0, 230.0, -132.0], dates)
        self.assertTrue(np.isnan(value))

    def test_xirr_rejects_unsorted_dates(self):
        dates = pd.to_datetime(['2024-01-01', '2023-01-01'])
        value = PerformanceEvaluator._calculate_xirr([-100.0, 110.0], dates)
        self.assertTrue(np.isnan(value))

    def test_xirr_rejects_mismatched_lengths_and_nonfinite_values(self):
        dates = pd.to_datetime(['2023-01-01', '2024-01-01'])
        self.assertTrue(np.isnan(PerformanceEvaluator._calculate_xirr([-100.0], dates)))
        self.assertTrue(np.isnan(PerformanceEvaluator._calculate_xirr([-100.0, np.inf], dates)))

    def test_split_event_does_not_double_adjust_split_adjusted_history(self):
        idx = pd.date_range('2024-01-02', periods=1, freq='B')
        data = pd.DataFrame({'close': [50.0], 'dividend': [0.0], 'split': [4.0]}, index=idx)
        shares, cash = BacktestEngine._apply_corporate_actions(data, 0, 2.0, 10.0)
        self.assertEqual(shares, 2.0)
        self.assertEqual(cash, 10.0)

    def test_zero_initial_capital_with_contribution_keeps_twr_neutral(self):
        idx = pd.date_range('2024-01-30', periods=4, freq='B')
        close = [100.0, 100.0, 100.0, 100.0]
        data = pd.DataFrame({'open': close, 'close': close, 'high': close, 'low': close,
                             'dividend': 0.0, 'split': 0.0}, index=idx)
        result = BacktestEngine.run_timing_strategy(
            data, pd.Series([0, 1, 1, 1], index=idx), 0.0, 0.0, 0.0, 0.0, 100.0
        )
        self.assertTrue(np.allclose(result['equity_net'].to_numpy(), 1.0))
        self.assertTrue(np.allclose(result['portfolio_value'], result['cash'] + result['stock_value']))


    def test_strategy_rejects_invalid_sma_periods(self):
        idx = pd.date_range('2024-01-01', periods=10, freq='B')
        data = pd.DataFrame({'close': np.arange(10, dtype=float) + 100}, index=idx)
        with self.assertRaises(ValueError):
            IndicatorCalculator.generate_signal(data, '價格突破均線買入/跌破賣出', {'period': 0})
        with self.assertRaises(ValueError):
            IndicatorCalculator.generate_signal(data, 'SMA 雙均線交叉', {'fast': 20, 'slow': 5})

    def test_strategy_rejects_invalid_rsi_thresholds(self):
        idx = pd.date_range('2024-01-01', periods=20, freq='B')
        data = pd.DataFrame({'close': np.arange(20, dtype=float) + 100}, index=idx)
        with self.assertRaises(ValueError):
            IndicatorCalculator.generate_signal(data, 'RSI 超買超賣', {'period': 14, 'buy': 80, 'sell': 20})

    def test_strategy_rejects_unknown_name(self):
        idx = pd.date_range('2024-01-01', periods=3, freq='B')
        data = pd.DataFrame({'close': [100.0, 101.0, 102.0]}, index=idx)
        with self.assertRaises(ValueError):
            IndicatorCalculator.generate_signal(data, '未知策略', {})

    def test_engine_rejects_invalid_costs_even_without_trades(self):
        idx = pd.date_range('2024-01-01', periods=3, freq='B')
        data = pd.DataFrame({'open': [100.0]*3, 'close': [100.0]*3,
                             'high': [100.0]*3, 'low': [100.0]*3}, index=idx)
        with self.assertRaises(ValueError):
            BacktestEngine.run_timing_strategy(data, pd.Series([0, 0, 0], index=idx),
                                               fee=-0.1, tax=0, slip=0,
                                               init_capital=1000, monthly_inflow=0)

    def test_engine_rejects_invalid_dca_amount(self):
        idx = pd.date_range('2024-01-01', periods=3, freq='B')
        data = pd.DataFrame({'close': [100.0]*3}, index=idx)
        with self.assertRaises(ValueError):
            BacktestEngine.run_dca_strategy(data, 1000, 0, '每月 (Monthly)', -1, 5, 0, 0)

    def test_price_validator_requires_datetime_index(self):
        data = pd.DataFrame({'close': [100.0, 101.0]})
        with self.assertRaises(ValueError):
            BacktestEngine._validate_price_columns(data)


    def test_randomized_paths_preserve_ledger_invariants(self):
        rng = np.random.default_rng(20261026)
        for case in range(50):
            n = int(rng.integers(30, 160))
            idx = pd.bdate_range('2022-01-03', periods=n)
            returns = rng.normal(0.0002, 0.018, n)
            close = 100.0 * np.exp(np.cumsum(returns))
            open_price = close * np.exp(rng.normal(0.0, 0.006, n))
            high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.0, 0.02, n))
            low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.0, 0.02, n))
            data = pd.DataFrame({'open': open_price, 'close': close, 'high': high, 'low': low,
                                 'dividend': 0.0, 'split': 0.0}, index=idx)
            signal = pd.Series(rng.integers(0, 2, n), index=idx)
            fee, tax, slip = rng.uniform(0, 0.003, 3)
            result = BacktestEngine.run_timing_strategy(
                data, signal, float(fee), float(tax), float(slip),
                float(rng.uniform(0, 100000)), float(rng.uniform(0, 5000))
            )
            with self.subTest(case=case):
                self.assertTrue(np.isfinite(result[['portfolio_value', 'stock_value', 'cash', 'equity_net']].to_numpy()).all())
                self.assertTrue(np.allclose(result['portfolio_value'], result['stock_value'] + result['cash'], atol=1e-6))
                self.assertGreaterEqual(float(result['cash'].min()), -1e-6)
                self.assertGreaterEqual(float(result['stock_value'].min()), -1e-6)
                self.assertTrue(PerformanceEvaluator.evaluate(result, '$', initial_capital=float(result['total_invested'].iloc[0])).get('總報酬率 (TWR)'))

    def test_higher_transaction_costs_do_not_improve_buy_hold_final_value(self):
        idx = pd.bdate_range('2023-01-02', periods=80)
        close = np.linspace(100.0, 130.0, len(idx))
        data = pd.DataFrame({'close': close}, index=idx)
        free = BacktestEngine.run_dca_strategy(data, 10000, 0, '每月 (Monthly)', 0, 20, 0, 0)
        costly = BacktestEngine.run_dca_strategy(data, 10000, 0, '每月 (Monthly)', 0, 20, 0.01, 0.01)
        self.assertLessEqual(float(costly['portfolio_value'].iloc[-1]), float(free['portfolio_value'].iloc[-1]))

    def test_cashflow_twr_and_accounting_invariant_under_monthly_contributions(self):
        idx = pd.bdate_range('2024-01-29', periods=90)
        close = np.full(len(idx), 100.0)
        data = pd.DataFrame({'open': close, 'close': close, 'high': close, 'low': close}, index=idx)
        signal = pd.Series(1, index=idx)
        result = BacktestEngine.run_timing_strategy(data, signal, 0, 0, 0, 5000, 1000)
        self.assertTrue(np.allclose(result['equity_net'].to_numpy(), 1.0))
        self.assertTrue(np.allclose(result['portfolio_value'], result['stock_value'] + result['cash']))
        self.assertTrue(np.allclose(result['total_invested'], 5000 + result['external_flow'].cumsum()))


    def test_data_quality_flags_large_price_move_without_dropping_data(self):
        idx = pd.bdate_range('2024-01-01', periods=5)
        data = pd.DataFrame({'open': [100, 101, 102, 150, 151],
                             'close': [100, 101, 102, 150, 151],
                             'high': [100, 101, 102, 150, 151],
                             'low': [100, 101, 102, 150, 151]}, index=idx)
        warnings = DataFetcher.data_quality_warnings(data, 'TEST', min_rows=3, extreme_return=0.35)
        self.assertTrue(any('單日收盤價變動' in w for w in warnings))
        self.assertEqual(len(data), 5)

    def test_data_quality_flags_long_data_gap(self):
        idx = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-20'])
        data = pd.DataFrame({'close': [100.0, 101.0, 102.0]}, index=idx)
        warnings = DataFetcher.data_quality_warnings(data, 'GAP', min_rows=2, gap_days=10)
        self.assertTrue(any('行情間隔' in w for w in warnings))

    def test_data_quality_clean_data_has_no_warnings_when_thresholds_satisfied(self):
        idx = pd.bdate_range('2024-01-01', periods=5)
        data = pd.DataFrame({'open': [100, 101, 102, 103, 104],
                             'close': [100, 101, 102, 103, 104],
                             'high': [100, 101, 102, 103, 104],
                             'low': [100, 101, 102, 103, 104]}, index=idx)
        warnings = DataFetcher.data_quality_warnings(data, 'CLEAN', min_rows=5)
        self.assertEqual(warnings, [])

    def test_all_strategy_routes_produce_valid_accounting(self):
        idx = pd.date_range('2023-01-02', periods=120, freq='B')
        close = 100.0 * np.exp(np.cumsum(np.sin(np.arange(120) / 8.0) * 0.005 + 0.0005))
        data = pd.DataFrame({
            'open': close * (1 + np.sin(np.arange(120)) * 0.002),
            'close': close, 'high': close * 1.01, 'low': close * 0.99,
            'dividend': 0.0, 'split': 0.0
        }, index=idx)
        cases = [
            ('DCA', BacktestEngine.run_dca_strategy(data, 10000, 500, '每月 (Monthly)', 500, 20, .001, .001)),
            ('DipDCA', BacktestEngine.run_dip_dca(data, 10000, 500,
                {'sma_period': 20, 'timeframe': '日線 (Daily)', 'buy_amount': 300, 'min_interval': 5}, .001, .001)),
        ]
        signal_cases = [
            ('價格突破均線買入/跌破賣出', {'period': 20, 'timeframe': '日線 (Daily)'}),
            ('SMA 雙均線交叉', {'fast': 5, 'slow': 20, 'timeframe': '日線 (Daily)'}),
            ('RSI 超買超賣', {'period': 14, 'buy': 30, 'sell': 70}),
            ('MACD 柱狀圖/交叉', {'fast': 12, 'slow': 26, 'signal': 9}),
            ('KD 隨機指標', {'period': 9, 'buy_k': 20, 'sell_k': 80}),
        ]
        for name, params in signal_cases:
            signal = IndicatorCalculator.generate_signal(data, name, params)
            result = BacktestEngine.run_timing_strategy(data, signal, .001, .001, .001, 10000, 500)
            cases.append((name, result))
        for name, result in cases:
            with self.subTest(strategy=name):
                self.assertEqual(len(result), len(data))
                self.assertTrue(np.isfinite(result[['portfolio_value', 'cash', 'stock_value', 'equity_net']].to_numpy()).all())
                self.assertTrue(np.allclose(result['portfolio_value'], result['cash'] + result['stock_value']))
                self.assertGreaterEqual(float(result['cash'].min()), -1e-6)
                metrics = PerformanceEvaluator.evaluate(result, '$', initial_capital=10000)
                self.assertTrue(metrics, f'{name} failed performance validation')

    def test_flat_market_kd_is_not_oversold(self):
        idx = pd.date_range('2024-01-01', periods=30, freq='B')
        data = pd.DataFrame({
            'close': np.full(30, 100.0), 'high': np.full(30, 100.0),
            'low': np.full(30, 100.0)
        }, index=idx)
        signal = IndicatorCalculator.generate_signal(
            data, 'KD 隨機指標', {'period': 9, 'buy_k': 20, 'sell_k': 80}
        )
        self.assertTrue((signal == 0).all())

    def test_performance_rejects_inconsistent_invested_cashflow(self):
        idx = pd.date_range('2024-01-02', periods=2, freq='B')
        frame = pd.DataFrame({
            'portfolio_value': [100.0, 110.0], 'stock_value': [100.0, 110.0],
            'cash': [0.0, 0.0], 'total_invested': [100.0, 150.0],
            'equity_net': [1.0, 1.1], 'external_flow': [0.0, 0.0], 'trades': [1, 1]
        }, index=idx)
        self.assertEqual(PerformanceEvaluator.evaluate(frame, '$', initial_capital=100.0), {})

    def test_performance_rejects_noninteger_trade_count(self):
        idx = pd.date_range('2024-01-02', periods=2, freq='B')
        frame = pd.DataFrame({
            'portfolio_value': [100.0, 110.0], 'stock_value': [100.0, 110.0],
            'cash': [0.0, 0.0], 'total_invested': [100.0, 100.0],
            'equity_net': [1.0, 1.1], 'external_flow': [0.0, 0.0], 'trades': [1.0, 1.5]
        }, index=idx)
        self.assertEqual(PerformanceEvaluator.evaluate(frame, '$', initial_capital=100.0), {})



    def test_holdout_period_returns_zero_for_flat_market(self):
        idx = pd.date_range('2020-01-01', periods=100, freq='B')
        frame = pd.DataFrame({'equity_net': np.ones(100)}, index=idx)
        metrics = PerformanceEvaluator.evaluate_holdout_period(frame, 0.70)
        self.assertTrue(metrics)
        self.assertAlmostEqual(metrics['驗證區間 TWR'], 0.0, places=10)
        self.assertAlmostEqual(metrics['驗證區間年化 TWR'], 0.0, places=10)
        self.assertAlmostEqual(metrics['驗證區間 MDD'], 0.0, places=10)

    def test_holdout_period_uses_equity_ratio_not_cash_inflows(self):
        idx = pd.date_range('2020-01-01', periods=100, freq='B')
        equity = np.ones(100)
        equity[70:] = np.linspace(1.0, 1.2, 30)
        frame = pd.DataFrame({'equity_net': equity}, index=idx)
        metrics = PerformanceEvaluator.evaluate_holdout_period(frame, 0.70)
        self.assertAlmostEqual(metrics['驗證區間 TWR'], 0.2, places=8)

    def test_holdout_period_includes_first_validation_day_return(self):
        idx = pd.date_range('2020-01-01', periods=100, freq='B')
        equity = np.ones(100)
        equity[70:] = 1.1
        metrics = PerformanceEvaluator.evaluate_holdout_period(pd.DataFrame({'equity_net': equity}, index=idx), 0.70)
        self.assertAlmostEqual(metrics['驗證區間 TWR'], 0.1, places=8)
        equity[69] = 1.0
        equity[70] = 1.05
        metrics = PerformanceEvaluator.evaluate_holdout_period(pd.DataFrame({'equity_net': equity}, index=idx), 0.70)
        self.assertAlmostEqual(metrics['驗證區間 TWR'], 0.1, places=8)

    def test_holdout_period_rejects_bad_ratio_or_short_data(self):
        idx = pd.date_range('2020-01-01', periods=8, freq='B')
        frame = pd.DataFrame({'equity_net': np.ones(8)}, index=idx)
        self.assertEqual(PerformanceEvaluator.evaluate_holdout_period(frame, 0.2), {})
        self.assertEqual(PerformanceEvaluator.evaluate_holdout_period(frame, 0.7), {})

    def test_holdout_period_rejects_nonfinite_or_invalid_equity(self):
        idx = pd.date_range('2020-01-01', periods=100, freq='B')
        frame = pd.DataFrame({'equity_net': np.ones(100)}, index=idx)
        frame.iloc[80, 0] = np.nan
        self.assertEqual(PerformanceEvaluator.evaluate_holdout_period(frame), {})
        frame.iloc[80, 0] = -1.0
        self.assertEqual(PerformanceEvaluator.evaluate_holdout_period(frame), {})


    def test_benchmark_comparison_identical_paths_has_zero_excess(self):
        idx = pd.date_range('2024-01-01', periods=4, freq='B')
        frame = pd.DataFrame({'equity_net': [1.0, 1.1, 1.05, 1.2]}, index=idx)
        result = PerformanceEvaluator.compare_to_benchmark(frame, frame.copy())
        self.assertAlmostEqual(result['benchmark_twr'], 0.2)
        self.assertAlmostEqual(result['excess_twr'], 0.0)

    def test_benchmark_comparison_detects_outperformance(self):
        idx = pd.date_range('2024-01-01', periods=3, freq='B')
        strategy = pd.DataFrame({'equity_net': [1.0, 1.1, 1.25]}, index=idx)
        benchmark = pd.DataFrame({'equity_net': [1.0, 1.05, 1.1]}, index=idx)
        result = PerformanceEvaluator.compare_to_benchmark(strategy, benchmark)
        self.assertAlmostEqual(result['excess_twr'], 0.15)

    def test_benchmark_comparison_rejects_mismatched_dates(self):
        idx1 = pd.date_range('2024-01-01', periods=3, freq='B')
        idx2 = pd.date_range('2024-01-02', periods=3, freq='B')
        strategy = pd.DataFrame({'equity_net': [1.0, 1.1, 1.2]}, index=idx1)
        benchmark = pd.DataFrame({'equity_net': [1.0, 1.1, 1.2]}, index=idx2)
        self.assertEqual(PerformanceEvaluator.compare_to_benchmark(strategy, benchmark), {})

    def test_benchmark_comparison_rejects_nonfinite_values(self):
        idx = pd.date_range('2024-01-01', periods=3, freq='B')
        strategy = pd.DataFrame({'equity_net': [1.0, np.nan, 1.2]}, index=idx)
        benchmark = pd.DataFrame({'equity_net': [1.0, 1.1, 1.2]}, index=idx)
        self.assertEqual(PerformanceEvaluator.compare_to_benchmark(strategy, benchmark), {})

    def test_benchmark_comparison_excludes_initial_capital_scale(self):
        idx = pd.date_range('2024-01-01', periods=3, freq='B')
        strategy = pd.DataFrame({'equity_net': [0.98, 1.1, 1.2]}, index=idx)
        benchmark = pd.DataFrame({'equity_net': [0.99, 1.05, 1.1]}, index=idx)
        result = PerformanceEvaluator.compare_to_benchmark(strategy, benchmark)
        self.assertAlmostEqual(result['excess_twr'], 0.1)


    def test_named_strategy_router_runs_dca(self):
        idx = pd.date_range('2024-01-01', periods=5, freq='B')
        data = pd.DataFrame({'open': 100.0, 'close': 100.0, 'high': 100.0, 'low': 100.0,
                             'dividend': 0.0, 'split': 0.0}, index=idx)
        params = {'DCA': {'freq': '每月 (Monthly)', 'val': 100.0, 'custom_days': 20}}
        result = BacktestEngine.run_named_strategy(
            data, '定期定額 / 買入持有 (DCA / Buy & Hold)', params, 1000, 0, 0, 0, 0
        )
        self.assertAlmostEqual(result['portfolio_value'].iloc[-1], 1000.0)

    def test_named_strategy_router_rejects_invalid_costs(self):
        idx = pd.date_range('2024-01-01', periods=3, freq='B')
        data = pd.DataFrame({'open': 100.0, 'close': 100.0, 'high': 100.0, 'low': 100.0}, index=idx)
        params = {'SMA': {'fast': 2, 'slow': 3}}
        with self.assertRaises(ValueError):
            BacktestEngine.run_named_strategy(data, 'SMA 雙均線交叉', params, 1000, 0, 1.0, 0, 0)

    def test_cost_sensitivity_higher_costs_do_not_improve_dca_value(self):
        idx = pd.date_range('2024-01-01', periods=20, freq='B')
        prices = np.linspace(100, 120, len(idx))
        data = pd.DataFrame({'open': prices, 'close': prices, 'high': prices, 'low': prices,
                             'dividend': 0.0, 'split': 0.0}, index=idx)
        params = {'DCA': {'freq': '每月 (Monthly)', 'val': 100.0, 'custom_days': 20}}
        base = BacktestEngine.run_named_strategy(data, '定期定額 / 買入持有 (DCA / Buy & Hold)', params, 1000, 100, 0.001, 0, 0.001)
        high = BacktestEngine.run_named_strategy(data, '定期定額 / 買入持有 (DCA / Buy & Hold)', params, 1000, 100, 0.002, 0, 0.002)
        self.assertLessEqual(high['portfolio_value'].iloc[-1], base['portfolio_value'].iloc[-1] + 1e-8)


    def test_rolling_periods_returns_expected_number_of_segments(self):
        idx = pd.date_range('2020-01-01', periods=100, freq='B')
        equity = np.linspace(1.0, 1.5, 100)
        result = PerformanceEvaluator.evaluate_rolling_periods(pd.DataFrame({'equity_net': equity}, index=idx), 4)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[-1]['區間 TWR'], equity[-1] / equity[74] - 1.0)

    def test_rolling_periods_first_segment_includes_initial_cost(self):
        idx = pd.date_range('2020-01-01', periods=40, freq='B')
        equity = np.full(40, 0.98)
        equity[0] = 0.98
        result = PerformanceEvaluator.evaluate_rolling_periods(pd.DataFrame({'equity_net': equity}, index=idx), 4)
        self.assertAlmostEqual(result[0]['區間 TWR'], -0.02)
        self.assertAlmostEqual(result[0]['區間 MDD'], -0.02)

    def test_rolling_periods_flat_market_is_zero_after_first_day(self):
        idx = pd.date_range('2020-01-01', periods=80, freq='B')
        result = PerformanceEvaluator.evaluate_rolling_periods(pd.DataFrame({'equity_net': np.ones(80)}, index=idx), 4)
        self.assertTrue(all(abs(row['區間 TWR']) < 1e-12 for row in result))

    def test_rolling_periods_rejects_bad_data(self):
        idx = pd.date_range('2020-01-01', periods=40, freq='B')
        frame = pd.DataFrame({'equity_net': np.ones(40)}, index=idx)
        self.assertEqual(PerformanceEvaluator.evaluate_rolling_periods(frame, 1), [])
        frame.iloc[10, 0] = np.nan
        self.assertEqual(PerformanceEvaluator.evaluate_rolling_periods(frame, 4), [])

    def test_rolling_periods_rejects_duplicate_dates(self):
        idx = pd.to_datetime(['2024-01-01'] * 40)
        frame = pd.DataFrame({'equity_net': np.ones(40)}, index=idx)
        self.assertEqual(PerformanceEvaluator.evaluate_rolling_periods(frame, 4), [])


    def test_benchmark_comparison_mdd_includes_first_day_cost(self):
        idx = pd.date_range('2024-01-01', periods=3, freq='B')
        strategy = pd.DataFrame({'equity_net': [0.98, 0.99, 1.02]}, index=idx)
        benchmark = pd.DataFrame({'equity_net': [0.99, 1.00, 1.01]}, index=idx)
        result = PerformanceEvaluator.compare_to_benchmark(strategy, benchmark)
        self.assertAlmostEqual(result['strategy_mdd'], -0.02)
        self.assertAlmostEqual(result['benchmark_mdd'], -0.01)


    def test_xirr_aggregates_same_day_terminal_value_and_contribution(self):
        dates = pd.to_datetime(['2024-01-01', '2024-12-31', '2024-12-31'])
        result = PerformanceEvaluator._calculate_xirr([-100.0, 120.0, -20.0], dates)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_xirr_same_day_cashflow_order_does_not_change_result(self):
        dates_a = pd.to_datetime(['2024-01-01', '2024-12-31', '2024-12-31'])
        dates_b = pd.to_datetime(['2024-01-01', '2024-12-31', '2024-12-31'])
        result_a = PerformanceEvaluator._calculate_xirr([-100.0, 120.0, -20.0], dates_a)
        result_b = PerformanceEvaluator._calculate_xirr([-100.0, -20.0, 120.0], dates_b)
        self.assertAlmostEqual(result_a, result_b, places=8)

    def test_xirr_same_day_flows_can_cancel_to_zero(self):
        dates = pd.to_datetime(['2024-01-01', '2024-12-31', '2024-12-31'])
        result = PerformanceEvaluator._calculate_xirr([-100.0, 20.0, -20.0], dates)
        self.assertTrue(np.isnan(result))


    def test_risk_metrics_flat_market_has_zero_volatility_and_no_ratios(self):
        idx = pd.date_range('2024-01-01', periods=10, freq='B')
        result = PerformanceEvaluator.calculate_risk_metrics(pd.DataFrame({'equity_net': np.ones(10)}, index=idx))
        self.assertAlmostEqual(result['annualized_volatility'], 0.0)
        self.assertTrue(np.isnan(result['sharpe_ratio']))
        self.assertTrue(np.isnan(result['sortino_ratio']))

    def test_risk_metrics_include_first_day_transaction_cost(self):
        idx = pd.date_range('2024-01-01', periods=4, freq='B')
        equity = np.array([0.98, 0.98, 0.98, 0.98])
        result = PerformanceEvaluator.calculate_risk_metrics(pd.DataFrame({'equity_net': equity}, index=idx))
        expected = np.array([-0.02, 0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(result['daily_twr_returns'], expected))
        self.assertGreater(result['annualized_volatility'], 0.0)

    def test_risk_metrics_reject_nonfinite_or_invalid_equity(self):
        idx = pd.date_range('2024-01-01', periods=4, freq='B')
        frame = pd.DataFrame({'equity_net': [1.0, 1.1, np.nan, 1.2]}, index=idx)
        self.assertEqual(PerformanceEvaluator.calculate_risk_metrics(frame), {})
        frame['equity_net'] = [1.0, 0.0, 0.0, 1.2]
        self.assertEqual(PerformanceEvaluator.calculate_risk_metrics(frame), {})

    def test_risk_metrics_reject_invalid_risk_free_rate(self):
        idx = pd.date_range('2024-01-01', periods=4, freq='B')
        frame = pd.DataFrame({'equity_net': [1.0, 1.01, 1.02, 1.03]}, index=idx)
        self.assertEqual(PerformanceEvaluator.calculate_risk_metrics(frame, -1.0), {})
        self.assertEqual(PerformanceEvaluator.calculate_risk_metrics(frame, np.nan), {})

    def test_risk_metrics_known_sharpe_calculation(self):
        idx = pd.date_range('2024-01-01', periods=4, freq='B')
        returns = np.array([0.01, -0.01, 0.02, 0.0])
        equity = np.cumprod(1 + returns)
        frame = pd.DataFrame({'equity_net': equity}, index=idx)
        result = PerformanceEvaluator.calculate_risk_metrics(frame)
        self.assertAlmostEqual(result['sharpe_ratio'], np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252), places=8)


    def test_dca_rejects_unknown_frequency(self):
        idx = pd.date_range('2024-01-01', periods=3, freq='B')
        data = pd.DataFrame({'open': 100.0, 'close': 100.0, 'high': 100.0, 'low': 100.0}, index=idx)
        with self.assertRaises(ValueError):
            BacktestEngine.run_dca_strategy(data, 1000, 0, 'unknown', 100, 10, 0, 0)

    def test_dca_rejects_noninteger_custom_days(self):
        idx = pd.date_range('2024-01-01', periods=3, freq='B')
        data = pd.DataFrame({'open': 100.0, 'close': 100.0, 'high': 100.0, 'low': 100.0}, index=idx)
        with self.assertRaises(ValueError):
            BacktestEngine.run_dca_strategy(data, 1000, 0, '自訂交易日數', 100, 1.5, 0, 0)

    def test_dip_strategy_rejects_invalid_period_or_amount(self):
        idx = pd.date_range('2024-01-01', periods=4, freq='B')
        data = pd.DataFrame({'open': 100.0, 'close': [100, 90, 80, 70], 'high': 100.0, 'low': [100, 90, 80, 70]}, index=idx)
        base = {'sma_period': 2, 'timeframe': '日線 (Daily)', 'buy_amount': 100, 'min_interval': 1}
        with self.assertRaises(ValueError):
            BacktestEngine.run_dip_dca(data, 1000, 0, {**base, 'sma_period': 1.5}, 0, 0)
        with self.assertRaises(ValueError):
            BacktestEngine.run_dip_dca(data, 1000, 0, {**base, 'buy_amount': np.nan}, 0, 0)

    def test_timing_strategy_rejects_nonbinary_signal(self):
        idx = pd.date_range('2024-01-01', periods=3, freq='B')
        data = pd.DataFrame({'open': 100.0, 'close': 100.0, 'high': 100.0, 'low': 100.0}, index=idx)
        signal = pd.Series([0, 0.5, 1], index=idx)
        with self.assertRaises(ValueError):
            BacktestEngine.run_timing_strategy(data, signal, 0, 0, 0, 1000, 0)

    def test_timing_strategy_rejects_infinite_signal(self):
        idx = pd.date_range('2024-01-01', periods=3, freq='B')
        data = pd.DataFrame({'open': 100.0, 'close': 100.0, 'high': 100.0, 'low': 100.0}, index=idx)
        signal = pd.Series([0, np.inf, 1], index=idx)
        with self.assertRaises(ValueError):
            BacktestEngine.run_timing_strategy(data, signal, 0, 0, 0, 1000, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
