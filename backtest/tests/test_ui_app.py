"""
Streamlit 端到端介面與互動測試 (AppTest E2E Suite)
測試 app.py 與雙核心分頁在真實 Streamlit 1.67 + Plotly 7.1 執行環境下的所有互動操作。
"""
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.data_engine import DataEngine
from strategies.registry import StrategyRegistry


def _make_market_df(ticker: str, seed: int, periods: int = 420) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=periods)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.014, len(dates)))
    open_p = close * 0.998
    high_p = np.maximum(open_p, close) * 1.01
    low_p = np.minimum(open_p, close) * 0.99
    curr = "TWD" if DataEngine.is_taiwan_ticker(ticker) else "USD"
    return pd.DataFrame(
        {
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close,
            "adj_open": open_p,
            "adj_high": high_p,
            "adj_low": low_p,
            "adj_close": close,
            "dividend": np.where(np.arange(len(dates)) % 60 == 0, 0.8, 0.0),
            "split": 0.0,
            "is_trading_day": True,
            "quote_currency": curr,
            "user_start_idx": dates >= pd.Timestamp("2022-09-01"),
        },
        index=dates,
    )


def _fake_load_aligned_market_data(tickers, start_date, end_date, target_currency="TWD"):
    indiv = {}
    for idx, t in enumerate(tickers):
        indiv[t] = _make_market_df(t, seed=100 + idx)

    panel_dict = {}
    for t, df in indiv.items():
        sub = df[df["user_start_idx"] == True].drop(columns=["user_start_idx"])
        panel_dict[t] = sub
    aligned_panel = pd.concat(panel_dict, axis=1)
    fx_idx = next(iter(indiv.values())).index
    fx_series = pd.Series(32.0, index=fx_idx)
    curr_sym = "NT$" if "TWD" in target_currency.upper() else "$"
    return indiv, aligned_panel, fx_series, curr_sym, []


def _find_button(at: AppTest, *, key: str = None, label_contains: str = None):
    for b in at.button:
        if key and b.key == key:
            return b
        if label_contains and label_contains in (b.label or ""):
            return b
    return None


def _find_selectbox(at: AppTest, *, key: str = None, label_contains: str = None):
    for sb in at.selectbox:
        if key and sb.key == key:
            return sb
        if label_contains and label_contains in (sb.label or ""):
            return sb
    return None


class TestStreamlitUI(unittest.TestCase):
    def _assert_clean_run(self, at: AppTest, context: str):
        excs = [str(e.value) for e in at.exception]
        errs = [str(e.value) for e in at.error]
        self.assertEqual(excs, [], f"[{context}] Uncaught exceptions: {excs}")
        self.assertEqual(errs, [], f"[{context}] UI errors: {errs}")

    def test_full_ui_scenarios_currency_and_both_tabs(self):
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            with patch("core.data_engine.DataEngine.load_aligned_market_data", side_effect=_fake_load_aligned_market_data):
                at = AppTest.from_file(str(ROOT_DIR / "app.py"), default_timeout=45)
                at.run()
                self._assert_clean_run(at, "initial load")
                self.assertGreaterEqual(len(at.tabs), 2)

                # 1. 切換一鍵情境範本
                for scenario in [
                    "小資定期定額 (0050 + QQQ)",
                    "退休股債全天候 (0050 + VT + 00679B)",
                    "積極科技成長 (0050 + 00631L + QQQ)",
                    "自由自訂模式",
                ]:
                    at.sidebar.selectbox(key="sb_scenario").set_value(scenario).run()
                    self._assert_clean_run(at, f"scenario={scenario}")

                # 2. 切換計價幣別 TWD <-> USD
                at.sidebar.selectbox(key="sb_currency").set_value("美元 (USD)").run()
                self._assert_clean_run(at, "currency=USD")
                at.sidebar.selectbox(key="sb_currency").set_value("新台幣 (TWD)").run()
                self._assert_clean_run(at, "currency=TWD")

                # 3. 模式 A：全策略同台 PK + 風控 + 對數座標 + 檢視帳本
                all_strats = StrategyRegistry.list_names()
                ms_strat = [m for m in at.multiselect if "參戰策略" in (m.label or "")][0]
                ms_strat.set_value(all_strats).run()
                self._assert_clean_run(at, "all 10 strategies")

                # 設定固定停損與移動停利
                sl_inputs = [n for n in at.number_input if (n.key or "").startswith("rk_sl_")]
                ts_inputs = [n for n in at.number_input if (n.key or "").startswith("rk_ts_")]
                if sl_inputs:
                    sl_inputs[0].set_value(8.0).run()
                    self._assert_clean_run(at, "stop loss 8%")
                if ts_inputs:
                    ts_inputs[0].set_value(12.0).run()
                    self._assert_clean_run(at, "trailing stop 12%")

                # 對數座標切換
                chk_log = [c for c in at.checkbox if c.key == "pk_log_scale"]
                if chk_log:
                    chk_log[0].set_value(True).run()
                    self._assert_clean_run(at, "pk log scale")

                # 帳本切換檢視 DCA 與 擇時策略
                sb_inspect = _find_selectbox(at, key="select_inspect_strat")
                self.assertIsNotNone(sb_inspect)
                for s_name in ["定期定額 / 買入持有 (DCA / Buy & Hold)", "布林通道策略 (Bollinger Bands)"]:
                    sb_inspect.set_value(s_name).run()
                    self._assert_clean_run(at, f"inspect={s_name}")

                # 4. 模式 A：執行 2D 參數尋優 (SMA, MACD, RSI, KD, Dip) 與三種目標，並套用最佳參數
                sb_opt = _find_selectbox(at, label_contains="選擇要執行 2D 尋優的策略")
                sb_metric = _find_selectbox(at, label_contains="尋優目標指標")
                self.assertIsNotNone(sb_opt)
                self.assertIsNotNone(sb_metric)

                grid_cases = [
                    ("SMA 雙均線交叉 (快線週期 × 慢線週期)", "總報酬率 (%)"),
                    ("MACD 柱狀圖/交叉 (EMA快線 × EMA慢線)", "夏普比率 (Sharpe)"),
                    ("RSI 超買超賣 (超賣買入門檻 × 超買賣出門檻)", "卡瑪比率 (Calmar)"),
                    ("KD 隨機指標 (超賣買入K × 超買賣出K)", "總報酬率 (%)"),
                    ("跌破均線且創新低買入 (基準均線 × 加碼間隔天數)", "夏普比率 (Sharpe)"),
                ]
                for opt_name, m_name in grid_cases:
                    sb_opt = _find_selectbox(at, key="pk_grid_opt_strategy")
                    sb_metric = _find_selectbox(at, key="pk_grid_metric_choice")
                    sb_opt.set_value(opt_name)
                    sb_metric.set_value(m_name)
                    at.run()
                    self._assert_clean_run(at, f"grid_select={opt_name}, metric={m_name}")
                    btn_grid = _find_button(at, key="btn_run_generic_grid")
                    self.assertIsNotNone(btn_grid)
                    btn_grid.click().run()
                    self._assert_clean_run(at, f"grid={opt_name}, metric={m_name}")

                btn_apply = _find_button(at, key="btn_apply_best_grid")
                self.assertIsNotNone(btn_apply)
                btn_apply.click().run()
                self._assert_clean_run(at, "apply best grid")

                # 5. 模式 B：快捷權重按鈕、現金儲備、再平衡週期、非 100% 防呆
                for btn_lbl in ["最高夏普", "最小波動", "風險平價", "等權重 1/N", "自動配平 100%"]:
                    b = _find_button(at, label_contains=btn_lbl)
                    self.assertIsNotNone(b, f"Missing button {btn_lbl}")
                    b.click().run()
                    self._assert_clean_run(at, f"portfolio button={btn_lbl}")

                # 設定現金儲備 15% 後自動配平 100%
                cash_w = [n for n in at.number_input if n.key == "weight_input_CASH_RESERVE"][0]
                cash_w.set_value(15.0).run()
                # 此時總和超過 100%，應僅跳 warning 且不崩潰
                self._assert_clean_run(at, "weights not 100% warning check")
                self.assertGreater(len(at.warning), 0)

                btn_bal = _find_button(at, label_contains="自動配平 100%")
                btn_bal.click().run()
                self._assert_clean_run(at, "rebalance with 15% cash reserve")

                # 切換為不進行再平衡 (Buy & Drift) 與偏差門檻 0%
                sb_rebal = _find_selectbox(at, label_contains="再平衡檢驗週期")
                sb_rebal.set_value("不進行再平衡 (Buy & Drift)").run()
                self._assert_clean_run(at, "rebal=Buy & Drift")
                sb_rebal.set_value("每季再平衡 (Quarterly)").run()
                self._assert_clean_run(at, "rebal=Quarterly")

                # 檢查是否出現未預期執行期警告 (如 Plotly add_vline / FutureWarning 等)
                dep_warns = [
                    f"{w.category.__name__}: {w.message}"
                    for w in w_list
                    if any(tag in str(w.message).lower() for tag in ("use_container_width", "add_vline", "futurewarning"))
                    or issubclass(w.category, FutureWarning)
                ]
                self._captured_warns = dep_warns

        # 若有任何嚴重警告將在測試報告中檢視；此處確保無 FutureWarning
        future_only = [w for w in self._captured_warns if w.startswith("FutureWarning")]
        self.assertEqual(future_only, [], f"Found FutureWarnings: {future_only}")

    def test_data_engine_two_otc_fallback(self):
        dates = pd.bdate_range("2024-01-02", periods=60)
        p = np.linspace(30.0, 32.0, len(dates))
        df_two = pd.DataFrame(
            {
                "Open": p,
                "High": p * 1.01,
                "Low": p * 0.99,
                "Close": p,
                "Adj Close": p,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            },
            index=dates,
        )
        df_fx = pd.DataFrame(
            {
                "Open": 32.0,
                "High": 32.1,
                "Low": 31.9,
                "Close": 32.0,
                "Adj Close": 32.0,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            },
            index=dates,
        )

        # 模擬批次下載時 00679B.TW 為空，fallback 抓取 00679B.TWO 成功
        def mock_yf_download(tickers, **kwargs):
            if isinstance(tickers, list):
                empty_tw = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close"])
                return pd.concat({"00679B.TW": empty_tw, "USDTWD=X": df_fx}, axis=1)
            if tickers == "00679B.TWO":
                return df_two
            return pd.DataFrame()

        with patch("core.data_engine.yf.download", side_effect=mock_yf_download):
            indiv, panel, fx_out, sym, warns = DataEngine.load_aligned_market_data(
                ["00679B"], "2024-02-01", "2024-03-31", "TWD"
            )
        self.assertIn("00679B", indiv)
        self.assertFalse(indiv["00679B"].empty)
        self.assertTrue(any("00679B.TWO" in w for w in warns))


if __name__ == "__main__":
    unittest.main()
