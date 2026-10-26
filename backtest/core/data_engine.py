"""
市場行情與數據對齊引擎 (Data Engine)
升級功能：
1. 智能 Ticker 辨識 (支援常見上櫃清單與 .TW/.TWO 自動 Fallback)。
2. 雙價分離 (Raw Open/Close 供交易撮合，Adj Close 供指標計算)。
3. 365 天靜默暖機期保留 (避免技術指標前期無效)。
4. 原生幣別與換匯雙軌架構 (單標的 PK 不受匯率污染，資產配置按統一幣別結算)。
5. 共同上市日自動檢測與截斷 (徹底杜絕 bfill 偽造未上市價格)。
"""
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import pandas as pd
import numpy as np


class DataEngine:
    """市場行情資料下載、清洗與跨市場對齊器。"""

    # 常用熱門資產分類
    PRESET_OPTIONS = {
        "美股指數與權值": ["QQQ (納斯達克100)", "AAPL (蘋果)", "NVDA (輝達)", "SPY (S&P 500)", "TSLA (特斯拉)", "MSFT (微軟)"],
        "台股熱門標的": ["0050 (元大台灣50)", "2330 (台積電)", "0056 (元大高股息)", "2317 (鴻海)", "6547 (高端疫苗-上櫃)", "3105 (穩懋-上櫃)"],
        "全球固定收益與商品": ["TLT (美債20年)", "IEF (美債7-10年)", "GLD (黃金現貨)", "DBC (大宗商品)"],
        "加密貨幣": ["BTC-USD (比特幣)", "ETH-USD (乙太幣)", "SOL-USD (索拉納)"]
    }

    # 常見台灣上櫃代號 (OTC) 靜態速查表，避免使用者輸入純數字時被誤加上 .TW
    KNOWN_OTC_TICKERS = {
        "3105", "3293", "6547", "8069", "6488", "5483", "3529", "4966", 
        "6274", "3264", "5347", "8299", "6180", "4147", "6446", "3558",
        "00679B", "00687B", "00720B", "00751B", "00772B", "00795B"
    }

    @classmethod
    def normalize_ticker(cls, ticker: str) -> str:
        """智能解析與標準化 Ticker，純數字自動判斷上市或上櫃。"""
        t = ticker.strip().upper()
        if t.endswith((".TW", ".TWO", "-USD")) or t.endswith("USDT") or "=" in t:
            return t

        if t in cls.KNOWN_OTC_TICKERS:
            return f"{t}.TWO"

        if t.isdigit():
            return f"{t}.TW"

        return t

    @classmethod
    def detect_quote_currency(cls, ticker: str, overrides: Optional[Dict[str, str]] = None) -> str:
        """判斷標的原生計價幣別。"""
        t = cls.normalize_ticker(ticker)
        if overrides and t in overrides:
            return overrides[t]
        if t.endswith((".TW", ".TWO")):
            return "TWD"
        return "USD"

    @classmethod
    def clean_ohlc(cls, df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        清洗單一標的的 OHLC 資料。
        重要修復：同時保留市場未還原價 (raw open/close) 與除息還原價 (adj_close)。
        """
        warnings = []
        if df is None or df.empty:
            return pd.DataFrame(), [f"{ticker}：未取得任何市場行情數據。"]

        data = df.copy()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

        data.columns = [str(c).lower().replace(" ", "_") for c in data.columns]

        # 核心欄位檢查
        req_cols = {"open", "high", "low", "close"}
        if not req_cols.issubset(data.columns):
            if "close" in data.columns:
                for c in ("open", "high", "low"):
                    if c not in data.columns:
                        data[c] = data["close"]
            else:
                return pd.DataFrame(), [f"{ticker}：缺少核心 Close 價格欄位。"]

        # 處理還原價 (Adj Close)
        if "adj_close" not in data.columns:
            data["adj_close"] = data["close"]

        # 數值型別轉換與過濾
        all_numeric_cols = ["open", "high", "low", "close", "adj_close"]
        for c in all_numeric_cols:
            data[c] = pd.to_numeric(data[c], errors="coerce")

        if "dividends" in data.columns:
            data["dividend"] = pd.to_numeric(data["dividends"], errors="coerce").fillna(0.0)
        elif "dividend" not in data.columns:
            data["dividend"] = 0.0

        if "stock_splits" in data.columns:
            data["split"] = pd.to_numeric(data["stock_splits"], errors="coerce").fillna(0.0)
        elif "split" not in data.columns:
            data["split"] = 0.0

        data = data.dropna(subset=all_numeric_cols)
        data = data[(data["close"] > 0) & (data["open"] > 0) & (data["adj_close"] > 0)]

        if data.empty:
            return pd.DataFrame(), [f"{ticker}：過濾異常值後價格數據為空。"]

        data.index = pd.to_datetime(data.index)
        data = data[~data.index.duplicated(keep="last")].sort_index()

        # 邊界校正
        data["high"] = data[["high", "open", "close"]].max(axis=1)
        data["low"] = data[["low", "open", "close"]].min(axis=1)

        return data[["open", "high", "low", "close", "adj_close", "dividend", "split"]], warnings

    @classmethod
    def _fetch_single_ticker(cls, formatted_ticker: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
        """單一標的抓取函式，若台股上市失敗則自動 fallback 嘗試上櫃。"""
        try:
            df = yf.download(
                formatted_ticker,
                start=start_dt,
                end=end_dt,
                auto_adjust=False,
                actions=True,
                progress=False
            )
            if (df is None or df.empty) and formatted_ticker.endswith(".TW"):
                fallback_ticker = formatted_ticker.replace(".TW", ".TWO")
                df = yf.download(
                    fallback_ticker,
                    start=start_dt,
                    end=end_dt,
                    auto_adjust=False,
                    actions=True,
                    progress=False
                )
            return df
        except Exception:
            return pd.DataFrame()

    @classmethod
    def load_aligned_market_data(
        cls,
        tickers: List[str],
        start_date: str,
        end_date: str,
        target_currency: str = "新台幣 (TWD)",
        currency_overrides: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, str, str, List[str]]:
        """
        載入並處理市場數據：
        - 保留 365 天靜默暖機期。
        - 自動檢測各標的最晚上市日，自動將回測起點安全平移至共同有效日。
        - individual_dict 保留原生幣別與暖機期（供 Tab 1 單標的無污染運算）。
        - aligned_panel 進行統一幣別折算與日期交集（供 Tab 2 多資產再平衡使用）。
        """
        all_warnings = []
        fmt_to_orig = {cls.normalize_ticker(t): t.strip().upper() for t in tickers if t.strip()}
        if not fmt_to_orig:
            return {}, pd.DataFrame(), "NT$", start_date, ["未提供有效標的代碼。"]

        target_curr_code = "TWD" if "TWD" in target_currency.upper() else "USD"
        curr_symbol = "NT$" if target_curr_code == "TWD" else "$"

        # 365 天暖機期起始日
        requested_start_ts = pd.Timestamp(start_date)
        warmup_start_ts = requested_start_ts - pd.Timedelta(days=365)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)

        # 1. 平行並行下載各標的數據
        download_targets = list(fmt_to_orig.keys())
        raw_results: Dict[str, pd.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=min(len(download_targets) + 1, 6)) as executor:
            future_map = {
                executor.submit(cls._fetch_single_ticker, fmt_t, warmup_start_ts, end_ts): fmt_t
                for fmt_t in download_targets
            }
            # 同步下載匯率數據
            future_fx = executor.submit(cls._fetch_single_ticker, "USDTWD=X", warmup_start_ts, end_ts)

            for fut in future_map:
                fmt_t = future_map[fut]
                raw_results[fmt_t] = fut.result()
            raw_fx = future_fx.result()

        # 解析匯率
        fx_rates: Optional[pd.Series] = None
        if raw_fx is not None and not raw_fx.empty:
            cleaned_fx, _ = cls.clean_ohlc(raw_fx, "USDTWD=X")
            if not cleaned_fx.empty:
                fx_rates = cleaned_fx["close"]

        # 2. 清洗各標的數據並辨別真實上市掛牌日
        individual_dict: Dict[str, pd.DataFrame] = {}
        first_valid_dates: Dict[str, pd.Timestamp] = {}

        for fmt_t, orig_t in fmt_to_orig.items():
            df_raw = raw_results.get(fmt_t)
            cleaned, warns = cls.clean_ohlc(df_raw, orig_t)
            all_warnings.extend(warns)
            if cleaned.empty:
                continue

            # 記錄標的原生幣別屬性
            cleaned.attrs["quote_currency"] = cls.detect_quote_currency(orig_t, currency_overrides)
            individual_dict[orig_t] = cleaned
            first_valid_dates[orig_t] = cleaned.index[0]

        if not individual_dict:
            return {}, pd.DataFrame(), curr_symbol, start_date, all_warnings

        # 3. 自動對齊回測起點 (解決上市日不齊與 bfill 穿梭問題)
        max_inception_date = max(first_valid_dates.values())
        effective_start_ts = requested_start_ts

        if max_inception_date > requested_start_ts:
            latest_asset = [k for k, v in first_valid_dates.items() if v == max_inception_date][0]
            effective_start_ts = max_inception_date
            all_warnings.append(
                f"ℹ️ 標的【{latest_asset}】於 {max_inception_date.strftime('%Y-%m-%d')} 才掛牌上市。"
                f" 系統已自動將全體資產的實際回測起點對齊至該日，以確保數據嚴謹真實！"
            )

        effective_start_str = effective_start_ts.strftime("%Y-%m-%d")

        # 4. 建構多資產換匯對齊面板 (aligned_panel)
        # 僅取共同上市日以後的有效日曆交集 (Union Calendar)
        all_dates = sorted(list({
            d for df in individual_dict.values() 
            for d in df.loc[df.index >= effective_start_ts].index
        }))
        unified_calendar = pd.DatetimeIndex(all_dates)

        panel_dict = {}
        for orig_t, df in individual_dict.items():
            # 截取共同上市日之後的資料
            df_active = df.loc[df.index >= effective_start_ts].copy()
            aligned = df_active.reindex(unified_calendar)
            aligned["close"] = aligned["close"].ffill().bfill()
            aligned["open"] = aligned["open"].fillna(aligned["close"])
            aligned["high"] = aligned["high"].fillna(aligned["close"])
            aligned["low"] = aligned["low"].fillna(aligned["close"])
            aligned["adj_close"] = aligned["adj_close"].ffill().bfill()
            aligned["dividend"] = aligned["dividend"].fillna(0.0)
            aligned["split"] = aligned["split"].fillna(0.0)

            # 換匯至統一目標貨幣 (供資產配置使用)
            quote_curr = df.attrs.get("quote_currency", "USD")
            if quote_curr != target_curr_code and fx_rates is not None:
                aligned_fx = fx_rates.reindex(unified_calendar).ffill().bfill()
                prior_fx = fx_rates.shift(1).reindex(unified_calendar).ffill().bfill()

                if target_curr_code == "TWD" and quote_curr == "USD":
                    aligned["open"] = aligned["open"] * prior_fx
                    for col in ("close", "high", "low", "adj_close", "dividend"):
                        aligned[col] = aligned[col] * aligned_fx
                elif target_curr_code == "USD" and quote_curr == "TWD":
                    aligned["open"] = aligned["open"] / prior_fx
                    for col in ("close", "high", "low", "adj_close", "dividend"):
                        aligned[col] = aligned[col] / aligned_fx

            panel_dict[orig_t] = aligned

        aligned_panel = pd.concat(panel_dict, axis=1)

        return individual_dict, aligned_panel, curr_symbol, effective_start_str, all_warnings
