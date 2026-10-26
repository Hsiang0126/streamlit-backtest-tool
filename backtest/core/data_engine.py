"""
市場行情與數據對齊引擎 (Data Engine - 5元組完整版)
1. 雙軌計價：保留各市場原生報價 (Native Currency)，杜絕匯率污染股價 K 棒與指標。
2. 開盤日標記 (is_trading_day)：真實反映各市場開閉市，杜絕休市日幽靈成交。
3. 靜默暖機期 (Warm-up Window)：自動預先抓取 365 天歷史資料，確保第 1 天指標成熟非 NaN。
4. 還原價格提取 (adj_close)：提供全報酬還原價格供技術指標計算，杜絕除權息假跌破。
"""
from typing import List, Dict, Tuple, Optional
import yfinance as yf
import pandas as pd
import numpy as np


class DataEngine:
    """市場行情資料下載、清洗與跨市場真實對齊器。"""

    PRESET_OPTIONS = {
        "美股指數與權值": ["QQQ (納斯達克100)", "AAPL (蘋果)", "NVDA (輝達)", "SPY (S&P 500)", "TSLA (特斯拉)", "MSFT (微軟)"],
        "台股代表標的": ["0050 (元大台灣50)", "2330 (台積電)", "0056 (元大高股息)", "2317 (鴻海)", "2454 (聯發科)"],
        "加密貨幣": ["BTC-USD (比特幣)", "ETH-USD (乙太幣)", "SOL-USD (索拉納)", "BNB-USD (幣安幣)"]
    }

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        t = ticker.strip().upper()
        if t.isdigit():
            return f"{t}.TW"
        return t

    @classmethod
    def detect_quote_currency(cls, ticker: str, overrides: Optional[Dict[str, str]] = None) -> str:
        t = cls.normalize_ticker(ticker)
        if overrides and t in overrides:
            return overrides[t]
        if t.endswith((".TW", ".TWO")):
            return "TWD"
        if "-USD" in t or t.endswith("USDT") or t.isalpha():
            return "USD"
        return "USD"

    @classmethod
    def clean_ohlc(cls, df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, List[str]]:
        warnings = []
        if df is None or df.empty:
            return pd.DataFrame(), [f"{ticker}：無行情數據。"]

        data = df.copy()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

        data.columns = [str(c).lower().replace(" ", "_") for c in data.columns]
        req_cols = {"open", "high", "low", "close"}
        if not req_cols.issubset(data.columns):
            if "close" in data.columns:
                for c in ("open", "high", "low"):
                    if c not in data.columns:
                        data[c] = data["close"]
            else:
                return pd.DataFrame(), [f"{ticker}：缺少核心 Close 價格欄位。"]

        for c in req_cols:
            data[c] = pd.to_numeric(data[c], errors="coerce")

        if "adj_close" in data.columns:
            data["adj_close"] = pd.to_numeric(data["adj_close"], errors="coerce").fillna(data["close"])
        else:
            data["adj_close"] = data["close"].copy()

        if "dividends" in data.columns:
            data["dividend"] = pd.to_numeric(data["dividends"], errors="coerce").fillna(0.0)
        elif "dividend" not in data.columns:
            data["dividend"] = 0.0

        if "stock_splits" in data.columns:
            data["split"] = pd.to_numeric(data["stock_splits"], errors="coerce").fillna(0.0)
        elif "split" not in data.columns:
            data["split"] = 0.0

        data = data.dropna(subset=["close", "open", "high", "low"])
        data = data[(data["close"] > 0) & (data["open"] > 0) & (data["high"] > 0) & (data["low"] > 0)]

        if data.empty:
            return pd.DataFrame(), [f"{ticker}：清洗後價格資料為空。"]

        data.index = pd.to_datetime(data.index)
        data = data[~data.index.duplicated(keep="last")].sort_index()

        data["high"] = data[["high", "open", "close"]].max(axis=1)
        data["low"] = data[["low", "open", "close"]].min(axis=1)
        data["is_trading_day"] = True

        return data[["open", "high", "low", "close", "adj_close", "dividend", "split", "is_trading_day"]], warnings

    @classmethod
    def load_aligned_market_data(
        cls,
        tickers: List[str],
        start_date: str,
        end_date: str,
        target_currency: str = "TWD",
        currency_overrides: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.Series, str, List[str]]:
        all_warnings = []
        fmt_to_orig = {cls.normalize_ticker(t): t.strip().upper() for t in tickers if t.strip()}
        if not fmt_to_orig:
            return {}, pd.DataFrame(), pd.Series(dtype=float), "NT$", ["未提供有效標的代碼。"]

        download_list = list(fmt_to_orig.keys())
        target_curr_code = "TWD" if "TWD" in target_currency.upper() else "USD"

        if "USDTWD=X" not in download_list:
            download_list.append("USDTWD=X")

        user_start_ts = pd.Timestamp(start_date)
        warmup_start_ts = user_start_ts - pd.Timedelta(days=365)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)

        try:
            raw_data = yf.download(
                download_list,
                start=warmup_start_ts,
                end=end_ts,
                auto_adjust=False,
                actions=True,
                group_by="ticker",
                progress=False,
                threads=True
            )
        except Exception as exc:
            return {}, pd.DataFrame(), pd.Series(dtype=float), ("NT$" if target_curr_code == "TWD" else "$"), [f"下載資料失敗：{exc}"]

        fx_rates = pd.Series(1.0, index=pd.date_range(warmup_start_ts, end_ts))
        try:
            if isinstance(raw_data.columns, pd.MultiIndex) and "USDTWD=X" in raw_data.columns.levels[0]:
                fx_frame = raw_data["USDTWD=X"]
            else:
                fx_frame = raw_data
            if "Close" in fx_frame.columns:
                extracted = pd.to_numeric(fx_frame["Close"], errors="coerce").dropna()
                extracted.index = pd.to_datetime(extracted.index)
                if not extracted.empty:
                    fx_rates = extracted
        except Exception:
            all_warnings.append("USD/TWD 匯率擷取異常，預設匯率為 1.0。")

        individual_dict: Dict[str, pd.DataFrame] = {}
        trading_calendar_set = set()

        for fmt_t, orig_t in fmt_to_orig.items():
            if orig_t == "USDTWD=X":
                continue
            try:
                if len(download_list) == 1 and not isinstance(raw_data.columns, pd.MultiIndex):
                    sub_df = raw_data.copy()
                elif isinstance(raw_data.columns, pd.MultiIndex):
                    if fmt_t in raw_data.columns.levels[0]:
                        sub_df = raw_data[fmt_t].copy()
                    else:
                        all_warnings.append(f"{orig_t}：無法自回傳資料中找到對應代碼。")
                        continue
                else:
                    sub_df = raw_data.copy()

                cleaned, warns = cls.clean_ohlc(sub_df, orig_t)
                all_warnings.extend(warns)
                if cleaned.empty:
                    continue

                quote_curr = cls.detect_quote_currency(orig_t, currency_overrides)
                cleaned["quote_currency"] = quote_curr
                cleaned["user_start_idx"] = (cleaned.index >= user_start_ts)

                individual_dict[orig_t] = cleaned
                active_days = cleaned[cleaned.index >= user_start_ts].index
                trading_calendar_set.update(active_days)
            except Exception as exc:
                all_warnings.append(f"{orig_t} 資料處理異常：{exc}")

        curr_symbol = "NT$" if target_curr_code == "TWD" else "$"
        if not individual_dict or not trading_calendar_set:
            return {}, pd.DataFrame(), fx_rates, curr_symbol, all_warnings

        unified_calendar = pd.DatetimeIndex(sorted(list(trading_calendar_set)))

        panel_dict = {}
        for orig_t, df in individual_dict.items():
            sub = df[df.index >= user_start_ts].copy()
            aligned = sub.reindex(unified_calendar)
            aligned["is_trading_day"] = aligned["is_trading_day"].fillna(False).astype(bool)
            aligned["close"] = aligned["close"].ffill().bfill()
            aligned["open"] = aligned["open"].fillna(aligned["close"])
            aligned["high"] = aligned["high"].fillna(aligned["close"])
            aligned["low"] = aligned["low"].fillna(aligned["close"])
            aligned["adj_close"] = aligned["adj_close"].ffill().bfill()
            aligned["dividend"] = aligned["dividend"].fillna(0.0)
            aligned["split"] = aligned["split"].fillna(0.0)
            aligned["quote_currency"] = df["quote_currency"].iloc[0]
            panel_dict[orig_t] = aligned

        aligned_panel = pd.concat(panel_dict, axis=1)
        aligned_fx = fx_rates.reindex(unified_calendar, method="ffill").fillna(31.0)

        return individual_dict, aligned_panel, aligned_fx, curr_symbol, all_warnings
