"""
市場行情與數據對齊引擎 (Data Engine)
支援 yfinance 批量下載、USD/TWD 匯率換算、跨市場日曆聯集 (Union Calendar) 與前值填補 (ffill)。
"""
from typing import List, Dict, Tuple, Optional
import yfinance as yf
import pandas as pd
import numpy as np


class DataEngine:
    """市場行情資料下載、清洗與跨市場對齊器。"""

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
        """清理與校正單一標的的 OHLC 資料。"""
        warnings = []
        if df is None or df.empty:
            return pd.DataFrame(), [f"{ticker}：無行情數據。"]

        data = df.copy()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

        data.columns = [str(c).lower() for c in data.columns]
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

        if "dividends" in data.columns:
            data["dividend"] = pd.to_numeric(data["dividends"], errors="coerce").fillna(0.0)
        elif "dividend" not in data.columns:
            data["dividend"] = 0.0

        if "stock splits" in data.columns:
            data["split"] = pd.to_numeric(data["stock splits"], errors="coerce").fillna(0.0)
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

        return data[["open", "high", "low", "close", "dividend", "split"]], warnings

    @classmethod
    def convert_currency(
        cls, 
        df: pd.DataFrame, 
        quote_curr: str, 
        target_curr: str, 
        fx_rates: Optional[pd.Series]
    ) -> pd.DataFrame:
        if quote_curr == target_curr or fx_rates is None or fx_rates.empty:
            return df.copy()

        rates = pd.to_numeric(fx_rates, errors="coerce").dropna().sort_index()
        aligned_fx = rates.reindex(df.index, method="ffill")
        prior_fx = rates.shift(1).reindex(df.index, method="ffill")
        valid_mask = aligned_fx.notna() & prior_fx.notna() & (aligned_fx > 0) & (prior_fx > 0)

        converted = df.loc[valid_mask].copy()
        fx_cur = aligned_fx.loc[valid_mask].astype(float)
        fx_pre = prior_fx.loc[valid_mask].astype(float)

        if target_curr == "TWD" and quote_curr == "USD":
            converted["open"] = converted["open"] * fx_pre
            for col in ("close", "high", "low", "dividend"):
                converted[col] = converted[col] * fx_cur
        elif target_curr == "USD" and quote_curr == "TWD":
            converted["open"] = converted["open"] / fx_pre
            for col in ("close", "high", "low", "dividend"):
                converted[col] = converted[col] / fx_cur

        converted["high"] = converted[["high", "open", "close"]].max(axis=1)
        converted["low"] = converted[["low", "open", "close"]].min(axis=1)
        return converted

    @classmethod
    def load_aligned_market_data(
        cls,
        tickers: List[str],
        start_date: str,
        end_date: str,
        target_currency: str = "TWD",
        currency_overrides: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, str, List[str]]:
        all_warnings = []
        fmt_to_orig = {cls.normalize_ticker(t): t.strip().upper() for t in tickers if t.strip()}
        if not fmt_to_orig:
            return {}, pd.DataFrame(), "NT$", ["未提供有效標的代碼。"]

        download_list = list(fmt_to_orig.keys())
        needed_fx = False
        target_curr_code = "TWD" if "TWD" in target_currency.upper() else "USD"

        for fmt_t, orig_t in fmt_to_orig.items():
            qc = cls.detect_quote_currency(orig_t, currency_overrides)
            if qc != target_curr_code:
                needed_fx = True
                break

        if needed_fx and "USDTWD=X" not in download_list:
            download_list.append("USDTWD=X")

        try:
            raw_data = yf.download(
                download_list,
                start=pd.Timestamp(start_date) - pd.Timedelta(days=35),
                end=pd.Timestamp(end_date) + pd.Timedelta(days=1),
                auto_adjust=False,
                actions=True,
                group_by="ticker",
                progress=False,
                threads=True
            )
        except Exception as exc:
            return {}, pd.DataFrame(), ("NT$" if target_curr_code == "TWD" else "$"), [f"下載資料失敗：{exc}"]

        fx_rates = None
        if needed_fx:
            try:
                if isinstance(raw_data.columns, pd.MultiIndex) and "USDTWD=X" in raw_data.columns.levels[0]:
                    fx_frame = raw_data["USDTWD=X"]
                else:
                    fx_frame = raw_data
                if "Close" in fx_frame.columns:
                    fx_rates = pd.to_numeric(fx_frame["Close"], errors="coerce").dropna()
                    fx_rates.index = pd.to_datetime(fx_rates.index)
            except Exception:
                all_warnings.append("USD/TWD 匯率擷取異常，換匯可能受影響。")

        individual_dict: Dict[str, pd.DataFrame] = {}
        all_dates_set = set()

        for fmt_t, orig_t in fmt_to_orig.items():
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
                converted = cls.convert_currency(cleaned, quote_curr, target_curr_code, fx_rates)
                
                user_start = pd.Timestamp(start_date)
                converted = converted[converted.index >= user_start]
                if converted.empty:
                    all_warnings.append(f"{orig_t}：在指定日期範圍內無有效數據。")
                    continue

                individual_dict[orig_t] = converted
                all_dates_set.update(converted.index)
            except Exception as exc:
                all_warnings.append(f"{orig_t} 資料處理異常：{exc}")

        if not individual_dict:
            curr_symbol = "NT$" if target_curr_code == "TWD" else "$"
            return {}, pd.DataFrame(), curr_symbol, all_warnings

        # 跨市場交易日聯集 (Union Calendar)
        unified_calendar = pd.DatetimeIndex(sorted(list(all_dates_set)))

        # 建立對齊總表 aligned_panel (MultiIndex: [Ticker, Metric])
        panel_dict = {}
        for orig_t, df in individual_dict.items():
            aligned_df = df.reindex(unified_calendar)
            aligned_df["close"] = aligned_df["close"].ffill().bfill()
            aligned_df["open"] = aligned_df["open"].fillna(aligned_df["close"])
            aligned_df["high"] = aligned_df["high"].fillna(aligned_df["close"])
            aligned_df["low"] = aligned_df["low"].fillna(aligned_df["close"])
            aligned_df["dividend"] = aligned_df["dividend"].fillna(0.0)
            aligned_df["split"] = aligned_df["split"].fillna(0.0)
            panel_dict[orig_t] = aligned_df

        aligned_panel = pd.concat(panel_dict, axis=1)
        curr_symbol = "NT$" if target_curr_code == "TWD" else "$"
        return individual_dict, aligned_panel, curr_symbol, all_warnings