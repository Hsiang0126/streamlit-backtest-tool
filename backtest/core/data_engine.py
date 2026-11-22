"""
市場行情與數據對齊引擎 (Data Engine)
1. 雙軌計價：保留各市場原生報價 (Native Currency)，杜絕匯率污染股價 K 棒與指標。
2. 開盤日標記 (is_trading_day)：真實反映各市場開閉市，杜絕休市日幽靈成交。
3. 靜默暖機期 (Warm-up Window)：自動預先抓取 365 天歷史資料，確保第 1 天指標成熟非 NaN。
4. 智慧代碼相容與上櫃備援：支援 4~6 碼純數字及含英文字尾台股代碼，.TW 無資料時自動嘗試 .TWO。
5. 消除前瞻偏差：多資產對齊時自動以共同有效起始日切齊，僅使用 forward-fill。
6. 一致性還原 OHLC：同步計算 adj_open / adj_high / adj_low / adj_close 避免指示器尺度混用。
"""
from typing import List, Dict, Tuple, Optional
import re
import yfinance as yf
import pandas as pd
import numpy as np


class DataEngine:
    """市場行情資料下載、清洗與跨市場真實對齊器。"""

    DEFAULT_USDTWD_RATE = 32.0

    PRESET_OPTIONS = {
        "美股指數與權值": [
            "QQQ (納斯達克100)",
            "SPY (S&P 500)",
            "VOO (標普500)",
            "VT (全世界股票)",
            "TLT (美國20年公債)",
            "NVDA (輝達)",
            "AAPL (蘋果)",
            "MSFT (微軟)",
            "TSLA (特斯拉)",
        ],
        "台股代表標的": [
            "0050 (元大台灣50)",
            "00631L (元大台灣50正2)",
            "00679B (元大美債20年)",
            "0056 (元大高股息)",
            "00878 (國泰永續高股息)",
            "2330 (台積電)",
            "2317 (鴻海)",
            "2454 (聯發科)",
        ],
        "加密貨幣": [
            "BTC-USD (比特幣)",
            "ETH-USD (乙太幣)",
            "SOL-USD (索拉納)",
        ],
    }

    @staticmethod
    def is_taiwan_ticker(ticker: str) -> bool:
        """判定是否為台股上市櫃代碼 (支援 4~6 碼及含英文字尾，如 2330, 0050, 00631L, 00679B, 00981A, 2881A)。"""
        clean = ticker.strip().upper()
        if clean.endswith((".TW", ".TWO")):
            return True
        return bool(re.match(r"^\d{4,6}[A-Z]?$", clean))

    @classmethod
    def normalize_ticker(cls, ticker: str) -> str:
        """將使用者輸入之標的代碼正規化，台股自動補上 .TW 後綴。"""
        t = ticker.strip().upper()
        if t.endswith((".TW", ".TWO")):
            return t
        if cls.is_taiwan_ticker(t):
            return f"{t}.TW"
        return t

    @classmethod
    def detect_quote_currency(cls, ticker: str, overrides: Optional[Dict[str, str]] = None) -> str:
        t = cls.normalize_ticker(ticker)
        orig = ticker.strip().upper()
        if overrides:
            if t in overrides:
                return overrides[t]
            if orig in overrides:
                return overrides[orig]
        if t.endswith((".TW", ".TWO")) or cls.is_taiwan_ticker(ticker):
            return "TWD"
        if "-USD" in t or t.endswith("USDT") or t.isalpha():
            return "USD"
        return "USD"

    @staticmethod
    def _normalize_dt_index(idx: pd.Index) -> pd.DatetimeIndex:
        dt_idx = pd.to_datetime(idx, errors="coerce")
        if getattr(dt_idx, "tz", None) is not None:
            dt_idx = dt_idx.tz_localize(None)
        return dt_idx.normalize()

    @classmethod
    def _extract_ticker_frame(cls, raw_data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        if raw_data is None or raw_data.empty:
            return None
        if not isinstance(raw_data.columns, pd.MultiIndex):
            return raw_data.copy()
        lvl0 = raw_data.columns.get_level_values(0)
        lvl1 = raw_data.columns.get_level_values(1)
        if symbol in lvl0:
            return raw_data[symbol].copy()
        if symbol in lvl1:
            return raw_data.xs(symbol, axis=1, level=1).copy()
        return None

    @classmethod
    def clean_ohlc(cls, df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, List[str]]:
        warnings: List[str] = []
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

        data.index = cls._normalize_dt_index(data.index)
        data = data[data.index.notna()]
        data = data[~data.index.duplicated(keep="last")].sort_index()

        data["high"] = data[["high", "open", "close"]].max(axis=1)
        data["low"] = data[["low", "open", "close"]].min(axis=1)

        # 計算一致尺度之還原除權息 OHLC，確保技術指標 (尤其 KD) 不因未還原高/低價混用而失真
        adj_ratio = (data["adj_close"] / data["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        adj_ratio = adj_ratio.where(adj_ratio > 0, 1.0)
        data["adj_open"] = data["open"] * adj_ratio
        data["adj_high"] = data["high"] * adj_ratio
        data["adj_low"] = data["low"] * adj_ratio
        data["adj_high"] = data[["adj_high", "adj_open", "adj_close"]].max(axis=1)
        data["adj_low"] = data[["adj_low", "adj_open", "adj_close"]].min(axis=1)

        data["is_trading_day"] = True

        cols = [
            "open", "high", "low", "close",
            "adj_open", "adj_high", "adj_low", "adj_close",
            "dividend", "split", "is_trading_day",
        ]
        return data[cols], warnings

    @classmethod
    def _download_single_fallback(
        cls, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp
    ) -> pd.DataFrame:
        try:
            df = yf.download(
                symbol,
                start=start_ts,
                end=end_ts,
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
            )
            return df if df is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    @classmethod
    def load_aligned_market_data(
        cls,
        tickers: List[str],
        start_date: str,
        end_date: str,
        target_currency: str = "TWD",
        currency_overrides: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.Series, str, List[str]]:
        all_warnings: List[str] = []
        fmt_to_orig: Dict[str, str] = {}
        for t in tickers:
            clean_orig = t.strip().upper()
            if not clean_orig or clean_orig == "USDTWD=X":
                continue
            fmt_to_orig[cls.normalize_ticker(clean_orig)] = clean_orig

        target_curr_code = "TWD" if "TWD" in target_currency.upper() else "USD"
        curr_symbol = "NT$" if target_curr_code == "TWD" else "$"

        if not fmt_to_orig:
            return {}, pd.DataFrame(), pd.Series(dtype=float), curr_symbol, ["未提供有效標的代碼。"]

        download_list = list(fmt_to_orig.keys())
        if "USDTWD=X" not in download_list:
            download_list.append("USDTWD=X")

        user_start_ts = pd.Timestamp(start_date).normalize()
        warmup_start_ts = user_start_ts - pd.Timedelta(days=365)
        end_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)

        try:
            raw_data = yf.download(
                download_list,
                start=warmup_start_ts,
                end=end_ts,
                auto_adjust=False,
                actions=True,
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception as exc:
            return {}, pd.DataFrame(), pd.Series(dtype=float), curr_symbol, [f"下載資料失敗：{exc}"]

        # 擷取 USD/TWD 匯率（含完整暖機期）
        fx_rates = pd.Series(
            cls.DEFAULT_USDTWD_RATE,
            index=pd.date_range(warmup_start_ts, end_ts, freq="D"),
            dtype=float,
        )
        fx_extracted = False
        try:
            fx_frame = cls._extract_ticker_frame(raw_data, "USDTWD=X")
            if fx_frame is not None and not fx_frame.empty:
                fx_cols = {str(c).lower(): c for c in fx_frame.columns}
                close_col = fx_cols.get("close")
                if close_col is not None:
                    extracted = pd.to_numeric(fx_frame[close_col], errors="coerce").dropna()
                    if not extracted.empty:
                        extracted.index = cls._normalize_dt_index(extracted.index)
                        extracted = extracted[extracted.index.notna()]
                        extracted = extracted[~extracted.index.duplicated(keep="last")].sort_index()
                        extracted = extracted[extracted > 0]
                        if not extracted.empty:
                            fx_rates = extracted
                            fx_extracted = True
        except Exception:
            fx_extracted = False

        if not fx_extracted:
            all_warnings.append(
                f"USD/TWD 匯率擷取異常，暫以預設匯率 {cls.DEFAULT_USDTWD_RATE:.1f} 計算。"
            )

        individual_dict: Dict[str, pd.DataFrame] = {}

        for fmt_t, orig_t in fmt_to_orig.items():
            try:
                sub_df = cls._extract_ticker_frame(raw_data, fmt_t)
                cleaned, warns = cls.clean_ohlc(sub_df, orig_t)

                # 若原本為自動補 .TW 的台股代碼且查無資料，自動嘗試上櫃 .TWO
                if cleaned.empty and fmt_t.endswith(".TW") and not orig_t.endswith(".TW"):
                    two_symbol = f"{fmt_t[:-3]}.TWO"
                    fallback_df = cls._download_single_fallback(two_symbol, warmup_start_ts, end_ts)
                    cleaned_two, warns_two = cls.clean_ohlc(fallback_df, orig_t)
                    if not cleaned_two.empty:
                        cleaned = cleaned_two
                        warns = [f"{orig_t}：上市 ({fmt_t}) 無資料，已自動切換上櫃代碼 ({two_symbol}) 載入。"]
                    else:
                        warns.extend(warns_two)

                all_warnings.extend(warns)
                if cleaned.empty:
                    continue

                quote_curr = cls.detect_quote_currency(orig_t, currency_overrides)
                cleaned["quote_currency"] = quote_curr
                cleaned["user_start_idx"] = cleaned.index >= user_start_ts
                individual_dict[orig_t] = cleaned
            except Exception as exc:
                all_warnings.append(f"{orig_t} 資料處理異常：{exc}")

        if not individual_dict:
            return {}, pd.DataFrame(), fx_rates, curr_symbol, all_warnings

        # 檢查每個標的在 user_start_ts 之後是否有足夠資料，並計算多資產共同有效起始日（消除 bfill 前瞻偏差）
        post_start_mins: Dict[str, pd.Timestamp] = {}
        for orig_t, df in individual_dict.items():
            post = df[df.index >= user_start_ts]
            if not post.empty:
                post_start_mins[orig_t] = pd.Timestamp(post.index.min())

        if not post_start_mins:
            return individual_dict, pd.DataFrame(), fx_rates, curr_symbol, all_warnings

        common_start_ts = max(user_start_ts, max(post_start_mins.values()))
        if common_start_ts > user_start_ts:
            limiting = [t for t, min_dt in post_start_mins.items() if min_dt == common_start_ts]
            # 僅當晚於起始日超過 7 天時才提醒（避開單純跨市場週末/假日 1~3 天落差）
            if (common_start_ts - user_start_ts).days > 7:
                all_warnings.append(
                    f"多資產共同起始日自動對齊至 {common_start_ts.strftime('%Y-%m-%d')}（受限於 {', '.join(limiting)} 掛牌日較晚，避免回測前瞻填值失真）。"
                )

        trading_calendar_set = set()
        for orig_t, df in individual_dict.items():
            active_days = df[df.index >= common_start_ts].index
            trading_calendar_set.update(active_days)

        if not trading_calendar_set:
            return individual_dict, pd.DataFrame(), fx_rates, curr_symbol, all_warnings

        unified_calendar = pd.DatetimeIndex(sorted(trading_calendar_set))

        panel_dict: Dict[str, pd.DataFrame] = {}
        for orig_t, df in individual_dict.items():
            # 先併入完整歷史 (含暖機期) 再僅向後 ffill()，徹底杜絕 bfill() 偷看未來價格
            union_idx = df.index.union(unified_calendar).sort_values()
            expanded = df.reindex(union_idx)

            is_open = unified_calendar.isin(df.index)
            expanded["close"] = expanded["close"].ffill()
            expanded["adj_close"] = expanded["adj_close"].ffill()

            sub = expanded.reindex(unified_calendar).copy()
            # 極端情況：若共同起始日當天恰巧為市場休市且暖機期完全無任何前值，才用當期最前筆有效收盤補齊
            if sub["close"].isna().any():
                sub["close"] = sub["close"].bfill()
            if sub["adj_close"].isna().any():
                sub["adj_close"] = sub["adj_close"].bfill()

            sub["open"] = sub["open"].fillna(sub["close"])
            sub["high"] = sub["high"].fillna(sub["close"])
            sub["low"] = sub["low"].fillna(sub["close"])

            adj_ratio = (sub["adj_close"] / sub["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            adj_ratio = adj_ratio.where(adj_ratio > 0, 1.0)
            sub["adj_open"] = sub["adj_open"].fillna(sub["open"] * adj_ratio)
            sub["adj_high"] = sub["adj_high"].fillna(sub["high"] * adj_ratio)
            sub["adj_low"] = sub["adj_low"].fillna(sub["low"] * adj_ratio)

            sub["dividend"] = sub["dividend"].fillna(0.0)
            sub["split"] = sub["split"].fillna(0.0)
            sub["is_trading_day"] = is_open.astype(bool)
            sub["quote_currency"] = df["quote_currency"].iloc[0]

            panel_dict[orig_t] = sub[
                [
                    "open", "high", "low", "close",
                    "adj_open", "adj_high", "adj_low", "adj_close",
                    "dividend", "split", "is_trading_day", "quote_currency",
                ]
            ]

        aligned_panel = pd.concat(panel_dict, axis=1)

        # 建立涵蓋暖機期與 unified_calendar 的完整匯率序列，確保單標的與多資產共用精準匯率
        full_fx_idx = fx_rates.index.union(unified_calendar)
        for df in individual_dict.values():
            full_fx_idx = full_fx_idx.union(df.index)
        full_fx_idx = full_fx_idx.sort_values()
        aligned_fx_full = (
            fx_rates.reindex(full_fx_idx).ffill().bfill().fillna(cls.DEFAULT_USDTWD_RATE)
        )

        return individual_dict, aligned_panel, aligned_fx_full, curr_symbol, all_warnings
