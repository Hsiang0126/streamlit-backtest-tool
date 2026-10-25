import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 0. 頁面基本設定 (Page Config)
# ==========================================
st.set_page_config(layout="wide", page_title="多市場與多策略回測系統 v32.1")


# ==========================================
# 1. 數據獲取與匯率處理模組 (Data Module)
# ==========================================
class DataFetcher:
    """下載市場價格、辨識報價幣別，並依使用者資金幣別進行歷史匯率換算。"""

    PRESET_OPTIONS = {
        "美股": ["QQQ (納斯達克100)", "AAPL (蘋果)", "NVDA (輝達)", "SPY (S&P 500)", "TSLA (特斯拉)", "MSFT (微軟)"],
        "台股": ["0050 (元大台灣50)", "2330 (台積電)", "0056 (元大高股息)", "2317 (鴻海)", "2454 (聯發科)"],
        "加密貨幣": ["BTC-USD (比特幣)", "ETH-USD (乙太幣)", "SOL-USD (索拉納)", "BNB-USD (幣安幣)"]
    }

    @staticmethod
    def get_asset_costs(ticker_symbol):
        clean_symbol = ticker_symbol.strip().upper()
        if clean_symbol.isdigit() or clean_symbol.endswith(".TW") or clean_symbol.endswith(".TWO"):
            return (0.00071, 0.0010, 0.0002) if clean_symbol.startswith("00") else (0.00071, 0.0030, 0.0003)
        elif "-USD" in clean_symbol:
            return (0.0010, 0.00, 0.0010)
        return (0.00, 0.00, 0.0003)

    @staticmethod
    def normalize_ticker(ticker):
        t = ticker.strip().upper()
        if t.isdigit():
            return f"{t}.TW"
        return t

    @staticmethod
    def _quote_currency(ticker, currency_overrides=None):
        """辨識 TWD/USD。無法可靠判斷的自訂代碼必須由使用者明確指定幣別。"""
        t = ticker.strip().upper()
        overrides = {str(k).strip().upper(): str(v).strip().upper() for k, v in (currency_overrides or {}).items()}
        if t in overrides:
            return overrides[t] if overrides[t] in {"TWD", "USD"} else "UNKNOWN"
        if t.endswith((".TW", ".TWO")) or t.isdigit():
            return "TWD"
        if t.endswith("=X"):
            return "UNKNOWN"
        
        unsupported_suffixes = (".HK", ".SS", ".SZ", ".L", ".TO", ".V", ".AX", ".SI", ".KS", ".KQ", ".T", ".PA", ".DE", ".MI", ".SW", ".AS", ".BR", ".HE", ".ST", ".OL", ".CO", ".NZ")
        if t.endswith(unsupported_suffixes):
            return "UNKNOWN"
            
        # 通用美股與加密貨幣代碼結構判斷
        if (t.isalpha() and 1 <= len(t) <= 5) or t.endswith("-USD"):
            return "USD"
        return "UNKNOWN"

    @staticmethod
    def _extract_ticker_frame(downloaded, ticker, single_ticker):
        if downloaded is None or downloaded.empty:
            return pd.DataFrame()
        if isinstance(downloaded.columns, pd.MultiIndex):
            if ticker in downloaded.columns.get_level_values(0):
                frame = downloaded[ticker].copy()
            elif ticker in downloaded.columns.get_level_values(-1):
                frame = downloaded.xs(ticker, axis=1, level=-1).copy()
            elif single_ticker:
                frame = downloaded.copy()
            else:
                return pd.DataFrame()
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            return frame
        return downloaded.copy() if single_ticker else pd.DataFrame()

    @staticmethod
    def data_quality_warnings(data, ticker="標的", min_rows=60, gap_days=10, extreme_return=0.35):
        warnings = []
        if data is None or data.empty:
            return [f"{ticker}：沒有可供品質檢查的價格資料。"]
        if not isinstance(data.index, pd.DatetimeIndex):
            return [f"{ticker}：資料索引不是日期格式，無法可靠檢查時間間隔。"]
        if not data.index.is_monotonic_increasing or data.index.has_duplicates:
            warnings.append(f"{ticker}：日期索引未排序或存在重複日期，回測前應檢查。")
        if len(data) < int(min_rows):
            warnings.append(f"{ticker}：只有 {len(data)} 筆行情，少於建議的 {min_rows} 筆。")
        if "close" not in data.columns:
            warnings.append(f"{ticker}：缺少 close 欄位，無法進行完整回測。")
            return warnings
        close = pd.to_numeric(data["close"], errors="coerce")
        if close.isna().any() or not np.isfinite(close.to_numpy(dtype=float)).all() or (close <= 0).any():
            warnings.append(f"{ticker}：收盤價包含缺失、非有限或非正值。")
        if len(data) > 1:
            gaps = data.index.to_series().diff().dt.total_seconds().div(86400.0)
            large_gaps = gaps[gaps > float(gap_days)]
            if not large_gaps.empty:
                dates = ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in large_gaps.index[:3])
                warnings.append(f"{ticker}：發現超過 {gap_days} 個日曆日的行情間隔（例如 {dates}）。")
            returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            extreme = returns[returns.abs() >= float(extreme_return)]
            if not extreme.empty:
                dates = ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in extreme.index[:3])
                warnings.append(f"{ticker}：發現單日收盤價變動達 ±{extreme_return:.0%} 的行情（例如 {dates}）。")
        required_ohlc = {"open", "high", "low", "close"}
        if required_ohlc.issubset(data.columns):
            values = data[list(required_ohlc)].apply(pd.to_numeric, errors="coerce")
            if not values.isna().any().any():
                upper = values[["open", "close"]].max(axis=1)
                lower = values[["open", "close"]].min(axis=1)
                if (values["high"] < upper).any() or (values["low"] > lower).any():
                    warnings.append(f"{ticker}：部分 OHLC 高低價未涵蓋開盤／收盤價。")
        return warnings

    @staticmethod
    def _convert_prices_to_target_currency(data, fx_close, target_currency):
        if data is None or data.empty or fx_close is None or fx_close.empty:
            return pd.DataFrame(columns=data.columns if data is not None else [])
        rates = pd.to_numeric(fx_close, errors="coerce")
        rates.index = pd.to_datetime(rates.index)
        rates = rates.replace([np.inf, -np.inf], np.nan).dropna()
        rates = rates[rates > 0].sort_index()
        current_fx = rates.reindex(data.index, method="ffill")
        prior_fx = rates.shift(1).reindex(data.index, method="ffill")
        valid = current_fx.notna() & prior_fx.notna() & (current_fx > 0) & (prior_fx > 0)
        converted = data.loc[valid].copy()
        current_fx = current_fx.loc[valid].astype(float)
        prior_fx = prior_fx.loc[valid].astype(float)
        if converted.empty:
            return converted
        if target_currency == "TWD":
            close_multiplier = current_fx
            open_multiplier = prior_fx
        else:
            close_multiplier = 1.0 / current_fx
            open_multiplier = 1.0 / prior_fx
        converted["open"] = converted["open"].astype(float) * open_multiplier
        for col in ("close", "high", "low"):
            converted[col] = converted[col].astype(float) * close_multiplier
        if "dividend" in converted.columns:
            converted["dividend"] = converted["dividend"].astype(float) * close_multiplier
        converted["high"] = converted[["high", "open", "close"]].max(axis=1)
        converted["low"] = converted[["low", "open", "close"]].min(axis=1)
        converted["fx_rate_usdtwd"] = current_fx
        return converted

    @classmethod
    @st.cache_data(show_spinner=False)
    def load_data_and_fx(cls, tickers, start, end, auto_adjust=True, user_curr="新台幣 (TWD)", currency_overrides=None):
        fmt_to_orig = {cls.normalize_ticker(t): t.strip().upper() for t in tickers}
        if not fmt_to_orig:
            return {}, ("NT$" if "TWD" in user_curr else "$"), ["沒有可下載的標的代碼。"]

        warnings = []
        try:
            raw = yf.download(
                list(fmt_to_orig.keys()), start=start, end=end, auto_adjust=False,
                actions=True, group_by="ticker", progress=False, threads=True
            )
        except Exception as exc:
            return {}, ("NT$" if "TWD" in user_curr else "$"), [f"Yahoo Finance 下載失敗：{exc}"]

        processed_data = {}
        for fmt_t, orig_t in fmt_to_orig.items():
            try:
                frame = cls._extract_ticker_frame(raw, fmt_t, len(fmt_to_orig) == 1)
                required = {"Close"}
                if frame.empty or not required.issubset(frame.columns):
                    warnings.append(f"{orig_t}：沒有可用的收盤價資料，已略過。")
                    continue
                close = pd.to_numeric(frame["Close"], errors="coerce")
                open_price = pd.to_numeric(frame["Open"], errors="coerce") if "Open" in frame else close.copy()
                high = pd.to_numeric(frame["High"], errors="coerce") if "High" in frame else close.copy()
                low = pd.to_numeric(frame["Low"], errors="coerce") if "Low" in frame else close.copy()
                dividends = pd.to_numeric(frame["Dividends"], errors="coerce") if "Dividends" in frame else pd.Series(0.0, index=frame.index)
                splits = pd.to_numeric(frame["Stock Splits"], errors="coerce") if "Stock Splits" in frame else pd.Series(0.0, index=frame.index)
                res = pd.DataFrame({
                    "open": open_price, "close": close, "high": high, "low": low,
                    "dividend": dividends.fillna(0.0), "split": splits.fillna(0.0)
                })
                res = res.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "close", "high", "low"])
                res = res[(res["close"] > 0) & (res["open"] > 0) & (res["high"] > 0) & (res["low"] > 0)]
                if res.empty:
                    warnings.append(f"{orig_t}：價格資料清理後為空，已略過。")
                    continue
                res.index = pd.to_datetime(res.index)
                res = res[~res.index.duplicated(keep="last")].sort_index()
                if not auto_adjust:
                    res["dividend"] = 0.0
                    res["split"] = 0.0
                override_map = {str(k).strip().upper(): str(v).strip().upper() for k, v in (currency_overrides or {}).items()}
                quote_currency = cls._quote_currency(orig_t, override_map)
                if quote_currency == "UNKNOWN":
                    warnings.append(f"{orig_t}：無法自動辨識報價幣別。請在選單中指定 TWD 或 USD。")
                res["quote_currency"] = quote_currency
                processed_data[orig_t] = res
            except Exception as exc:
                warnings.append(f"{orig_t}：整理資料時發生錯誤（{exc}）。")

        target_currency = "TWD" if "TWD" in user_curr else "USD"
        for ticker in list(processed_data):
            quote_currency = processed_data[ticker]["quote_currency"].iloc[0]
            if quote_currency not in {"TWD", "USD"}:
                warnings.append(f"{ticker}：報價幣別無法可靠辨識；已略過。")
                del processed_data[ticker]

        needed_fx = any(df["quote_currency"].iloc[0] != target_currency for df in processed_data.values())
        fx_close = None
        if needed_fx:
            try:
                fx_start = pd.Timestamp(start) - pd.Timedelta(days=30)
                fx_raw = yf.download("USDTWD=X", start=fx_start, end=end, auto_adjust=True, progress=False)
                fx_frame = cls._extract_ticker_frame(fx_raw, "USDTWD=X", True)
                if not fx_frame.empty and "Close" in fx_frame.columns:
                    fx_close = pd.to_numeric(fx_frame["Close"], errors="coerce")
                    fx_close.index = pd.to_datetime(fx_close.index)
                    fx_close = fx_close.replace([np.inf, -np.inf], np.nan).dropna()
                    fx_close = fx_close[fx_close > 0].sort_index()
                if fx_close is None or fx_close.empty:
                    warnings.append("無法取得 USD/TWD 歷史匯率；需要換匯的標的將略過。")
            except Exception as exc:
                warnings.append(f"下載 USD/TWD 匯率失敗：{exc}。")

        for ticker in list(processed_data):
            data = processed_data[ticker]
            quote_currency = data["quote_currency"].iloc[0]
            if quote_currency == target_currency:
                continue
            if fx_close is None or fx_close.empty:
                warnings.append(f"{ticker}：缺少必要匯率資料，已略過。")
                del processed_data[ticker]
                continue
            original_rows = len(data)
            data = cls._convert_prices_to_target_currency(data, fx_close, target_currency)
            if data.empty:
                warnings.append(f"{ticker}：缺少可用的當日／前一交易日匯率，已略過。")
                del processed_data[ticker]
                continue
            if len(data) < original_rows:
                warnings.append(f"{ticker}：排除缺少前置匯率的 {original_rows - len(data)} 筆資料。")
            processed_data[ticker] = data

        for data in processed_data.values():
            if "quote_currency" in data:
                data.drop(columns=["quote_currency"], inplace=True)
        curr_symbol = "NT$" if target_currency == "TWD" else "$"
        return processed_data, curr_symbol, warnings


# ==========================================
# 2. 技術指標與訊號產生模組 (Indicator Module)
# ==========================================
class IndicatorCalculator:
    """計算技術指標與買賣訊號。"""

    @staticmethod
    def calculate_ma(df, period, timeframe="日線 (Daily)"):
        if timeframe == "週線 (Weekly)":
            weekly_close = df['close'].resample('W-FRI').last().dropna()
            weekly_ma = weekly_close.rolling(period).mean()
            return weekly_ma.reindex(df.index, method='ffill')
        return df['close'].rolling(period).mean()

    @classmethod
    def generate_signal(cls, df, strat_name, p):
        data = df.copy()
        if data.empty or "close" not in data.columns:
            raise ValueError("指標計算需要非空資料及 close 欄位。")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("指標資料索引必須是 DatetimeIndex。")
        close = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError("指標計算的 close 必須全部是有限正數。")
        if not isinstance(p, dict):
            raise ValueError("策略參數必須是字典。")

        def positive_int(key):
            try:
                value = float(p[key])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"策略參數 {key} 必須是正整數。")
            if not np.isfinite(value) or value < 1 or value != int(value):
                raise ValueError(f"策略參數 {key} 必須是正整數。")
            return int(value)

        if strat_name == "價格突破均線買入/跌破賣出":
            positive_int("period")
        elif strat_name == "SMA 雙均線交叉":
            fast, slow = positive_int("fast"), positive_int("slow")
            if fast >= slow:
                raise ValueError("SMA 雙均線交叉要求 fast 小於 slow。")
        elif strat_name == "RSI 超買超賣":
            positive_int("period")
            try:
                buy, sell = float(p["buy"]), float(p["sell"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("RSI 買入與賣出門檻必須是數值。")
            if not np.isfinite([buy, sell]).all() or not (0 <= buy < sell <= 100):
                raise ValueError("RSI 門檻必須符合 0 ≤ buy < sell ≤ 100。")
        elif strat_name == "MACD 柱狀圖/交叉":
            fast, slow, signal_period = positive_int("fast"), positive_int("slow"), positive_int("signal")
            if fast >= slow:
                raise ValueError("MACD 要求 fast 小於 slow。")
        elif strat_name == "KD 隨機指標":
            positive_int("period")
            if not {"high", "low"}.issubset(data.columns):
                raise ValueError("KD 指標需要 high 與 low 欄位。")
            try:
                buy, sell = float(p["buy_k"]), float(p["sell_k"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("KD 買入與賣出門檻必須是數值。")
            if not np.isfinite([buy, sell]).all() or not (0 <= buy < sell ≤ 100):
                raise ValueError("KD 門檻必須符合 0 ≤ buy_k < sell_k ≤ 100。")
        else:
            raise ValueError(f"不支援的策略名稱：{strat_name}")

        if strat_name == "價格突破均線買入/跌破賣出":
            sma = cls.calculate_ma(data, p['period'], p.get('timeframe', '日線 (Daily)'))
            return (data['close'] > sma).astype(int)
        elif strat_name == "SMA 雙均線交叉":
            fast_m = cls.calculate_ma(data, p['fast'], p.get('timeframe', '日線 (Daily)'))
            slow_m = cls.calculate_ma(data, p['slow'], p.get('timeframe', '日線 (Daily)'))
            return (fast_m > slow_m).astype(int)
        elif strat_name == "RSI 超買超賣":
            period = max(1, int(p['period']))
            delta = data['close'].diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
            avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
            rsi = rsi.mask((avg_gain > 0) & (avg_loss == 0), 100.0)
            rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
            sig = pd.Series(np.nan, index=data.index)
            sig[rsi < p['buy']] = 1
            sig[rsi > p['sell']] = 0
            return sig.ffill().fillna(0).astype(int)
        elif strat_name == "MACD 柱狀圖/交叉":
            ema_fast = data['close'].ewm(span=p['fast'], adjust=False).mean()
            ema_slow = data['close'].ewm(span=p['slow'], adjust=False).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=p['signal'], adjust=False).mean()
            return (macd > signal).astype(int)
        elif strat_name == "KD 隨機指標":
            low_min = data['low'].rolling(p['period']).min()
            high_max = data['high'].rolling(p['period']).max()
            price_range = high_max - low_min
            rsv = (data['close'] - low_min) / price_range.replace(0, np.nan) * 100
            rsv = rsv.mask(price_range == 0, 50.0)
            k = rsv.ewm(com=2, adjust=False).mean()
            sig = pd.Series(np.nan, index=data.index)
            sig[k < p['buy_k']] = 1
            sig[k > p['sell_k']] = 0
            return sig.ffill().fillna(0).astype(int)
        return pd.Series(0, index=data.index)


# ==========================================
# 3. 核心交易回測引擎模組 (Backtest Engine Module)
# ==========================================
class BacktestEngine:
    """模擬交易過程，逐日計算資產帳本與交易成本。"""

    @staticmethod
    def _validate_capital_inputs(init_capital, monthly_inflow):
        initial = float(init_capital)
        monthly = float(monthly_inflow)
        if not np.isfinite(initial) or not np.isfinite(monthly):
            raise ValueError("期初本金與每月投入金額必須是有限數值。")
        if initial < 0 or monthly < 0:
            raise ValueError("期初本金與每月投入金額不得為負數。")
        return initial, monthly

    @staticmethod
    def _validate_cost_inputs(fee, tax=0.0, slip=0.0):
        try:
            values = np.asarray([fee, tax, slip], dtype=float)
        except (TypeError, ValueError):
            raise ValueError("手續費、交易稅與滑價必須是數值。")
        if not np.isfinite(values).all() or (values < 0).any() or (values >= 1).any():
            raise ValueError("手續費、交易稅與滑價必須介於 0（含）與 100%（不含）之間。")
        return tuple(float(v) for v in values)

    @staticmethod
    def _validate_price_columns(df, require_open=False):
        if df is None or df.empty:
            raise ValueError("價格資料為空。")
        if "close" not in df.columns:
            raise ValueError("價格資料缺少 close 欄位。")
        checked = ["close"]
        if require_open:
            if "open" not in df.columns:
                raise ValueError("此策略需要 open 欄位才能模擬下一交易日開盤成交。")
            checked.append("open")
        checked.extend(c for c in ("open", "high", "low") if c in df.columns and c not in checked)

        arrays = {}
        for col in checked:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"{col} 價格必須全部是有限正數。")
            arrays[col] = values

        if all(c in arrays for c in ("open", "high", "low", "close")):
            op, hi, lo, cl = (arrays[c] for c in ("open", "high", "low", "close"))
            tol = np.maximum(np.maximum(np.abs(hi), np.abs(lo)), 1.0) * 1e-8
            if np.any(hi + tol < np.maximum(op, cl)):
                raise ValueError("OHLC 資料不一致：high 低於 open 或 close。")
            if np.any(lo - tol > np.minimum(op, cl)):
                raise ValueError("OHLC 資料不一致：low 高於 open 或 close。")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("價格資料索引必須是 DatetimeIndex。")
        if not df.index.is_monotonic_increasing or df.index.has_duplicates:
            raise ValueError("價格資料索引必須依日期遞增且不得重複。")

    @staticmethod
    def _buy_fill_price(reference_price, fee, slip):
        price = float(reference_price)
        fee, slip = float(fee), float(slip)
        if not np.isfinite(price) or price <= 0:
            raise ValueError("買進參考價格必須是有限正數。")
        if not (0 <= fee < 1 and 0 <= slip < 1):
            raise ValueError("手續費率與滑點率必須介於 0（含）與 100%（不含）之間。")
        return price * (1.0 + slip) * (1.0 + fee)

    @staticmethod
    def _sell_net_price(reference_price, fee, tax, slip):
        price = float(reference_price)
        fee, tax, slip = float(fee), float(tax), float(slip)
        if not np.isfinite(price) or price <= 0:
            raise ValueError("賣出參考價格必須是有限正數。")
        if not (0 <= fee < 1 and 0 <= tax < 1 and 0 <= slip < 1):
            raise ValueError("手續費率、交易稅率與滑點率必須介於 0（含）與 100%（不含）之間。")
        return price * (1.0 - slip) * (1.0 - fee) * (1.0 - tax)

    @staticmethod
    def _apply_corporate_actions(data, i, shares, cash):
        dividend = float(data["dividend"].iloc[i]) if "dividend" in data.columns else 0.0
        split_ratio = float(data["split"].iloc[i]) if "split" in data.columns else 0.0
        if not np.isfinite(dividend) or dividend < 0:
            raise ValueError("股利資料必須是有限且非負的數值。")
        if not np.isfinite(split_ratio) or split_ratio < 0:
            raise ValueError("拆股比例資料必須是有限且非負的數值。")
        if dividend > 0 and shares > 0:
            cash += shares * dividend
        return shares, cash

    @staticmethod
    def _attach_portfolio_history(data, portfolio_vals, cash_hist, stock_hist,
                                 external_flows, init_capital, trade_count):
        data = data.copy()
        data["portfolio_value"] = np.asarray(portfolio_vals, dtype=float)
        data["cash"] = np.asarray(cash_hist, dtype=float)
        data["stock_value"] = np.asarray(stock_hist, dtype=float)
        data["external_flow"] = np.asarray(external_flows, dtype=float)
        data["total_invested"] = data["external_flow"].cumsum()
        data["trades"] = int(trade_count)

        equity = []
        index_value = 1.0
        previous_value = None
        for i, (value, flow) in enumerate(zip(data["portfolio_value"], data["external_flow"])):
            if i == 0:
                if init_capital > 0:
                    index_value = max(value / init_capital, 0.0)
                else:
                    index_value = 1.0
            elif previous_value is not None:
                capital_base = previous_value + flow
                if capital_base > 0:
                    daily_return = value / capital_base - 1.0
                    index_value *= max(0.0, 1.0 + daily_return)
                elif previous_value == 0 and flow > 0:
                    index_value *= max(0.0, value / flow)
            equity.append(index_value)
            previous_value = value

        data["equity_net"] = np.asarray(equity, dtype=float)
        return data

    @staticmethod
    def run_dca_strategy(df, init_capital, monthly_inflow, freq_type, dca_amount,
                         custom_days, fee, slip, reinvest_idle=True):
        """定期定額／買入持有策略。

        Parameters:
            reinvest_idle (bool): 若為 True，扣款日會將帳戶所有可用現金（含股利與積累資金）
                                 全額滾動投入，消除 Cash Drag；若為 False 則僅扣固定金額。
        """
        data = df.copy()
        init_capital, monthly_inflow = BacktestEngine._validate_capital_inputs(init_capital, monthly_inflow)
        fee, _, slip = BacktestEngine._validate_cost_inputs(fee, 0.0, slip)
        try:
            dca_amount = float(dca_amount)
        except (TypeError, ValueError):
            raise ValueError("定期投入金額必須是有限非負數值。")
        if not np.isfinite(dca_amount) or dca_amount < 0:
            raise ValueError("定期投入金額必須是有限非負數值。")

        total_days = len(data)
        if total_days == 0:
            return data
        BacktestEngine._validate_price_columns(data)

        month_changed = pd.Series(
            data.index.to_period("M"), index=data.index
        ).ne(pd.Series(data.index.to_period("M"), index=data.index).shift(1))

        if freq_type == "每月 (Monthly)":
            buy_mask = month_changed
        elif freq_type == "每週 (Weekly)":
            iso = data.index.isocalendar()
            week_key = iso["year"].astype(str) + "-" + iso["week"].astype(str)
            buy_mask = week_key.ne(week_key.shift(1))
        elif freq_type == "自訂交易日數":
            try:
                step = float(custom_days)
            except (TypeError, ValueError):
                raise ValueError("自訂交易日數必須是正整數。")
            if not np.isfinite(step) or step < 1 or step != int(step):
                raise ValueError("自訂交易日數必須是正整數。")
            buy_mask = pd.Series(np.arange(total_days) % int(step) == 0, index=data.index)
        else:
            raise ValueError(f"不支援的 DCA 扣款頻率：{freq_type}")

        cash, shares, trade_count = float(init_capital), 0.0, 0
        portfolio_vals, cash_hist, stock_hist, external_flows = [], [], [], []

        for i, date in enumerate(data.index):
            shares, cash = BacktestEngine._apply_corporate_actions(data, i, shares, cash)
            flow = 0.0
            if i > 0 and bool(month_changed.iloc[i]):
                flow = float(monthly_inflow)
                cash += flow

            price = float(data["close"].iloc[i])
            if i == 0 and cash > 0:
                buy_price = BacktestEngine._buy_fill_price(price, fee, slip)
                shares += cash / buy_price
                cash = 0.0
                trade_count += 1
            elif bool(buy_mask.iloc[i]) and cash > 0:
                actual_buy = cash if reinvest_idle else min(cash, float(dca_amount))
                if actual_buy > 0:
                    buy_price = BacktestEngine._buy_fill_price(price, fee, slip)
                    shares += actual_buy / buy_price
                    cash -= actual_buy
                    trade_count += 1

            stock_val = shares * price
            portfolio_vals.append(cash + stock_val)
            cash_hist.append(cash)
            stock_hist.append(stock_val)
            external_flows.append(flow)

        result = BacktestEngine._attach_portfolio_history(
            data, portfolio_vals, cash_hist, stock_hist, external_flows,
            init_capital, trade_count
        )
        result["total_invested"] = float(init_capital) + result["external_flow"].cumsum()
        return result

    @staticmethod
    def run_dip_dca(df, init_capital, monthly_inflow, p, fee, slip):
        data = df.copy()
        init_capital, monthly_inflow = BacktestEngine._validate_capital_inputs(init_capital, monthly_inflow)
        fee, _, slip = BacktestEngine._validate_cost_inputs(fee, 0.0, slip)
        if not isinstance(p, dict):
            raise ValueError("DipDCA 參數必須是字典。")
        try:
            sma_p = float(p.get("sma_period", 60))
            buy_amt = float(p.get("buy_amount", 0.0))
        except (TypeError, ValueError):
            raise ValueError("DipDCA 參數無效。")
        if not np.isfinite(sma_p) or sma_p < 1 or sma_p != int(sma_p):
            raise ValueError("DipDCA 均線週期必須是正整數。")
        if not np.isfinite(buy_amt) or buy_amt < 0:
            raise ValueError("DipDCA 抄底觸發金額必須是有限非負數值。")

        total_days = len(data)
        if total_days == 0:
            return data
        BacktestEngine._validate_price_columns(data, require_open=True)

        sma = IndicatorCalculator.calculate_ma(
            data, int(sma_p), p.get("timeframe", "日線 (Daily)")
        )
        month_changed = pd.Series(
            data.index.to_period("M"), index=data.index
        ).ne(pd.Series(data.index.to_period("M"), index=data.index).shift(1))

        cash, shares, trade_count = float(init_capital), 0.0, 0
        portfolio_vals, cash_hist, stock_hist, external_flows = [], [], [], []
        min_interval = max(1, int(p.get("min_interval", 1)))
        last_buy_idx = -min_interval
        min_price_in_dip = float("inf")
        pending_buy = False

        for i, date in enumerate(data.index):
            shares, cash = BacktestEngine._apply_corporate_actions(data, i, shares, cash)
            flow = 0.0
            if i > 0 and bool(month_changed.iloc[i]):
                flow = float(monthly_inflow)
                cash += flow

            if pending_buy and (i - last_buy_idx >= min_interval) and cash > 0:
                amount = min(cash, buy_amt)
                if amount > 0:
                    buy_price = BacktestEngine._buy_fill_price(float(data["open"].iloc[i]), fee, slip)
                    shares += amount / buy_price
                    cash -= amount
                    last_buy_idx = i
                    trade_count += 1
            pending_buy = False

            price = float(data["close"].iloc[i])
            sma_val = sma.iloc[i]
            if pd.notna(sma_val) and price < float(sma_val):
                is_new_low = price < min_price_in_dip
                min_price_in_dip = min(min_price_in_dip, price)
                pending_buy = bool(is_new_low)
            else:
                min_price_in_dip = float("inf")

            stock_val = shares * price
            portfolio_vals.append(cash + stock_val)
            cash_hist.append(cash)
            stock_hist.append(stock_val)
            external_flows.append(flow)

        result = BacktestEngine._attach_portfolio_history(
            data, portfolio_vals, cash_hist, stock_hist, external_flows,
            init_capital, trade_count
        )
        result["total_invested"] = float(init_capital) + result["external_flow"].cumsum()
        return result

    @staticmethod
    def run_timing_strategy(df, signal_series, fee, tax, slip, init_capital, monthly_inflow):
        data = df.copy()
        init_capital, monthly_inflow = BacktestEngine._validate_capital_inputs(init_capital, monthly_inflow)
        fee, tax, slip = BacktestEngine._validate_cost_inputs(fee, tax, slip)
        if signal_series is None or len(signal_series) == 0:
            raise ValueError("訊號資料為空。")
        sig_vals = pd.to_numeric(signal_series, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(sig_vals).all() or not np.isin(sig_vals, [0, 1]).all():
            raise ValueError("擇時策略訊號必須只包含 0 或 1 的有限二元數值。")

        total_days = len(data)
        if total_days == 0:
            return data
        BacktestEngine._validate_price_columns(data, require_open=True)

        month_changed = pd.Series(
            data.index.to_period("M"), index=data.index
        ).ne(pd.Series(data.index.to_period("M"), index=data.index).shift(1))
        position = signal_series.reindex(data.index).shift(1).fillna(0)

        cash, shares, trade_count = float(init_capital), 0.0, 0
        portfolio_vals, cash_hist, stock_hist, external_flows = [], [], [], []

        for i, date in enumerate(data.index):
            shares, cash = BacktestEngine._apply_corporate_actions(data, i, shares, cash)
            flow = 0.0
            if i > 0 and bool(month_changed.iloc[i]):
                flow = float(monthly_inflow)
                cash += flow

            pos = int(position.iloc[i])
            prev_pos = int(position.iloc[i - 1]) if i > 0 else 0
            execution_price = float(data["open"].iloc[i]) if "open" in data.columns else float(data["close"].iloc[i])
            close_price = float(data["close"].iloc[i])

            if pos > 0 and cash > 0:
                buy_price = BacktestEngine._buy_fill_price(execution_price, fee, slip)
                shares += cash / buy_price
                cash = 0.0
                trade_count += 1
            elif pos == 0 and prev_pos > 0 and shares > 0:
                sell_price = BacktestEngine._sell_net_price(execution_price, fee, tax, slip)
                cash += shares * sell_price
                shares = 0.0
                trade_count += 1

            stock_val = shares * close_price
            portfolio_vals.append(cash + stock_val)
            cash_hist.append(cash)
            stock_hist.append(stock_val)
            external_flows.append(flow)

        result = BacktestEngine._attach_portfolio_history(
            data, portfolio_vals, cash_hist, stock_hist, external_flows,
            init_capital, trade_count
        )
        result["total_invested"] = float(init_capital) + result["external_flow"].cumsum()
        return result

    @classmethod
    def run_named_strategy(cls, df, strat_name, params_dict, init_capital, monthly_inflow, fee, tax, slip):
        """具名策略執行介面。"""
        fee, tax, slip = cls._validate_cost_inputs(fee, tax, slip)
        if strat_name == "定期定額 / 買入持有 (DCA / Buy & Hold)":
            p = params_dict.get("DCA", params_dict)
            reinvest_idle = bool(p.get("reinvest_idle", True))
            return cls.run_dca_strategy(
                df, init_capital, monthly_inflow, 
                p.get("freq", "每月 (Monthly)"), p.get("val", 0.0), 
                p.get("custom_days", 20), fee, slip, reinvest_idle=reinvest_idle
            )
        elif strat_name == "跌破均線且創新低買入":
            p = params_dict.get("DipDCA", params_dict)
            return cls.run_dip_dca(df, init_capital, monthly_inflow, p, fee, slip)
        else:
            p_key = "PriceSMA" if "突破" in strat_name else ("SMA" if "SMA" in strat_name else ("RSI" if "RSI" in strat_name else ("MACD" if "MACD" in strat_name else "KD")))
            p = params_dict.get(p_key, params_dict)
            sig = IndicatorCalculator.generate_signal(df, strat_name, p)
            return cls.run_timing_strategy(df, sig, fee, tax, slip, init_capital, monthly_inflow)


# ==========================================
# 4. 績效評估與統計模組 (Performance Module)
# ==========================================
class PerformanceEvaluator:
    """績效統計、XIRR、夏普/索提諾比率、Holdout 驗證與基準對比。"""

    @staticmethod
    def _calculate_xirr(cashflows, dates):
        try:
            flows = np.asarray(cashflows, dtype=float).reshape(-1)
            dates = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce"))
        except (TypeError, ValueError, OverflowError):
            return np.nan
        if len(flows) < 2 or len(flows) != len(dates):
            return np.nan
        if not np.isfinite(flows).all() or dates.isna().any():
            return np.nan
        if not dates.is_monotonic_increasing:
            return np.nan

        df_cf = pd.DataFrame({"flow": flows, "date": dates}).groupby("date", as_index=False).sum()
        flows = df_cf["flow"].to_numpy(dtype=float)
        dates = pd.DatetimeIndex(df_cf["date"])
        if len(flows) < 2:
            return np.nan

        nonzero_signs = np.sign(flows[flows != 0])
        if len(nonzero_signs) < 2 or not (np.any(flows < 0) and np.any(flows > 0)):
            return np.nan
        sign_changes = int(np.count_nonzero(nonzero_signs[1:] != nonzero_signs[:-1]))
        if sign_changes != 1:
            return np.nan

        years = np.asarray((dates - dates[0]).days, dtype=float) / 365.0
        if years[-1] <= 0:
            return np.nan

        def npv(rate):
            try:
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    value = float(np.sum(flows / np.power(1.0 + rate, years)))
                return value if np.isfinite(value) else np.nan
            except (FloatingPointError, OverflowError, ZeroDivisionError):
                return np.nan

        low, high = -0.999999, 10.0
        f_low, f_high = npv(low), npv(high)
        while np.isfinite(f_low) and np.isfinite(f_high) and np.sign(f_low) == np.sign(f_high) and high < 1_000_000:
            high = high * 2 + 1
            f_high = npv(high)
        if not np.isfinite(f_low) or not np.isfinite(f_high) or np.sign(f_low) == np.sign(f_high):
            return np.nan

        for _ in range(250):
            mid = (low + high) / 2
            f_mid = npv(mid)
            if not np.isfinite(f_mid):
                high = mid
                continue
            if abs(f_mid) < 1e-8 or abs(high - low) < 1e-11:
                return mid
            if np.sign(f_low) != np.sign(f_mid):
                high, f_high = mid, f_mid
            else:
                low, f_low = mid, f_mid
        return (low + high) / 2

    @staticmethod
    def calculate_risk_metrics(df, risk_free_rate=0.0):
        """計算每日 TWR 報酬率、年化波動度、夏普比率與索提諾比率。"""
        if df is None or df.empty or "equity_net" not in df.columns:
            return {}
        try:
            rf = float(risk_free_rate)
        except (TypeError, ValueError):
            return {}
        if not np.isfinite(rf) or rf < 0:
            return {}
        equity = pd.to_numeric(df["equity_net"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(equity).all() or (equity <= 0).any():
            return {}

        daily_returns = np.empty(len(equity), dtype=float)
        daily_returns[0] = equity[0] - 1.0
        if len(equity) > 1:
            daily_returns[1:] = equity[1:] / equity[:-1] - 1.0

        vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
        mean_ret = float(np.mean(daily_returns))

        if vol == 0.0:
            sharpe = np.nan
            sortino = np.nan
        else:
            rf_daily = rf / 252.0
            excess = daily_returns - rf_daily
            sharpe = float(np.mean(excess) / np.std(daily_returns, ddof=1) * np.sqrt(252))

            downside = daily_returns[daily_returns < rf_daily] - rf_daily
            if len(downside) > 0 and np.sum(downside**2) > 0:
                downside_std = float(np.sqrt(np.mean(downside**2)) * np.sqrt(252))
                sortino = float((mean_ret * 252.0 - rf) / downside_std)
            else:
                sortino = np.nan

        return {
            "daily_twr_returns": daily_returns,
            "annualized_volatility": vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino
        }

    @staticmethod
    def evaluate_holdout_period(df, train_ratio=0.70):
        try:
            ratio = float(train_ratio)
        except (TypeError, ValueError):
            return {}
        if not np.isfinite(ratio) or not (0.5 <= ratio <= 0.9):
            return {}
        if df is None or len(df) < 10 or not isinstance(df.index, pd.DatetimeIndex):
            return {}
        if not df.index.is_monotonic_increasing or df.index.has_duplicates or "equity_net" not in df:
            return {}
        equity = pd.to_numeric(df["equity_net"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(equity).all() or (equity < 0).any():
            return {}
        split = int(len(df) * ratio)
        if split < 2 or len(df) - split < 2:
            return {}
        segment = equity[split:]
        base_equity = float(equity[split - 1])
        if base_equity <= 0:
            return {}
        normalized = segment / base_equity
        total_return = float(normalized[-1] - 1.0)
        elapsed_days = max((pd.Timestamp(df.index[-1]) - pd.Timestamp(df.index[split - 1])).days, 1)
        cagr = float(normalized[-1] ** (365.25 / elapsed_days) - 1.0) if normalized[-1] > 0 else -1.0
        with_initial = np.concatenate(([1.0], normalized))
        peaks = np.maximum.accumulate(with_initial)
        drawdowns = with_initial / np.where(peaks == 0, np.nan, peaks) - 1.0
        mdd = float(np.nanmin(drawdowns)) if np.isfinite(drawdowns).any() else 0.0
        return {
            "驗證開始日": pd.Timestamp(df.index[split]).date().isoformat(),
            "驗證結束日": pd.Timestamp(df.index[-1]).date().isoformat(),
            "驗證區間 TWR": total_return,
            "驗證區間年化 TWR": cagr,
            "驗證區間 MDD": mdd,
            "驗證交易日數": int(len(df) - split),
        }

    @classmethod
    def evaluate(cls, df, curr_symbol, initial_capital=None):
        required_cols = {"portfolio_value", "stock_value", "cash", "total_invested", "equity_net", "external_flow", "trades"}
        if df is None or len(df) < 2 or not required_cols.issubset(df.columns):
            return {}

        numeric_cols = ["portfolio_value", "stock_value", "cash", "total_invested", "equity_net", "external_flow", "trades"]
        numeric_values = df[numeric_cols].to_numpy(dtype=float)
        if not np.isfinite(numeric_values).all():
            return {}
        if (df[["portfolio_value", "stock_value", "cash", "total_invested"]] < -1e-8).any().any():
            return {}
        if (df["equity_net"] < -1e-8).any() or (df["trades"] < 0).any():
            return {}
        trades = df["trades"].to_numpy(dtype=float)
        if not np.allclose(trades, np.round(trades), rtol=0, atol=1e-8):
            return {}
        reconciled = df["stock_value"].astype(float) + df["cash"].astype(float)
        if not np.allclose(df["portfolio_value"].astype(float), reconciled, rtol=1e-7, atol=1e-6):
            return {}
        if not df.index.is_monotonic_increasing or df.index.has_duplicates:
            return {}

        if initial_capital is None:
            initial_capital = float(df["total_invested"].iloc[0] - df["external_flow"].iloc[0])
        try:
            initial_capital = float(initial_capital)
        except (TypeError, ValueError):
            return {}
        if not np.isfinite(initial_capital) or initial_capital < 0:
            return {}
        expected_invested = initial_capital + df["external_flow"].astype(float).cumsum()
        if not np.allclose(df["total_invested"].astype(float), expected_invested, rtol=1e-7, atol=1e-6):
            return {}

        final_value = float(df["portfolio_value"].iloc[-1])
        final_stock_val = float(df["stock_value"].iloc[-1])
        final_cash = float(df["cash"].iloc[-1])
        gross_invested = float(df["total_invested"].iloc[-1])
        total_return = float(df["equity_net"].iloc[-1] - 1.0)

        elapsed_days = max((pd.Timestamp(df.index[-1]) - pd.Timestamp(df.index[0])).days, 1)
        ending_index = max(float(df["equity_net"].iloc[-1]), 0.0)
        cagr = (ending_index ** (365.25 / elapsed_days) - 1.0) if ending_index > 0 else -1.0

        equity = df["equity_net"].astype(float)
        equity_with_initial_base = pd.concat([
            pd.Series([1.0], index=[pd.Timestamp(df.index[0]) - pd.Timedelta(nanoseconds=1)]),
            equity
        ])
        running_peak = equity_with_initial_base.cummax()
        drawdown = equity_with_initial_base.div(running_peak.replace(0, np.nan)).sub(1.0)
        mdd = float(drawdown.min()) if drawdown.notna().any() else 0.0

        cashflows = [-float(initial_capital)]
        dates = [df.index[0]]
        for date, flow in df["external_flow"].iloc[1:].items():
            if float(flow) != 0:
                cashflows.append(-float(flow))
                dates.append(date)
        cashflows.append(final_value)
        dates.append(df.index[-1])
        xirr = cls._calculate_xirr(cashflows, dates)

        # 計算風險指標 (Sharpe, Sortino, Volatility)
        risk_metrics = cls.calculate_risk_metrics(df)
        sharpe_val = risk_metrics.get("sharpe_ratio", np.nan)
        sortino_val = risk_metrics.get("sortino_ratio", np.nan)
        vol_val = risk_metrics.get("annualized_volatility", np.nan)

        return {
            "累積投入本金": f"{curr_symbol}{gross_invested:,.0f}",
            "最終資產金額": f"{curr_symbol}{final_value:,.0f}",
            "最終持股市值": f"{curr_symbol}{final_stock_val:,.0f}",
            "最終剩餘現金": f"{curr_symbol}{final_cash:,.0f}",
            "總報酬率 (TWR)": f"{total_return:.2%}",
            "CAGR 年化報酬 (TWR)": f"{cagr:.2%}",
            "XIRR 資金加權年化報酬": f"{xirr:.2%}" if np.isfinite(xirr) else "無法計算",
            "夏普比率 (Sharpe)": f"{sharpe_val:.2f}" if np.isfinite(sharpe_val) else "-",
            "索提諾比率 (Sortino)": f"{sortino_val:.2f}" if np.isfinite(sortino_val) else "-",
            "年化波動度": f"{vol_val:.2%}" if np.isfinite(vol_val) else "-",
            "Max Drawdown (MDD)": f"{mdd:.2%}",
            "累積交易次數": f"{int(df['trades'].iloc[-1]) if isinstance(df['trades'], pd.Series) else int(df['trades'])} 次",
            "_raw_final": final_value,
            "_raw_return": total_return,
            "_raw_cagr": cagr,
            "_raw_xirr": xirr,
            "_raw_mdd": mdd
        }

    @staticmethod
    def compare_to_benchmark(strategy_df, benchmark_df):
        if strategy_df is None or benchmark_df is None or strategy_df.empty or benchmark_df.empty:
            return {}
        if not strategy_df.index.equals(benchmark_df.index):
            return {}
        if "equity_net" not in strategy_df.columns or "equity_net" not in benchmark_df.columns:
            return {}
        s_eq = pd.to_numeric(strategy_df["equity_net"], errors="coerce").to_numpy(dtype=float)
        b_eq = pd.to_numeric(benchmark_df["equity_net"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(s_eq).all() or not np.isfinite(b_eq).all() or (s_eq <= 0).any() or (b_eq <= 0).any():
            return {}

        strat_twr = float(s_eq[-1] - 1.0)
        bench_twr = float(b_eq[-1] - 1.0)
        excess_twr = strat_twr - bench_twr

        def calc_mdd(eq_arr):
            arr_with_base = np.concatenate(([1.0], eq_arr))
            peaks = np.maximum.accumulate(arr_with_base)
            dds = arr_with_base / np.where(peaks == 0, np.nan, peaks) - 1.0
            return float(np.nanmin(dds))

        return {
            "strategy_twr": strat_twr,
            "benchmark_twr": bench_twr,
            "excess_twr": excess_twr,
            "strategy_mdd": calc_mdd(s_eq),
            "benchmark_mdd": calc_mdd(b_eq)
        }

    @staticmethod
    def evaluate_rolling_periods(df, num_periods):
        try:
            k = int(num_periods)
        except (TypeError, ValueError):
            return []
        if k <= 1 or df is None or len(df) < k or not isinstance(df.index, pd.DatetimeIndex):
            return []
        if not df.index.is_monotonic_increasing or df.index.has_duplicates or "equity_net" not in df.columns:
            return []
        equity = pd.to_numeric(df["equity_net"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(equity).all() or (equity <= 0).any():
            return []

        n = len(df)
        chunk_size = n // k
        if chunk_size < 1:
            return []

        results = []
        for i in range(k):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size - 1 if i < k - 1 else n - 1
            base_eq = 1.0 if start_idx == 0 else float(equity[start_idx - 1])
            seg_eq = equity[start_idx:end_idx + 1]
            seg_twr = float(seg_eq[-1] / base_eq - 1.0)

            with_base = np.concatenate(([base_eq], seg_eq))
            peaks = np.maximum.accumulate(with_base)
            dds = with_base / np.where(peaks == 0, np.nan, peaks) - 1.0
            seg_mdd = float(np.nanmin(dds))

            results.append({
                "區間": f"第 {i+1} 區間",
                "開始日": pd.Timestamp(df.index[start_idx]).date().isoformat(),
                "結束日": pd.Timestamp(df.index[end_idx]).date().isoformat(),
                "區間 TWR": seg_twr,
                "區間 MDD": seg_mdd
            })
        return results


# ==========================================
# 5. UI 介面與控制主流程 (Streamlit UI Controller)
# ==========================================
st.title("📈 多市場與多策略比較回測系統 v32.1")

# --- 側邊欄控制區 1：選擇標的與資金 ---
st.sidebar.header("1. 選擇標的與資產")

all_flat_presets = []
for m_list in DataFetcher.PRESET_OPTIONS.values():
    all_flat_presets.extend(m_list)

selected_presets = st.sidebar.multiselect(
    "下拉選取熱門標的 (可跨市場多選)",
    options=all_flat_presets,
    default=["0050 (元大台灣50)", "QQQ (納斯達克100)", "BTC-USD (比特幣)"]
)

custom_input_extra = st.sidebar.text_input(
    "額外補充自訂代碼 (逗號隔開)",
    value="",
    help="例如：2330, NVDA, VOO, AMD。未列入預設清單的代碼，需在下方指定幣別。"
)
currency_override_text = st.sidebar.text_input(
    "自訂代碼幣別對照（選填）",
    value="",
    help="格式：代碼:TWD 或 代碼:USD，多組以逗號分隔，例如：7203.T:TWD。"
)
currency_overrides = {}
for item in currency_override_text.split(","):
    if not item.strip():
        continue
    if ":" not in item:
        st.sidebar.warning(f"幣別對照格式錯誤：{item.strip()}，請使用 代碼:TWD 或 代碼:USD。")
        continue
    override_ticker, override_currency = [part.strip().upper() for part in item.split(":", 1)]
    if override_ticker and override_currency in {"TWD", "USD"}:
        currency_overrides[override_ticker] = override_currency
    else:
        st.sidebar.warning(f"幣別對照無效：{item.strip()}。目前只接受 TWD 或 USD。")

preset_symbols = [p.split(" ")[0].strip().upper() for p in selected_presets]
extra_symbols = [s.strip().upper() for s in custom_input_extra.split(",") if s.strip()]
target_symbols = list(dict.fromkeys(preset_symbols + extra_symbols))

user_currency = st.sidebar.selectbox("輸入資金幣別", ["新台幣 (TWD)", "美元 (USD)"])
initial_capital = st.sidebar.number_input("期初投入本金", value=300000 if "TWD" in user_currency else 10000, step=10000)
monthly_inflow = st.sidebar.number_input("每月固定新增可投入資金", value=30000 if "TWD" in user_currency else 1000, step=1000)
reinvest_dividends = st.sidebar.checkbox(
    "模擬股利入帳（拆股由歷史價格基準反映）",
    value=True
)

time_option = st.sidebar.selectbox("回測時間跨度", ["近 3 年", "近 5 年", "近 10 年", "自訂日期範圍"], index=1)
today = pd.to_datetime("today")
if time_option == "近 3 年":
    start_date, end_date = today - pd.DateOffset(years=3), today
elif time_option == "近 5 年":
    start_date, end_date = today - pd.DateOffset(years=5), today
elif time_option == "近 10 年":
    start_date, end_date = today - pd.DateOffset(years=10), today
else:
    start_date = st.sidebar.date_input("開始日期", value=pd.to_datetime("2015-01-01"))
    end_date = st.sidebar.date_input("結束日期", value=today)

# --- 側邊欄控制區 2：交易成本與資金再投資 ---
st.sidebar.header("2. 交易成本與資金管理")
auto_cost_toggle = st.sidebar.checkbox("依據各資產類別自動設定交易成本", value=True)
reinvest_idle_toggle = st.sidebar.checkbox(
    "閒置資金與股利全額再投資 (Reinvestment)", 
    value=True, 
    help="開啟時扣款日會將帳戶所有現金（含股利與積累資金）全額投入，消除 Cash Drag；關閉時僅扣固定金額。"
)

if not auto_cost_toggle:
    fee_input = st.sidebar.number_input("手續費率 (%)", value=0.071, step=0.01, format="%.3f") / 100
    tax_input = st.sidebar.number_input("賣出證交稅 (%)", value=0.10, step=0.01, format="%.2f") / 100
    slip_input = st.sidebar.number_input("單邊預估滑點 (%)", value=0.02, step=0.01, format="%.2f") / 100

# --- 側邊欄控制區 3：策略選擇與設定 ---
st.sidebar.header("3. 選擇與設定策略")
available_strats = [
    "定期定額 / 買入持有 (DCA / Buy & Hold)", 
    "跌破均線且創新低買入", 
    "價格突破均線買入/跌破賣出", 
    "SMA 雙均線交叉", 
    "RSI 超買超賣", 
    "MACD 柱狀圖/交叉", 
    "KD 隨機指標"
]
selected_strategies = st.sidebar.multiselect("選擇策略", available_strats, default=["定期定額 / 買入持有 (DCA / Buy & Hold)", "跌破均線且創新低買入"])
show_holdout_analysis = st.sidebar.checkbox("顯示後段驗證區間分析（前 70%／後 30%）", value=True)

params = {}
if "定期定額 / 買入持有 (DCA / Buy & Hold)" in selected_strategies:
    st.sidebar.markdown("**[DCA / 買入持有 設定]**")
    params["DCA"] = {
        "freq": st.sidebar.selectbox("DCA 扣款頻率", ["每月 (Monthly)", "每週 (Weekly)", "自訂交易日數"]),
        "val": float(st.sidebar.number_input("每期 DCA 扣款金額", value=30000 if "TWD" in user_currency else 1000, step=1000)),
        "custom_days": 20,
        "reinvest_idle": reinvest_idle_toggle
    }

if "跌破均線且創新低買入" in selected_strategies:
    st.sidebar.markdown("**[跌破均線創新低買入 設定]**")
    params["DipDCA"] = {
        "sma_period": int(st.sidebar.number_input("DipDCA 均線週期", value=60, min_value=1)),
        "timeframe": st.sidebar.selectbox("DipDCA 單位", ["週線 (Weekly)", "日線 (Daily)"]),
        "buy_amount": float(st.sidebar.number_input("抄底觸發金額", value=50000 if "TWD" in user_currency else 1000)),
        "min_interval": int(st.sidebar.number_input("買入最小間隔(日)", value=1, min_value=1))
    }

if "價格突破均線買入/跌破賣出" in selected_strategies:
    params["PriceSMA"] = {"period": int(st.sidebar.number_input("突破策略均線週期", value=20, min_value=1)), "timeframe": "日線 (Daily)"}

if "SMA 雙均線交叉" in selected_strategies:
    params["SMA"] = {"fast": int(st.sidebar.number_input("SMA 快線", value=20, min_value=1)), "slow": int(st.sidebar.number_input("SMA 慢線", value=60, min_value=1)), "timeframe": "日線 (Daily)"}

if "RSI 超買超賣" in selected_strategies:
    params["RSI"] = {"period": 14, "buy": 30, "sell": 70}

if "MACD 柱狀圖/交叉" in selected_strategies:
    params["MACD"] = {"fast": 12, "slow": 26, "signal": 9}

if "KD 隨機指標" in selected_strategies:
    params["KD"] = {"period": 9, "buy_k": 20, "sell_k": 80}

# --- 側邊欄控制區 4：自訂顯示欄位 ---
st.sidebar.header("4. 自訂顯示欄位")
all_metrics_cols = [
    "累積投入本金", "最終資產金額", "最終持股市值", "最終剩餘現金", 
    "總報酬率 (TWR)", "CAGR 年化報酬 (TWR)", "XIRR 資金加權年化報酬", 
    "夏普比率 (Sharpe)", "索提諾比率 (Sortino)", "年化波動度", "Max Drawdown (MDD)", "累積交易次數"
]
selected_cols = st.sidebar.multiselect("表格數據欄位", all_metrics_cols, default=all_metrics_cols)

# --- 主程式執行按鈕與畫面渲染 ---
if st.sidebar.button("開始多策略與多資產回測"):
    if not target_symbols:
        st.error("請至少選擇或輸入一個標的代碼！")
    elif not selected_strategies:
        st.error("請至少選擇一種回測策略。")
    elif initial_capital < 0 or monthly_inflow < 0:
        st.error("期初本金與每月新增資金不可為負數。")
    elif end_date <= start_date:
        st.error("結束日期必須晚於開始日期。")
    elif not auto_cost_toggle and any((not np.isfinite(v) or v < 0 or v >= 1) for v in (fee_input, tax_input, slip_input)):
        st.error("手續費、交易稅與滑點率必須介於 0（含）與 100%（不含）之間。")
    elif "SMA 雙均線交叉" in selected_strategies and params["SMA"]["fast"] >= params["SMA"]["slow"]:
        st.error("SMA 雙均線策略的快線週期必須小於慢線週期。")
    elif "MACD 柱狀圖/交叉" in selected_strategies and params["MACD"]["fast"] >= params["MACD"]["slow"]:
        st.error("MACD 的快線週期必須小於慢線週期。")
    elif "定期定額 / 買入持有 (DCA / Buy & Hold)" in selected_strategies and params["DCA"]["val"] < 0:
        st.error("DCA 每期扣款金額不可為負數。")
    elif "跌破均線且創新低買入" in selected_strategies and params["DipDCA"]["buy_amount"] < 0:
        st.error("DipDCA 抄底觸發金額不可為負數。")
    else:
        with st.spinner("正在獲取數據與運算回測結果..."):
            data_dict, curr_sym, data_warnings = DataFetcher.load_data_and_fx(
                target_symbols,
                start_date,
                pd.Timestamp(end_date) + pd.Timedelta(days=1),
                auto_adjust=reinvest_dividends,
                user_curr=user_currency,
                currency_overrides=currency_overrides
            )

            for warning in data_warnings:
                st.warning(warning)
            for quality_ticker, quality_data in data_dict.items():
                for quality_warning in DataFetcher.data_quality_warnings(quality_data, quality_ticker):
                    st.warning(quality_warning)

            if not data_dict:
                st.error("無法獲取對應標的數據，請確認代碼與日期。")
            else:
                matrix_results = []
                holdout_results = []
                fig = go.Figure()
                
                for sym, data in data_dict.items():
                    if auto_cost_toggle:
                        fee_rate, tax_rate, slippage = DataFetcher.get_asset_costs(sym)
                    else:
                        fee_rate, tax_rate, slippage = fee_input, tax_input, slip_input

                    for strat in selected_strategies:
                        res = BacktestEngine.run_named_strategy(
                            data, strat, params, initial_capital, monthly_inflow, fee_rate, tax_rate, slippage
                        )
                        
                        m = PerformanceEvaluator.evaluate(res, curr_sym, initial_capital=initial_capital)
                        if not m:
                            st.warning(f"{sym}／{strat}：績效資料未通過完整性或帳務驗證，已略過該組結果。")
                            continue
                        m_row = {"標的": sym, "策略": strat, **m}
                        matrix_results.append(m_row)
                        if show_holdout_analysis:
                            holdout = PerformanceEvaluator.evaluate_holdout_period(res, train_ratio=0.70)
                            if holdout:
                                holdout_results.append({"標的": sym, "策略": strat, **holdout})

                        curve_label = f"{sym}-{strat}" if len(data_dict) > 1 else strat
                        fig.add_trace(go.Scatter(x=res.index, y=res['equity_net'] * 100.0, name=f"{curve_label} ({m['_raw_return']:.2%} TWR)"))

                st.subheader("📊 績效與金額對比總表")
                results_df = pd.DataFrame(matrix_results)
                if results_df.empty:
                    st.error("沒有任何策略通過績效驗證，請檢查資料、價格及帳務警告。")
                else:
                    display_cols = ["標的", "策略"] + [c for c in selected_cols if c in results_df.columns]
                    st.dataframe(results_df[display_cols].set_index(["標的", "策略"]), use_container_width=True)

                    # 數據 CSV 匯出
                    csv_data = results_df[display_cols].to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        label="📥 下載績效對比總表 (CSV)",
                        data=csv_data,
                        file_name="backtest_performance_summary.csv",
                        mime="text/csv"
                    )

                if show_holdout_analysis:
                    st.subheader("🧪 後段驗證區間（固定參數診斷）")
                    if holdout_results:
                        holdout_df = pd.DataFrame(holdout_results)
                        st.dataframe(holdout_df, use_container_width=True)
                    else:
                        st.warning("資料長度不足或驗證區間無有效淨值，無法計算後段驗證指標。")

                if len(data_dict) > 1 and not results_df.empty:
                    st.subheader("🔀 多資產 × 多策略 總報酬率矩陣 (%)")
                    pivot_df = results_df.pivot(index="標的", columns="策略", values="_raw_return") * 100
                    
                    def format_matrix_with_crown(df):
                        formatted_df = df.copy().astype(object)
                        for row in df.index:
                            max_val = df.loc[row].max()
                            for col in df.columns:
                                val = df.loc[row, col]
                                if pd.isna(val):
                                    formatted_df.loc[row, col] = "-"
                                elif val == max_val:
                                    formatted_df.loc[row, col] = f"👑 {val:.2f}%"
                                else:
                                    formatted_df.loc[row, col] = f"{val:.2f}%"
                        return formatted_df

                    formatted_matrix = format_matrix_with_crown(pivot_df)
                    st.dataframe(formatted_matrix, use_container_width=True)

                st.subheader("📈 資產成長曲線圖")
                fig.update_layout(xaxis_title="日期", yaxis_title="TWR 淨值指數（期初 = 100）", height=500)
                st.plotly_chart(fig, use_container_width=True)
