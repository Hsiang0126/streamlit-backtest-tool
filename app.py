import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 0. 頁面基本設定 (Page Config)
# ==========================================
# 設定 Streamlit 為寬螢幕模式，並設定網頁頁籤標題
st.set_page_config(layout="wide", page_title="多市場與多策略回測工具")


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
        # 常見非美元交易所後綴：未支援對應匯率路由前拒絕自動換算。
        unsupported_suffixes = (".HK", ".SS", ".SZ", ".L", ".TO", ".V", ".AX", ".SI", ".KS", ".KQ", ".T", ".PA", ".DE", ".MI", ".SW", ".AS", ".BR", ".HE", ".ST", ".OL", ".CO", ".NZ")
        if t.endswith(unsupported_suffixes):
            return "UNKNOWN"
        # 僅自動辨識已列入本工具美股預設清單的無後綴代碼；其他自訂代碼需明確指定。
        known_us = {"QQQ", "AAPL", "NVDA", "SPY", "TSLA", "MSFT"}
        if t in known_us or t.endswith("-USD"):
            return "USD"
        return "UNKNOWN"

    @staticmethod
    def _extract_ticker_frame(downloaded, ticker, single_ticker):
        if downloaded is None or downloaded.empty:
            return pd.DataFrame()
        if isinstance(downloaded.columns, pd.MultiIndex):
            # yfinance 不同版本可能將 ticker 放在第一層或第二層。
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

    @classmethod
    @st.cache_data(show_spinner=False)
    def load_data_and_fx(cls, tickers, start, end, auto_adjust=True, user_curr="新台幣 (TWD)", currency_overrides=None):
        """回傳 (價格資料字典, 顯示幣別符號, 資料警告)。只支援 TWD/USD；未知自訂代碼需明確指定幣別。"""
        fmt_to_orig = {cls.normalize_ticker(t): t.strip().upper() for t in tickers}
        if not fmt_to_orig:
            return {}, ("NT$" if "TWD" in user_curr else "$"), ["沒有可下載的標的代碼。"]

        warnings = []
        try:
            # 始終下載未調整 OHLC；股利與拆股透過事件帳本明確處理，避免調整價格再加股利造成重複計算。
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
                    warnings.append(f"{orig_t}：沒有可用的收盤價資料，已略過。請確認代碼及日期範圍。")
                    continue
                close = pd.to_numeric(frame["Close"], errors="coerce")
                open_price = pd.to_numeric(frame["Open"], errors="coerce") if "Open" in frame else close.copy()
                high = pd.to_numeric(frame["High"], errors="coerce") if "High" in frame else close.copy()
                low = pd.to_numeric(frame["Low"], errors="coerce") if "Low" in frame else close.copy()
                # 保留來源 OHLC 作為回測價格欄位；若 yfinance 使用 auto_adjust=True，
                # 這些 OHLC 可能已按股利／拆股調整，不能等同歷史實際成交價。
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
                # 此處明確使用 yf.download(auto_adjust=False) 取得未調整 OHLC。
                # 未調整歷史價格已反映實際歷史交易時的價格尺度，不應再手動乘拆股比例；
                # 拆股日由 BacktestEngine._apply_corporate_actions 調整持股數即可。
                # auto_adjust 參數為維持舊介面相容，實際意義是「是否模擬企業行動」。
                if not auto_adjust:
                    res["dividend"] = 0.0
                    res["split"] = 0.0
                override_map = {str(k).strip().upper(): str(v).strip().upper() for k, v in (currency_overrides or {}).items()}
                quote_currency = cls._quote_currency(orig_t, override_map)
                if quote_currency == "UNKNOWN":
                    warnings.append(f"{orig_t}：無法自動辨識報價幣別。請在「自訂代碼幣別對照」中指定 TWD 或 USD；若標的實際幣別不是 TWD/USD，目前尚未支援。")
                res["quote_currency"] = quote_currency
                processed_data[orig_t] = res
            except Exception as exc:
                warnings.append(f"{orig_t}：整理資料時發生錯誤，已略過（{exc}）。")

        if auto_adjust:
            warnings.append(
                "目前使用 Yahoo Finance 未調整 OHLC；股利按事件日持有股數計入現金，拆股日調整持股數。"
                "股利不會自動再投資；若來源未提供完整企業行動資料，總報酬仍可能不完整。"
            )
        else:
            warnings.append(
                "目前使用未調整 OHLC，且已關閉股利／拆股事件處理。若回測期間發生企業行動，績效將不代表完整總報酬。"
            )

        target_currency = "TWD" if "TWD" in user_curr else "USD"
        # 先移除不支援的報價幣別，避免例如港股/日股被錯當美元資產估值。
        for ticker in list(processed_data):
            quote_currency = processed_data[ticker]["quote_currency"].iloc[0]
            if quote_currency not in {"TWD", "USD"}:
                warnings.append(
                    f"{ticker}：報價幣別無法可靠辨識或尚未支援；目前只支援 TWD 與 USD 報價，已略過。"
                )
                del processed_data[ticker]
        needed_fx = any(df["quote_currency"].iloc[0] != target_currency for df in processed_data.values())
        fx_close = None
        if needed_fx:
            try:
                fx_raw = yf.download("USDTWD=X", start=start, end=end, auto_adjust=True, progress=False)
                fx_frame = cls._extract_ticker_frame(fx_raw, "USDTWD=X", True)
                if not fx_frame.empty and "Close" in fx_frame.columns:
                    fx_close = pd.to_numeric(fx_frame["Close"], errors="coerce")
                    fx_close.index = pd.to_datetime(fx_close.index)
                    fx_close = fx_close.replace([np.inf, -np.inf], np.nan).dropna()
                    fx_close = fx_close[fx_close > 0].sort_index()
                if fx_close is None or fx_close.empty:
                    warnings.append("無法取得 USD/TWD 歷史匯率；需要換匯的標的將略過，避免使用固定匯率造成失真。")
            except Exception as exc:
                warnings.append(f"下載 USD/TWD 匯率失敗：{exc}。需要換匯的標的將略過。")

        for ticker in list(processed_data):
            data = processed_data[ticker]
            quote_currency = data["quote_currency"].iloc[0]
            if quote_currency == "UNKNOWN":
                warnings.append(f"{ticker}：無法辨識報價幣別，已略過。")
                del processed_data[ticker]
                continue
            if quote_currency == target_currency:
                continue
            if fx_close is None or fx_close.empty:
                warnings.append(f"{ticker}：缺少必要匯率資料，已略過。")
                del processed_data[ticker]
                continue
            fx = fx_close.reindex(data.index, method="ffill")
            valid = fx.notna() & (fx > 0)
            data = data.loc[valid].copy()
            fx = fx.loc[valid]
            if data.empty:
                warnings.append(f"{ticker}：價格日期無法對應到有效匯率，已略過。")
                del processed_data[ticker]
                continue
            # 對每個交易日分別換算估值價格，才能反映持有期間的匯率變化。
            # 股利也按事件當日匯率換算；split 是股數比例，不能換匯。
            multiplier = fx if target_currency == "TWD" else (1.0 / fx)
            for col in ("open", "close", "high", "low", "dividend"):
                data[col] = data[col] * multiplier
            data["fx_rate_usdtwd"] = fx.astype(float)
            processed_data[ticker] = data

        # quote_currency 僅供轉換內部使用，移除以免被指標或策略誤當數值欄位。
        for data in processed_data.values():
            if "quote_currency" in data:
                data.drop(columns=["quote_currency"], inplace=True)
        curr_symbol = "NT$" if target_currency == "TWD" else "$"
        return processed_data, curr_symbol, warnings


# ==========================================
# 2. 技術指標與訊號產生模組 (Indicator Module)
# ==========================================
class IndicatorCalculator:
    """
    【指標模組】負責計算移動平均線 (MA)、RSI、MACD、KD 等技術指標，
    並根據指標規則產生買賣訊號 (1 表示持有股票，0 表示空倉手握現金)。
    """
    
    @staticmethod
    def calculate_ma(df, period, timeframe="日線 (Daily)"):
        """【功能】計算日均線或週均線 (SMA)"""
        if timeframe == "週線 (Weekly)":
            weekly_close = df['close'].resample('W-FRI').last().dropna()
            weekly_ma = weekly_close.rolling(period).mean()
            return weekly_ma.reindex(df.index, method='ffill')
        return df['close'].rolling(period).mean()

    @classmethod
    def generate_signal(cls, df, strat_name, p):
        """【功能】根據選擇的技術指標策略，計算每日的買賣訊號 (1=買進持有, 0=賣出空倉)"""
        data = df.copy()
        
        # 策略 1：價格突破均線買入，跌破均線賣出
        if strat_name == "價格突破均線買入/跌破賣出":
            sma = cls.calculate_ma(data, p['period'], p.get('timeframe', '日線 (Daily)'))
            return (data['close'] > sma).astype(int)
            
        # 策略 2：SMA 快慢雙均線黃金交叉買入，死亡交叉賣出
        elif strat_name == "SMA 雙均線交叉":
            fast_m = cls.calculate_ma(data, p['fast'], p.get('timeframe', '日線 (Daily)'))
            slow_m = cls.calculate_ma(data, p['slow'], p.get('timeframe', '日線 (Daily)'))
            return (fast_m > slow_m).astype(int)
            
        # 策略 3：RSI 超賣 (低於買入界線) 買入，超買 (高於賣出界線) 賣出
        elif strat_name == "RSI 超買超賣":
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(p['period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(p['period']).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            sig = pd.Series(np.nan, index=data.index)
            sig[rsi < p['buy']] = 1
            sig[rsi > p['sell']] = 0
            return sig.ffill().fillna(0).astype(int)
            
        # 策略 4：MACD 快線突破慢線 (柱狀圖 > 0) 買入，反之賣出
        elif strat_name == "MACD 柱狀圖/交叉":
            ema_fast = data['close'].ewm(span=p['fast'], adjust=False).mean()
            ema_slow = data['close'].ewm(span=p['slow'], adjust=False).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=p['signal'], adjust=False).mean()
            return (macd > signal).astype(int)
            
        # 策略 5：KD 指標 K 值低於超賣區買入，高於超買區賣出
        elif strat_name == "KD 隨機指標":
            low_min = data['low'].rolling(p['period']).min()
            high_max = data['high'].rolling(p['period']).max()
            rsv = (data['close'] - low_min) / (high_max - low_min + 1e-9) * 100
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
    """
    【回測引擎模組】模擬真實交易過程，每日逐筆計算「現金」、「持股股數」、「總資產價值」與「交易手續費」。
    """
    
    @staticmethod
    def _validate_capital_inputs(init_capital, monthly_inflow):
        """驗證期初本金與每月入金，避免負資金或非有限數值破壞資產帳本。"""
        initial = float(init_capital)
        monthly = float(monthly_inflow)
        if not np.isfinite(initial) or not np.isfinite(monthly):
            raise ValueError("期初本金與每月投入金額必須是有限數值。")
        if initial < 0 or monthly < 0:
            raise ValueError("期初本金與每月投入金額不得為負數。")
        return initial, monthly

    @staticmethod
    def _validate_price_columns(df, require_open=False):
        """驗證回測所需價格欄位，拒絕非正數或無效收盤價。"""
        if "close" not in df.columns:
            raise ValueError("價格資料缺少 close 欄位。")
        close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError("close 價格必須全部是有限正數。")
        if require_open and "open" in df.columns:
            op = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(op).all() or (op <= 0).any():
                raise ValueError("open 價格必須全部是有限正數。")

    @staticmethod
    def _buy_fill_price(reference_price, fee, slip):
        """買進成交價：先套用不利滑點，再計入手續費；各項成本分開建模。"""
        price = float(reference_price)
        fee, slip = float(fee), float(slip)
        if not np.isfinite(price) or price <= 0:
            raise ValueError("買進參考價格必須是有限正數。")
        if not (0 <= fee < 1 and 0 <= slip < 1):
            raise ValueError("手續費率與滑點率必須介於 0（含）與 100%（不含）之間。")
        return price * (1.0 + slip) * (1.0 + fee)

    @staticmethod
    def _sell_net_price(reference_price, fee, tax, slip):
        """賣出每股淨收入：依序扣除不利滑點、手續費與交易稅。"""
        price = float(reference_price)
        fee, tax, slip = float(fee), float(tax), float(slip)
        if not np.isfinite(price) or price <= 0:
            raise ValueError("賣出參考價格必須是有限正數。")
        if not (0 <= fee < 1 and 0 <= tax < 1 and 0 <= slip < 1):
            raise ValueError("手續費率、交易稅率與滑點率必須介於 0（含）與 100%（不含）之間。")
        return price * (1.0 - slip) * (1.0 - fee) * (1.0 - tax)

    @staticmethod
    def _apply_corporate_actions(data, i, shares, cash):
        """在當日交易前處理企業行動：股利按前收盤持股入帳，拆股調整股數。"""
        dividend = float(data["dividend"].iloc[i]) if "dividend" in data.columns else 0.0
        split_ratio = float(data["split"].iloc[i]) if "split" in data.columns else 0.0
        if not np.isfinite(dividend) or dividend < 0:
            raise ValueError("股利資料必須是有限且非負的數值。")
        if not np.isfinite(split_ratio) or split_ratio < 0:
            raise ValueError("拆股比例資料必須是有限且非負的數值。")
        # Yahoo Finance 的 Stock Splits 通常為 2.0、0.5 等比例；0 代表無拆股事件。
        if dividend > 0 and shares > 0:
            cash += shares * dividend
        if split_ratio > 0 and shares > 0:
            shares *= split_ratio
        return shares, cash

    @staticmethod
    def _attach_portfolio_history(data, portfolio_vals, cash_hist, stock_hist,
                                  external_flows, init_capital, trade_count):
        """統一寫入資產歷史、外部資金流與扣除入金影響的淨值曲線。"""
        data = data.copy()
        data["portfolio_value"] = np.asarray(portfolio_vals, dtype=float)
        data["cash"] = np.asarray(cash_hist, dtype=float)
        data["stock_value"] = np.asarray(stock_hist, dtype=float)
        data["external_flow"] = np.asarray(external_flows, dtype=float)
        data["total_invested"] = data["external_flow"].cumsum()
        data["trades"] = int(trade_count)

        # TWR 淨值：入金在當日交易前發生，因此當日報酬分母為「前日資產 + 當日外部入金」。
        # 第 0 日以期初本金為基準；後續日避免將新增資金誤認成投資報酬。
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
                    # 從零資產開始的第一筆入金：以入金作為當日資本基礎。
                    index_value *= max(0.0, value / flow)
                # 前日資產與當日入金皆為 0 時，保留原淨值，避免除以 0。
            equity.append(index_value)
            previous_value = value

        data["equity_net"] = np.asarray(equity, dtype=float)
        return data

    @staticmethod
    def run_dca_strategy(df, init_capital, monthly_inflow, freq_type, dca_amount,
                         custom_days, fee, slip):
        """定期定額／買入持有。外部入金與交易損益分開記錄。"""
        data = df.copy()
        init_capital, monthly_inflow = BacktestEngine._validate_capital_inputs(init_capital, monthly_inflow)
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
        else:
            step = max(1, int(custom_days))
            buy_mask = pd.Series(np.arange(total_days) % step == 0, index=data.index)

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
            elif bool(buy_mask.iloc[i]) and cash > 0 and dca_amount > 0:
                actual_buy = min(cash, float(dca_amount))
                buy_price = BacktestEngine._buy_fill_price(price, fee, slip)
                shares += actual_buy / buy_price
                cash -= actual_buy
                trade_count += 1

            stock_val = shares * price
            portfolio_vals.append(cash + stock_val)
            cash_hist.append(cash)
            stock_hist.append(stock_val)
            external_flows.append(flow)

        # 初始本金也屬於外部資金流，但不重複加入每日 flow 欄位的累積入金計算。
        result = BacktestEngine._attach_portfolio_history(
            data, portfolio_vals, cash_hist, stock_hist, external_flows,
            init_capital, trade_count
        )
        result["total_invested"] = float(init_capital) + result["external_flow"].cumsum()
        return result

    @staticmethod
    def run_dip_dca(df, init_capital, monthly_inflow, p, fee, slip):
        """跌破均線且創新低加碼；最低價追蹤與買入限制分開處理。"""
        data = df.copy()
        init_capital, monthly_inflow = BacktestEngine._validate_capital_inputs(init_capital, monthly_inflow)
        total_days = len(data)
        if total_days == 0:
            return data
        BacktestEngine._validate_price_columns(data)

        sma = IndicatorCalculator.calculate_ma(
            data, p["sma_period"], p.get("timeframe", "日線 (Daily)")
        )
        month_changed = pd.Series(
            data.index.to_period("M"), index=data.index
        ).ne(pd.Series(data.index.to_period("M"), index=data.index).shift(1))

        cash, shares, trade_count = float(init_capital), 0.0, 0
        portfolio_vals, cash_hist, stock_hist, external_flows = [], [], [], []
        last_buy_idx = -max(1, int(p.get("min_interval", 1)))
        min_price_in_dip = float("inf")

        for i, date in enumerate(data.index):
            shares, cash = BacktestEngine._apply_corporate_actions(data, i, shares, cash)
            flow = 0.0
            if i > 0 and bool(month_changed.iloc[i]):
                flow = float(monthly_inflow)
                cash += flow

            price = float(data["close"].iloc[i])
            sma_val = sma.iloc[i]
            if pd.notna(sma_val) and price < float(sma_val):
                is_new_low = price < min_price_in_dip
                # 無論是否允許下單，都要更新區間最低價。
                min_price_in_dip = min(min_price_in_dip, price)
                if is_new_low and (i - last_buy_idx >= max(1, int(p.get("min_interval", 1)))) and cash > 0:
                    amount = min(cash, max(0.0, float(p.get("buy_amount", 0.0))))
                    if amount > 0:
                        buy_price = BacktestEngine._buy_fill_price(price, fee, slip)
                        shares += amount / buy_price
                        cash -= amount
                        last_buy_idx = i
                        trade_count += 1
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
        """擇時回測：以前一日訊號決定當日交易，並逐日記錄外部入金。

        成交成本分開套用：買進價 = 開盤參考價 × (1+滑點) × (1+手續費)；
        賣出淨價 = 開盤參考價 × (1-滑點) × (1-手續費) × (1-交易稅)。
        """
        data = df.copy()
        init_capital, monthly_inflow = BacktestEngine._validate_capital_inputs(init_capital, monthly_inflow)
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
            # 訊號來自前一交易日收盤資料；當日開盤成交，避免同日收盤價前視偏誤。
            execution_price = float(data["open"].iloc[i]) if "open" in data.columns else float(data["close"].iloc[i])
            close_price = float(data["close"].iloc[i])

            if pos > 0 and cash > 0:
                # 只要當日目標持倉為多頭，就把可用現金投入；包含持倉期間的新月度入金。
                # 交易價格採當日開盤價，估值仍採當日收盤價。
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

# ==========================================
# 4. 績效評估與統計模組 (Performance Module)
# ==========================================
class PerformanceEvaluator:
    """
    【績效模組】計算總資產、TWR、XIRR、CAGR 與最大回撤。
    TWR 用於比較策略；XIRR 用於反映不同日期投入資金的資金加權報酬。
    """
    
    @staticmethod
    def _calculate_xirr(cashflows, dates):
        """以 365 日為一年計算 XIRR（符合常見 XIRR day-count 慣例）；無有效解時回傳 NaN。"""
        flows = np.asarray(cashflows, dtype=float)
        dates = pd.to_datetime(dates)
        if len(flows) < 2 or not (np.any(flows < 0) and np.any(flows > 0)):
            return np.nan
        years = np.asarray((dates - dates[0]).days, dtype=float) / 365.0

        def npv(rate):
            try:
                return float(np.sum(flows / np.power(1.0 + rate, years)))
            except (FloatingPointError, OverflowError, ZeroDivisionError):
                return np.nan

        low, high = -0.9999, 10.0
        f_low, f_high = npv(low), npv(high)
        while np.isfinite(f_low) and np.isfinite(f_high) and f_low * f_high > 0 and high < 1_000_000:
            high = high * 2 + 1
            f_high = npv(high)
        if not np.isfinite(f_low) or not np.isfinite(f_high) or f_low * f_high > 0:
            return np.nan

        for _ in range(200):
            mid = (low + high) / 2
            f_mid = npv(mid)
            if not np.isfinite(f_mid):
                high = mid
                continue
            if abs(f_mid) < 1e-7 or abs(high - low) < 1e-10:
                return mid
            if f_low * f_mid <= 0:
                high, f_high = mid, f_mid
            else:
                low, f_low = mid, f_mid
        return (low + high) / 2

    @classmethod
    def evaluate(cls, df, curr_symbol, initial_capital=None):
        """評估總資產、TWR、XIRR 與基於 TWR 淨值曲線的最大回撤。"""
        required_cols = {"portfolio_value", "stock_value", "cash", "total_invested", "equity_net", "external_flow", "trades"}
        if df is None or len(df) < 2 or not required_cols.issubset(df.columns):
            return {}

        # 排除非有限數值，避免損壞資料將績效指標傳染成無限值或錯誤結果。
        numeric_cols = ["portfolio_value", "stock_value", "cash", "total_invested", "equity_net", "external_flow"]
        numeric_values = df[numeric_cols].to_numpy(dtype=float)
        if not np.isfinite(numeric_values).all():
            return {}
        # 資產負值或資產帳務無法對帳時，拒絕輸出績效，避免呈現看似精確的錯誤結果。
        if (df[["portfolio_value", "stock_value", "cash", "total_invested"]] < -1e-8).any().any():
            return {}
        reconciled = df["stock_value"].astype(float) + df["cash"].astype(float)
        if not np.allclose(df["portfolio_value"].astype(float), reconciled, rtol=1e-7, atol=1e-6):
            return {}
        if not df.index.is_monotonic_increasing or df.index.has_duplicates:
            return {}

        final_value = float(df["portfolio_value"].iloc[-1])
        final_stock_val = float(df["stock_value"].iloc[-1])
        final_cash = float(df["cash"].iloc[-1])
        gross_invested = float(df["total_invested"].iloc[-1])
        total_return = float(df["equity_net"].iloc[-1] - 1.0)

        elapsed_days = max((pd.Timestamp(df.index[-1]) - pd.Timestamp(df.index[0])).days, 1)
        ending_index = max(float(df["equity_net"].iloc[-1]), 0.0)
        cagr = (ending_index ** (365.25 / elapsed_days) - 1.0) if ending_index > 0 else -1.0

        # MDD 必須包含期初本金基準 1.0。若第一天因手續費或滑點而立即虧損，
        # 不能只從第一筆已交易後的 equity_net 開始計算，否則會漏掉首日回撤。
        equity = df["equity_net"].astype(float)
        equity_with_initial_base = pd.concat([
            pd.Series([1.0], index=[pd.Timestamp(df.index[0]) - pd.Timedelta(nanoseconds=1)]),
            equity
        ])
        running_peak = equity_with_initial_base.cummax()
        drawdown = equity_with_initial_base.div(running_peak.replace(0, np.nan)).sub(1.0)
        mdd = float(drawdown.min()) if drawdown.notna().any() else 0.0

        # XIRR：初始本金與後續入金視為負現金流，期末總資產視為正現金流。
        if initial_capital is None:
            initial_capital = float(df["total_invested"].iloc[0])
        cashflows = [-float(initial_capital)]
        dates = [df.index[0]]
        for date, flow in df["external_flow"].iloc[1:].items():
            if float(flow) != 0:
                cashflows.append(-float(flow))
                dates.append(date)
        cashflows.append(final_value)
        dates.append(df.index[-1])
        xirr = cls._calculate_xirr(cashflows, dates)

        return {
            "累積投入本金": f"{curr_symbol}{gross_invested:,.0f}",
            "最終資產金額": f"{curr_symbol}{final_value:,.0f}",
            "最終持股市值": f"{curr_symbol}{final_stock_val:,.0f}",
            "最終剩餘現金": f"{curr_symbol}{final_cash:,.0f}",
            "總報酬率 (TWR)": f"{total_return:.2%}",
            "CAGR 年化報酬 (TWR)": f"{cagr:.2%}",
            "XIRR 資金加權年化報酬": f"{xirr:.2%}" if np.isfinite(xirr) else "無法計算",
            "Max Drawdown (MDD)": f"{mdd:.2%}",
            "累積交易次數": f"{int(df['trades'].iloc[-1]) if isinstance(df['trades'], pd.Series) else int(df['trades'])} 次",
            "_raw_final": final_value,
            "_raw_return": total_return,
            "_raw_cagr": cagr,
            "_raw_xirr": xirr,
            "_raw_mdd": mdd
        }



# ==========================================
# 5. UI 介面與控制主流程 (Streamlit UI Controller)
# ==========================================
st.title("📈 多市場與多策略比較回測系統")

# --- 側邊欄控制區 1：選擇標的與資金 ---
st.sidebar.header("1. 選擇標的與資產")

all_flat_presets = []
for m_list in DataFetcher.PRESET_OPTIONS.values():
    all_flat_presets.extend(m_list)

# 熱門標的多選下拉選單
selected_presets = st.sidebar.multiselect(
    "下拉選取熱門標的 (可跨市場多選)",
    options=all_flat_presets,
    default=["0050 (元大台灣50)", "QQQ (納斯達克100)", "BTC-USD (比特幣)"]
)

# 額外自訂代碼輸入框
custom_input_extra = st.sidebar.text_input(
    "額外補充自訂代碼 (逗號隔開)",
    value="",
    help="如需補充下拉選單沒有的標的，請在此輸入（例如：2330, NVDA）。未列入預設清單的代碼，需在下方指定報價幣別。"
)
currency_override_text = st.sidebar.text_input(
    "自訂代碼幣別對照（選填）",
    value="",
    help="格式：代碼:TWD 或 代碼:USD，多組以逗號分隔，例如：XYZ:USD, 7203.T:TWD。此版本只支援 TWD/USD；非這兩種幣別的標的仍不支援。"
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

# 解析並合併標的（自動去重）
preset_symbols = [p.split(" ")[0].strip().upper() for p in selected_presets]
extra_symbols = [s.strip().upper() for s in custom_input_extra.split(",") if s.strip()]
target_symbols = list(dict.fromkeys(preset_symbols + extra_symbols))

user_currency = st.sidebar.selectbox("輸入資金幣別", ["新台幣 (TWD)", "美元 (USD)"])
initial_capital = st.sidebar.number_input("期初投入本金", value=300000 if "TWD" in user_currency else 10000, step=10000)
monthly_inflow = st.sidebar.number_input("每月固定新增可投入資金", value=30000 if "TWD" in user_currency else 1000, step=1000)
reinvest_dividends = st.sidebar.checkbox(
    "模擬股利入帳與拆股持股數調整",
    value=True,
    help="使用未調整 OHLC。開啟時，股利依除息日持股計入現金、拆股依事件比例調整持股數；關閉時不處理企業行動。這不是股利自動再投資，且依賴 Yahoo Finance 事件資料完整性。"
)

# 選擇時間跨度
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

# --- 側邊欄控制區 2：交易成本 ---
st.sidebar.header("2. 交易成本設定")
auto_cost_toggle = st.sidebar.checkbox("依據各資產類別自動設定交易成本", value=True, help="勾選後，台股、美股、加密貨幣將分別套用各自合理的預設手續費與證交稅率。")

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

params = {}
if "定期定額 / 買入持有 (DCA / Buy & Hold)" in selected_strategies:
    st.sidebar.markdown("**[DCA / 買入持有 設定]**")
    params["DCA"] = {
        "freq": st.sidebar.selectbox("DCA 扣款頻率", ["每月 (Monthly)", "每週 (Weekly)", "自訂交易日數"]),
        "val": float(st.sidebar.number_input("每期 DCA 扣款金額", value=30000 if "TWD" in user_currency else 1000, step=1000)),
        "custom_days": 20
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

# --- 側邊欄控制區 4：自訂表格顯示欄位 ---
st.sidebar.header("4. 自訂顯示欄位")
all_metrics_cols = ["累積投入本金", "最終資產金額", "最終持股市值", "最終剩餘現金", "總報酬率 (TWR)", "CAGR 年化報酬 (TWR)", "XIRR 資金加權年化報酬", "Max Drawdown (MDD)", "累積交易次數"]
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
            # 1. 向數據模組請求數據
            data_dict, curr_sym, data_warnings = DataFetcher.load_data_and_fx(
                target_symbols,
                start_date,
                end_date,
                auto_adjust=reinvest_dividends,
                user_curr=user_currency,
                currency_overrides=currency_overrides
            )

            for warning in data_warnings:
                st.warning(warning)

            if not data_dict:
                st.error("無法獲取對應標的數據，請確認代碼與日期。")
            else:
                matrix_results = []
                fig = go.Figure()
                
                # 2. 針對每個標的與選定的策略進行交叉回測
                for sym, data in data_dict.items():
                    # 決定交易成本
                    if auto_cost_toggle:
                        fee_rate, tax_rate, slippage = DataFetcher.get_asset_costs(sym)
                    else:
                        fee_rate, tax_rate, slippage = fee_input, tax_input, slip_input

                    for strat in selected_strategies:
                        # 路由到對應的策略回測引擎
                        if strat == "定期定額 / 買入持有 (DCA / Buy & Hold)":
                            res = BacktestEngine.run_dca_strategy(
                                data, initial_capital, monthly_inflow, 
                                params["DCA"]["freq"], params["DCA"]["val"], 
                                params["DCA"]["custom_days"], fee_rate, slippage
                            )
                        elif strat == "跌破均線且創新低買入":
                            res = BacktestEngine.run_dip_dca(data, initial_capital, monthly_inflow, params["DipDCA"], fee_rate, slippage)
                        else:
                            p_key = "PriceSMA" if "突破" in strat else ("SMA" if "SMA" in strat else ("RSI" if "RSI" in strat else ("MACD" if "MACD" in strat else "KD")))
                            sig = IndicatorCalculator.generate_signal(data, strat, params.get(p_key, {}))
                            res = BacktestEngine.run_timing_strategy(data, sig, fee_rate, tax_rate, slippage, initial_capital, monthly_inflow)
                        
                        # 3. 計算績效數據
                        m = PerformanceEvaluator.evaluate(res, curr_sym)
                        m_row = {"標的": sym, "策略": strat, **m}
                        matrix_results.append(m_row)
                        
                        # 4. 繪製資產成長曲線
                        curve_label = f"{sym}-{strat}" if len(data_dict) > 1 else strat
                        fig.add_trace(go.Scatter(x=res.index, y=res['portfolio_value'], name=f"{curve_label} ({curr_sym}{m['_raw_final']:,.0f})"))

                # 5. 輸出績效對比總表
                st.subheader("📊 績效與金額對比總表")
                st.caption("TWR 用於比較策略表現；每月新增資金於當日交易前入帳。系統使用未調整 OHLC，並可依 Yahoo Finance 企業行動資料處理股利現金入帳與拆股持股數調整；股利目前留在現金、不自動再投資。資料來源若漏列企業行動，回測仍可能偏離實際結果。買賣成本以獨立乘數估算，並非券商撮合模擬。")
                results_df = pd.DataFrame(matrix_results)
                display_cols = ["標的", "策略"] + [c for c in selected_cols if c in results_df.columns]
                st.dataframe(results_df[display_cols].set_index(["標的", "策略"]), use_container_width=True)

                # 6. 若有多個標的，輸出『多資產 x 多策略』總報酬率對比矩陣
                if len(data_dict) > 1:
                    st.subheader("🔀 多資產 × 多策略 總報酬率矩陣 (%)")
                    pivot_df = results_df.pivot(index="標的", columns="策略", values="_raw_return") * 100
                    
                    # 格式化矩陣：最高報酬率自動加上 👑 符號
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

                # 7. 渲染資產成長曲線圖
                st.subheader("📈 資產成長曲線圖")
                fig.update_layout(xaxis_title="日期", yaxis_title=f"資產總額 ({curr_sym})", height=500)
                st.plotly_chart(fig, use_container_width=True)
