import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="多市場與多策略回測工具")

# ==========================================
# 1. 市場與數據處理模組 (Data Module)
# ==========================================
class DataFetcher:
    PRESET_OPTIONS = {
        "美股": ["QQQ (納斯達克100)", "AAPL (蘋果)", "NVDA (輝達)", "SPY (S&P 500)", "TSLA (特斯拉)", "MSFT (微軟)"],
        "台股": ["0050 (元大台灣50)", "2330 (台積電)", "0056 (元大高股息)", "2317 (鴻海)", "2454 (聯發科)"],
        "加密貨幣": ["BTC-USD (比特幣)", "ETH-USD (乙太幣)", "SOL-USD (索拉納)", "BNB-USD (幣安幣)"]
    }

    @staticmethod
    def get_default_costs(market_type, ticker_symbol):
        clean_symbol = ticker_symbol.strip().upper()
        if market_type == "台股":
            return (0.071, 0.10, 0.02) if clean_symbol.startswith("00") else (0.071, 0.30, 0.03)
        elif market_type == "美股":
            return 0.00, 0.00, 0.03
        else:
            return 0.10, 0.00, 0.10

    @staticmethod
    def format_ticker(ticker, market_type):
        ticker = ticker.strip().upper()
        if market_type == "台股" and not (ticker.endswith(".TW") or ticker.endswith(".TWO")):
            return f"{ticker}.TW"
        return ticker

    @classmethod
    @st.cache_data
    def load_data_and_fx(cls, tickers, market_type, start, end, auto_adjust=True, user_curr="新台幣 (TWD)"):
        formatted_tickers = [cls.format_ticker(t, market_type) for t in tickers]
        df = yf.download(formatted_tickers, start=start, end=end, auto_adjust=auto_adjust, group_by='ticker', progress=False)
        
        processed_data = {}
        for original_t, fmt_t in zip(tickers, formatted_tickers):
            try:
                sub_df = df[fmt_t] if len(formatted_tickers) > 1 else df
                if isinstance(sub_df.columns, pd.MultiIndex):
                    sub_df.columns = sub_df.columns.get_level_values(0)
                
                close_data = sub_df['Close']
                high_data = sub_df['High'] if 'High' in sub_df else close_data
                low_data = sub_df['Low'] if 'Low' in sub_df else close_data
                
                res_df = pd.DataFrame({'close': close_data, 'high': high_data, 'low': low_data}).dropna()
                res_df = res_df[res_df['close'] > 0.01]
                
                if not res_df.empty:
                    processed_data[original_t] = res_df
            except Exception:
                continue

        is_usd_asset = (market_type in ["美股", "加密貨幣"])
        curr_symbol = "NT$" if "TWD" in user_curr else "$"
        
        if is_usd_asset and "TWD" in user_curr:
            fx_df = yf.download("USDTWD=X", start=start, end=end, auto_adjust=True, progress=False)
            fx_close = fx_df['Close'].iloc[:, 0] if isinstance(fx_df['Close'], pd.DataFrame) else fx_df['Close']
            
            for t in processed_data:
                fx_series = fx_close.reindex(processed_data[t].index, method='ffill').fillna(30.0)
                processed_data[t]['close'] *= fx_series
                processed_data[t]['high'] *= fx_series
                processed_data[t]['low'] *= fx_series
                
        return processed_data, curr_symbol


# ==========================================
# 2. 技術指標與訊號模組 (Indicator Module)
# ==========================================
class IndicatorCalculator:
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
        if strat_name == "價格突破均線買入/跌破賣出":
            sma = cls.calculate_ma(data, p['period'], p.get('timeframe', '日線 (Daily)'))
            return (data['close'] > sma).astype(int)
            
        elif strat_name == "SMA 雙均線交叉":
            fast_m = cls.calculate_ma(data, p['fast'], p.get('timeframe', '日線 (Daily)'))
            slow_m = cls.calculate_ma(data, p['slow'], p.get('timeframe', '日線 (Daily)'))
            return (fast_m > slow_m).astype(int)
            
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
            
        elif strat_name == "MACD 柱狀圖/交叉":
            ema_fast = data['close'].ewm(span=p['fast'], adjust=False).mean()
            ema_slow = data['close'].ewm(span=p['slow'], adjust=False).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=p['signal'], adjust=False).mean()
            return (macd > signal).astype(int)
            
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
# 3. 回測引擎模組 (Backtest Engine Module)
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_buy_and_hold(df, init_capital, monthly_inflow, fee, slip):
        data = df.copy()
        total_days = len(data)
        month_changed = (pd.Series(data.index.month, index=data.index) != pd.Series(data.index.month, index=data.index).shift(1))
        
        cash, total_invested, shares = init_capital, init_capital, 0.0
        portfolio_vals, cash_hist, stock_hist = [], [], []
        
        for i in range(total_days):
            if i > 0 and month_changed.iloc[i]:
                cash += monthly_inflow
                total_invested += monthly_inflow
                
            if cash > 0:
                buy_price = data['close'].iloc[i] * (1 + fee + slip)
                shares += cash / buy_price
                cash = 0.0
                
            stock_val = shares * data['close'].iloc[i]
            portfolio_vals.append(cash + stock_val)
            cash_hist.append(cash)
            stock_hist.append(stock_val)
            
        data['portfolio_value'] = portfolio_vals
        data['total_invested'] = total_invested
        data['cash'] = cash_hist
        data['stock_value'] = stock_hist
        data['equity_net'] = data['portfolio_value'] / total_invested
        data['trades'] = 1
        return data

    @staticmethod
    def run_dca(df, init_capital, monthly_inflow, freq_type, dca_amount, custom_days, fee, slip):
        data = df.copy()
        total_days = len(data)
        month_changed = (pd.Series(data.index.month, index=data.index) != pd.Series(data.index.month, index=data.index).shift(1))
        
        if freq_type == "每月 (Monthly)":
            buy_mask = month_changed
        elif freq_type == "每週 (Weekly)":
            buy_mask = (pd.Series(data.index.isocalendar().week, index=data.index) != pd.Series(data.index.isocalendar().week, index=data.index).shift(1))
        else:
            step = max(1, int(custom_days))
            buy_mask = (pd.Series(range(total_days), index=data.index) % step == 0)
            
        buy_indices = set(data.index[buy_mask])
        cash, total_invested, shares = init_capital, init_capital, 0.0
        portfolio_vals, cash_hist, stock_hist = [], [], []
        trade_count = 0
        
        for i in range(total_days):
            curr_date = data.index[i]
            if i > 0 and month_changed.iloc[i]:
                cash += monthly_inflow
                total_invested += monthly_inflow
                
            if curr_date in buy_indices and cash > 0:
                actual_buy = min(cash, dca_amount)
                buy_price = data['close'].iloc[i] * (1 + fee + slip)
                shares += actual_buy / buy_price
                cash -= actual_buy
                trade_count += 1
                
            stock_val = shares * data['close'].iloc[i]
            portfolio_vals.append(cash + stock_val)
            cash_hist.append(cash)
            stock_hist.append(stock_val)
            
        data['portfolio_value'] = portfolio_vals
        data['total_invested'] = total_invested
        data['cash'] = cash_hist
        data['stock_value'] = stock_hist
        data['equity_net'] = data['portfolio_value'] / total_invested
        data['trades'] = trade_count
        return data

    @staticmethod
    def run_dip_dca(df, init_capital, monthly_inflow, p, fee, slip):
        data = df.copy()
        sma = IndicatorCalculator.calculate_ma(data, p['sma_period'], p['timeframe'])
        total_days = len(data)
        month_changed = (pd.Series(data.index.month, index=data.index) != pd.Series(data.index.month, index=data.index).shift(1))
        
        cash, total_invested, shares = init_capital, init_capital, 0.0
        portfolio_vals, cash_hist, stock_hist = [], [], []
        last_buy_idx = -p['min_interval']
        min_price_in_dip = float('inf')
        trade_count = 0
        
        for i in range(total_days):
            if i > 0 and month_changed.iloc[i]:
                cash += monthly_inflow
                total_invested += monthly_inflow
                
            price, sma_val = data['close'].iloc[i], sma.iloc[i]
            if not np.isnan(sma_val) and price < sma_val:
                if price < min_price_in_dip:
                    if (i - last_buy_idx >= p['min_interval']) and cash > 0:
                        actual_buy = min(cash, p['buy_amount'])
                        buy_price = price * (1 + fee + slip)
                        shares += actual_buy / buy_price
                        cash -= actual_buy
                        last_buy_idx = i
                        min_price_in_dip = price
                        trade_count += 1
            else:
                min_price_in_dip = float('inf')
                
            stock_val = shares * price
            portfolio_vals.append(cash + stock_val)
            cash_hist.append(cash)
            stock_hist.append(stock_val)
            
        data['portfolio_value'] = portfolio_vals
        data['total_invested'] = total_invested
        data['cash'] = cash_hist
        data['stock_value'] = stock_hist
        data['equity_net'] = data['portfolio_value'] / total_invested
        data['trades'] = trade_count
        return data

    @staticmethod
    def run_timing_strategy(df, signal_series, fee, tax, slip, init_capital, monthly_inflow):
        """
        修正版擇時策略引擎：
        - 每月新增資金進入現金池。
        - 只有當策略發出「買入訊號（空轉多）」時，才會把帳上累積的所有現金買入。
        - 持倉期間（Position = 1）新進來的資金會積攢在現金池中，直到下一次觸發買入訊號（平倉後重新進場）時一併投入。
        """
        data = df.copy()
        total_days = len(data)
        month_changed = (pd.Series(data.index.month, index=data.index) != pd.Series(data.index.month, index=data.index).shift(1))
        position = signal_series.shift(1).fillna(0)
        
        cash, total_invested, shares = init_capital, init_capital, 0.0
        portfolio_vals, cash_hist, stock_hist = [], [], []
        trade_count = 0
        
        for i in range(total_days):
            # 每月注資注入現金儲蓄池
            if i > 0 and month_changed.iloc[i]:
                cash += monthly_inflow
                total_invested += monthly_inflow
                
            pos = position.iloc[i]
            prev_pos = position.iloc[i-1] if i > 0 else 0.0
            price = data['close'].iloc[i]
            
            # 策略觸發【買入訊號】(無倉位轉為有倉位)：把目前手頭所有現金全數投入
            if pos > 0 and prev_pos == 0:
                if cash > 0:
                    buy_price = price * (1 + fee + slip)
                    shares += cash / buy_price
                    cash = 0.0
                    trade_count += 1
                    
            # 策略觸發【賣出訊號】(有倉位轉為無倉位)：全數賣出換回現金
            elif pos == 0 and prev_pos > 0 and shares > 0:
                sell_price = price * (1 - fee - tax - slip)
                cash += shares * sell_price
                shares = 0.0
                trade_count += 1
                
            stock_val = shares * price
            portfolio_vals.append(cash + stock_val)
            cash_hist.append(cash)
            stock_hist.append(stock_val)
            
        data['portfolio_value'] = portfolio_vals
        data['total_invested'] = total_invested
        data['cash'] = cash_hist
        data['stock_value'] = stock_hist
        data['equity_net'] = data['portfolio_value'] / total_invested
        data['trades'] = trade_count
        return data


# ==========================================
# 4. 績效評估模組 (Performance Module)
# ==========================================
class PerformanceEvaluator:
    @staticmethod
    def evaluate(df, curr_symbol):
        total_days = len(df)
        if total_days < 2:
            return {}
            
        actual_invested = df['total_invested'].iloc[-1]
        final_value = df['portfolio_value'].iloc[-1]
        final_stock_val = df['stock_value'].iloc[-1]
        final_cash = df['cash'].iloc[-1]
        
        net_profit = final_value - actual_invested
        total_return = net_profit / actual_invested if actual_invested > 0 else 0.0
        cagr = (((final_value / actual_invested) ** (252 / total_days)) - 1) if actual_invested > 0 else 0.0
        
        cum_max = df['equity_net'].cummax()
        dd = (df['equity_net'] - cum_max) / (cum_max + 1e-9)
        mdd = dd.min()
        
        return {
            "實際總投入本金": f"{curr_symbol}{actual_invested:,.0f}",
            "最終資產金額": f"{curr_symbol}{final_value:,.0f}",
            "最終持股市值": f"{curr_symbol}{final_stock_val:,.0f}",
            "最終剩餘現金": f"{curr_symbol}{final_cash:,.0f}",
            "總報酬率": f"{total_return:.2%}",
            "CAGR 年化報酬": f"{cagr:.2%}",
            "Max Drawdown (MDD)": f"{mdd:.2%}",
            "累積交易次數": f"{df['trades'].iloc[-1] if isinstance(df['trades'], pd.Series) else df['trades']} 次",
            "_raw_final": final_value,
            "_raw_return": total_return,
            "_raw_cagr": cagr,
            "_raw_mdd": mdd
        }


# ==========================================
# 5. UI 介面與主流程控制 (Streamlit UI)
# ==========================================
st.title("📈 多市場與多策略比較回測系統 (嚴謹策略現金流版)")

# --- 側邊欄 1：選擇標的與市場 ---
st.sidebar.header("1. 選擇標的與市場")
market = st.sidebar.selectbox("市場類別", ["美股", "台股", "加密貨幣"])

select_mode = st.sidebar.radio("標的選擇模式", ["熱門標的單選/多選", "自訂輸入"])
if select_mode == "熱門標的單選/多選":
    selected_presets = st.sidebar.multiselect("選擇標的（可多選進行交叉對比）", DataFetcher.PRESET_OPTIONS[market], default=[DataFetcher.PRESET_OPTIONS[market][0]])
    target_symbols = [p.split(" ")[0].strip() for p in selected_presets]
else:
    custom_input = st.sidebar.text_input("輸入自訂代碼 (多個請用逗號隔開)", value="QQQ, AAPL" if market == "美股" else "0050, 2330")
    target_symbols = [s.strip().upper() for s in custom_input.split(",") if s.strip()]

user_currency = st.sidebar.selectbox("輸入資金幣別", ["新台幣 (TWD)", "美元 (USD)"])
initial_capital = st.sidebar.number_input("期初投入本金", value=300000 if "TWD" in user_currency else 10000, step=10000)
monthly_inflow = st.sidebar.number_input("每月固定新增可投入資金", value=30000 if "TWD" in user_currency else 1000, step=1000)
reinvest_dividends = st.sidebar.checkbox("計入股利再投入", value=True)

# 時間跨度
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

# --- 側邊欄 2：交易成本 ---
st.sidebar.header("2. 交易成本設定")
auto_fee, auto_tax, auto_slip = DataFetcher.get_default_costs(market, target_symbols[0] if target_symbols else "")
fee_rate = st.sidebar.number_input("手續費率 (%)", value=auto_fee, step=0.01) / 100
tax_rate = st.sidebar.number_input("賣出證交稅 (%)", value=auto_tax, step=0.01) / 100
slippage = st.sidebar.number_input("預估滑點 (%)", value=auto_slip, step=0.01) / 100

# --- 側邊欄 3：策略選擇與設定 ---
st.sidebar.header("3. 選擇與設定策略")
available_strats = ["買入持有 (Buy & Hold)", "定期定額 (DCA)", "跌破均線且創新低買入", "價格突破均線買入/跌破賣出", "SMA 雙均線交叉", "RSI 超買超賣", "MACD 柱狀圖/交叉", "KD 隨機指標"]
selected_strategies = st.sidebar.multiselect("選擇策略", available_strats, default=["買入持有 (Buy & Hold)", "定期定額 (DCA)", "跌破均線且創新低買入"])

params = {}
if "定期定額 (DCA)" in selected_strategies:
    params["DCA"] = {
        "freq": st.sidebar.selectbox("DCA 頻率", ["每月 (Monthly)", "每週 (Weekly)", "自訂交易日數"]),
        "val": float(st.sidebar.number_input("DCA 每期金額", value=10000 if "TWD" in user_currency else 1000)),
        "custom_days": 20
    }

if "跌破均線且創新低買入" in selected_strategies:
    params["DipDCA"] = {
        "sma_period": int(st.sidebar.number_input("DipDCA 均線週期", value=60)),
        "timeframe": st.sidebar.selectbox("DipDCA 單位", ["週線 (Weekly)", "日線 (Daily)"]),
        "buy_amount": float(st.sidebar.number_input("抄底觸發金額", value=10000 if "TWD" in user_currency else 1000)),
        "min_interval": int(st.sidebar.number_input("買入最小間隔(日)", value=1))
    }

if "價格突破均線買入/跌破賣出" in selected_strategies:
    params["PriceSMA"] = {"period": int(st.sidebar.number_input("突破策略均線週期", value=20)), "timeframe": "日線 (Daily)"}

if "SMA 雙均線交叉" in selected_strategies:
    params["SMA"] = {"fast": int(st.sidebar.number_input("SMA 快線", value=20)), "slow": int(st.sidebar.number_input("SMA 慢線", value=60)), "timeframe": "日線 (Daily)"}

if "RSI 超買超賣" in selected_strategies:
    params["RSI"] = {"period": 14, "buy": 30, "sell": 70}

if "MACD 柱狀圖/交叉" in selected_strategies:
    params["MACD"] = {"fast": 12, "slow": 26, "signal": 9}

if "KD 隨機指標" in selected_strategies:
    params["KD"] = {"period": 9, "buy_k": 20, "sell_k": 80}

# --- 側邊欄 4：自訂欄位 ---
st.sidebar.header("4. 自訂顯示欄位")
all_metrics_cols = ["實際總投入本金", "最終資產金額", "最終持股市值", "最終剩餘現金", "總報酬率", "CAGR 年化報酬", "Max Drawdown (MDD)", "累積交易次數"]
selected_cols = st.sidebar.multiselect("表格數據欄位", all_metrics_cols, default=all_metrics_cols)

# --- 主程式執行區塊 ---
if st.sidebar.button("開始多策略與多資產回測"):
    if not target_symbols:
        st.error("請至少選擇或輸入一個標的代碼！")
    else:
        with st.spinner("正在獲取數據與運算回測結果..."):
            data_dict, curr_sym = DataFetcher.load_data_and_fx(target_symbols, market, start_date, end_date, auto_adjust=reinvest_dividends, user_curr=user_currency)
            
            if not data_dict:
                st.error("無法獲取對應標的數據，請確認代碼與日期。")
            else:
                matrix_results = []
                fig = go.Figure()
                
                for sym, data in data_dict.items():
                    for strat in selected_strategies:
                        if strat == "買入持有 (Buy & Hold)":
                            res = BacktestEngine.run_buy_and_hold(data, initial_capital, monthly_inflow, fee_rate, slippage)
                        elif strat == "定期定額 (DCA)":
                            res = BacktestEngine.run_dca(data, initial_capital, monthly_inflow, params["DCA"]["freq"], params["DCA"]["val"], params["DCA"]["custom_days"], fee_rate, slippage)
                        elif strat == "跌破均線且創新低買入":
                            res = BacktestEngine.run_dip_dca(data, initial_capital, monthly_inflow, params["DipDCA"], fee_rate, slippage)
                        else:
                            p_key = "PriceSMA" if "突破" in strat else ("SMA" if "SMA" in strat else ("RSI" if "RSI" in strat else ("MACD" if "MACD" in strat else "KD")))
                            sig = IndicatorCalculator.generate_signal(data, strat, params.get(p_key, {}))
                            res = BacktestEngine.run_timing_strategy(data, sig, fee_rate, tax_rate, slippage, initial_capital, monthly_inflow)
                        
                        m = PerformanceEvaluator.evaluate(res, curr_sym)
                        m_row = {"標的": sym, "策略": strat, **m}
                        matrix_results.append(m_row)
                        
                        curve_label = f"{sym}-{strat}" if len(data_dict) > 1 else strat
                        fig.add_trace(go.Scatter(x=res.index, y=res['portfolio_value'], name=f"{curve_label} ({curr_sym}{m['_raw_final']:,.0f})"))

                st.subheader("📊 績效與金額對比總表")
                results_df = pd.DataFrame(matrix_results)
                display_cols = ["標的", "策略"] + [c for c in selected_cols if c in results_df.columns]
                st.dataframe(results_df[display_cols].set_index(["標的", "策略"]), use_container_width=True)

                if len(data_dict) > 1:
                    st.subheader("🔀 多資產 × 多策略 總報酬率矩陣 (%)")
                    pivot_df = results_df.pivot(index="標的", columns="策略", values="_raw_return") * 100
                    st.dataframe(pivot_df.style.highlight_max(axis=1, color="lightgreen").format("{:.2f}%"), use_container_width=True)

                st.subheader("📈 資產成長曲線圖")
                fig.update_layout(xaxis_title="日期", yaxis_title=f"資產總額 ({curr_sym})", height=500)
                st.plotly_chart(fig, use_container_width=True)
