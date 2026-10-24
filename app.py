import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="多市場與多策略回測工具")

# ==========================================
# 1. 跨市場數據與動態成本模組 (Data Module)
# ==========================================
class DataFetcher:
    PRESET_OPTIONS = {
        "美股": ["QQQ (納斯達克100)", "AAPL (蘋果)", "NVDA (輝達)", "SPY (S&P 500)", "TSLA (特斯拉)", "MSFT (微軟)"],
        "台股": ["0050 (元大台灣50)", "2330 (台積電)", "0056 (元大高股息)", "2317 (鴻海)", "2454 (聯發科)"],
        "加密貨幣": ["BTC-USD (比特幣)", "ETH-USD (乙太幣)", "SOL-USD (索拉納)", "BNB-USD (幣安幣)"]
    }

    @staticmethod
    def get_asset_costs(ticker_symbol):
        """依據標的特性動態回傳 (手續費率, 證交稅率, 預估滑點)"""
        clean_symbol = ticker_symbol.strip().upper()
        if clean_symbol.isdigit() or clean_symbol.endswith(".TW") or clean_symbol.endswith(".TWO"):
            # 台股 ETF 證交稅 0.1%，個股 0.3%
            return (0.00071, 0.0010, 0.0002) if clean_symbol.startswith("00") else (0.00071, 0.0030, 0.0003)
        elif "-USD" in clean_symbol:
            # 加密貨幣常見 0.1% 手續費，無證交稅
            return 0.0010, 0.00, 0.0010
        else:
            # 美股多數免手續費與證交稅
            return 0.00, 0.00, 0.0003

    @staticmethod
    def normalize_ticker(ticker):
        t = ticker.strip().upper()
        if t.isdigit():
            return f"{t}.TW"
        return t

    @classmethod
    @st.cache_data
    def load_data_and_fx(cls, tickers, start, end, auto_adjust=True, user_curr="新台幣 (TWD)"):
        fmt_to_orig = {cls.normalize_ticker(t): t.strip().upper() for t in tickers}
        formatted_tickers = list(fmt_to_orig.keys())
        
        # yfinance 安全下載
        df = yf.download(formatted_tickers, start=start, end=end, auto_adjust=auto_adjust, group_by='ticker', progress=False)
        
        processed_data = {}
        for fmt_t, orig_t in fmt_to_orig.items():
            try:
                # 解決單標的 vs 多標的欄位結構差異
                if len(formatted_tickers) == 1:
                    sub_df = df.copy()
                else:
                    sub_df = df[fmt_t] if fmt_t in df else pd.DataFrame()

                if isinstance(sub_df.columns, pd.MultiIndex):
                    sub_df.columns = sub_df.columns.get_level_values(-1)
                
                if 'Close' not in sub_df or sub_df['Close'].dropna().empty:
                    continue

                close_data = sub_df['Close']
                high_data = sub_df['High'] if 'High' in sub_df else close_data
                low_data = sub_df['Low'] if 'Low' in sub_df else close_data
                
                res_df = pd.DataFrame({'close': close_data, 'high': high_data, 'low': low_data}).dropna()
                res_df = res_df[res_df['close'] > 0.01]
                
                if not res_df.empty:
                    processed_data[orig_t] = res_df
            except Exception:
                continue

        curr_symbol = "NT$" if "TWD" in user_curr else "$"
        
        # 匯率自動換算
        if "TWD" in user_curr:
            try:
                fx_df = yf.download("USDTWD=X", start=start, end=end, auto_adjust=True, progress=False)
                fx_close = fx_df['Close'].iloc[:, 0] if isinstance(fx_df['Close'], pd.DataFrame) else fx_df['Close']
                
                for orig_t in processed_data:
                    if not (orig_t.endswith(".TW") or orig_t.endswith(".TWO") or orig_t.isdigit()):
                        fx_series = fx_close.reindex(processed_data[orig_t].index, method='ffill').fillna(30.0)
                        processed_data[orig_t]['close'] *= fx_series
                        processed_data[orig_t]['high'] *= fx_series
                        processed_data[orig_t]['low'] *= fx_series
            except Exception:
                pass
                
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
    def run_dca_strategy(df, init_capital, monthly_inflow, freq_type, dca_amount, custom_days, fee, slip):
        """
        合併 Buy & Hold 與 DCA 的統一引擎：
        - 第 1 天若有初始本金 (init_capital > 0)，先單筆一次性投入買入。
        - 每月新增可投入資金注入現金池。
        - 依照設定頻率 (每月/每週/自訂天數)，若現金池有資金則進行 DCA 買入。
        """
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
            
            # 月份變更：新增每月可投入資金
            if i > 0 and month_changed.iloc[i]:
                cash += monthly_inflow
                total_invested += monthly_inflow
                
            # 第 1 天：若有初始本金，單筆一次性全部投入買入
            if i == 0 and cash > 0:
                buy_price = data['close'].iloc[i] * (1 + fee + slip)
                shares += cash / buy_price
                cash = 0.0
                trade_count += 1
            # 後續扣款日：執行定期定額買入
            elif curr_date in buy_indices and cash > 0 and dca_amount > 0:
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
        data = df.copy()
        total_days = len(data)
        month_changed = (pd.Series(data.index.month, index=data.index) != pd.Series(data.index.month, index=data.index).shift(1))
        position = signal_series.shift(1).fillna(0)
        
        cash, total_invested, shares = init_capital, init_capital, 0.0
        portfolio_vals, cash_hist, stock_hist = [], [], []
        trade_count = 0
        
        for i in range(total_days):
            if i > 0 and month_changed.iloc[i]:
                cash += monthly_inflow
                total_invested += monthly_inflow
                
            pos = position.iloc[i]
            prev_pos = position.iloc[i-1] if i > 0 else 0.0
            price = data['close'].iloc[i]
            
            if pos > 0 and prev_pos == 0:
                if cash > 0:
                    buy_price = price * (1 + fee + slip)
                    shares += cash / buy_price
                    cash = 0.0
                    trade_count += 1
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
            
        gross_invested = df['total_invested'].iloc[-1]
        final_value = df['portfolio_value'].iloc[-1]
        final_stock_val = df['stock_value'].iloc[-1]
        final_cash = df['cash'].iloc[-1]
        
        # 實際投入本金 = 累積注入總資金 - 最終未投入之剩餘現金
        actual_invested = gross_invested - final_cash
        
        if actual_invested > 0:
            net_profit = final_stock_val - actual_invested
            total_return = net_profit / actual_invested
            cagr = (((final_stock_val / actual_invested) ** (252 / total_days)) - 1)
        else:
            net_profit = 0.0
            total_return = 0.0
            cagr = 0.0
            
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
st.title("📈 多市場與多策略比較回測系統 (DCA與單筆合併版)")

# --- 側邊欄 1：選擇標的與資產 ---
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
    help="如需補充下拉選單沒有的標的，請在此輸入（例如：2330, NVDA）"
)

preset_symbols = [p.split(" ")[0].strip().upper() for p in selected_presets]
extra_symbols = [s.strip().upper() for s in custom_input_extra.split(",") if s.strip()]
target_symbols = list(dict.fromkeys(preset_symbols + extra_symbols))

user_currency = st.sidebar.selectbox("輸入資金幣別", ["新台幣 (TWD)", "美元 (USD)"])
initial_capital = st.sidebar.number_input("期初投入本金", value=300000 if "TWD" in user_currency else 10000, step=10000)
monthly_inflow = st.sidebar.number_input("每月固定新增可投入資金", value=30000 if "TWD" in user_currency else 1000, step=1000)
reinvest_dividends = st.sidebar.checkbox("計入股利再投入", value=True)

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

# --- 側邊欄 2：交易成本設定 ---
st.sidebar.header("2. 交易成本設定")
auto_cost_toggle = st.sidebar.checkbox("依據各資產類別自動設定交易成本", value=True, help="勾選後，台股、美股、加密貨幣將分別套用各自合理的預設手續費與證交稅率。")

if not auto_cost_toggle:
    fee_input = st.sidebar.number_input("手續費率 (%)", value=0.071, step=0.01, format="%.3f") / 100
    tax_input = st.sidebar.number_input("賣出證交稅 (%)", value=0.10, step=0.01, format="%.2f") / 100
    slip_input = st.sidebar.number_input("單邊預估滑點 (%)", value=0.02, step=0.01, format="%.2f") / 100

# --- 側邊欄 3：選擇與設定策略 ---
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
        "sma_period": int(st.sidebar.number_input("DipDCA 均線週期", value=60)),
        "timeframe": st.sidebar.selectbox("DipDCA 單位", ["週線 (Weekly)", "日線 (Daily)"]),
        "buy_amount": float(st.sidebar.number_input("抄底觸發金額", value=50000 if "TWD" in user_currency else 1000)),
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

# --- 側邊欄 4：自訂顯示欄位 ---
st.sidebar.header("4. 自訂顯示欄位")
all_metrics_cols = ["實際總投入本金", "最終資產金額", "最終持股市值", "最終剩餘現金", "總報酬率", "CAGR 年化報酬", "Max Drawdown (MDD)", "累積交易次數"]
selected_cols = st.sidebar.multiselect("表格數據欄位", all_metrics_cols, default=all_metrics_cols)

# --- 主程式執行區塊 ---
if st.sidebar.button("開始多策略與多資產回測"):
    if not target_symbols:
        st.error("請至少選擇或輸入一個標的代碼！")
    else:
        with st.spinner("正在獲取數據與運算回測結果..."):
            data_dict, curr_sym = DataFetcher.load_data_and_fx(
                target_symbols, 
                start_date, 
                end_date, 
                auto_adjust=reinvest_dividends, 
                user_curr=user_currency
            )
            
            if not data_dict:
                st.error("無法獲取對應標的數據，請確認代碼與日期。")
            else:
                matrix_results = []
                fig = go.Figure()
                
                for sym, data in data_dict.items():
                    if auto_cost_toggle:
                        fee_rate, tax_rate, slippage = DataFetcher.get_asset_costs(sym)
                    else:
                        fee_rate, tax_rate, slippage = fee_input, tax_input, slip_input

                    for strat in selected_strategies:
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
                fig.update_layout(xaxis_title="日期", yaxis_title=f"資產總額 ({curr_sym})", height=500)
                st.plotly_chart(fig, use_container_width=True)
