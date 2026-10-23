import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="多市場與多策略回測工具")

st.title("📈 多市場與多策略比較回測系統 (支援日/週均線切換版)")

# --- 1. 側邊欄控制項：標的與市場 ---
st.sidebar.header("1. 選擇標的與市場")
market = st.sidebar.selectbox("市場類別", ["美股", "台股", "加密貨幣"])

preset_options = {
    "台股": ["0050 (元大台灣50)", "2330 (台積電)", "0056 (元大高股息)", "2317 (鴻海)", "2454 (聯發科)", "自訂輸入..."],
    "美股": ["QQQ (納斯達克100)", "AAPL (蘋果)", "NVDA (輝達)", "SPY (S&P 500)", "TSLA (特斯拉)", "MSFT (微軟)", "自訂輸入..."],
    "加密貨幣": ["BTC-USD (比特幣)", "ETH-USD (乙太幣)", "SOL-USD (索拉納)", "BNB-USD (幣安幣)", "自訂輸入..."]
}

selected_preset = st.sidebar.selectbox("選擇熱門標的（可打字搜尋）", preset_options[market])

if "自訂輸入" in selected_preset:
    symbol_input = st.sidebar.text_input("輸入自訂代碼", value="QQQ" if market == "美股" else ("0050" if market == "台股" else "BTC-USD"))
    symbol = symbol_input.strip().upper()
else:
    symbol = selected_preset.split(" ")[0].strip()

# 投入本金設定
initial_capital = st.sidebar.number_input("預計總投入本金 ($)", value=100000, step=10000)

time_option = st.sidebar.selectbox("回測時間跨度", ["近 5 年", "近 10 年", "近 20 年", "自訂日期範圍"], index=2)
today = pd.to_datetime("today")

if time_option == "近 5 年":
    start_date = today - pd.DateOffset(years=5)
    end_date = today
elif time_option == "近 10 年":
    start_date = today - pd.DateOffset(years=10)
    end_date = today
elif time_option == "近 20 年":
    start_date = today - pd.DateOffset(years=20)
    end_date = today
else:
    start_date = st.sidebar.date_input("開始日期", pd.to_datetime("2005-01-01"))
    end_date = st.sidebar.date_input("結束日期", today)

# --- 2. 側邊欄控制項：交易成本 ---
st.sidebar.header("2. 設定交易成本與滑點")
def get_default_costs(market_type, ticker_symbol):
    if market_type == "台股":
        clean_symbol = ticker_symbol.strip().upper()
        if clean_symbol.startswith("00"):
            return 0.071, 0.10, 0.02
        else:
            return 0.071, 0.30, 0.03
    elif market_type == "美股":
        return 0.00, 0.00, 0.03
    else:
        return 0.10, 0.00, 0.10

auto_fee, auto_tax, auto_slip = get_default_costs(market, symbol)
fee_input = st.sidebar.number_input("手續費率 (%)", value=auto_fee, step=0.01, format="%.3f")
tax_input = st.sidebar.number_input("賣出證交稅 (%)", value=auto_tax, step=0.01, format="%.2f")
slip_input = st.sidebar.number_input("單邊預估滑點 (%)", value=auto_slip, step=0.01, format="%.2f")

fee_rate = fee_input / 100
tax_rate = tax_input / 100
slippage = slip_input / 100

# --- 3. 側邊欄控制項：擴充交易策略列表 ---
st.sidebar.header("3. 選擇與設定回測策略")
available_strats = [
    "定期定額 (DCA)",
    "跌破均線分批買入並持有",
    "價格突破均線買入/跌破賣出",
    "SMA 雙均線交叉", 
    "RSI 超買超賣", 
    "MACD 柱狀圖/交叉", 
    "布林通道 (Bollinger Bands)", 
    "KD 隨機指標", 
    "唐奇安通道 (海龜突破)"
]

selected_strategies = st.sidebar.multiselect(
    "選擇要比較的策略（可多選）",
    available_strats,
    default=["定期定額 (DCA)", "跌破均線分批買入並持有", "價格突破均線買入/跌破賣出"]
)

params = {}

if "定期定額 (DCA)" in selected_strategies:
    st.sidebar.markdown("**[定期定額 (DCA) 精細設定]**")
    dca_freq = st.sidebar.selectbox("扣款頻率", ["每月 (Monthly)", "每週 (Weekly)", "雙週 (Bi-weekly)", "自訂交易日數"])
    custom_dca_days = 20
    if dca_freq == "自訂交易日數":
        custom_dca_days = int(st.sidebar.number_input("相隔交易日數", value=20, min_value=1))
        
    dca_mode = st.sidebar.radio("扣款金額計算方式", ["按總本金平均分期", "固定每期扣款金額"])
    if dca_mode == "按總本金平均分期":
        dca_val = int(st.sidebar.number_input("預計總扣款期數 (次)", value=60, min_value=1))
    else:
        dca_val = float(st.sidebar.number_input("每期扣款金額 ($)", value=3000, min_value=100))
        
    params["DCA"] = {
        "freq": dca_freq,
        "mode": dca_mode,
        "val": dca_val,
        "custom_days": custom_dca_days
    }

if "跌破均線分批買入並持有" in selected_strategies:
    st.sidebar.markdown("**[跌破均線分批買入設定]**")
    params["DipDCA"] = {
        "sma_period": int(st.sidebar.number_input("跌破基準均線 (週期數)", value=52, min_value=1)),
        "timeframe": st.sidebar.selectbox("均線單位 (DipDCA)", ["週線 (Weekly)", "日線 (Daily)"], key="dip_tf"),
        "min_interval": int(st.sidebar.number_input("買入最小間隔 (交易日)", value=20, min_value=1)),
        "periods": int(st.sidebar.number_input("總本金拆為幾期買入 (次)", value=10, min_value=1))
    }

if "價格突破均線買入/跌破賣出" in selected_strategies:
    st.sidebar.markdown("**[價格突破均線策略]**")
    params["PriceSMA"] = {
        "period": int(st.sidebar.number_input("基準均線 (週期數)", value=20, min_value=1)),
        "timeframe": st.sidebar.selectbox("均線單位 (突破策略)", ["日線 (Daily)", "週線 (Weekly)"], key="price_tf")
    }

if "SMA 雙均線交叉" in selected_strategies:
    st.sidebar.markdown("**[SMA 雙均線]**")
    params["SMA"] = {
        "fast": int(st.sidebar.number_input("SMA 快線 (週期)", value=20)),
        "slow": int(st.sidebar.number_input("SMA 慢線 (週期)", value=60)),
        "timeframe": st.sidebar.selectbox("均線單位 (雙均線)", ["日線 (Daily)", "週線 (Weekly)"], key="sma_tf")
    }

if "RSI 超買超賣" in selected_strategies:
    st.sidebar.markdown("**[RSI 策略]**")
    params["RSI"] = {
        "period": int(st.sidebar.number_input("RSI 週期", value=14)),
        "buy": st.sidebar.number_input("RSI 超賣買入 (低於)", value=30),
        "sell": st.sidebar.number_input("RSI 超買賣出 (高於)", value=70)
    }

if "MACD 柱狀圖/交叉" in selected_strategies:
    st.sidebar.markdown("**[MACD 參數]**")
    params["MACD"] = {
        "fast": int(st.sidebar.number_input("MACD 快線", value=12)),
        "slow": int(st.sidebar.number_input("MACD 慢線", value=26)),
        "signal": int(st.sidebar.number_input("MACD 訊號線", value=9))
    }

if "布林通道 (Bollinger Bands)" in selected_strategies:
    st.sidebar.markdown("**[布林通道]**")
    params["BB"] = {
        "period": int(st.sidebar.number_input("BB 週期", value=20)),
        "std_dev": st.sidebar.number_input("BB 標準差倍數", value=2.0, step=0.1)
    }

if "KD 隨機指標" in selected_strategies:
    st.sidebar.markdown("**[KD 指標]**")
    params["KD"] = {
        "period": int(st.sidebar.number_input("KD RSV週期", value=9)),
        "buy_k": st.sidebar.number_input("K低於超賣區買入", value=20),
        "sell_k": st.sidebar.number_input("K高於超買區賣出", value=80)
    }

if "唐奇安通道 (海龜突破)" in selected_strategies:
    st.sidebar.markdown("**[唐奇安通道]**")
    params["Donchian"] = {
        "entry_period": int(st.sidebar.number_input("突破創 N 日高點進場", value=20)),
        "exit_period": int(st.sidebar.number_input("跌破創 N 日低點出場", value=10))
    }

# --- 數據獲取與均線計算工具 ---
@st.cache_data
def load_data(ticker, market_type, start, end):
    formatted_ticker = ticker
    if market_type == "台股" and not ticker.endswith(".TW") and not ticker.endswith(".TWO"):
        formatted_ticker = ticker + ".TW"
    
    df = yf.download(formatted_ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        close_data = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high_data = df['High'].iloc[:, 0] if 'High' in df and isinstance(df['High'], pd.DataFrame) else (df['High'] if 'High' in df else close_data)
        low_data = df['Low'].iloc[:, 0] if 'Low' in df and isinstance(df['Low'], pd.DataFrame) else (df['Low'] if 'Low' in df else close_data)
    else:
        close_data = df['Close']
        high_data = df['High'] if 'High' in df else close_data
        low_data = df['Low'] if 'Low' in df else close_data
        
    res_df = pd.DataFrame({'close': close_data, 'high': high_data, 'low': low_data}).dropna()
    res_df = res_df[res_df['close'] > 5.0]  # 數據清洗：自動剔除歷史髒資料壞點
    return res_df

def calculate_ma(df, period, timeframe="日線 (Daily)"):
    if timeframe == "週線 (Weekly)":
        weekly_close = df['close'].resample('W-FRI').last().dropna()
        weekly_ma = weekly_close.rolling(period).mean()
        return weekly_ma.reindex(df.index, method='ffill')
    else:
        return df['close'].rolling(period).mean()

# --- 精細 DCA 定期定額回測引擎 ---
def run_dca_backtest_advanced(df, total_capital, freq_type, dca_mode, mode_val, custom_days, fee, slip):
    data = df.copy()
    total_days = len(data)
    
    if freq_type == "每月 (Monthly)":
        month_series = pd.Series(data.index.month, index=data.index)
        buy_mask = (month_series != month_series.shift(1))
    elif freq_type == "每週 (Weekly)":
        week_series = pd.Series(data.index.isocalendar().week, index=data.index)
        buy_mask = (week_series != week_series.shift(1))
    elif freq_type == "雙週 (Bi-weekly)":
        buy_mask = (pd.Series(range(total_days), index=data.index) % 10 == 0)
    else:
        step = max(1, int(custom_days))
        buy_mask = (pd.Series(range(total_days), index=data.index) % step == 0)
        
    buy_indices = set(data.index[buy_mask])
    
    if dca_mode == "按總本金平均分期":
        n_periods = int(mode_val)
        dca_amount = total_capital / n_periods
    else:
        dca_amount = float(mode_val)
        n_periods = int(np.ceil(total_capital / dca_amount)) if dca_amount > 0 else 1
        
    cash = total_capital
    shares = 0.0
    portfolio_values = []
    daily_positions = []
    daily_trades = [0] * total_days
    trades_count = 0
    
    for i in range(len(data)):
        current_date = data.index[i]
        if current_date in buy_indices and cash > 0 and trades_count < n_periods:
            actual_buy = min(cash, dca_amount)
            cost_factor = 1 + fee + slip
            buy_price = data['close'].iloc[i] * cost_factor
            bought_shares = actual_buy / buy_price
            shares += bought_shares
            cash -= actual_buy
            trades_count += 1
            daily_trades[i] = 1
            
        current_stock_val = shares * data['close'].iloc[i]
        current_value = cash + current_stock_val
        portfolio_values.append(current_value)
        pos_ratio = (current_stock_val / current_value) if current_value > 0 else 0.0
        daily_positions.append(pos_ratio)
        
    data['portfolio_value'] = portfolio_values
    data['equity_net'] = data['portfolio_value'] / total_capital
    data['strategy_ret'] = data['equity_net'].pct_change().fillna(0)
    data['position'] = daily_positions
    data['trades'] = daily_trades
    return data

# --- 跌破均線分批買入並持有引擎 (支援週線抄底) ---
def run_dip_dca_backtest(df, total_capital, sma_period, timeframe, min_interval_days, n_periods, fee, slip):
    data = df.copy()
    sma = calculate_ma(data, sma_period, timeframe)
    total_days = len(data)
    
    dca_amount = total_capital / n_periods
    cash = total_capital
    shares = 0.0
    portfolio_values = []
    daily_positions = []
    daily_trades = [0] * total_days
    trades_count = 0
    last_buy_idx = -min_interval_days
    
    for i in range(len(data)):
        price = data['close'].iloc[i]
        sma_val = sma.iloc[i]
        
        if not np.isnan(sma_val) and price < sma_val and (i - last_buy_idx >= min_interval_days) and cash > 0 and trades_count < n_periods:
            actual_buy = min(cash, dca_amount)
            cost_factor = 1 + fee + slip
            buy_price = price * cost_factor
            bought_shares = actual_buy / buy_price
            shares += bought_shares
            cash -= actual_buy
            trades_count += 1
            last_buy_idx = i
            daily_trades[i] = 1
            
        current_stock_val = shares * price
        current_value = cash + current_stock_val
        portfolio_values.append(current_value)
        pos_ratio = (current_stock_val / current_value) if current_value > 0 else 0.0
        daily_positions.append(pos_ratio)
        
    data['portfolio_value'] = portfolio_values
    data['equity_net'] = data['portfolio_value'] / total_capital
    data['strategy_ret'] = data['equity_net'].pct_change().fillna(0)
    data['position'] = daily_positions
    data['trades'] = daily_trades
    return data

# --- 一般擇時策略回測引擎 ---
def run_backtest(df, signal_series, fee, tax, slip, init_capital):
    data = df.copy()
    data['position'] = signal_series.shift(1).fillna(0)
    data['trades'] = data['position'].diff().abs().fillna(0)
    data['market_ret'] = data['close'].pct_change().fillna(0)
    
    cost_per_trade = fee + slip
    data['cost'] = 0.0
    data.loc[data['trades'] > 0, 'cost'] = cost_per_trade
    data.loc[data['position'].diff() < 0, 'cost'] += tax
    
    data['strategy_ret'] = (data['market_ret'] * data['position']) - data['cost']
    data['equity_net'] = (1 + data['strategy_ret']).cumprod()
    data['portfolio_value'] = data['equity_net'] * init_capital
    return data

# --- 策略訊號產生器 ---
def generate_signal(df, strat_name, p):
    data = df.copy()
    
    if strat_name == "價格突破均線買入/跌破賣出":
        tf = p.get('timeframe', '日線 (Daily)')
        sma = calculate_ma(data, p['period'], tf)
        data['signal'] = (data['close'] > sma).astype(int)
        
    elif strat_name == "SMA 雙均線交叉":
        tf = p.get('timeframe', '日線 (Daily)')
        fast_m = calculate_ma(data, p['fast'], tf)
        slow_m = calculate_ma(data, p['slow'], tf)
        data['signal'] = (fast_m > slow_m).astype(int)
        
    elif strat_name == "RSI 超買超賣":
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(p['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p['period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        sig = pd.Series(np.nan, index=data.index)
        sig[rsi < p['buy']] = 1
        sig[rsi > p['sell']] = 0
        data['signal'] = sig.ffill().fillna(0).astype(int)
        
    elif strat_name == "MACD 柱狀圖/交叉":
        ema_fast = data['close'].ewm(span=p['fast'], adjust=False).mean()
        ema_slow = data['close'].ewm(span=p['slow'], adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=p['signal'], adjust=False).mean()
        data['signal'] = (macd > signal).astype(int)
        
    elif strat_name == "布林通道 (Bollinger Bands)":
        ma = data['close'].rolling(p['period']).mean()
        std = data['close'].rolling(p['period']).std()
        upper = ma + (p['std_dev'] * std)
        lower = ma - (p['std_dev'] * std)
        sig = pd.Series(np.nan, index=data.index)
        sig[data['close'] > upper] = 1
        sig[data['close'] < lower] = 0
        data['signal'] = sig.ffill().fillna(0).astype(int)
        
    elif strat_name == "KD 隨機指標":
        low_min = data['low'].rolling(p['period']).min()
        high_max = data['high'].rolling(p['period']).max()
        rsv = (data['close'] - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        sig = pd.Series(np.nan, index=data.index)
        sig[k < p['buy_k']] = 1
        sig[k > p['sell_k']] = 0
        data['signal'] = sig.ffill().fillna(0).astype(int)
        
    elif strat_name == "唐奇安通道 (海龜突破)":
        upper_channel = data['high'].shift(1).rolling(p['entry_period']).max()
        lower_channel = data['low'].shift(1).rolling(p['exit_period']).min()
        sig = pd.Series(np.nan, index=data.index)
        sig[data['close'] > upper_channel] = 1
        sig[data['close'] < lower_channel] = 0
        data['signal'] = sig.ffill().fillna(0).astype(int)

    return data['signal']

# --- 績效指標計算 ---
def eval_performance(df, init_capital):
    ret = df['strategy_ret']
    total_days = len(df)
    if total_days < 2:
        return {}
        
    final_equity = max(1e-6, df['equity_net'].iloc[-1])
    cagr = (final_equity) ** (252 / total_days) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / vol if vol != 0 else 0
    
    cum_max = df['equity_net'].cummax()
    dd = (df['equity_net'] - cum_max) / cum_max
    mdd = dd.min()
    
    if isinstance(df['trades'], pd.Series):
        trade_count = int(df['trades'].sum())
    else:
        trade_count = int(df['trades'])
        
    exposure_time = df['position'].mean()
    final_value = df['portfolio_value'].iloc[-1]
    
    return {
        "預計總投入": f"${init_capital:,.0f}",
        "最終金額": f"${final_value:,.0f}",
        "CAGR 年化報酬": f"{cagr:.2%}",
        "Sharpe 比率": f"{sharpe:.2f}",
        "Max Drawdown (MDD)": f"{mdd:.2%}",
        "累積扣款/交易次數": f"{trade_count} 次",
        "平均持倉水位": f"{exposure_time:.1%}",
        "_raw_cagr": cagr,
        "_raw_mdd": mdd,
        "_raw_trades": trade_count,
        "_raw_exposure": exposure_time,
        "_raw_final": final_value
    }

# --- 執行與顯示 ---
if st.sidebar.button("開始多策略回測與對比"):
    with st.spinner("正在進行多策略回測運算..."):
        data = load_data(symbol, market, start_date, end_date)
        if len(data) > 0:
            data['buy_hold_equity'] = (1 + data['close'].pct_change().fillna(0)).cumprod()
            data['buy_hold_value'] = data['buy_hold_equity'] * initial_capital
            
            bh_days = len(data)
            bh_cagr = (data['buy_hold_equity'].iloc[-1]) ** (252 / bh_days) - 1
            bh_cum_max = data['buy_hold_equity'].cummax()
            bh_mdd = ((data['buy_hold_equity'] - bh_cum_max) / bh_cum_max).min()
            bh_final = data['buy_hold_value'].iloc[-1]
            
            fig = go.Figure()
            metrics_display_list = []
            strategy_raw_data = {}
            
            # 1. 第一列加入【單筆買入並持有 (Buy & Hold)】
            metrics_display_list.append({
                "策略名稱": "單筆買入持有 (Buy & Hold)",
                "預計總投入": f"${initial_capital:,.0f}",
                "最終金額": f"${bh_final:,.0f}",
                "CAGR 年化報酬": f"{bh_cagr:.2%}",
                "Sharpe 比率": "-",
                "Max Drawdown (MDD)": f"{bh_mdd:.2%}",
                "累積扣款/交易次數": "1 次",
                "平均持倉水位": "100.0%"
            })
            
            fig.add_trace(go.Scatter(x=data.index, y=data['buy_hold_value'], name=f"單筆買入持有 (${bh_final:,.0f})", line=dict(color='gray', dash='dash')))
            
            for strat in selected_strategies:
                if strat == "定期定額 (DCA)":
                    d_freq = params["DCA"]["freq"]
                    d_mode = params["DCA"]["mode"]
                    d_val = params["DCA"]["val"]
                    d_custom = params["DCA"]["custom_days"]
                    res = run_dca_backtest_advanced(data, initial_capital, d_freq, d_mode, d_val, d_custom, fee_rate, slippage)
                elif strat == "跌破均線分批買入並持有":
                    d_sma = params["DipDCA"]["sma_period"]
                    d_tf = params["DipDCA"]["timeframe"]
                    d_interval = params["DipDCA"]["min_interval"]
                    d_periods = params["DipDCA"]["periods"]
                    res = run_dip_dca_backtest(data, initial_capital, d_sma, d_tf, d_interval, d_periods, fee_rate, slippage)
                else:
                    p_key = "PriceSMA" if "突破" in strat else ("SMA" if "SMA" in strat else ("RSI" if "RSI" in strat else ("MACD" if "MACD" in strat else ("BB" if "BB" in strat else ("KD" if "KD" in strat else "Donchian")))))
                    sig = generate_signal(data, strat, params.get(p_key, {}))
                    res = run_backtest(data, sig, fee_rate, tax_rate, slippage, initial_capital)
                
                m = eval_performance(res, initial_capital)
                strategy_raw_data[strat] = m
                
                metrics_display_list.append({
                    "策略名稱": strat,
                    "預計總投入": m["預計總投入"],
                    "最終金額": m["最終金額"],
                    "CAGR 年化報酬": m["CAGR 年化報酬"],
                    "Sharpe 比率": m["Sharpe 比率"],
                    "Max Drawdown (MDD)": m["Max Drawdown (MDD)"],
                    "累積扣款/交易次數": m["累積扣款/交易次數"],
                    "平均持倉水位": m["平均持倉水位"]
                })
                
                fig.add_trace(go.Scatter(x=res.index, y=res['portfolio_value'], name=f"{strat} (${m['_raw_final']:,.0f})", line=dict(width=2)))
            
            fig.update_layout(title=f"{symbol} 投入 ${initial_capital:,.0f} 之累積資產金額變化 ($)", xaxis_title="日期", yaxis_title="資產總額 ($)", height=500)
            
            # 1. 顯示全策略績效對比表格
            st.subheader("📊 各策略與 Buy & Hold 績效與金額對比表")
            metrics_df = pd.DataFrame(metrics_display_list).set_index("策略名稱")
            st.dataframe(metrics_df, use_container_width=True)
            
            # 2. 顯示資產總額曲線圖
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. 自動化解讀診斷模組
            st.subheader("💡 戰術診斷：分批投入與單筆投入對比分析")
            
            diagnostic_text = f"總投入本金 **${initial_capital:,.0f}**，在長達 **{bh_days/252:.1f} 年** 的回測中：\n"
            diagnostic_text += f"* **單筆買入持有 (Lump-sum Buy & Hold)** 最終金額為 **${bh_final:,.0f}**（年化報酬率 **{bh_cagr:.2%}**）。\n\n"
            
            for strat, m in strategy_raw_data.items():
                final_val = m["_raw_final"]
                profit = final_val - initial_capital
                
                diagnostic_text += f"#### 📌 【{strat}】診斷報告\n"
                diagnostic_text += f"* **最終資產**: **${final_val:,.0f}**（淨賺 **${profit:,.0f}**）。\n"
                if strat == "定期定額 (DCA)":
                    d_freq = params["DCA"]["freq"]
                    diagnostic_text += f"* **時間頻率**: 採用 **{d_freq}** 設定進行定期扣款，共完成了 **{m['_raw_trades']} 次** 扣款。\n"
                    diagnostic_text += f"* **平滑效果**: 分批進場有效降低了單筆高點套牢的心理壓力，雖然資金在場時間較短導致牛市報酬低於單筆買入，但整體風險曲線更平滑。\n"
                elif strat == "跌破均線分批買入並持有":
                    d_sma = params["DipDCA"]["sma_period"]
                    d_tf = params["DipDCA"]["timeframe"]
                    diagnostic_text += f"* **抄底策略特性**: 設定當價格跌破 **{d_sma} {d_tf}** 時才觸發扣款，全期共於市場回檔低檔成功分批建倉 **{m['_raw_trades']} 次**。\n"
                    diagnostic_text += f"* **避高就低**: 此策略成功避開了高檔追高，並專注在價格低於均線時累積廉價籌碼。\n"
                else:
                    cagr_diff = m["_raw_cagr"] - bh_cagr
                    if cagr_diff < 0:
                        diagnostic_text += f"* **落後主因**: 報酬比單筆買入少了 **${bh_final - final_val:,.0f}**，主要因為持倉水位平均僅有 **{m['_raw_exposure']:.1%}**，空倉時錯過了資產主升段。\n"
                    else:
                        diagnostic_text += f"* **勝出優勢**: 成功超越單筆買入 **${final_val - bh_final:,.0f}**！避開了主要下跌波段。\n"
                diagnostic_text += "\n"
            
            st.markdown(diagnostic_text)
        else:
            st.error("未能獲取數據，請確認標的與日期設定。")
