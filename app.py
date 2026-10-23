import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="多市場與多策略回測工具")

st.title("📈 多市場與多策略比較回測系統 (完整金額與策略版)")

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
initial_capital = st.sidebar.number_input("期初投入本金 ($)", value=100000, step=10000)

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
    "價格突破均線買入/跌破賣出",
    "定期定額 (DCA)",
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
    default=["價格突破均線買入/跌破賣出", "定期定額 (DCA)", "SMA 雙均線交叉"]
)

params = {}
if "價格突破均線買入/跌破賣出" in selected_strategies:
    st.sidebar.markdown("**[價格突破均線策略]**")
    params["PriceSMA"] = {
        "period": int(st.sidebar.number_input("基準均線天數 (天)", value=20, min_value=1))
    }

if "SMA 雙均線交叉" in selected_strategies:
    st.sidebar.markdown("**[SMA 雙均線]**")
    params["SMA"] = {
        "fast": int(st.sidebar.number_input("SMA 快線 (天)", value=20)),
        "slow": int(st.sidebar.number_input("SMA 慢線 (天)", value=60))
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

# --- 數據獲取模組 ---
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
        
    res_df = pd.DataFrame({'close': close_data, 'high': high_data, 'low': low_data})
    return res_df.dropna()

# --- 策略訊號產生器 ---
def generate_signal(df, strat_name, p):
    data = df.copy()
    
    if strat_name == "價格突破均線買入/跌破賣出":
        sma = data['close'].rolling(p['period']).mean()
        data['signal'] = (data['close'] > sma).astype(int)
        
    elif strat_name == "定期定額 (DCA)":
        data['signal'] = 1 # 定期定額為持續在場
        
    elif strat_name == "SMA 雙均線交叉":
        fast_m = data['close'].rolling(p['fast']).mean()
        slow_m = data['close'].rolling(p['slow']).mean()
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

# --- 回測核心引擎 ---
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

# --- 績效指標計算 ---
def eval_performance(df, init_capital):
    ret = df['strategy_ret']
    total_days = len(df)
    if total_days < 2:
        return {}
    cagr = (df['equity_net'].iloc[-1]) ** (252 / total_days) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / vol if vol != 0 else 0
    
    cum_max = df['equity_net'].cummax()
    dd = (df['equity_net'] - cum_max) / cum_max
    mdd = dd.min()
    
    trade_count = int(df['trades'].sum())
    exposure_time = df['position'].mean()
    final_value = df['portfolio_value'].iloc[-1]
    
    return {
        "期初投入": f"${init_capital:,.0f}",
        "最終金額": f"${final_value:,.0f}",
        "CAGR 年化報酬": f"{cagr:.2%}",
        "Sharpe 比率": f"{sharpe:.2f}",
        "Max Drawdown (MDD)": f"{mdd:.2%}",
        "總交易次數": f"{trade_count} 次",
        "在場持倉時間": f"{exposure_time:.1%}",
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
            
            # 1. 第一列加入【買入並持有 (Buy & Hold)】
            metrics_display_list.append({
                "策略名稱": "買入並持有 (Buy & Hold)",
                "期初投入": f"${initial_capital:,.0f}",
                "最終金額": f"${bh_final:,.0f}",
                "CAGR 年化報酬": f"{bh_cagr:.2%}",
                "Sharpe 比率": "-",
                "Max Drawdown (MDD)": f"{bh_mdd:.2%}",
                "總交易次數": "1 次",
                "在場持倉時間": "100.0%"
            })
            
            fig.add_trace(go.Scatter(x=data.index, y=data['buy_hold_value'], name=f"買入持有 (${bh_final:,.0f})", line=dict(color='gray', dash='dash')))
            
            for strat in selected_strategies:
                p_key = "PriceSMA" if "突破" in strat else ("SMA" if "SMA" in strat else ("RSI" if "RSI" in strat else ("MACD" if "MACD" in strat else ("BB" if "BB" in strat else ("KD" if "KD" in strat else "Donchian")))))
                sig = generate_signal(data, strat, params.get(p_key, {}))
                res = run_backtest(data, sig, fee_rate, tax_rate, slippage, initial_capital)
                
                m = eval_performance(res, initial_capital)
                strategy_raw_data[strat] = m
                
                metrics_display_list.append({
                    "策略名稱": strat,
                    "期初投入": m["期初投入"],
                    "最終金額": m["最終金額"],
                    "CAGR 年化報酬": m["CAGR 年化報酬"],
                    "Sharpe 比率": m["Sharpe 比率"],
                    "Max Drawdown (MDD)": m["Max Drawdown (MDD)"],
                    "總交易次數": m["總交易次數"],
                    "在場持倉時間": m["在場持倉時間"]
                })
                
                fig.add_trace(go.Scatter(x=res.index, y=res['portfolio_value'], name=f"{strat} (${m['_raw_final']:,.0f})", line=dict(width=2)))
            
            fig.update_layout(title=f"{symbol} 投入 ${initial_capital:,.0f} 之累積資產金額變化 ($)", xaxis_title="日期", yaxis_title="資產總額 ($)", height=500)
            
            # 1. 顯示全策略績效對比表格 (含 Buy & Hold 與實際金額)
            st.subheader("📊 各策略與 Buy & Hold 績效與金額對比表")
            metrics_df = pd.DataFrame(metrics_display_list).set_index("策略名稱")
            st.dataframe(metrics_df, use_container_width=True)
            
            # 2. 顯示資產總額曲線圖
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. 自動化解讀診斷模組
            st.subheader("💡 戰術診斷：本金變化與策略績效分析")
            
            diagnostic_text = f"期初投入 **${initial_capital:,.0f}**，在長達 **{bh_days/252:.1f} 年** 的回測中：\n"
            diagnostic_text += f"* **買入並持有 (Buy & Hold)** 最終變成了 **${bh_final:,.0f}**（淨賺 **${bh_final - initial_capital:,.0f}**，年化報酬率 **{bh_cagr:.2%}**）。\n\n"
            
            for strat, m in strategy_raw_data.items():
                final_val = m["_raw_final"]
                profit = final_val - initial_capital
                cagr_diff = m["_raw_cagr"] - bh_cagr
                
                diagnostic_text += f"#### 📌 【{strat}】診斷報告\n"
                diagnostic_text += f"* **最終資產**: **${final_val:,.0f}**（淨賺 **${profit:,.0f}**）。\n"
                if cagr_diff < 0:
                    diagnostic_text += f"* **落後主因**: 報酬比 Buy & Hold 少了 **${bh_final - final_val:,.0f}**。主要源於持倉時間僅有 **{m['_raw_exposure']:.1%}**，空倉期間錯過了資產暴漲段，且進出場 **{m['_raw_trades']} 次** 累積了摩擦成本損耗。\n"
                else:
                    diagnostic_text += f"* **勝出優勢**: 成功超越 Buy & Hold **${final_val - bh_final:,.0f}**！避開了主要下跌段並保持資金高效率運作。\n"
                diagnostic_text += "\n"
            
            st.markdown(diagnostic_text)
        else:
            st.error("未能獲取數據，請確認標的與日期設定。")
