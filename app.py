import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="多市場與多策略回測工具")

st.title("📈 多市場與多策略比較回測系統 (進階診斷版)")

# --- 1. 側邊欄控制項：標的與市場 (含輔助輸入與預設選單) ---
st.sidebar.header("1. 選擇標的與市場")
market = st.sidebar.selectbox("市場類別", ["美股", "台股", "加密貨幣"])

# 針對不同市場提供「熱門標的清單」，支援打字搜尋與自訂輸入
preset_options = {
    "台股": ["0050 (元大台灣50)", "2330 (台積電)", "0056 (元大高股息)", "2317 (鴻海)", "2454 (聯發科)", "自訂輸入..."],
    "美股": ["QQQ (納斯達克100)", "AAPL (蘋果)", "NVDA (輝達)", "SPY (S&P 500)", "TSLA (特斯拉)", "MSFT (微軟)", "自訂輸入..."],
    "加密貨幣": ["BTC-USD (比特幣)", "ETH-USD (乙太幣)", "SOL-USD (索拉納)", "BNB-USD (幣安幣)", "自訂輸入..."]
}

selected_preset = st.sidebar.selectbox("選擇熱門標的（可打字搜尋）", preset_options[market])

if "自訂輸入" in selected_preset:
    symbol_input = st.sidebar.text_input("輸入自訂代碼 (例如: AMD, 2603, SOL-USD)", value="QQQ" if market == "美股" else ("0050" if market == "台股" else "BTC-USD"))
    symbol = symbol_input.strip().upper()
else:
    # 自動擷取括號前的純代碼
    symbol = selected_preset.split(" ")[0].strip()

# 時間跨度選擇
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
    "SMA 均線交叉", 
    "RSI 超買超賣", 
    "MACD 柱狀圖/交叉", 
    "布林通道 (Bollinger Bands)", 
    "KD 隨機指標", 
    "唐奇安通道 (海龜突破)"
]

selected_strategies = st.sidebar.multiselect(
    "選擇要比較的策略（可多選）",
    available_strats,
    default=["SMA 均線交叉", "RSI 超買超賣", "唐奇安通道 (海龜突破)"]
)

params = {}
if "SMA 均線交叉" in selected_strategies:
    st.sidebar.markdown("**[SMA 均線]**")
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
    
    if strat_name == "SMA 均線交叉":
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
        sig[data['close'] > upper] = 1   # 突破上軌順勢做多
        sig[data['close'] < lower] = 0   # 跌破下軌出場
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
        sig[data['close'] > upper_channel] = 1 # 創新高買入
        sig[data['close'] < lower_channel] = 0 # 創新低賣出
        data['signal'] = sig.ffill().fillna(0).astype(int)

    return data['signal']

# --- 回測核心引擎 ---
def run_backtest(df, signal_series, fee, tax, slip):
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
    return data

# --- 績效指標計算 ---
def eval_performance(df):
    ret = df['strategy_ret']
    total_days = len(df)
    if total_days < 2:
        return {}
    cagr = (df['equity_net'].iloc[-1]) ** (252 / total_days) - 1
    vol = ret.std() * np.sqrt(252)
    downside_vol = ret[ret < 0].std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / vol if vol != 0 else 0
    sortino = (ret.mean() * 252) / downside_vol if downside_vol != 0 and not np.isnan(downside_vol) else 0
    
    cum_max = df['equity_net'].cummax()
    dd = (df['equity_net'] - cum_max) / cum_max
    mdd = dd.min()
    
    pos_ret = ret[ret > 0].sum()
    neg_ret = abs(ret[ret < 0].sum())
    profit_factor = (pos_ret / neg_ret) if neg_ret != 0 else np.nan
    trade_count = int(df['trades'].sum())
    exposure_time = df['position'].mean()
    
    return {
        "CAGR 年化報酬": f"{cagr:.2%}",
        "Sharpe 比率": f"{sharpe:.2f}",
        "Sortino 比率": f"{sortino:.2f}",
        "Max Drawdown (MDD)": f"{mdd:.2%}",
        "Profit Factor 獲利因子": f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "N/A",
        "總交易次數": f"{trade_count} 次",
        "在場持倉時間比例": f"{exposure_time:.1%}",
        "_raw_cagr": cagr,
        "_raw_mdd": mdd,
        "_raw_trades": trade_count,
        "_raw_exposure": exposure_time
    }

# --- 執行與顯示 ---
if st.sidebar.button("開始多策略回測與對比"):
    if not selected_strategies:
        st.error("請至少在左側邊欄選擇一種策略！")
    else:
        with st.spinner("正在進行多策略回測運算..."):
            data = load_data(symbol, market, start_date, end_date)
            if len(data) > 0:
                data['buy_hold'] = (1 + data['close'].pct_change().fillna(0)).cumprod()
                
                # 計算 Buy & Hold 指標
                bh_total_days = len(data)
                bh_cagr = (data['buy_hold'].iloc[-1]) ** (252 / bh_total_days) - 1
                bh_cum_max = data['buy_hold'].cummax()
                bh_mdd = ((data['buy_hold'] - bh_cum_max) / bh_cum_max).min()
                
                fig = go.Figure()
                metrics_display_list = []
                strategy_raw_data = {}
                
                fig.add_trace(go.Scatter(x=data.index, y=data['buy_hold'], name=f"買入持有 (CAGR: {bh_cagr:.1%})", line=dict(color='gray', dash='dash')))
                
                for strat in selected_strategies:
                    p_key = "SMA" if "SMA" in strat else ("RSI" if "RSI" in strat else ("MACD" if "MACD" in strat else ("BB" if "BB" in strat else ("KD" if "KD" in strat else "Donchian"))))
                    sig = generate_signal(data, strat, params[p_key])
                    res = run_backtest(data, sig, fee_rate, tax_rate, slippage)
                    
                    m = eval_performance(res)
                    strategy_raw_data[strat] = m
                    
                    # 整理顯示表格欄位
                    disp_m = {
                        "策略名稱": strat,
                        "CAGR 年化報酬": m["CAGR 年化報酬"],
                        "Sharpe 比率": m["Sharpe 比率"],
                        "Max Drawdown (MDD)": m["Max Drawdown (MDD)"],
                        "總交易次數": m["總交易次數"],
                        "持倉時間比例": m["在場持倉時間比例"]
                    }
                    metrics_display_list.append(disp_m)
                    
                    fig.add_trace(go.Scatter(x=res.index, y=res['equity_net'], name=f"{strat} ({m['CAGR 年化報酬']})", line=dict(width=2)))
                
                fig.update_layout(title=f"{symbol} 多策略資產曲線對比", xaxis_title="日期", yaxis_title="淨值 (Base=1.0)", height=500)
                
                # 1. 顯示全策略績效對比表格
                st.subheader("📊 各策略與 Buy & Hold 績效指標對比表")
                metrics_df = pd.DataFrame(metrics_display_list).set_index("策略名稱")
                st.dataframe(metrics_df, use_container_width=True)
                
                # 2. 顯示資產曲線圖
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. 自動化解讀診斷模組
                st.subheader("💡 戰術診斷：為什麼策略表現會落後於「買入並持有」？")
                
                diagnostic_text = f"在本次長達 **{bh_total_days/252:.1f} 年** 的回測週期中，**買入並持有 (Buy & Hold)** 實現了 **{bh_cagr:.2%}** 的年化報酬率，最大回撤為 **{bh_mdd:.2%}**。\n\n"
                
                for strat, m in strategy_raw_data.items():
                    cagr_diff = m["_raw_cagr"] - bh_cagr
                    cash_time = (1 - m["_raw_exposure"])
                    trades = m["_raw_trades"]
                    
                    diagnostic_text += f"#### 📌 【{strat}】診斷報告\n"
                    if cagr_diff < 0:
                        diagnostic_text += f"* **報酬差距**：年化報酬率比 Buy & Hold 低了 **{abs(cagr_diff):.2%}**。\n"
                        diagnostic_text += f"* **核心敗因一 (空倉代價)**：該策略有 **{cash_time:.1%}** 的時間處於空倉狀態。對於 {symbol} 這類具備長期向上趨勢的資產，空倉意味著徹底錯過市場暴漲段的複利累積。\n"
                        diagnostic_text += f"* **核心敗因二 (摩擦損耗)**：共進行了 **{trades} 次** 進出場。每次交易產生的手續費、證交稅與滑點在長週期下產生嚴重的暗扣效應。\n"
                        if m["_raw_mdd"] > bh_mdd:
                            diagnostic_text += f"* **優點/避險效果**：雖然總報酬落後，但最大回撤成功控制在 **{m['_raw_mdd']:.2%}**（優於大盤的 {bh_mdd:.2%}），有效減緩熊市恐慌。\n"
                    else:
                        diagnostic_text += f"* **勝出優勢**：年化報酬率超越 Buy & Hold **{cagr_diff:.2%}**！成功靠著指標避開多數下跌波段，並保持良好的獲利能力。\n"
                    
                    diagnostic_text += "\n"
                
                st.markdown(diagnostic_text)
            else:
                st.error("未能獲取數據，請確認標的與日期設定。")
