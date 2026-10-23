import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="多市場與多策略回測工具")

st.title("📈 多市場與多策略比較回測系統 (0 元版)")

# --- 1. 側邊欄控制項：標的與市場 ---
st.sidebar.header("1. 選擇標的與市場")
market = st.sidebar.selectbox("市場類別", ["台股", "美股", "加密貨幣"])

default_symbol = "0050" if market == "台股" else ("AAPL" if market == "美股" else "BTC-USD")
symbol = st.sidebar.text_input("輸入股票/幣種代碼", value=default_symbol)

# 新增 10~20 年快捷時間選擇器
time_option = st.sidebar.selectbox(
    "回測時間跨度",
    ["近 5 年", "近 10 年", "近 20 年", "自訂日期範圍"],
    index=2  # 預設選取近 20 年
)

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
if market == "台股":
    fee_rate = st.sidebar.number_input("手續費率 (%)", value=0.071, step=0.01) / 100
    default_tax = 0.10 if symbol.startswith("00") else 0.30
    tax_rate = st.sidebar.number_input("賣出證交稅 (%)", value=default_tax, step=0.05) / 100
    slippage = st.sidebar.number_input("單邊預估滑點 (%)", value=0.03, step=0.01) / 100
elif market == "美股":
    fee_rate = st.sidebar.number_input("手續費率 (%)", value=0.00, step=0.01) / 100
    tax_rate = 0.0
    slippage = st.sidebar.number_input("單邊預估滑點 (%)", value=0.03, step=0.01) / 100
else:  # 加密貨幣
    fee_rate = st.sidebar.number_input("手續費率 (%)", value=0.10, step=0.01) / 100
    tax_rate = 0.0
    slippage = st.sidebar.number_input("單邊預估滑點 (%)", value=0.10, step=0.01) / 100

# --- 3. 側邊欄控制項：多策略選擇與參數設定 ---
st.sidebar.header("3. 選擇與設定回測策略")
selected_strategies = st.sidebar.multiselect(
    "選擇要比較的策略（可多選）",
    ["SMA 均線交叉", "RSI 超買超賣", "MACD 柱狀圖/交叉"],
    default=["SMA 均線交叉", "RSI 超買超賣"]
)

params = {}
if "SMA 均線交叉" in selected_strategies:
    st.sidebar.markdown("**[SMA 均線參數]**")
    sma_fast = st.sidebar.number_input("快線 (天)", value=20, min_value=1)
    sma_slow = st.sidebar.number_input("慢線 (天)", value=60, min_value=2)
    params["SMA"] = {"fast": int(sma_fast), "slow": int(sma_slow)}

if "RSI 超買超賣" in selected_strategies:
    st.sidebar.markdown("**[RSI 策略參數]**")
    rsi_p = st.sidebar.number_input("RSI 週期", value=14, min_value=1)
    rsi_buy = st.sidebar.number_input("超賣買入臨界值", value=30, max_value=50)
    rsi_sell = st.sidebar.number_input("超買賣出臨界值", value=70, min_value=50)
    params["RSI"] = {"period": int(rsi_p), "buy": rsi_buy, "sell": rsi_sell}

if "MACD 柱狀圖/交叉" in selected_strategies:
    st.sidebar.markdown("**[MACD 參數]**")
    macd_fast = st.sidebar.number_input("快線 EMA", value=12)
    macd_slow = st.sidebar.number_input("慢線 EMA", value=26)
    macd_signal = st.sidebar.number_input("訊號線", value=9)
    params["MACD"] = {"fast": int(macd_fast), "slow": int(macd_slow), "signal": int(macd_signal)}

# --- 數據獲取模組 ---
@st.cache_data
def load_data(ticker, market_type, start, end):
    formatted_ticker = ticker
    if market_type == "台股" and not ticker.endswith(".TW") and not ticker.endswith(".TWO"):
        formatted_ticker = ticker + ".TW"
    
    df = yf.download(formatted_ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        close_data = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
    else:
        close_data = df['Close']
        
    res_df = pd.DataFrame({'close': close_data})
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
        sig[rsi < p['buy']] = 1   # 低於超賣區買入
        sig[rsi > p['sell']] = 0  # 高於超買區賣出
        data['signal'] = sig.ffill().fillna(0).astype(int)
        
    elif strat_name == "MACD 柱狀圖/交叉":
        ema_fast = data['close'].ewm(span=p['fast'], adjust=False).mean()
        ema_slow = data['close'].ewm(span=p['slow'], adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=p['signal'], adjust=False).mean()
        data['signal'] = (macd_line > signal_line).astype(int)
        
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
    
    return {
        "CAGR 年化報酬": f"{cagr:.2%}",
        "Sharpe 比率": f"{sharpe:.2f}",
        "Sortino 比率": f"{sortino:.2f}",
        "Max Drawdown (MDD)": f"{mdd:.2%}",
        "Profit Factor 獲利因子": f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "N/A",
        "總交易次數": f"{trade_count} 次"
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
                
                fig = go.Figure()
                metrics_table = []
                
                # 繪製對照基準：買入持有 (Buy & Hold)
                fig.add_trace(go.Scatter(x=data.index, y=data['buy_hold'], name="買入持有 (Buy & Hold)", line=dict(color='gray', dash='dash')))
                
                # 依序計算並繪製各個選取的策略
                for strat in selected_strategies:
                    p_key = "SMA" if "SMA" in strat else ("RSI" if "RSI" in strat else "MACD")
                    sig = generate_signal(data, strat, params[p_key])
                    res = run_backtest(data, sig, fee_rate, tax_rate, slippage)
                    
                    m = eval_performance(res)
                    m["策略名稱"] = strat
                    metrics_table.append(m)
                    
                    fig.add_trace(go.Scatter(x=res.index, y=res['equity_net'], name=strat, line=dict(width=2)))
                
                fig.update_layout(title=f"{symbol} 多策略資產曲線對比", xaxis_title="日期", yaxis_title="淨值 (Base=1.0)", height=500)
                
                # 1. 顯示全策略績效對比表格
                st.subheader("📊 各策略績效指標對比表")
                metrics_df = pd.DataFrame(metrics_table).set_index("策略名稱")
                st.dataframe(metrics_df, use_container_width=True)
                
                # 2. 顯示多策略資產曲線比較圖
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("未能獲取數據，請確認標的與日期設定。")
