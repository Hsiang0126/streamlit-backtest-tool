import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="免費多市場交易策略回測工具")

st.title("📈 多市場策略回測與優缺點診斷系統 (0 元版)")

# --- 側邊欄控制項 ---
st.sidebar.header("1. 選擇標的與市場")
market = st.sidebar.selectbox("市場類別", ["台股", "美股", "加密貨幣"])

default_symbol = "2330" if market == "台股" else ("AAPL" if market == "美股" else "BTC-USD")
symbol = st.sidebar.text_input("輸入股票/幣種代碼", value=default_symbol)

start_date = st.sidebar.date_input("開始日期", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("結束日期", pd.to_datetime("2026-01-01"))

st.sidebar.header("2. 設定交易成本與滑點")
if market == "台股":
    fee_rate = st.sidebar.number_input("手續費率 (%)", value=0.071, step=0.01) / 100
    tax_rate = st.sidebar.number_input("賣出證交稅 (%)", value=0.3, step=0.05) / 100
    slippage = st.sidebar.number_input("單邊預估滑點 (%)", value=0.05, step=0.01) / 100
elif market == "美股":
    fee_rate = st.sidebar.number_input("手續費率 (%)", value=0.00, step=0.01) / 100
    tax_rate = 0.0
    slippage = st.sidebar.number_input("單邊預估滑點 (%)", value=0.03, step=0.01) / 100
else:  # 加密貨幣
    fee_rate = st.sidebar.number_input("手續費率 (%)", value=0.10, step=0.01) / 100
    tax_rate = 0.0
    slippage = st.sidebar.number_input("單邊預估滑點 (%)", value=0.10, step=0.01) / 100

# --- 數據獲取模組 ---
@st.cache_data
def load_data(ticker, market_type, start, end):
    formatted_ticker = ticker
    if market_type == "台股" and not ticker.endswith(".TW") and not ticker.endswith(".TWO"):
        formatted_ticker = ticker + ".TW"
    
    df = yf.download(formatted_ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close'].to_frame(name='close')
    else:
        df = df[['Close']].rename(columns={'Close': 'close'})
    return df.dropna()

# --- 核心回測引擎 (含成本計算) ---
def run_backtest_with_costs(df, fast_p=20, slow_p=60, fee=0.0, tax=0.0, slip=0.0):
    data = df.copy()
    data['fast_sma'] = data['close'].rolling(fast_p).mean()
    data['slow_sma'] = data['close'].rolling(slow_p).mean()
    
    # 訊號生成：1為持倉，0為空倉
    data['signal'] = (data['fast_sma'] > data['slow_sma']).astype(int)
    data['position'] = data['signal'].shift(1).fillna(0)
    
    # 判斷交易發生點 (買入或賣出)
    data['trades'] = data['position'].diff().abs().fillna(0)
    
    # 每日市場原始報酬率
    data['market_ret'] = data['close'].pct_change().fillna(0)
    
    # 扣除手續費與滑點 (每次進出場扣除 fee + slip，賣出時額外加扣 tax)
    cost_per_trade = fee + slip
    data['cost'] = 0.0
    
    # 持倉變動即觸發成本
    trade_mask = data['trades'] > 0
    data.loc[trade_mask, 'cost'] = cost_per_trade
    
    # 賣出點加算證交稅
    sell_mask = (data['position'].diff() < 0)
    data.loc[sell_mask, 'cost'] += tax
    
    # 策略每日淨報酬率
    data['strategy_ret'] = (data['market_ret'] * data['position']) - data['cost']
    
    # 累積資產淨值
    data['equity_raw'] = (1 + (data['market_ret'] * data['position'])).cumprod()
    data['equity_net'] = (1 + data['strategy_ret']).cumprod()
    data['buy_hold'] = (1 + data['market_ret']).cumprod()
    
    return data

# --- 績效指標計算 ---
def eval_performance(df):
    ret = df['strategy_ret']
    total_days = len(df)
    if total_days < 2:
        return {}
    
    cagr = (df['equity_net'].iloc[-1]) ** (252 / total_days) - 1
    
    # 夏普比率與索提諾比率
    vol = ret.std() * np.sqrt(252)
    downside_vol = ret[ret < 0].std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / vol if vol != 0 else 0
    sortino = (ret.mean() * 252) / downside_vol if downside_vol != 0 and not np.isnan(downside_vol) else 0
    
    # 最大回撤 (MDD)
    cum_max = df['equity_net'].cummax()
    dd = (df['equity_net'] - cum_max) / cum_max
    mdd = dd.min()
    
    # 勝率與獲利因子
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
if st.sidebar.button("開始回測與評估"):
    with st.spinner("正在抓取數據並進行雙重回測..."):
        data = load_data(symbol, market, start_date, end_date)
        
        if len(data) > 0:
            res = run_backtest_with_costs(data, fee=fee_rate, tax=tax_rate, slip=slippage)
            metrics = eval_performance(res)
            
            # 1. 績效指標顯示
            st.subheader("📊 策略實踐績效 (已扣除交易成本與滑點)")
            col_list = st.columns(len(metrics))
            for col, (k, v) in zip(col_list, metrics.items()):
                col.metric(k, v)
                
            # 2. 資產曲線比較圖
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.index, y=res['equity_net'], name="扣除成本後淨值", line=dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=res.index, y=res['equity_raw'], name="未扣成本淨值 (理想狀態)", line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=res.index, y=res['buy_hold'], name="買入持有 (Buy & Hold)", line=dict(color='blue', width=1.5)))
            
            fig.update_layout(title=f"{symbol} 雙重資產曲線對比", xaxis_title="日期", yaxis_title="淨值 (Base=1.0)", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. 策略優缺點診斷報告
            st.subheader("💡 策略優缺點與適合市況診斷")
            
            cagr_val = float(metrics["CAGR 年化報酬"].replace("%", "")) / 100
            mdd_val = float(metrics["Max Drawdown (MDD)"].replace("%", "")) / 100
            trades_val = int(metrics["總交易次數"].replace(" 次", ""))
            
            st.markdown(f"""
            * **核心優點**：
              * **趨勢捕捉力**：當 {symbol} 出現長達數月的大順風趨勢時，均線交叉能完整抓住主升段。
              * **下行風險防護**：在深度熊市（如 2022 年大盤暴跌）中，策略能及時切換為空倉，有效縮減最大回撤。
            * **核心缺點與潛在風險**：
              * **盤整洗盤磨損**：在橫盤震盪市況下，會頻繁觸發交叉買賣，產生高達 **{trades_val} 次** 的摩擦成本損耗。
              * **訊號滯後性**：均線屬於落後指標，暴漲或暴跌發生的前幾天無法第一時間反應，頂部與底部均會有一定利潤回吐。
            * **總體實戰建議**：適合應用於**高波動、強趨勢**標的（如加密貨幣或科技龍頭股）；若應用於低波動的盤整股，建議加大均線週期或搭配 RSI 等震盪指標輔助過濾假訊號。
            """)
        else:
            st.error("未能獲取數據，請確認代碼與日期設定是否正確。")