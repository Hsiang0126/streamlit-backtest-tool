import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="多市場與多策略回測工具")

st.title("📈 多市場與多策略比較回測系統 (跌破均線創新低抄底版)")

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

# 資金與現金流設定
initial_capital = st.sidebar.number_input("期初投入本金 ($)", value=100000, step=10000)
monthly_inflow = st.sidebar.number_input("每月固定新增可投入資金 ($)", value=10000, step=1000)
reinvest_dividends = st.sidebar.checkbox("計入股利再投入 (還原總報酬股價)", value=True)

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
    "跌破均線且創新低買入",
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
    default=["定期定額 (DCA)", "跌破均線且創新低買入", "價格突破均線買入/跌破賣出"]
)

params = {}

if "定期定額 (DCA)" in selected_strategies:
    st.sidebar.markdown("**[定期定額 (DCA) 設定]**")
    dca_freq = st.sidebar.selectbox("扣款頻率", ["每月 (Monthly)", "每週 (Weekly)", "雙週 (Bi-weekly)", "自訂交易日數"])
    custom_dca_days = 20
    if dca_freq == "自訂交易日數":
        custom_dca_days = int(st.sidebar.number_input("相隔交易日數", value=20, min_value=1))
        
    dca_val = float(st.sidebar.number_input("定額扣款金額 ($)", value=10000, min_value=100))
    params["DCA"] = {
        "freq": dca_freq,
        "val": dca_val,
        "custom_days": custom_dca_days
    }

if "跌破均線且創新低買入" in selected_strategies:
    st.sidebar.markdown("**[跌破均線創新低買入設定]**")
    params["DipDCA"] = {
        "sma_period": int(st.sidebar.number_input("基準均線 (週期數)", value=60, min_value=1)),
        "timeframe": st.sidebar.selectbox("均線單位 (DipDCA)", ["週線 (Weekly)", "日線 (Daily)"], key="dip_tf"),
        "buy_amount": float(st.sidebar.number_input("每次觸發買入金額 ($)", value=10000, min_value=100)),
        "min_interval": int(st.sidebar.number_input("買入最小間隔 (交易日)", value=1, min_value=1))
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

# --- 數據獲取與均線計算工具 (含股利再投入切換與階梯斷層修正) ---
@st.cache_data
def load_data(ticker, market_type, start, end, auto_adjust=True):
    formatted_ticker = ticker
    if market_type == "台股" and not ticker.endswith(".TW") and not ticker.endswith(".TWO"):
        formatted_ticker = ticker + ".TW"
    
    df = yf.download(formatted_ticker, start=start, end=end, auto_adjust=auto_adjust)
    
    if isinstance(df.columns, pd.MultiIndex):
        close_data = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high_data = df['High'].iloc[:, 0] if 'High' in df and isinstance(df['High'], pd.DataFrame) else (df['High'] if 'High' in df else close_data)
        low_data = df['Low'].iloc[:, 0] if 'Low' in df and isinstance(df['Low'], pd.DataFrame) else (df['Low'] if 'Low' in df else close_data)
    else:
        close_data = df['Close']
        high_data = df['High'] if 'High' in df else close_data
        low_data = df['Low'] if 'Low' in df else close_data
        
    res_df = pd.DataFrame({'close': close_data, 'high': high_data, 'low': low_data}).dropna()
    res_df = res_df[res_df['close'] > 1.0]
    
    # 校正 Yahoo Finance 0050.TW 2014 年底結構性斷層
    if market_type == "台股":
        pct = res_df['close'].pct_change()
        crash_points = res_df.index[pct < -0.30]
        
        for crash_idx in crash_points:
            loc = res_df.index.get_loc(crash_idx)
            if loc > 0:
                prev_price = res_df['close'].iloc[loc - 1]
                after_avg = res_df['close'].iloc[loc:min(loc + 5, len(res_df))].mean()
                if after_avg / prev_price < 0.7:
                    ratio = res_df['close'].iloc[loc] / prev_price
                    close_col = res_df.columns.get_loc('close')
                    high_col = res_df.columns.get_loc('high')
                    low_col = res_df.columns.get_loc('low')
                    res_df.iloc[:loc, close_col] *= ratio
                    res_df.iloc[:loc, high_col] *= ratio
                    res_df.iloc[:loc, low_col] *= ratio
                    
    return res_df.dropna()

def calculate_ma(df, period, timeframe="日線 (Daily)"):
    if timeframe == "週線 (Weekly)":
        weekly_close = df['close'].resample('W-FRI').last().dropna()
        weekly_ma = weekly_close.rolling(period).mean()
        return weekly_ma.reindex(df.index, method='ffill')
    else:
        return df['close'].rolling(period).mean()

# --- 單筆買入持有 (含每月定期注資) ---
def run_buy_and_hold_with_inflow(df, init_capital, monthly_inflow, fee, slip):
    data = df.copy()
    total_days = len(data)
    month_series = pd.Series(data.index.month, index=data.index)
    month_changed = (month_series != month_series.shift(1))
    
    cash = init_capital
    total_invested = init_capital
    shares = 0.0
    portfolio_values = []
    invested_history = []
    
    for i in range(total_days):
        if i > 0 and month_changed.iloc[i]:
            cash += monthly_inflow
            total_invested += monthly_inflow
            
        if cash > 0:
            cost_factor = 1 + fee + slip
            buy_price = data['close'].iloc[i] * cost_factor
            bought_shares = cash / buy_price
            shares += bought_shares
            cash = 0.0
            
        current_value = shares * data['close'].iloc[i]
        portfolio_values.append(current_value)
        invested_history.append(total_invested)
        
    data['portfolio_value'] = portfolio_values
    data['total_invested'] = invested_history
    data['equity_net'] = data['portfolio_value'] / data['total_invested']
    data['strategy_ret'] = data['equity_net'].pct_change().fillna(0)
    data['position'] = 1.0
    data['trades'] = 1
    return data

# --- DCA 定期定額回測引擎 (含每月現金流注入) ---
def run_dca_backtest_advanced(df, init_capital, monthly_inflow, freq_type, dca_amount, custom_days, fee, slip):
    data = df.copy()
    total_days = len(data)
    
    month_series = pd.Series(data.index.month, index=data.index)
    month_changed = (month_series != month_series.shift(1))
    
    if freq_type == "每月 (Monthly)":
        buy_mask = month_changed
    elif freq_type == "每週 (Weekly)":
        week_series = pd.Series(data.index.isocalendar().week, index=data.index)
        buy_mask = (week_series != week_series.shift(1))
    elif freq_type == "雙週 (Bi-weekly)":
        buy_mask = (pd.Series(range(total_days), index=data.index) % 10 == 0)
    else:
        step = max(1, int(custom_days))
        buy_mask = (pd.Series(range(total_days), index=data.index) % step == 0)
        
    buy_indices = set(data.index[buy_mask])
    
    cash = init_capital
    total_invested = init_capital
    shares = 0.0
    portfolio_values = []
    invested_history = []
    daily_positions = []
    daily_trades = [0] * total_days
    
    for i in range(total_days):
        current_date = data.index[i]
        
        if i > 0 and month_changed.iloc[i]:
            cash += monthly_inflow
            total_invested += monthly_inflow
            
        if current_date in buy_indices and cash > 0:
            actual_buy = min(cash, dca_amount)
            cost_factor = 1 + fee + slip
            buy_price = data['close'].iloc[i] * cost_factor
            bought_shares = actual_buy / buy_price
            shares += bought_shares
            cash -= actual_buy
            daily_trades[i] = 1
            
        current_stock_val = shares * data['close'].iloc[i]
        current_value = cash + current_stock_val
        portfolio_values.append(current_value)
        invested_history.append(total_invested)
        pos_ratio = (current_stock_val / current_value) if current_value > 0 else 0.0
        daily_positions.append(pos_ratio)
        
    data['portfolio_value'] = portfolio_values
    data['total_invested'] = invested_history
    data['equity_net'] = data['portfolio_value'] / data['total_invested']
    data['strategy_ret'] = data['equity_net'].pct_change().fillna(0)
    data['position'] = daily_positions
    data['trades'] = daily_trades
    return data

# --- 跌破均線且創新低買入引擎 (已移除均線需下降限制) ---
def run_dip_dca_backtest(df, init_capital, monthly_inflow, sma_period, timeframe, buy_amount, min_interval_days, fee, slip):
    data = df.copy()
    sma = calculate_ma(data, sma_period, timeframe)
    total_days = len(data)
    
    month_series = pd.Series(data.index.month, index=data.index)
    month_changed = (month_series != month_series.shift(1))
    
    cash = init_capital
    total_invested = init_capital
    shares = 0.0
    portfolio_values = []
    invested_history = []
    daily_positions = []
    daily_trades = [0] * total_days
    last_buy_idx = -min_interval_days
    
    min_price_in_dip = float('inf')  # 追蹤當前跌破區間的最低價
    
    for i in range(total_days):
        if i > 0 and month_changed.iloc[i]:
            cash += monthly_inflow
            total_invested += monthly_inflow
            
        price = data['close'].iloc[i]
        sma_val = sma.iloc[i]
        
        # 條件 1：價格低於均線
        is_below_sma = not np.isnan(sma_val) and (price < sma_val)
        
        if is_below_sma:
            # 條件 2：價格創出該次跌破區間的新低點
            if price < min_price_in_dip:
                if (i - last_buy_idx >= min_interval_days) and cash > 0:
                    actual_buy = min(cash, buy_amount)
                    cost_factor = 1 + fee + slip
                    buy_price = price * cost_factor
                    bought_shares = actual_buy / buy_price
                    shares += bought_shares
                    cash -= actual_buy
                    last_buy_idx = i
                    daily_trades[i] = 1
                    min_price_in_dip = price  # 更新該次跌破區間的最低價
        else:
            # 當價格重回均線上方時，重置區域最低價門檻
            min_price_in_dip = float('inf')
            
        current_stock_val = shares * price
        current_value = cash + current_stock_val
        portfolio_values.append(current_value)
        invested_history.append(total_invested)
        pos_ratio = (current_stock_val / current_value) if current_value > 0 else 0.0
        daily_positions.append(pos_ratio)
        
    data['portfolio_value'] = portfolio_values
    data['total_invested'] = invested_history
    data['equity_net'] = data['portfolio_value'] / data['total_invested']
    data['strategy_ret'] = data['equity_net'].pct_change().fillna(0)
    data['position'] = daily_positions
    data['trades'] = daily_trades
    return data

# --- 一般擇時策略回測引擎 (含每月現金注入) ---
def run_backtest(df, signal_series, fee, tax, slip, init_capital, monthly_inflow):
    data = df.copy()
    total_days = len(data)
    month_series = pd.Series(data.index.month, index=data.index)
    month_changed = (month_series != month_series.shift(1))
    
    position = signal_series.shift(1).fillna(0)
    
    cash = init_capital
    total_invested = init_capital
    shares = 0.0
    portfolio_values = []
    invested_history = []
    daily_trades = [0] * total_days
    
    for i in range(total_days):
        if i > 0 and month_changed.iloc[i]:
            cash += monthly_inflow
            total_invested += monthly_inflow
            
        pos = position.iloc[i]
        prev_pos = position.iloc[i-1] if i > 0 else 0.0
        price = data['close'].iloc[i]
        
        if pos > 0 and prev_pos == 0 and cash > 0:
            cost_factor = 1 + fee + slip
            buy_price = price * cost_factor
            shares = cash / buy_price
            cash = 0.0
            daily_trades[i] = 1
        elif pos == 0 and prev_pos > 0 and shares > 0:
            cost_factor = 1 - fee - tax - slip
            sell_price = price * cost_factor
            cash = shares * sell_price
            shares = 0.0
            daily_trades[i] = 1
        elif pos > 0 and cash > 0:
            cost_factor = 1 + fee + slip
            buy_price = price * cost_factor
            bought_shares = cash / buy_price
            shares += bought_shares
            cash = 0.0
            
        current_val = cash + (shares * price)
        portfolio_values.append(current_val)
        invested_history.append(total_invested)
        
    data['portfolio_value'] = portfolio_values
    data['total_invested'] = invested_history
    data['equity_net'] = data['portfolio_value'] / data['total_invested']
    data['strategy_ret'] = data['equity_net'].pct_change().fillna(0)
    data['position'] = position
    data['trades'] = daily_trades
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
def eval_performance(df):
    total_days = len(df)
    if total_days < 2:
        return {}
        
    final_invested = df['total_invested'].iloc[-1]
    final_value = df['portfolio_value'].iloc[-1]
    net_profit = final_value - final_invested
    total_return = net_profit / final_invested if final_invested > 0 else 0.0
    
    cagr = ((final_value / final_invested) ** (252 / total_days) - 1) if final_invested > 0 else 0.0
    
    cum_max = df['equity_net'].cummax()
    dd = (df['equity_net'] - cum_max) / cum_max
    mdd = dd.min()
    
    if isinstance(df['trades'], pd.Series):
        trade_count = int(df['trades'].sum())
    else:
        trade_count = int(df['trades'])
        
    exposure_time = df['position'].mean()
    
    return {
        "累積總投入本金": f"${final_invested:,.0f}",
        "最終資產金額": f"${final_value:,.0f}",
        "總報酬率": f"{total_return:.2%}",
        "CAGR 年化報酬": f"{cagr:.2%}",
        "Max Drawdown (MDD)": f"{mdd:.2%}",
        "累積交易次數": f"{trade_count} 次",
        "平均持倉水位": f"{exposure_time:.1%}",
        "_raw_final": final_value,
        "_raw_invested": final_invested,
        "_raw_return": total_return,
        "_raw_trades": trade_count
    }

# --- 執行與顯示 ---
if st.sidebar.button("開始多策略回測與對比"):
    with st.spinner("正在進行多策略回測運算..."):
        data = load_data(symbol, market, start_date, end_date, auto_adjust=reinvest_dividends)
        if len(data) > 0:
            bh_res = run_buy_and_hold_with_inflow(data, initial_capital, monthly_inflow, fee_rate, slippage)
            bh_m = eval_performance(bh_res)
            
            fig = go.Figure()
            metrics_display_list = []
            strategy_raw_data = {}
            
            dividend_label = " (含息總報酬)" if reinvest_dividends else " (未還原價格)"
            
            metrics_display_list.append({
                "策略名稱": f"買入持有 (Buy & Hold){dividend_label}",
                "累積總投入本金": bh_m["累積總投入本金"],
                "最終資產金額": bh_m["最終資產金額"],
                "總報酬率": bh_m["總報酬率"],
                "CAGR 年化報酬": bh_m["CAGR 年化報酬"],
                "Max Drawdown (MDD)": bh_m["Max Drawdown (MDD)"],
                "累積交易次數": bh_m["累積交易次數"],
                "平均持倉水位": bh_m["平均持倉水位"]
            })
            
            fig.add_trace(go.Scatter(x=bh_res.index, y=bh_res['portfolio_value'], name=f"買入持有 (${bh_m['_raw_final']:,.0f})", line=dict(color='gray', dash='dash')))
            
            for strat in selected_strategies:
                if strat == "定期定額 (DCA)":
                    d_freq = params["DCA"]["freq"]
                    d_val = params["DCA"]["val"]
                    d_custom = params["DCA"]["custom_days"]
                    res = run_dca_backtest_advanced(data, initial_capital, monthly_inflow, d_freq, d_val, d_custom, fee_rate, slippage)
                elif strat == "跌破均線且創新低買入":
                    d_sma = params["DipDCA"]["sma_period"]
                    d_tf = params["DipDCA"]["timeframe"]
                    d_buy_amt = params["DipDCA"]["buy_amount"]
                    d_interval = params["DipDCA"]["min_interval"]
                    res = run_dip_dca_backtest(data, initial_capital, monthly_inflow, d_sma, d_tf, d_buy_amt, d_interval, fee_rate, slippage)
                else:
                    p_key = "PriceSMA" if "突破" in strat else ("SMA" if "SMA" in strat else ("RSI" if "RSI" in strat else ("MACD" if "MACD" in strat else ("BB" if "BB" in strat else ("KD" if "KD" in strat else "Donchian")))))
                    sig = generate_signal(data, strat, params.get(p_key, {}))
                    res = run_backtest(data, sig, fee_rate, tax_rate, slippage, initial_capital, monthly_inflow)
                
                m = eval_performance(res)
                strategy_raw_data[strat] = m
                
                metrics_display_list.append({
                    "策略名稱": strat,
                    "累積總投入本金": m["累積總投入本金"],
                    "最終資產金額": m["最終資產金額"],
                    "總報酬率": m["總報酬率"],
                    "CAGR 年化報酬": m["CAGR 年化報酬"],
                    "Max Drawdown (MDD)": m["Max Drawdown (MDD)"],
                    "累積交易次數": m["累積交易次數"],
                    "平均持倉水位": m["平均持倉水位"]
                })
                
                fig.add_trace(go.Scatter(x=res.index, y=res['portfolio_value'], name=f"{strat} (${m['_raw_final']:,.0f})", line=dict(width=2)))
            
            fig.update_layout(title=f"{symbol} 資產總額成長曲線 ($)", xaxis_title="日期", yaxis_title="資產總額 ($)", height=500)
            
            st.subheader("📊 各策略與 Buy & Hold 績效與金額對比表")
            metrics_df = pd.DataFrame(metrics_display_list).set_index("策略名稱")
            st.dataframe(metrics_df, use_container_width=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("💡 戰術診斷與現金流分析")
            diagnostic_text = f"期初本金 **${initial_capital:,.0f}** 搭配每月注資 **${monthly_inflow:,.0f}**，全期累計總投入本金達到 **{bh_m['累積總投入本金']}**：\n"
            diagnostic_text += f"* **買入持有 (Buy & Hold)** 最終變成了 **{bh_m['最終資產金額']}**（總報酬率 **{bh_m['總報酬率']}**）。\n\n"
            
            for strat, m in strategy_raw_data.items():
                diagnostic_text += f"#### 📌 【{strat}】診斷報告\n"
                diagnostic_text += f"* **最終資產**: **{m['最終資產金額']}**（總報酬率 **{m['總報酬率']}**）。\n"
                if strat == "跌破均線且創新低買入":
                    diagnostic_text += f"* **抄底機制**: 僅在「價格低於均線 ＋ 價格創出該次跌破區間新低」時觸發建倉，全期共精準抄底 **{m['_raw_trades']} 次**。\n"
                diagnostic_text += "\n"
            st.markdown(diagnostic_text)
        else:
            st.error("未能獲取數據，請確認標的與日期設定。")
