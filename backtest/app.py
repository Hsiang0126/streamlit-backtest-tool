"""
量化回測與多資產配置終端 (Main Application Entry - 智慧解構防呆版)
徹底解決 ValueError (unpacking mismatch) 與參數不對齊問題。
"""
import sys
import os
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_engine import DataEngine
from core.cost_model import CostModel, AssetCostStructure
import strategies.dca
import strategies.ma_cross
import strategies.oscillators
from ui.theme import apply_dark_theme
from ui.tab_strategy_pk import render_strategy_pk_tab
from ui.tab_portfolio_rebalance import render_portfolio_rebalance_tab

st.set_page_config(
    layout="wide",
    page_title="PRO 量化回測與多資產配置終端",
    page_icon="📈"
)
apply_dark_theme()

st.sidebar.title("🎛️ 控制中心")

# --- 1. 標的選擇 ---
st.sidebar.markdown("### 1. 選擇標的資產")
all_presets = []
for group_name, items in DataEngine.PRESET_OPTIONS.items():
    all_presets.extend(items)

selected_presets = st.sidebar.multiselect(
    "熱門跨市場預設標的：",
    options=all_presets,
    default=["0050 (元大台灣50)", "QQQ (納斯達克100)", "BTC-USD (比特幣)"]
)

custom_input = st.sidebar.text_input(
    "補充自訂代碼 (逗號分隔)：",
    value="",
    help="例如：AAPL, 2330, NVDA, ETH-USD"
)

preset_tickers = [p.split(" ")[0].strip().upper() for p in selected_presets]
# 🛡️ 修復：自動清洗中文逗號、頓號與全形/半形空格
clean_custom = custom_input.replace("，", ",").replace("、", ",").replace(" ", ",")
custom_tickers = [s.strip().upper() for s in clean_custom.split(",") if s.strip()]
target_symbols = list(dict.fromkeys(preset_tickers + custom_tickers))

# --- 2. 帳戶資金設定 ---
st.sidebar.markdown("### 2. 帳戶資金設定")
user_currency = st.sidebar.selectbox("計價幣別", ["新台幣 (TWD)", "美元 (USD)"], index=0)
is_twd = "TWD" in user_currency
default_init = 300000.0 if is_twd else 10000.0
default_monthly = 30000.0 if is_twd else 1000.0

initial_capital = st.sidebar.number_input(
    "期初投入本金",
    value=default_init,
    min_value=0.0,
    step=10000.0 if is_twd else 500.0
)
monthly_inflow = st.sidebar.number_input(
    "每月固定新注資 (定期定額)",
    value=default_monthly,
    min_value=0.0,
    step=5000.0 if is_twd else 200.0
)

# --- 3. 回測時間區間 ---
st.sidebar.markdown("### 3. 回測時間區間")
time_span = st.sidebar.selectbox("回測時長", ["近 3 年", "近 5 年", "近 10 年", "自訂日期"], index=1)
today = pd.to_datetime("today")

if time_span == "近 3 年":
    start_date = (today - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
elif time_span == "近 5 年":
    start_date = (today - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
elif time_span == "近 10 年":
    start_date = (today - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
else:
    c_s, c_e = st.sidebar.columns(2)
    with c_s:
        start_date = st.date_input("開始", value=pd.to_datetime("2018-01-01")).strftime("%Y-%m-%d")
    with c_e:
        end_date = st.date_input("結束", value=today).strftime("%Y-%m-%d")

# --- 4. 交易摩擦成本 ---
st.sidebar.markdown("### 4. 交易手續費與摩擦成本")
auto_cost = st.sidebar.checkbox("依標的類別自動匹配官方稅費 (推薦)", value=True)
cost_override = None

if not auto_cost:
    f_rate = st.sidebar.number_input("單邊手續費率 (%)", value=0.071, step=0.01) / 100.0
    t_rate = st.sidebar.number_input("賣出交易稅率 (%)", value=0.10, step=0.01) / 100.0
    s_rate = st.sidebar.number_input("單邊滑價率 (%)", value=0.02, step=0.01) / 100.0
    cost_override = AssetCostStructure(fee_rate=f_rate, tax_rate=t_rate, slippage_rate=s_rate)


@st.cache_data(show_spinner=False, ttl=3600)
def load_market_data(symbols, s_date, e_date, curr):
    return DataEngine.load_aligned_market_data(symbols, s_date, e_date, target_currency=curr)

st.title("⚡ PRO 量化回測與多資產配置終端")
st.caption("跨市場數據對齊 (Union Calendar) • 零前瞻偏差撮合 • 真實獲利金額與利潤歸因拆解")

if not target_symbols:
    st.info("👈 請於側邊欄選擇至少一個交易標的。")
    st.stop()

with st.spinner("正在自市場載入歷史行情並進行跨市場對齊與換匯..."):
    raw_res = load_market_data(
        target_symbols,
        start_date,
        end_date,
        user_currency
    )

    # 智慧解構防呆
    if len(raw_res) == 5:
        individual_dict, aligned_panel, aligned_fx, curr_symbol, warnings = raw_res
    elif len(raw_res) == 4:
        individual_dict, aligned_panel, curr_symbol, warnings = raw_res
        aligned_fx = pd.Series(31.0, index=aligned_panel.index if not aligned_panel.empty else None)
    else:
        st.error("載入數據格式異常，請確認 DataEngine 模組。")
        st.stop()

for w in warnings:
    st.sidebar.warning(w)

if not individual_dict:
    st.error("未能成功載入任何標的數據，請確認標的代碼是否正確。")
    st.stop()

tab_pk, tab_rebalance = st.tabs([
    "⚔️ 模式 A：單標的 × 多策略 PK (找最強策略)",
    "💼 模式 B：多資產配置 × 定期再平衡 (看賺多少錢、誰最賺)"
])

with tab_pk:
    render_strategy_pk_tab(
        individual_dict=individual_dict,
        curr_symbol=curr_symbol,
        initial_capital=initial_capital,
        monthly_inflow=monthly_inflow,
        cost_override=cost_override,
        fx_series=aligned_fx,
        target_currency=user_currency
    )

with tab_rebalance:
    render_portfolio_rebalance_tab(
        aligned_panel=aligned_panel,
        curr_symbol=curr_symbol,
        initial_capital=initial_capital,
        monthly_inflow=monthly_inflow,
        cost_override=cost_override,
        aligned_fx=aligned_fx,
        target_currency=user_currency
    )
