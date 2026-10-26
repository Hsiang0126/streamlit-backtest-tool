"""
量化回測與多資產配置終端 (Main Application Entry)
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

st.sidebar.markdown("### 1. 選擇標的資產")

# 經典資產配置主題包快捷選擇
PORTFOLIO_PRESETS = {
    "自選標的組合": [],
    "全天候防禦組合 (All Weather)": ["SPY", "TLT", "IEF", "GLD", "DBC"],
    "經典股債 60/40": ["0050", "TLT"],
    "科技巨頭動能": ["QQQ", "NVDA", "AAPL", "2330"]
}
selected_pack = st.sidebar.selectbox("一鍵套用經典配置組合包：", list(PORTFOLIO_PRESETS.keys()))

all_presets = []
for group_name, items in DataEngine.PRESET_OPTIONS.items():
    all_presets.extend(items)

default_selection = ["0050 (元大台灣50)", "QQQ (納斯達克100)", "BTC-USD (比特幣)"]
if PORTFOLIO_PRESETS[selected_pack]:
    default_selection = [p for p in all_presets if any(p.startswith(sym) for sym in PORTFOLIO_PRESETS[selected_pack])]

selected_presets = st.sidebar.multiselect(
    "熱門跨市場預設標的：",
    options=all_presets,
    default=default_selection
)

custom_input = st.sidebar.text_input(
    "補充自訂代碼 (逗號分隔，台美股、上櫃皆可)：",
    value=",".join([s for s in PORTFOLIO_PRESETS[selected_pack] if not any(p.startswith(s) for p in all_presets)]) if PORTFOLIO_PRESETS[selected_pack] else "",
    help="例如：AAPL, 2330, 3105, NVDA, ETH-USD"
)

preset_tickers = [p.split(" ")[0].strip().upper() for p in selected_presets]
custom_tickers = [s.strip().upper() for s in custom_input.split(",") if s.strip()]
target_symbols = list(dict.fromkeys(preset_tickers + custom_tickers))

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
st.caption("365天暖機指標預熱 • 智能上市日防穿梭對齊 • 雙價分離下單 • 真實獲利金額拆解")

if not target_symbols:
    st.info("👈 請於側邊欄選擇至少一個交易標的。")
    st.stop()

with st.spinner("正在下載行情、校驗上市日並計算跨市場對齊..."):
    individual_dict, aligned_panel, curr_symbol, effective_start, warnings = load_market_data(
        target_symbols,
        start_date,
        end_date,
        user_currency
    )

for w in warnings:
    if "⚠️" in w or "ℹ️" in w:
        st.sidebar.info(w)
    else:
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
        effective_start_date=effective_start,
        initial_capital=initial_capital,
        monthly_inflow=monthly_inflow,
        cost_override=cost_override
    )

with tab_rebalance:
    render_portfolio_rebalance_tab(
        aligned_panel=aligned_panel,
        curr_symbol=curr_symbol,
        effective_start_date=effective_start,
        initial_capital=initial_capital,
        monthly_inflow=monthly_inflow,
        cost_override=cost_override
    )
