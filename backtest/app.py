"""
量化回測與多資產配置終端 (Main Application Entry - 新手極簡引導與情境範本版)
包含：
- 🚀 30 秒新手一鍵情境範本 (小資定投、退休穩健全天候、積極成長槓桿型)
- 步驟化引導提示 (Step 1 選擇情境與資金 -> Step 2 查看診斷與曲線)
- 智慧解構與全形標點符號容錯
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

# --- 0. 🚀 快速上手：一鍵情境範本 ---
st.sidebar.markdown("### 0. 🚀 快速上手：一鍵情境範本")
scenario_choice = st.sidebar.selectbox(
    "選擇預設情境 (新手推薦直接選一項)：",
    options=[
        "自由自訂模式 (自選標的與資金)",
        "🚀 【小資佛系無腦存】台美大盤定期定額 (0050 + QQQ)",
        "🛡️ 【退休穩健全天候】股債現極致平衡 (0050 + VT + 00679B + 活存)",
        "⚡ 【積極成長衝刺型】科技雙核心 + 槓桿 (0050 + 00631L + QQQ)"
    ],
    index=0,
    help="選擇經典組合後，系統會自動幫您帶入標的、資金與配置，30 秒內即可看到專業回測！"
)

# 依情境自動判定預設值
default_presets = ["0050 (元大台灣50)", "QQQ (納斯達克100)", "BTC-USD (比特幣)"]
default_custom = ""
default_init_val = 300000.0
default_monthly_val = 30000.0
default_span_idx = 1 # 近 5 年

if "小資佛系" in scenario_choice:
    default_presets = ["0050 (元大台灣50)", "QQQ (納斯達克100)"]
    default_custom = ""
    default_init_val = 0.0
    default_monthly_val = 10000.0
    default_span_idx = 1
elif "退休穩健" in scenario_choice:
    default_presets = ["0050 (元大台灣50)"]
    default_custom = "VT, 00679B"
    default_init_val = 500000.0
    default_monthly_val = 20000.0
    default_span_idx = 1
elif "積極成長" in scenario_choice:
    default_presets = ["0050 (元大台灣50)", "00631L (元大台灣50正2)", "QQQ (納斯達克100)"]
    default_custom = ""
    default_init_val = 100000.0
    default_monthly_val = 20000.0
    default_span_idx = 0 # 近 3 年

# --- 1. 標的選擇 ---
st.sidebar.markdown("### 1. 選擇標的資產")
all_presets = []
for group_name, items in DataEngine.PRESET_OPTIONS.items():
    all_presets.extend(items)

valid_defaults = [p for p in default_presets if p in all_presets]
selected_presets = st.sidebar.multiselect(
    "熱門跨市場預設標的：",
    options=all_presets,
    default=valid_defaults,
    help="支援台股個股/ETF (如 0050, 00631L, 2330)、美股 (如 QQQ, NVDA) 及加密貨幣。"
)

custom_input = st.sidebar.text_input(
    "補充自訂代碼 (逗號分隔)：",
    value=default_custom,
    help="例如：AAPL, 2330, 00631L, 00679B, NVDA, ETH-USD (支援中英文逗號與空格自動清洗)"
)

preset_tickers = [p.split(" ")[0].strip().upper() for p in selected_presets]
clean_custom = custom_input.replace("，", ",").replace("、", ",").replace(" ", ",")
custom_tickers = [s.strip().upper() for s in clean_custom.split(",") if s.strip()]
target_symbols = list(dict.fromkeys(preset_tickers + custom_tickers))

# --- 2. 帳戶資金設定 ---
st.sidebar.markdown("### 2. 帳戶資金設定")
user_currency = st.sidebar.selectbox("計價幣別", ["新台幣 (TWD)", "美元 (USD)"], index=0)
is_twd = "TWD" in user_currency

init_val = default_init_val if is_twd else round(default_init_val / 30.0, 0)
monthly_val = default_monthly_val if is_twd else round(default_monthly_val / 30.0, 0)

initial_capital = st.sidebar.number_input(
    "期初投入本金",
    value=float(init_val),
    min_value=0.0,
    step=10000.0 if is_twd else 500.0,
    help="回測第 1 天立即全額買入建倉的單筆總資金。若只想純定期定額、期初不投大筆資金，請填 0。"
)
monthly_inflow = st.sidebar.number_input(
    "每月固定新注資 (定期定額)",
    value=float(monthly_val),
    min_value=0.0,
    step=5000.0 if is_twd else 200.0,
    help="每個月第一個開市日固定拿來扣款買入的新增資金。"
)

if initial_capital == 0.0 and monthly_inflow > 0:
    st.sidebar.caption("💡 **扣款時機提示**：期初本金設為 0 時，第 1 個月為開戶期，首筆定期定額將在**次月首個開市日**才扣款。若希望第 1 天就扣第一筆，請填入「期初投入本金」。")

# --- 3. 回測時間區間 ---
st.sidebar.markdown("### 3. 回測時間區間")
time_span = st.sidebar.selectbox("回測時長", ["近 3 年", "近 5 年", "近 10 年", "自訂日期"], index=default_span_idx)
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

st.sidebar.caption("💡 **時間結算提示**：所有獲利、資產與年化報酬嚴格結算至結束日收盤。若遇假日，自動對齊至前一真實開市日。")

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
st.caption("跨市場數據對齊 (Union Calendar) • 零前瞻偏差撮合 • 現金活存儲備計息 • 真實獲利金額與利潤歸因拆解")

if not target_symbols:
    st.info("👈 請於側邊欄選擇至少一個交易標的，或直接點選上方的「一鍵情境範本」。")
    st.stop()

with st.spinner("正在自市場載入歷史行情並進行跨市場對齊與換匯..."):
    raw_res = load_market_data(
        target_symbols,
        start_date,
        end_date,
        user_currency
    )

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
