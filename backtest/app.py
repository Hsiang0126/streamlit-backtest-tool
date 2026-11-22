"""
量化回測與多資產配置終端 (Main Application Entry)
設計原則：
1. 簡潔分層側邊欄 + 一鍵情境範本 on_change 即時連動
2. 支援單標的策略 PK 與多資產配置再平衡雙核心模式
3. 完整支援台美股稅費、券商手續費折數、上櫃 .TWO 自動備援與無前瞻偏差對齊
"""
import sys
import os
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_engine import DataEngine
from core.cost_model import AssetCostStructure
import strategies.dca
import strategies.ma_cross
import strategies.oscillators
import strategies.advanced
from ui.theme import apply_dark_theme, render_context_banner
from ui.tab_strategy_pk import render_strategy_pk_tab
from ui.tab_portfolio_rebalance import render_portfolio_rebalance_tab

st.set_page_config(
    layout="wide",
    page_title="PRO 量化回測與多資產配置終端",
    page_icon="📈",
)
apply_dark_theme()

SCENARIO_OPTIONS = [
    "自由自訂模式",
    "小資定期定額 (0050 + QQQ)",
    "退休股債全天候 (0050 + VT + 00679B)",
    "積極科技成長 (0050 + 00631L + QQQ)",
]

ALL_PRESETS = []
for _group_name, _items in DataEngine.PRESET_OPTIONS.items():
    ALL_PRESETS.extend(_items)


def _find_preset(prefix: str) -> str:
    for item in ALL_PRESETS:
        if item.startswith(prefix + " "):
            return item
    return ""


def _init_sidebar_state():
    defaults = {
        "sb_scenario": SCENARIO_OPTIONS[0],
        "sb_presets": [x for x in [_find_preset("0050"), _find_preset("QQQ"), _find_preset("BTC-USD")] if x],
        "sb_custom": "",
        "sb_currency": "新台幣 (TWD)",
        "sb_init_cap": 300000.0,
        "sb_monthly": 30000.0,
        "sb_invest_day_one": True,
        "sb_timespan": "近 5 年",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _on_scenario_change():
    choice = st.session_state.get("sb_scenario", SCENARIO_OPTIONS[0])
    is_twd = "TWD" in st.session_state.get("sb_currency", "新台幣 (TWD)")
    fx = 1.0 if is_twd else (1.0 / 32.0)

    if "小資定期定額" in choice:
        st.session_state["sb_presets"] = [x for x in [_find_preset("0050"), _find_preset("QQQ")] if x]
        st.session_state["sb_custom"] = ""
        st.session_state["sb_init_cap"] = 0.0
        st.session_state["sb_monthly"] = round(10000.0 * fx, 0)
        st.session_state["sb_invest_day_one"] = True
        st.session_state["sb_timespan"] = "近 5 年"
    elif "退休股債全天候" in choice:
        st.session_state["sb_presets"] = [
            x for x in [_find_preset("0050"), _find_preset("VT"), _find_preset("00679B")] if x
        ]
        st.session_state["sb_custom"] = ""
        st.session_state["sb_init_cap"] = round(500000.0 * fx, 0)
        st.session_state["sb_monthly"] = round(20000.0 * fx, 0)
        st.session_state["sb_invest_day_one"] = True
        st.session_state["sb_timespan"] = "近 5 年"
        # 同步套用穩健現金預留權重
        st.session_state["weight_input_CASH_RESERVE"] = 10.0
    elif "積極科技成長" in choice:
        st.session_state["sb_presets"] = [
            x for x in [_find_preset("0050"), _find_preset("00631L"), _find_preset("QQQ")] if x
        ]
        st.session_state["sb_custom"] = ""
        st.session_state["sb_init_cap"] = round(100000.0 * fx, 0)
        st.session_state["sb_monthly"] = round(20000.0 * fx, 0)
        st.session_state["sb_invest_day_one"] = True
        st.session_state["sb_timespan"] = "近 3 年"


def _on_currency_change():
    prev_curr = st.session_state.get("_last_currency", "新台幣 (TWD)")
    curr = st.session_state.get("sb_currency", "新台幣 (TWD)")
    if prev_curr == curr:
        return
    factor = (1.0 / 32.0) if "USD" in curr else 32.0
    st.session_state["sb_init_cap"] = round(float(st.session_state.get("sb_init_cap", 0.0)) * factor, 0)
    st.session_state["sb_monthly"] = round(float(st.session_state.get("sb_monthly", 0.0)) * factor, 0)
    st.session_state["_last_currency"] = curr


_init_sidebar_state()
st.session_state.setdefault("_last_currency", st.session_state["sb_currency"])

# --- 側邊欄控制中心 ---
st.sidebar.title("控制中心")

st.sidebar.selectbox(
    "一鍵情境範本",
    options=SCENARIO_OPTIONS,
    key="sb_scenario",
    on_change=_on_scenario_change,
    help="切換範本會即時自動更新標的、資金設定與回測區間。",
)

st.sidebar.markdown("##### 1. 標的資產")
selected_presets = st.sidebar.multiselect(
    "預設熱門標的",
    options=ALL_PRESETS,
    key="sb_presets",
    help="支援台股上市/上櫃 (0050, 00631L, 00679B, 2330)、美股與加密貨幣。",
)

custom_input = st.sidebar.text_input(
    "補充代碼 (逗號分隔)",
    key="sb_custom",
    placeholder="例如：VTI, 00878, 2881A",
    help="支援中英文逗號或空格分隔，台股代碼自動判定上市 .TW / 上櫃 .TWO。",
)

preset_tickers = [p.split(" ")[0].strip().upper() for p in selected_presets]
clean_custom = custom_input.replace("，", ",").replace("、", ",").replace(" ", ",")
custom_tickers = [s.strip().upper() for s in clean_custom.split(",") if s.strip()]
target_symbols = list(dict.fromkeys(preset_tickers + custom_tickers))

st.sidebar.markdown("##### 2. 資金與幣別")
user_currency = st.sidebar.selectbox(
    "計價幣別",
    ["新台幣 (TWD)", "美元 (USD)"],
    key="sb_currency",
    on_change=_on_currency_change,
)
is_twd = "TWD" in user_currency

c_cap1, c_cap2 = st.sidebar.columns(2)
with c_cap1:
    initial_capital = st.number_input(
        "期初本金",
        min_value=0.0,
        step=10000.0 if is_twd else 500.0,
        key="sb_init_cap",
        help="第 1 天開盤即全額投入的單筆資金；純定期定額請填 0。",
    )
with c_cap2:
    monthly_inflow = st.number_input(
        "每月定投",
        min_value=0.0,
        step=5000.0 if is_twd else 200.0,
        key="sb_monthly",
        help="每月固定注入帳戶之新資金。",
    )

invest_day_one = False
if initial_capital == 0.0 and monthly_inflow > 0.0:
    invest_day_one = st.sidebar.checkbox(
        "首日即扣首筆定投",
        key="sb_invest_day_one",
        help="勾選後回測第 1 天即注入並買入第一筆定期定額；取消則自次月首個開市日起扣款。",
    )

st.sidebar.markdown("##### 3. 回測區間")
time_span = st.sidebar.selectbox(
    "回測時長",
    ["近 3 年", "近 5 年", "近 10 年", "自訂日期"],
    key="sb_timespan",
)
today = pd.to_datetime("today").normalize()

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

with st.sidebar.expander("4. 交易摩擦成本設定", expanded=False):
    auto_cost = st.checkbox("自動匹配各市場官方稅費 (推薦)", value=True)
    cost_override = None
    tw_fee_discount = 0.5
    if auto_cost:
        discount_label = st.selectbox(
            "台股券商手續費折數",
            ["2.8 折 (電子下單常見)", "3 折", "5 折 (系統預設)", "6 折", "無折扣 (10 折)"],
            index=2,
            help="台股公定手續費 0.1425% 乘以折數；債券 ETF 證交稅自動設為 0%、股票 ETF 0.1%、個股 0.3%。",
        )
        discount_map = {
            "2.8 折 (電子下單常見)": 0.28,
            "3 折": 0.30,
            "5 折 (系統預設)": 0.50,
            "6 折": 0.60,
            "無折扣 (10 折)": 1.00,
        }
        tw_fee_discount = discount_map.get(discount_label, 0.5)
    else:
        f_rate = st.number_input("單邊手續費率 (%)", value=0.071, min_value=0.0, step=0.01) / 100.0
        t_rate = st.number_input("賣出交易稅率 (%)", value=0.10, min_value=0.0, step=0.01) / 100.0
        s_rate = st.number_input("單邊滑價率 (%)", value=0.02, min_value=0.0, step=0.01) / 100.0
        cost_override = AssetCostStructure(fee_rate=f_rate, tax_rate=t_rate, slippage_rate=s_rate)


@st.cache_data(show_spinner=False, ttl=3600)
def load_market_data(symbols, s_date, e_date, curr):
    return DataEngine.load_aligned_market_data(symbols, s_date, e_date, target_currency=curr)


if st.sidebar.button("重新抓取最新行情", width="stretch"):
    load_market_data.clear()
    st.rerun()


# --- 主標題與橫幅 ---
st.title("⚡ 量化回測與多資產配置終端")
st.caption("跨市場日曆對齊 • 零前瞻偏差撮合 • 除權息還原指標 • 真實利潤歸因與風險評估")

if not target_symbols:
    st.info("👈 請於左側選擇至少一個交易標的，或直接切換上方的「一鍵情境範本」。")
    st.stop()

with st.spinner("正在載入市場歷史行情與進行跨市場匯率對齊..."):
    raw_res = load_market_data(
        tuple(target_symbols),
        start_date,
        end_date,
        user_currency,
    )
    individual_dict, aligned_panel, aligned_fx, curr_symbol, warnings = raw_res

for w in warnings:
    st.sidebar.warning(w)

if not individual_dict:
    st.error("未能成功載入任何標的數據，請確認標的代碼是否正確或稍後重試。")
    st.stop()

cap_summary = f"期初 {curr_symbol} {initial_capital:,.0f} + 每月 {curr_symbol} {monthly_inflow:,.0f}"
render_context_banner(
    tickers=list(individual_dict.keys()),
    start_str=start_date,
    end_str=end_date,
    capital_summary=cap_summary,
    currency_label="TWD 新台幣" if is_twd else "USD 美元",
)

tab_pk, tab_rebalance = st.tabs(
    [
        "⚔️ 單標的 × 多策略 PK",
        "💼 多資產配置 × 再平衡",
    ]
)

with tab_pk:
    render_strategy_pk_tab(
        individual_dict=individual_dict,
        curr_symbol=curr_symbol,
        initial_capital=initial_capital,
        monthly_inflow=monthly_inflow,
        cost_override=cost_override,
        tw_fee_discount=tw_fee_discount,
        fx_series=aligned_fx,
        target_currency=user_currency,
        invest_day_one=invest_day_one,
    )

with tab_rebalance:
    render_portfolio_rebalance_tab(
        aligned_panel=aligned_panel,
        curr_symbol=curr_symbol,
        initial_capital=initial_capital,
        monthly_inflow=monthly_inflow,
        cost_override=cost_override,
        tw_fee_discount=tw_fee_discount,
        aligned_fx=aligned_fx,
        target_currency=user_currency,
        invest_day_one=invest_day_one,
    )
