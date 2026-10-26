"""分頁 2：多資產配置與定期再平衡模式 (Multi-Asset Allocation & Rebalancing)。"""
from typing import Dict, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.cost_model import CostModel, AssetCostStructure
from core.metrics import PerformanceMetrics
from core.portfolio_engine import BacktestEngine
from ui.theme import render_metric_card


def render_portfolio_rebalance_tab(
    aligned_panel: pd.DataFrame,
    curr_symbol: str,
    effective_start_date: str,
    initial_capital: float,
    monthly_inflow: float,
    cost_override: Optional[AssetCostStructure] = None
):
    st.header("💼 多資產權重配置與定期再平衡")
    st.caption(f"以統一幣別 ({curr_symbol}) 進行跨市場配置，回測區間自 {effective_start_date} 開始。")

    if aligned_panel.empty:
        st.info("👈 請在側邊欄選擇至少兩個標的。")
        return

    if isinstance(aligned_panel.columns, pd.MultiIndex):
        tickers = list(dict.fromkeys(aligned_panel.columns.get_level_values(0)))
    else:
        tickers = []

    if len(tickers) < 2:
        st.warning("⚠️ 多資產配置建議至少選擇 2 檔標的（例如：美股 QQQ + 台股 0050）。")

    st.subheader("1. 設定各資產配置目標比例")
    cols = st.columns(min(len(tickers), 5))
    raw_weights = {}

    eq_w = round(100.0 / len(tickers), 1)
    for idx, t in enumerate(tickers):
        with cols[idx % len(cols)]:
            raw_weights[t] = st.number_input(
                f"{t} 權重 (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(eq_w),
                step=5.0,
                key=f"weight_input_{t}"
            )

    total_weight = sum(raw_weights.values())
    if abs(total_weight - 100.0) > 0.1:
        st.error(f"⚠️ 目前設定權重總和為 **{total_weight:.1f}%**，請微調使其等於 100%。")
        return

    target_weights = {t: raw_weights[t] / 100.0 for t in tickers}

    c_freq, c_info = st.columns([1, 2])
    with c_freq:
        rebal_freq = st.selectbox(
            "定期再平衡頻率",
            options=["每季再平衡 (Quarterly)", "每半年再平衡 (Semi-Annually)", "每年再平衡 (Annually)", "每月再平衡 (Monthly)"],
            index=0
        )
    with c_info:
        st.info("💡 **再平衡機制**：週期日開盤將超配資產部分賣出獲利了結，加碼跌深落後資產，嚴格控制風險。")

    cost_dict = {}
    for t in tickers:
        cost_dict[t] = cost_override if cost_override else CostModel.get_default_costs(t)

    with st.spinner("正在計算多資產軌跡與利潤歸因..."):
        res_df, attribution = BacktestEngine.run_multi_asset_portfolio(
            aligned_panel,
            target_weights,
            initial_capital,
            monthly_inflow,
            rebalance_freq=rebal_freq,
            cost_dict=cost_dict
        )

        m = PerformanceMetrics.evaluate(res_df, curr_symbol=curr_symbol, initial_capital=initial_capital)

    st.subheader("📊 投資組合整體獲利總覽")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    net_gain = m["_raw_net_profit"]
    with kpi1:
        render_metric_card("實賺獲利總額", f"{curr_symbol} {net_gain:+,.0f}", "真金白銀淨收益", "positive" if net_gain >= 0 else "negative")
    with kpi2:
        render_metric_card("期末總資產價值", f"{curr_symbol} {m['_raw_final']:,.0f}", f"包含現金 {curr_symbol} {res_df['cash'].iloc[-1]:,.0f}", "neutral")
    with kpi3:
        render_metric_card("累積自付總本金", f"{curr_symbol} {m['_raw_invested']:,.0f}", f"期初投入 + 定期注資", "neutral")
    with kpi4:
        render_metric_card("年化複合成長 (CAGR)", f"{m['_raw_cagr']:+.2%}", f"最大歷史回撤 {m['_raw_mdd']:.2%}", "positive" if m['_raw_cagr'] > 0 else "negative")

    st.subheader("🎯 哪項資產最賺錢？各標的利潤貢獻拆解 (Profit Attribution)")
    attrib_rows = []
    for t, attr in attribution.items():
        attrib_rows.append({
            "標的代碼": t,
            "目標權重": f"{target_weights[t]:.1%}",
            "期末持股市值": f"{curr_symbol} {attr['final_value']:,.0f}",
            "累計投入本金": f"{curr_symbol} {attr['cum_bought']:,.0f}",
            "實賺淨獲利": f"{curr_symbol} {attr['net_profit']:+,.0f}",
            "累積報酬率": f"{attr['roi']:+.2%}",
            "_raw_profit": attr['net_profit'],
            "_raw_roi": attr['roi']
        })

    df_attrib = pd.DataFrame(attrib_rows).sort_values(by="_raw_profit", ascending=False).reset_index(drop=True)

    c_table, c_chart = st.columns([1.2, 1.8])
    with c_table:
        st.dataframe(
            df_attrib[["標的代碼", "目標權重", "實賺淨獲利", "累積報酬率", "期末持股市值"]].set_index("標的代碼"),
            use_container_width=True
        )

    with c_chart:
        colors = ["#3fb950" if p >= 0 else "#f85149" for p in df_attrib["_raw_profit"]]
        fig_bar = go.Figure(go.Bar(
            x=df_attrib["_raw_profit"],
            y=df_attrib["標的代碼"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{curr_symbol} {v:+,.0f} ({r:+.1%})" for v, r in zip(df_attrib["_raw_profit"], df_attrib["_raw_roi"])],
            textposition="auto"
        ))
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title=f"淨貢獻獲利 ({curr_symbol})", showgrid=True, gridcolor="#21262d"),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📈 總資產增長歷程 vs 累積投入本金")
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(
        x=res_df.index,
        y=res_df["portfolio_value"],
        name="投資組合總價值",
        line=dict(color="#58a6ff", width=2.5),
        hovertemplate="日期: %{x|%Y-%m-%d}<br>總資產: " + curr_symbol + " %{y:,.0f}<extra></extra>"
    ))
    fig_growth.add_trace(go.Scatter(
        x=res_df.index,
        y=res_df["total_invested"],
        name="累積掏出本金",
        line=dict(color="#8b949e", dash="dash", width=1.5),
        fill="tonexty",
        fillcolor="rgba(88, 166, 255, 0.08)",
        hovertemplate="日期: %{x|%Y-%m-%d}<br>投入本金: " + curr_symbol + " %{y:,.0f}<extra></extra>"
    ))
    fig_growth.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title=f"金額 ({curr_symbol})"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    st.subheader("🥞 各資產持股市值動態演變")
    fig_stack = go.Figure()
    for t in tickers:
        fig_stack.add_trace(go.Scatter(
            x=res_df.index,
            y=res_df[f"val_{t}"],
            name=t,
            stackgroup="one",
            hovertemplate=t + ": " + curr_symbol + " %{y:,.0f}<extra></extra>"
        ))
    fig_stack.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title=f"各標的市值 ({curr_symbol})"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_stack, use_container_width=True)
