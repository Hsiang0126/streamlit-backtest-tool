"""
分頁 2：多資產配置與定期再平衡模式 (Multi-Asset Allocation & Rebalancing - 防呆完全相容版)
徹底解決 TypeError 參數漏傳問題，支援門檻偏差再平衡、節省手續費統計與 CSV 匯出。
"""
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
    curr_symbol: str = "NT$",
    initial_capital: float = 300000.0,
    monthly_inflow: float = 30000.0,
    cost_override: Optional[AssetCostStructure] = None,
    aligned_fx: Optional[pd.Series] = None,
    target_currency: str = "TWD"
):
    st.header("💼 多資產權重配置與定期再平衡")
    st.caption("將資金分散投資於多個跨市場標的，透過紀律性的定期再平衡 (Rebalancing)，獲取資產配置紅利。")

    if aligned_panel is None or aligned_panel.empty:
        st.info("👈 請在側邊欄選擇至少兩個標的並加載市場數據。")
        return

    # 防呆提取可用標的
    if isinstance(aligned_panel.columns, pd.MultiIndex):
        tickers = list(dict.fromkeys(aligned_panel.columns.get_level_values(0)))
    else:
        tickers = []
    if len(tickers) < 2:
        st.warning("多資產配置模式建議至少選擇 2 檔標的（例如：美股 QQQ + 台股 0050）。")

    # 防呆補齊匯率
    if aligned_fx is None or aligned_fx.empty:
        aligned_fx = pd.Series(31.0, index=aligned_panel.index)
    else:
        aligned_fx = aligned_fx.reindex(aligned_panel.index, method="ffill").fillna(31.0)

    # 1. 配置目標權重與再平衡頻率
    st.subheader("1. 設定各資產配置目標比例")
    cols = st.columns(min(len(tickers), 4))
    raw_weights = {}

    eq_w = round(100.0 / len(tickers), 1)
    for idx, t in enumerate(tickers):
        with cols[idx % len(cols)]:
            raw_weights[t] = st.number_input(
                f"{t} 目標權重 (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(eq_w),
                step=5.0,
                key=f"weight_input_{t}"
            )

    total_weight = sum(raw_weights.values())
    if abs(total_weight - 100.0) > 0.1:
        st.error(f"⚠️ 目前設定權重總和為 **{total_weight:.1f}%**，請調整使其剛好等於 100%。")
        return

    target_weights = {t: raw_weights[t] / 100.0 for t in tickers}

    # 再平衡機制與省錢門檻設定
    st.subheader("2. 再平衡機制與省錢門檻設定")
    c_freq, c_band = st.columns([1, 1])
    with c_freq:
        rebal_freq = st.selectbox(
            "定期再平衡檢驗週期",
            options=["每季再平衡 (Quarterly)", "每半年再平衡 (Semi-Annually)", "每年再平衡 (Annually)", "每月再平衡 (Monthly)", "不進行再平衡 (Buy & Drift)"],
            index=0
        )
    with c_band:
        tolerance_pct = st.slider(
            "🛡️ 再平衡偏差容忍度 (Tolerance Band ±%)",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=1.0,
            help="若各資產實際偏離目標權重未超過此門檻，系統自動略過調倉，避免被頻繁扣除手續費與證交稅！設為 0% 則強制時間一到就調倉。"
        )
    tolerance_band = tolerance_pct / 100.0

    cost_dict = {}
    for t in tickers:
        cost_dict[t] = cost_override if cost_override else CostModel.get_default_costs(t)

    with st.spinner("正在計算投資組合資產軌跡、休市順延撮合與各標的利潤歸因..."):
        res_df, attribution, rebal_stats = BacktestEngine.run_multi_asset_portfolio(
            aligned_panel,
            aligned_fx,
            target_weights,
            initial_capital,
            monthly_inflow,
            rebalance_freq=rebal_freq,
            cost_dict=cost_dict,
            target_currency=target_currency,
            tolerance_band=tolerance_band
        )

        m = PerformanceMetrics.evaluate(res_df, curr_symbol=curr_symbol, initial_capital=initial_capital)

    # 3. 核心指標與省錢成果展示
    st.subheader("📊 投資組合整體獲利總覽")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    net_gain = m["_raw_net_profit"]
    with kpi1:
        render_metric_card("實賺獲利總額", f"{curr_symbol} {net_gain:+,.0f}", "真金白銀淨收益", "positive" if net_gain >= 0 else "negative")
    with kpi2:
        render_metric_card("期末總資產價值", f"{curr_symbol} {m['_raw_final']:,.0f}", f"包含現金 {curr_symbol} {res_df['cash'].iloc[-1]:,.0f}", "neutral")
    with kpi3:
        render_metric_card("累積自付總本金", f"{curr_symbol} {m['_raw_invested']:,.0f}", f"期初 {curr_symbol} {initial_capital:,.0f} + 定期注資", "neutral")
    with kpi4:
        render_metric_card("年化複合成長 (CAGR)", f"{m['_raw_cagr']:+.2%}", f"最大回撤 {m['_raw_mdd']:.2%}", "positive" if m['_raw_cagr'] > 0 else "negative")

    if tolerance_pct > 0:
        st.success(
            f"🛡️ **偏差門檻調倉省錢保護已啟動**：檢驗週期內實際執行調倉 **{rebal_stats['觸發調倉次數']} 次**，"
            f"成功略過微小偏差調倉 **{rebal_stats['成功略過調倉次數']} 次**，"
            f"估計為您少繳約 **{curr_symbol} {rebal_stats['估計省下摩擦手續費']} 元** 的摩擦手續費與證交稅！"
        )

    # 4. 利潤拆解圖
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

    # 5. 總資產成長 vs 累計投入本金
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

    # 下載投資組合資產軌跡 CSV
    csv_port = res_df.to_csv(index=True, encoding="utf-8-sig")
    st.download_button(
        label="📥 下載投資組合每日資產軌跡與明細 (CSV)",
        data=csv_port,
        file_name="portfolio_daily_equity_history.csv",
        mime="text/csv",
        key="btn_download_portfolio"
    )
