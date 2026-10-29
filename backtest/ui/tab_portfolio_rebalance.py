"""
分頁 2：多資產配置與定期再平衡模式 (Multi-Asset Allocation & Rebalancing - AI 智能健檢與權重自動平衡版)
包含：
- ⚖️ 一鍵等比補齊剩餘比例至 100% (Auto-Balance Weights)
- 🤖 AI 投資組合白話健檢總結 (獲利評價、抗跌風險、架構師建議)
- 💡 新手速查手冊：多資產配置、4 大 AI 權重推薦、現金活存儲備、門檻再平衡之白話意義
- 💵 方案 B 核心：虛擬現金活存儲備資產 (可自訂年化利率、每日複利計息、參與再平衡)
- 🧠 MPT 現代投資組合最佳化：一鍵帶入【最高夏普】、【最小波動】、【風險平價】、【等權重】
- 🛡️ 門檻偏差再平衡 (Tolerance Band Rebalancing)，自動略過微小調倉並統計節省手續費
- 🎯 各資產利潤拆解 (Profit Attribution - 包含現金儲備利息收益)
"""
from typing import Dict, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.cost_model import CostModel, AssetCostStructure
from core.metrics import PerformanceMetrics
from core.portfolio_engine import BacktestEngine
from core.portfolio_optimizer import PortfolioOptimizer
from ui.theme import render_metric_card
from ui.components_chart import render_efficient_frontier


def generate_portfolio_diagnosis(cagr: float, mdd: float, net_profit: float, rebal_stats: dict, cash_reserve_w: float, curr_symbol: str = "NT$"):
    """生成 AI 投資組合白話健檢評語。"""
    if cagr >= 0.20:
        profit_eval = f"🚀 **超凡爆發力 (年化複利 {cagr:+.1%})**：資產增長速度極快，顯著超越絕大多數傳統基金與大盤指數！"
    elif cagr >= 0.10:
        profit_eval = f"💪 **穩健強勁 (年化複利 {cagr:+.1%})**：超越多數定存與通貨膨脹，長期複利效益顯著，非常適合滾雪球累積財富。"
    elif cagr >= 0.04:
        profit_eval = f"🛡️ **保守保值 (年化複利 {cagr:+.1%})**：走勢平順溫和，主要著重於資產防禦與抗通膨。"
    else:
        profit_eval = f"⚠️ **成長偏緩 (年化複利 {cagr:+.1%})**：年化動能較低，建議適度提高成長型資產比例以增強獲利動能。"

    if mdd > -0.15:
        risk_eval = f"🌟 **極致抗震 (最大跌幅 {mdd:.1%})**：歷史回撤極小，暴跌時幾乎無感，宛如穩固堡壘，心理壓力極低！"
    elif mdd > -0.30:
        risk_eval = f"👌 **中度抗跌 (最大跌幅 {mdd:.1%})**：回撤在合理健康的波動範圍內，多數長期投資人皆能從容承受。"
    else:
        risk_eval = f"⚡ **劇烈心跳 (最大跌幅 {mdd:.1%})**：歷史上曾承受過顯著回調，潛在報酬雖高，但必須具備足夠耐心切勿在低點恐慌殺跌！"

    advice_points = []
    if rebal_stats.get("成功略過調倉次數", 0) > 0:
        advice_points.append(f"門檻保護已為您省下約 {curr_symbol} {rebal_stats.get('估計省下摩擦手續費', '0')} 元的摩擦磨損")
    if cash_reserve_w > 0:
        advice_points.append(f"{cash_reserve_w:.0%} 現金活存儲備成功在回檔時發揮乾柴 (Dry Powder) 與穩定生息效益")
    
    advice_text = "；".join(advice_points) if advice_points else "維持紀律定投與定期再平衡是發揮資產配置威力的核心基石。"
    return profit_eval, risk_eval, advice_text


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
    st.caption("將資金分散投資於多個跨市場標的與現金儲備，透過紀律性的定期再平衡 (Rebalancing)，獲取資產配置紅利。")

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
        return

    # 💡 新手白話教學手冊
    with st.expander("💡 第一次使用？點擊展開「3 分鐘多資產配置與資金池運作手冊」", expanded=False):
        st.markdown(
            """
            #### ⚠️ 關鍵細節：多資產資金池與扣款撮合運作說明
            1. **首期資金與定期注資時機**：
               * **期初本金**：會在回測**第 1 個開市日**，直接按設定比例分配買入各資產。
               * **每月固定新注資**：會在**每個月第 1 個開市日**注入。若「期初本金 = 0」，第 1 個月為開戶期，首筆注資會於**次月首日**開始分配投入。
            2. **領到的現金股利去哪裡了？**：
               * 標的除息時，股利會換算為目標幣別存入「現金池」。
               * **股利不會只買回原本標的**，而是在次月定投扣款或再平衡日時，與新本金合併**依照你的目標權重重新分配買入所有資產**！
            3. **期末總資產中的「剩餘現金」是什麼？**：
               * 主要是回測結束前剛入帳、還沒到下個月扣款日的尾期股息，以及扣除手續費後的零星找零（不足 10 元不重複下單）。
            4. **跨市場休市調倉如何處理？**：
               * 美股開市、台股休市時，系統會先保留掛單，次日台股開盤賣出資金到帳後自動補齊買單，絕不產生資金斷鏈死鎖！

            ---
            #### ⚖️ 什麼是「定期再平衡 (Rebalancing)」？
            假設你的目標是「50% 0050 + 50% QQQ」。一年後因為美股大漲，比例變成了「40% 0050 + 60% QQQ」。
            * **此時執行再平衡**：系統會自動**賣出漲多的 QQQ（高檔停利），並買入落後的 0050（低檔補倉）**，強制讓比例回到 50% : 50%。
            * **核心好處**：克服人性追高殺跌的恐懼，自動落實**「高賣低買」**，在控制風險的同時賺取資產輪動的超額紅利！
            """
        )

    # 防呆補齊匯率
    if aligned_fx is None or aligned_fx.empty:
        aligned_fx = pd.Series(31.0, index=aligned_panel.index)
    else:
        aligned_fx = aligned_fx.reindex(aligned_panel.index, method="ffill").fillna(31.0)

    # MPT 智慧權重推薦模組
    w_max_sharpe, w_min_vol, w_rp, w_eq, frontier_df, opt_summary = PortfolioOptimizer.optimize_portfolio(aligned_panel)

    st.subheader("1. 設定各資產配置目標比例")

    # 💵 現金活存儲備控制區塊
    st.markdown("##### 💵 資本市場無風險資產配置 (現金活存儲備)：")
    c_cash1, c_cash2 = st.columns([1, 1])
    with c_cash1:
        cash_weight_pct = st.number_input(
            "💵 現金活存目標權重 (%)",
            min_value=0.0,
            max_value=95.0,
            value=float(st.session_state.get("weight_input_CASH_RESERVE", 0.0)),
            step=1.0,
            key="weight_input_CASH_RESERVE",
            help="設定保留在銀行或券商的活存儲備比例。在熊市大跌時作為抄底乾柴 (Dry Powder)，平時享有年化活存利息。"
        )
    with c_cash2:
        default_rate = 1.8 if "TWD" in target_currency.upper() else 4.0
        cash_rate_pct = st.number_input(
            f"📈 現金活存預估年利率 (%) [{target_currency}]",
            min_value=0.0,
            max_value=15.0,
            value=float(st.session_state.get("cash_rate_pct", default_rate)),
            step=0.1,
            key="cash_rate_pct",
            help="現金儲備部位每日按此年化利率複利計息 (TWD 建議 1.5%~2.0%，USD 建議 4.0%~5.0%)。"
        )

    st.markdown("##### 🧠 快捷比例助手 (點擊一鍵自動填入)：")
    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)

    scale_risky = max(0.0, (100.0 - cash_weight_pct) / 100.0)

    def apply_ai_weights(raw_w_dict):
        scaled_w = {}
        for t in tickers:
            scaled_w[t] = round(float(raw_w_dict.get(t, 0.0)) * scale_risky, 1)
        tot_scaled = sum(scaled_w.values())
        diff = round((100.0 - cash_weight_pct) - tot_scaled, 1)
        if tickers:
            scaled_w[tickers[0]] = round(scaled_w[tickers[0]] + diff, 1)
        for t in tickers:
            st.session_state[f"weight_input_{t}"] = float(scaled_w.get(t, 0.0))

    if btn_col1.button("🎯【最高夏普】", help="性價比最高，以最少波動換取最大報酬"):
        apply_ai_weights(w_max_sharpe)
        st.rerun()

    if btn_col2.button("🛡️【最小波動】", help="走勢最平穩、最抗跌的極致防禦組合"):
        apply_ai_weights(w_min_vol)
        st.rerun()

    if btn_col3.button("⚖️【風險平價】", help="高波動標的權重變小，各資產風險貢獻均等"):
        apply_ai_weights(w_rp)
        st.rerun()

    if btn_col4.button("🔄【等權重 1/N】", help="所有標的平均分配"):
        apply_ai_weights(w_eq)
        st.rerun()

    # ⚖️ 一鍵等比補齊權重功能 (徹底消滅湊不齊 100% 的挫折感)
    if btn_col5.button("✨【自動補齊 100%】", help="一鍵將目前各標的依比例放大或縮小，剛好湊齊 100%！"):
        cur_w_map = {t: float(st.session_state.get(f"weight_input_{t}", 0.0)) for t in tickers}
        cur_sum = sum(cur_w_map.values())
        target_risky = max(0.0, 100.0 - cash_weight_pct)
        if cur_sum > 0:
            scale = target_risky / cur_sum
            new_w = {t: round(cur_w_map[t] * scale, 1) for t in tickers}
        else:
            eq = round(target_risky / len(tickers), 1)
            new_w = {t: eq for t in tickers}
        diff = round(target_risky - sum(new_w.values()), 1)
        if tickers:
            new_w[tickers[0]] = round(new_w[tickers[0]] + diff, 1)
        for t in tickers:
            st.session_state[f"weight_input_{t}"] = float(new_w[t])
        st.rerun()

    # 預設各標的初始權重
    eq_w = round((100.0 - cash_weight_pct) / len(tickers), 1)
    for t in tickers:
        if f"weight_input_{t}" not in st.session_state:
            st.session_state[f"weight_input_{t}"] = float(eq_w)

    cols = st.columns(min(len(tickers), 4))
    raw_weights = {}
    for idx, t in enumerate(tickers):
        with cols[idx % len(cols)]:
            raw_weights[t] = st.number_input(
                f"{t} 目標權重 (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key=f"weight_input_{t}"
            )

    total_weight = sum(raw_weights.values()) + cash_weight_pct
    diff_from_100 = round(100.0 - total_weight, 1)

    # 動態權重檢核提示條
    if abs(diff_from_100) > 0.1:
        st.warning(
            f"⚠️ 目前設定權重總和為 **{total_weight:.1f}%**"
            f"（風險資產合計 {sum(raw_weights.values()):.1f}% + 現金活存 {cash_weight_pct:.1f}%，差 **{diff_from_100:+.1f}%**）。"
            f"請點擊上方【✨ 自動補齊 100%】按鈕一鍵配平！"
        )
        return
    else:
        st.caption(f"✅ **權重配置完美平衡**：各標的合計 {sum(raw_weights.values()):.1f}% + 現金活存 {cash_weight_pct:.1f}% = 100.0%")

    target_weights = {t: raw_weights[t] / 100.0 for t in tickers}
    cash_reserve_w = cash_weight_pct / 100.0
    cash_rate = cash_rate_pct / 100.0

    if not frontier_df.empty:
        with st.expander("🌈 展開查看「馬科維茨效率前緣」彩虹散佈圖 (Efficient Frontier)", expanded=False):
            st.caption("展示 5,000 組隨機投資組合之性價比分佈。金色星星 ⭐ 代表最高夏普點，藍色菱形 🛡️ 代表最小波動點。")
            render_efficient_frontier(frontier_df, opt_summary)

    st.subheader("2. 再平衡機制與省錢門檻設定")
    c_freq, c_band = st.columns([1, 1])
    with c_freq:
        rebal_freq = st.selectbox(
            "定期再平衡檢驗週期",
            options=["每季再平衡 (Quarterly)", "每半年再平衡 (Semi-Annually)", "每年再平衡 (Annually)", "每月再平衡 (Monthly)", "不進行再平衡 (Buy & Drift)"],
            index=0,
            help="設定多久打開帳本檢查一次權重偏離程度。季再平衡 (Quarterly) 或半年再平衡是學術界公認兼顧效益與手續費的最佳平衡點。"
        )
    with c_band:
        tolerance_pct = st.slider(
            "🛡️ 再平衡偏差容忍度 (Tolerance Band ±%)",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=1.0,
            help="若各資產（含現金活存儲備）偏離目標權重未超過此門檻，系統自動略過調倉，避免被頻繁扣除手續費與證交稅！設為 0% 則強制時間一到就調倉。"
        )
    tolerance_band = tolerance_pct / 100.0

    cost_dict = {}
    for t in tickers:
        cost_dict[t] = cost_override if cost_override else CostModel.get_default_costs(t)

    with st.spinner("正在計算投資組合資產軌跡、現金活存複利計息、休市順延撮合與各標的利潤歸因..."):
        res_df, attribution, rebal_stats = BacktestEngine.run_multi_asset_portfolio(
            aligned_panel,
            aligned_fx,
            target_weights,
            initial_capital,
            monthly_inflow,
            rebalance_freq=rebal_freq,
            cost_dict=cost_dict,
            target_currency=target_currency,
            tolerance_band=tolerance_band,
            cash_reserve_weight=cash_reserve_w,
            cash_annual_rate=cash_rate
        )

        m = PerformanceMetrics.evaluate(res_df, curr_symbol=curr_symbol, initial_capital=initial_capital)

    # 3. 核心指標與 AI 白話健檢
    st.subheader("📊 投資組合整體獲利總覽")

    # 🤖 AI 白話健檢卡片
    p_badge, r_badge, adv_badge = generate_portfolio_diagnosis(
        m["_raw_cagr"], m["_raw_mdd"], m["_raw_net_profit"], rebal_stats, cash_reserve_w, curr_symbol=curr_symbol
    )
    st.info(
        f"**🤖 AI 架構師健檢白話總評：**\n"
        f"* {p_badge}\n"
        f"* {r_badge}\n"
        f"* 💡 **操作建議**：{adv_badge}"
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    net_gain = m["_raw_net_profit"]
    with kpi1:
        render_metric_card("實賺獲利總額", f"{curr_symbol} {net_gain:+,.0f}", "真金白銀淨收益 (已扣稅費)", "positive" if net_gain >= 0 else "negative")
    with kpi2:
        render_metric_card("期末總資產價值", f"{curr_symbol} {m['_raw_final']:,.0f}", f"包含現金活存儲備 {curr_symbol} {res_df['cash'].iloc[-1]:,.0f}", "neutral")
    with kpi3:
        render_metric_card("累積自付總本金", f"{curr_symbol} {m['_raw_invested']:,.0f}", f"期初 {curr_symbol} {initial_capital:,.0f} + 定期注資", "neutral")
    with kpi4:
        render_metric_card("年化複合成長 (CAGR)", f"{m['_raw_cagr']:+.2%}", f"最大回撤 (MDD) {m['_raw_mdd']:.2%}", "positive" if m['_raw_cagr'] > 0 else "negative")

    if tolerance_pct > 0:
        st.success(
            f"🛡️ **偏差門檻調倉省錢保護已啟動**：檢驗週期內實際執行調倉 **{rebal_stats['觸發調倉次數']} 次**，"
            f"成功略過微小偏差調倉 **{rebal_stats['成功略過調倉次數']} 次**，"
            f"估計為您少繳約 **{curr_symbol} {rebal_stats['估計省下摩擦手續費']} 元** 的摩擦手續費與證交稅！"
        )

    # 4. 利潤拆解圖
    st.subheader("🎯 哪項資產最賺錢？各標的利潤貢獻拆解 (Profit Attribution)")
    st.caption("詳細拆解投資組合中每一檔標的（含現金活存儲備利息）的真實盈虧貢獻與報酬率。")
    attrib_rows = []
    for t, attr in attribution.items():
        if t in target_weights:
            w_disp = f"{target_weights[t]:.1%}"
        elif "現金" in t:
            w_disp = f"{cash_reserve_w:.1%}"
        else:
            w_disp = "-"

        attrib_rows.append({
            "標的代碼": t,
            "目標權重": w_disp,
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

    csv_port = res_df.to_csv(index=True, encoding="utf-8-sig")
    st.download_button(
        label="📥 下載投資組合每日資產軌跡與明細 (CSV)",
        data=csv_port,
        file_name="portfolio_daily_equity_history.csv",
        mime="text/csv",
        key="btn_download_portfolio"
    )
