"""
分頁 2：多資產配置與定期再平衡模式 (Multi-Asset Allocation & Rebalancing)
架構設計：
- 直覺權重配置列：支援現金活存儲備 (依計價幣別切換預設年利率)、AI 一鍵權重推薦、自動配平至 100%
- 背景同步計算「不進行再平衡 (Buy & Drift)」基準線，精準量化再平衡對超額報酬與回撤的貢獻
- 四個清爽子分頁：
  1. 績效總覽與成長曲線 (健檢面板、KPI 卡片、再平衡 vs Buy & Drift 對照、省費統計、成長曲線 & 水下回撤)
  2. 利潤歸因與權重軌跡 (各資產貢獻表與橫條圖、動態權重漂移堆疊面積圖)
  3. 效率前緣與相關性 (MPT 效率前緣標示目前配置點、週報酬資產相關係數熱力圖、各單一資產年化對照)
  4. 月度報酬熱力圖 (投資組合年度 × 月度報酬矩陣)
"""
from typing import Dict, Any, Optional, Tuple
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.cost_model import CostModel, AssetCostStructure
from core.metrics import PerformanceMetrics
from core.portfolio_engine import BacktestEngine
from core.portfolio_optimizer import PortfolioOptimizer
from ui.theme import render_metric_grid, render_insight_panel
from ui.components_chart import (
    PLOTLY_CONFIG,
    _apply_terminal_layout,
    render_equity_comparison_chart,
    render_drawdown_chart,
    render_monthly_returns_heatmap,
    render_efficient_frontier,
    render_correlation_heatmap,
    render_weight_drift_chart,
)


@st.cache_data(show_spinner=False, ttl=1800)
def _cached_mpt_optimize(
    aligned_panel: pd.DataFrame,
    aligned_fx: pd.Series,
    target_currency: str,
):
    return PortfolioOptimizer.optimize_portfolio(
        aligned_panel=aligned_panel,
        aligned_fx=aligned_fx,
        target_currency=target_currency,
    )


def _generate_diagnosis_bullets(
    m: Dict[str, Any],
    m_drift: Dict[str, Any],
    rebal_stats: Dict[str, Any],
    cash_reserve_w: float,
    curr_symbol: str,
) -> Tuple[list, str]:
    cagr = float(m.get("_raw_cagr", 0.0))
    mdd = float(m.get("_raw_mdd", 0.0))
    sharpe = float(m.get("_raw_sharpe", 0.0))

    if cagr >= 0.15:
        p_eval = f"**強勁複利表現**：年化成長率達 **{cagr:+.2%}**（夏普比率 **{sharpe:.2f}**）。"
    elif cagr >= 0.08:
        p_eval = f"**穩健增長表現**：年化成長率 **{cagr:+.2%}**（夏普比率 **{sharpe:.2f}**），兼顧收益與波動平衡。"
    else:
        p_eval = f"**保守平緩走勢**：年化成長率 **{cagr:+.2%}**（夏普比率 **{sharpe:.2f}**），若欲強化收益可適度增加成長性資產。"

    if mdd > -0.15:
        r_eval = f"**極致抗震能力**：最大歷史回撤僅 **{mdd:.2%}**，最長套牢 **{m.get('最長套牢天數', '-')}**。"
    elif mdd > -0.30:
        r_eval = f"**健康波動範圍**：最大歷史回撤 **{mdd:.2%}**，最長套牢 **{m.get('最長套牢天數', '-')}**。"
    else:
        r_eval = f"**較大回撤提醒**：最大歷史回撤達 **{mdd:.2%}**，建議搭配債券或現金活存儲備降低震盪。"

    bullets = [p_eval, r_eval]

    if m_drift:
        diff_profit = float(m["_raw_net_profit"]) - float(m_drift["_raw_net_profit"])
        diff_mdd = (abs(float(m_drift["_raw_mdd"])) - abs(float(m["_raw_mdd"]))) * 100.0
        bullets.append(
            f"**再平衡效應對比 (vs 不調倉 Buy & Drift)**：淨獲利差異 **{curr_symbol} {diff_profit:+,.0f}**，"
            f"最大回撤改善 **{diff_mdd:+.2f}%**。"
        )

    if cash_reserve_w > 0:
        bullets.append(
            f"**現金活存儲備 ({cash_reserve_w:.0%})**：累計產生複利利息 **{curr_symbol} {rebal_stats.get('累積現金利息', 0.0):,.0f}**，並作為調倉乾柴穩定運作。"
        )

    tone = "success" if cagr >= 0.08 and mdd > -0.30 else ("warning" if mdd <= -0.30 else "info")
    return bullets, tone


def render_portfolio_rebalance_tab(
    aligned_panel: pd.DataFrame,
    curr_symbol: str = "NT$",
    initial_capital: float = 300000.0,
    monthly_inflow: float = 30000.0,
    cost_override: Optional[AssetCostStructure] = None,
    tw_fee_discount: float = 0.5,
    aligned_fx: Optional[pd.Series] = None,
    target_currency: str = "TWD",
    invest_day_one: bool = False,
):
    if aligned_panel is None or aligned_panel.empty or not isinstance(aligned_panel.columns, pd.MultiIndex):
        st.info("請在左側控制中心選擇至少兩個標的並載入市場數據。")
        return

    tickers = list(dict.fromkeys(aligned_panel.columns.get_level_values(0)))
    if len(tickers) < 2:
        st.warning("多資產配置模式需要至少 2 檔標的（例如：0050 + QQQ + 00679B）。")
        return

    if aligned_fx is None or aligned_fx.empty:
        aligned_fx = pd.Series(32.0, index=aligned_panel.index)
    else:
        aligned_fx = aligned_fx.reindex(aligned_panel.index).ffill().bfill().fillna(32.0)

    # 快取執行 MPT 最佳化
    w_max_sharpe, w_min_vol, w_rp, w_eq, frontier_df, opt_summary = _cached_mpt_optimize(
        aligned_panel, aligned_fx, target_currency
    )

    # 切換計價幣別時同步更新預設活存年利率
    default_rate = 1.8 if "TWD" in target_currency.upper() else 4.2
    if st.session_state.get("_port_last_curr") != target_currency:
        st.session_state["cash_rate_pct"] = default_rate
        st.session_state["_port_last_curr"] = target_currency
    if "cash_rate_pct" not in st.session_state:
        st.session_state["cash_rate_pct"] = default_rate

    with st.expander("配置比例與再平衡參數設定", expanded=True):
        c_cash1, c_cash2 = st.columns(2)
        with c_cash1:
            if "weight_input_CASH_RESERVE" not in st.session_state:
                st.session_state["weight_input_CASH_RESERVE"] = 0.0
            cash_weight_pct = st.number_input(
                "現金活存儲備 (%)",
                min_value=0.0,
                max_value=95.0,
                step=1.0,
                key="weight_input_CASH_RESERVE",
                help="設定保留於高利活存的防禦資金比例，跌市時用於再平衡逢低加碼。",
            )
        with c_cash2:
            cash_rate_pct = st.number_input(
                "活存預估年利率 (%)",
                min_value=0.0,
                max_value=15.0,
                step=0.1,
                key="cash_rate_pct",
                help="依真實日曆天數按日複利計息。",
            )

        c_freq, c_band = st.columns(2)
        with c_freq:
            rebal_freq = st.selectbox(
                "再平衡檢驗週期",
                options=[
                    "每季再平衡 (Quarterly)",
                    "每半年再平衡 (Semi-Annually)",
                    "每年再平衡 (Annually)",
                    "每月再平衡 (Monthly)",
                    "不進行再平衡 (Buy & Drift)",
                ],
                index=0,
            )
        with c_band:
            tolerance_pct = st.slider(
                "偏差容忍門檻 (±%)",
                min_value=0.0,
                max_value=15.0,
                value=5.0,
                step=1.0,
                help="僅當任一資產偏離目標權重超過此門檻時才調倉，避免無效手續費。",
            )

        # 一鍵套用權重按鈕列
        st.markdown("###### 一鍵套用配置策略")
        btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)
        scale_risky = max(0.0, (100.0 - cash_weight_pct) / 100.0)

        def _apply_weights(raw_w_dict: Dict[str, float]):
            scaled_w = {t: round(float(raw_w_dict.get(t, 0.0)) * scale_risky, 1) for t in tickers}
            diff = round((100.0 - cash_weight_pct) - sum(scaled_w.values()), 1)
            if tickers:
                scaled_w[tickers[0]] = round(scaled_w[tickers[0]] + diff, 1)
            for t in tickers:
                st.session_state[f"weight_input_{t}"] = float(scaled_w.get(t, 0.0))

        if btn_col1.button("最高夏普", width="stretch", help="性價比最高，追求最佳風險調整後報酬"):
            _apply_weights(w_max_sharpe)
            st.rerun()
        if btn_col2.button("最小波動", width="stretch", help="追求整體淨值波動度最低之防禦配置"):
            _apply_weights(w_min_vol)
            st.rerun()
        if btn_col3.button("風險平價", width="stretch", help="依波動度倒數加權，平衡各資產風險貢獻"):
            _apply_weights(w_rp)
            st.rerun()
        if btn_col4.button("等權重 1/N", width="stretch", help="平均分配至各風險標的"):
            _apply_weights(w_eq)
            st.rerun()
        if btn_col5.button("自動配平 100%", width="stretch", help="依目前輸入比例等比縮放至剛好 100%"):
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

        # 若標的組合改變或未初始化，給定等權重初始值
        eq_w = round((100.0 - cash_weight_pct) / len(tickers), 1)
        for t in tickers:
            if f"weight_input_{t}" not in st.session_state:
                st.session_state[f"weight_input_{t}"] = float(eq_w)

        raw_weights: Dict[str, float] = {}
        for start_i in range(0, len(tickers), 3):
            chunk = tickers[start_i : start_i + 3]
            cols = st.columns(len(chunk))
            for c_idx, t in enumerate(chunk):
                with cols[c_idx]:
                    raw_weights[t] = st.number_input(
                        f"{t} 權重 (%)",
                        min_value=0.0,
                        max_value=100.0,
                        step=1.0,
                        key=f"weight_input_{t}",
                    )

    total_weight = sum(raw_weights.values()) + cash_weight_pct
    diff_from_100 = round(100.0 - total_weight, 1)

    if abs(diff_from_100) > 0.1:
        st.warning(
            f"目前總配置權重為 **{total_weight:.1f}%**（差 **{diff_from_100:+.1f}%**），請調整至 100% 或點擊「自動配平 100%」。"
        )
        return

    target_weights = {t: raw_weights[t] / 100.0 for t in tickers}
    cash_reserve_w = cash_weight_pct / 100.0
    cash_rate = cash_rate_pct / 100.0
    tolerance_band = tolerance_pct / 100.0

    cost_dict = {
        t: (cost_override if cost_override else CostModel.get_default_costs(t, tw_fee_discount=tw_fee_discount))
        for t in tickers
    }

    with st.spinner("正在運算多資產配置軌跡、複利活存與基準對照組..."):
        res_df, attribution, rebal_stats = BacktestEngine.run_multi_asset_portfolio(
            aligned_panel=aligned_panel,
            aligned_fx=aligned_fx,
            target_weights=target_weights,
            initial_capital=initial_capital,
            monthly_inflow=monthly_inflow,
            rebalance_freq=rebal_freq,
            cost_dict=cost_dict,
            target_currency=target_currency,
            tolerance_band=tolerance_band,
            cash_reserve_weight=cash_reserve_w,
            cash_annual_rate=cash_rate,
            invest_day_one=invest_day_one,
        )
        m = PerformanceMetrics.evaluate(res_df, curr_symbol=curr_symbol, initial_capital=initial_capital)

        # 自動計算 Buy & Drift 對照組 (僅當目前非不調倉模式時)
        drift_df = None
        m_drift = {}
        if "Buy & Drift" not in rebal_freq:
            drift_df, _, _ = BacktestEngine.run_multi_asset_portfolio(
                aligned_panel=aligned_panel,
                aligned_fx=aligned_fx,
                target_weights=target_weights,
                initial_capital=initial_capital,
                monthly_inflow=monthly_inflow,
                rebalance_freq="不進行再平衡 (Buy & Drift)",
                cost_dict=cost_dict,
                target_currency=target_currency,
                tolerance_band=0.0,
                cash_reserve_weight=cash_reserve_w,
                cash_annual_rate=cash_rate,
                invest_day_one=invest_day_one,
            )
            m_drift = PerformanceMetrics.evaluate(drift_df, curr_symbol=curr_symbol, initial_capital=initial_capital)

    # 4 個清爽子分頁
    sub_ov, sub_attr, sub_mpt, sub_month = st.tabs(
        [
            "績效總覽與成長曲線",
            "利潤歸因與權重漂移",
            "效率前緣與相關性",
            "月度報酬熱力圖",
        ]
    )

    # --- 子分頁 1：績效總覽與成長曲線 ---
    with sub_ov:
        net_gain = float(m["_raw_net_profit"])
        render_metric_grid(
            [
                {
                    "title": "實賺獲利總額",
                    "value": f"{curr_symbol} {net_gain:+,.0f}",
                    "sub": f"累積投入 {m['累積投入本金']}",
                    "state": "positive" if net_gain >= 0 else "negative",
                },
                {
                    "title": "期末總資產價值",
                    "value": f"{curr_symbol} {m['_raw_final']:,.0f}",
                    "sub": f"含現金部位 {curr_symbol} {res_df['cash'].iloc[-1]:,.0f}",
                    "state": "neutral",
                },
                {
                    "title": "CAGR / XIRR 年化",
                    "value": f"{m['_raw_cagr']:+.2%}",
                    "sub": f"XIRR {m['XIRR 資金加權']} ｜ Sharpe {m['夏普比率 (Sharpe)']}",
                    "state": "positive" if m["_raw_cagr"] >= 0 else "negative",
                },
                {
                    "title": "最大回撤 (MDD) / Calmar",
                    "value": f"{m['_raw_mdd']:.2%}",
                    "sub": f"Calmar {m['卡瑪比率 (Calmar)']} ｜ 套牢 {m['最長套牢天數']}",
                    "state": "negative" if m["_raw_mdd"] < -0.2 else "neutral",
                },
            ]
        )

        bullets, tone = _generate_diagnosis_bullets(m, m_drift, rebal_stats, cash_reserve_w, curr_symbol)
        render_insight_panel("投資組合健康度總評", bullets, tone=tone)

        if tolerance_pct > 0 and "Buy & Drift" not in rebal_freq:
            st.caption(
                f"門檻保護調倉統計：實際調倉 **{rebal_stats['觸發調倉次數']} 次**，"
                f"略過微小偏離 **{rebal_stats['成功略過調倉次數']} 次**，"
                f"估計省下摩擦成本 **{curr_symbol} {rebal_stats['估計省下摩擦手續費']}**（累計實付交易成本 {curr_symbol} {rebal_stats.get('累計實付摩擦成本', 0.0):,.0f}）。"
            )

        c_hdr, c_log = st.columns([3, 1])
        with c_hdr:
            st.markdown("##### 總資產增長歷程（含再平衡 vs 不調倉對照）")
        with c_log:
            use_log = st.checkbox("對數座標 (Log Scale)", value=False, key="port_log_scale")

        curves = {"目前配置 (再平衡)": res_df}
        if drift_df is not None:
            curves["對照組：不調倉 (Buy & Drift)"] = drift_df
        render_equity_comparison_chart(
            curves,
            curr_symbol=curr_symbol,
            champion_name="目前配置 (再平衡)",
            log_scale=use_log,
        )

        st.markdown("##### 水下歷史回撤曲線")
        dd_map = {"目前配置 (再平衡)": PerformanceMetrics.calculate_drawdown_series(res_df)}
        if drift_df is not None:
            dd_map["對照組：不調倉 (Buy & Drift)"] = PerformanceMetrics.calculate_drawdown_series(drift_df)
        render_drawdown_chart(dd_map, champion_name="目前配置 (再平衡)", height=320)

        csv_port = res_df.to_csv(index=True, encoding="utf-8-sig")
        st.download_button(
            label="下載投資組合每日淨值與明細 (CSV)",
            data=csv_port,
            file_name="portfolio_daily_equity_history.csv",
            mime="text/csv",
            key="btn_download_portfolio",
        )

    # --- 子分頁 2：利潤歸因與權重漂移 ---
    with sub_attr:
        attrib_rows = []
        for t, attr in attribution.items():
            disp_t = t.replace("💵 ", "")
            if t in target_weights:
                w_disp = f"{target_weights[t]:.1%}"
            elif "現金" in t:
                w_disp = f"{cash_reserve_w:.1%}"
            else:
                w_disp = "-"

            attrib_rows.append(
                {
                    "標的代碼": disp_t,
                    "目標權重": w_disp,
                    "期末市值": f"{curr_symbol} {attr['final_value']:,.0f}",
                    "累計買入投入": f"{curr_symbol} {attr['cum_bought']:,.0f}",
                    "累計股利/利息": f"{curr_symbol} {attr['cum_dividend']:,.0f}",
                    "實賺淨獲利": f"{curr_symbol} {attr['net_profit']:+,.0f}",
                    "累積報酬率": f"{attr['roi']:+.2%}",
                    "_raw_profit": attr["net_profit"],
                    "_raw_roi": attr["roi"],
                }
            )

        df_attrib = pd.DataFrame(attrib_rows).sort_values(by="_raw_profit", ascending=False).reset_index(drop=True)

        c_table, c_chart = st.columns([1.25, 1.75])
        with c_table:
            st.markdown("##### 各資產利潤貢獻表")
            st.dataframe(
                df_attrib[["標的代碼", "目標權重", "實賺淨獲利", "累積報酬率", "累計股利/利息", "期末市值"]].set_index("標的代碼"),
                width="stretch",
            )

        with c_chart:
            st.markdown("##### 淨利潤貢獻橫條圖")
            colors = ["#15803D" if p >= 0 else "#B91C1C" for p in df_attrib["_raw_profit"]]
            fig_bar = go.Figure(
                go.Bar(
                    x=df_attrib["_raw_profit"],
                    y=df_attrib["標的代碼"],
                    orientation="h",
                    marker=dict(color=colors),
                    text=[f"{curr_symbol} {v:+,.0f} ({r:+.1%})" for v, r in zip(df_attrib["_raw_profit"], df_attrib["_raw_roi"])],
                    textposition="auto",
                )
            )
            _apply_terminal_layout(
                fig_bar,
                x_title=f"淨貢獻獲利 ({curr_symbol})",
                height=350,
                hovermode="closest",
                show_legend_bottom=False,
            )
            fig_bar.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_bar, width="stretch", config=PLOTLY_CONFIG)

        st.markdown("##### 實際資產權重漂移軌跡（虛線為實際觸發調倉日）")
        render_weight_drift_chart(
            res_df,
            tickers=tickers,
            rebalance_dates=rebal_stats.get("rebalance_dates", []),
            has_cash_reserve=(cash_reserve_w > 0),
        )

    # --- 子分頁 3：效率前緣與相關性 ---
    with sub_mpt:
        current_pt = PortfolioOptimizer.evaluate_custom_weights(opt_summary, target_weights)
        if not frontier_df.empty:
            st.markdown("##### 馬科維茨效率前緣 (Efficient Frontier — 週報酬與匯率調整後)")
            st.caption("顯示 6,000 組隨機組合、各單一資產位置，以及您目前配置的紅色標記位置。")
            render_efficient_frontier(frontier_df, opt_summary, current_point=current_pt)

        c_corr, c_stats = st.columns([1.15, 1])
        with c_corr:
            st.markdown("##### 資產間週報酬相關係數矩陣")
            st.caption("相關係數越低（偏藍），組合越能發揮分散風險與再平衡降波動的紅利。")
            render_correlation_heatmap(opt_summary.get("corr_matrix"))

        with c_stats:
            st.markdown("##### 各單一資產年化報酬與風險對照")
            asset_stats = opt_summary.get("asset_stats", {})
            if asset_stats:
                stat_rows = [
                    {
                        "標的": t,
                        "年化報酬": f"{s['return']:+.2%}",
                        "年化波動": f"{s['volatility']:.2%}",
                        "夏普比率": f"{s['sharpe']:.2f}",
                    }
                    for t, s in asset_stats.items()
                ]
                st.dataframe(pd.DataFrame(stat_rows).set_index("標的"), width="stretch")

    # --- 子分頁 4：月度報酬熱力圖 ---
    with sub_month:
        monthly_df = PerformanceMetrics.calculate_monthly_returns(res_df)
        render_monthly_returns_heatmap(monthly_df, title="多資產配置 — 各年度月報酬矩陣")

    with st.expander("多資產資金池與再平衡機制說明", expanded=False):
        st.markdown(
            """
            1. **現金活存儲備隔離保護**：您設定的現金活存比例會嚴格保留，每月固定新注資僅投入風險性資產目標比例，不會擅自挪用活存或把全部餘額買光；且活存依實際日曆天數複利計息。
            2. **跨市場休市順延掛單**：當美股開盤、台股休市時觸發調倉或注資，台股買賣單將安全掛單並於次個開盤日開盤自動補買完成，絕不產生休市日假價格成交。
            3. **效率前緣計價統一**：MPT 最佳化與相關係數矩陣皆先將各資產換算至您選擇的計價幣別並採用週報酬計算，徹底避免匯率缺漏與跨市場休市造成的假低波動。
            """
        )
