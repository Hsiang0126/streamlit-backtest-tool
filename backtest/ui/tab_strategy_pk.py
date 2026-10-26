"""
分頁 1：單標的 × 多策略 PK 模式 (Single Asset Multi-Strategy PK - 頂級增強版)
包含：
- 完整支援 DCA、Dip DCA 與擇時策略之逐筆成交/購買歷史 (Purchase Ledger) ＋ 📥 下載 CSV
- K 線進出場標記圖 (TradingView 質感 🟢買入 / 🔴賣出箭頭)
- 功能 C：快慢線 2D 尋優熱力圖 (僅在選取 SMA 雙均線策略時智慧顯示，其餘隱藏)
"""
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.cost_model import CostModel, AssetCostStructure
from core.metrics import PerformanceMetrics
from core.portfolio_engine import BacktestEngine
from core.trade_ledger import TradeLedger
from core.grid_optimizer import GridOptimizer
from strategies.registry import StrategyRegistry
from ui.theme import render_metric_card
from ui.components_chart import render_candlestick_with_markers, render_parameter_heatmap


def render_strategy_pk_tab(
    individual_dict: Dict[str, pd.DataFrame],
    curr_symbol: str = "NT$",
    initial_capital: float = 300000.0,
    monthly_inflow: float = 30000.0,
    cost_override: Optional[AssetCostStructure] = None,
    fx_series: Optional[pd.Series] = None,
    target_currency: str = "TWD"
):
    st.header("⚔️ 單標的 × 多策略獲利 PK 模式")
    st.caption("評估同一檔資產在不同交易策略下的獲利金額與防禦能力，迅速找出最強量化策略。")

    if not individual_dict:
        st.info("👈 請在左側側邊欄選擇或輸入標的代碼，並點擊「開始加載市場數據」。")
        return

    target_ticker = st.selectbox("選擇要進行多策略 PK 的標的：", list(individual_dict.keys()))
    df_target = individual_dict[target_ticker]

    all_strategies = StrategyRegistry.list_names()
    default_strats = [
        "定期定額 / 買入持有 (DCA / Buy & Hold)",
        "價格突破均線買入/跌破賣出",
        "SMA 雙均線交叉 (黃金交叉)"
    ]
    valid_defaults = [s for s in default_strats if s in all_strategies]

    selected_strats = st.multiselect(
        "勾選要進行對決的策略 (可多選)：",
        options=all_strategies,
        default=valid_defaults if valid_defaults else all_strategies[:2]
    )

    if not selected_strats:
        st.warning("請至少勾選一種策略進行評估。")
        return

    with st.expander("⚙️ 展開設定選取策略之參數 (含日線/週線切換)", expanded=False):
        strategy_params_dict = {}
        for strat_name in selected_strats:
            strat_cls = StrategyRegistry.get(strat_name)
            specs = strat_cls.get_param_specs()
            st.markdown(f"**[{strat_name}]** - {strat_cls.description}")
            params = {}
            if specs:
                cols = st.columns(min(len(specs), 3))
                for idx, spec in enumerate(specs):
                    col = cols[idx % len(cols)]
                    with col:
                        key_id = f"{target_ticker}_{strat_cls.strategy_id}_{spec.key}"
                        if spec.param_type == "int":
                            params[spec.key] = st.number_input(
                                spec.label,
                                value=int(spec.default),
                                min_value=int(spec.min_value) if spec.min_value is not None else 1,
                                max_value=int(spec.max_value) if spec.max_value is not None else 500,
                                step=int(spec.step) if spec.step is not None else 1,
                                key=key_id,
                                help=spec.help_text
                            )
                        elif spec.param_type == "float":
                            params[spec.key] = st.number_input(
                                spec.label,
                                value=float(spec.default),
                                min_value=float(spec.min_value) if spec.min_value is not None else 0.0,
                                max_value=float(spec.max_value) if spec.max_value is not None else 1000000.0,
                                step=float(spec.step) if spec.step is not None else 1.0,
                                key=key_id,
                                help=spec.help_text
                            )
                        elif spec.param_type == "select":
                            params[spec.key] = st.selectbox(
                                spec.label,
                                options=spec.options or [],
                                index=spec.options.index(spec.default) if spec.options and spec.default in spec.options else 0,
                                key=key_id,
                                help=spec.help_text
                            )
                        elif spec.param_type == "bool":
                            params[spec.key] = st.checkbox(
                                spec.label,
                                value=bool(spec.default),
                                key=key_id,
                                help=spec.help_text
                            )
            strategy_params_dict[strat_name] = params
            st.markdown("---")

    cost_struct = cost_override if cost_override else CostModel.get_default_costs(target_ticker)

    results_summary = []
    equity_curves = {}
    strategy_pos_dict = {}

    for strat_name in selected_strats:
        p = strategy_params_dict.get(strat_name, {})
        try:
            res_df = BacktestEngine.run_single_strategy(
                df_target,
                strat_name,
                p,
                initial_capital,
                monthly_inflow,
                cost_struct,
                fx_series=fx_series,
                target_currency=target_currency
            )
            m = PerformanceMetrics.evaluate(res_df, curr_symbol=curr_symbol, initial_capital=initial_capital)
            m["策略名稱"] = strat_name
            results_summary.append(m)
            equity_curves[strat_name] = res_df

            strat_cls = StrategyRegistry.get(strat_name)
            if strat_cls.is_timing:
                sig = strat_cls.generate_signals(df_target, p)
                strategy_pos_dict[strat_name] = sig.shift(1).fillna(0).astype(int)
        except Exception as exc:
            st.error(f"策略【{strat_name}】執行失敗：{exc}")

    if not results_summary:
        st.error("未能產出任何策略的有效績效，請檢查數據或參數。")
        return

    df_results = pd.DataFrame(results_summary)
    df_results = df_results.sort_values(by="_raw_net_profit", ascending=False).reset_index(drop=True)

    champion_name = df_results.iloc[0]["策略名稱"]
    champion_profit = df_results.iloc[0]["_raw_net_profit"]
    champion_final = df_results.iloc[0]["_raw_final"]
    champion_mdd = df_results.iloc[0]["_raw_mdd"]

    st.subheader(f"🏆 最賺錢策略冠軍：【{champion_name}】")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("冠軍實賺獲利金額", f"{curr_symbol} {champion_profit:+,.0f}", "實收淨利潤", "positive" if champion_profit >= 0 else "negative")
    with c2:
        render_metric_card("冠軍最終總資產", f"{curr_symbol} {champion_final:,.0f}", "帳戶期末總值", "neutral")
    with c3:
        render_metric_card("冠軍總報酬率", f"{df_results.iloc[0]['_raw_roi']:+.2%}", "基於累積自付本金", "positive" if champion_profit >= 0 else "negative")
    with c4:
        render_metric_card("冠軍最大歷史回撤", f"{champion_mdd:.2%}", "策略承受的最大跌幅", "negative" if champion_mdd < -0.2 else "neutral")

    st.subheader("📋 策略對抗排列表")
    display_cols = [
        "策略名稱", "實賺獲利金額", "最終總資產", "累積投入本金", 
        "淨獲利報酬率", "TWR 總報酬率", "CAGR 年化報酬", "最大歷史回撤 (MDD)", "累積交易次數"
    ]
    display_df = df_results[display_cols].copy()
    display_df.loc[0, "策略名稱"] = f"👑 {champion_name} (冠軍)"
    st.dataframe(display_df.set_index("策略名稱"), use_container_width=True)

    st.subheader("📈 策略資產成長曲線對比 ($)")
    fig = go.Figure()
    for strat_name in selected_strats:
        if strat_name in equity_curves:
            curve = equity_curves[strat_name]
            is_champ = (strat_name == champion_name)
            fig.add_trace(go.Scatter(
                x=curve.index,
                y=curve["portfolio_value"],
                name=f"{'👑 ' if is_champ else ''}{strat_name}",
                line=dict(width=3 if is_champ else 1.5),
                hovertemplate="日期: %{x|%Y-%m-%d}<br>總資產: " + curr_symbol + " %{y:,.0f}<extra></extra>"
            ))

    first_curve = list(equity_curves.values())[0]
    fig.add_trace(go.Scatter(
        x=first_curve.index,
        y=first_curve["total_invested"],
        name="累計自付本金",
        line=dict(color="#8b949e", dash="dot", width=1.5),
        hovertemplate="日期: %{x|%Y-%m-%d}<br>投入本金: " + curr_symbol + " %{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title=f"帳戶總資產價值 ({curr_symbol})"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------
    # 逐筆成交 / 扣款購買歷史帳本 (全面支援 DCA 與所有策略)
    # -------------------------------------------------------------
    st.subheader("📖 策略成交明細帳本 (Trade / Purchase Log)")
    inspect_strat = st.selectbox("選擇要檢視詳細交易/購買紀錄的策略：", selected_strats, key="select_inspect_strat")

    if inspect_strat in equity_curves:
        res_curve = equity_curves[inspect_strat]

        # DCA 或 Dip DCA：展示逐筆扣款購買歷史表
        if inspect_strat not in strategy_pos_dict:
            buys_df, buy_summary = TradeLedger.extract_buy_records(res_curve, curr_symbol)
            if not buys_df.empty:
                b_c1, b_c2, b_c3, b_c4 = st.columns(4)
                with b_c1:
                    render_metric_card("累積扣款次數", buy_summary["累積購買次數"], "實際成功扣款筆數", "neutral")
                with b_c2:
                    render_metric_card("累積持有股數", buy_summary["累積持有股數"], "截至期末總持股", "neutral")
                with b_c3:
                    render_metric_card("平均買入成本", buy_summary["平均持股成本"], "每股加權買入成本", "positive")
                with b_c4:
                    render_metric_card("期末持股市值", buy_summary["期末持股市值"], "最新市值結算", "neutral")

                st.markdown(f"##### 📅 【{inspect_strat}】逐筆扣款購買紀錄表 (共 {len(buys_df)} 筆)")
                st.dataframe(buys_df.set_index("購買序號"), use_container_width=True)
                csv_buys = buys_df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label=f"📥 下載【{inspect_strat}】購買紀錄 (CSV)",
                    data=csv_buys,
                    file_name=f"{target_ticker}_{inspect_strat}_purchase_history.csv",
                    mime="text/csv",
                    key="btn_download_buys"
                )
            else:
                st.info("該策略在回測期間內無購買紀錄。")

        # 擇時策略：展示進出場配對表與 K 線圖
        else:
            pos_s = strategy_pos_dict[inspect_strat]
            trades_df, summary = TradeLedger.extract_trades(
                df_target, pos_s, cost_struct.fee_rate, cost_struct.tax_rate, cost_struct.slippage_rate, curr_symbol
            )

            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            with m_c1:
                render_metric_card("策略勝率 (Win Rate)", summary["勝率"], f"獲利 {summary['獲利筆數']} 筆 / 虧損 {summary['虧損筆數']} 筆", "positive" if summary["_raw_win_rate"] >= 0.5 else "negative")
            with m_c2:
                render_metric_card("盈虧比 (Profit Factor)", summary["盈虧比 (Profit Factor)"], "總賺 / 總賠金額比值", "positive" if summary["_raw_pf"] >= 1.5 else "neutral")
            with m_c3:
                render_metric_card("平均持倉天數", summary["平均持倉天數"], "每筆單平均抱單時間", "neutral")
            with m_c4:
                render_metric_card("已結算交易", f"{summary['已平倉交易']} 筆", f"總訊號次數 {summary['總交易次數']} 筆", "neutral")

            with st.expander("🕯️ 展開查看 K 線買賣點標記圖 (🟢 買進 / 🔴 賣出箭頭)", expanded=False):
                render_candlestick_with_markers(df_target, pos_s, target_ticker, curr_symbol)

            if not trades_df.empty:
                st.dataframe(trades_df.set_index("進場日期"), use_container_width=True)
                csv_trades = trades_df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label=f"📥 下載【{inspect_strat}】逐筆交易明細 (CSV)",
                    data=csv_trades,
                    file_name=f"{target_ticker}_{inspect_strat}_trade_log.csv",
                    mime="text/csv",
                    key="btn_download_trades"
                )

    # -------------------------------------------------------------
    # 功能 C：快慢線 2D 尋優熱力圖 (【智慧判定】：只有在選取 SMA 雙均線交叉時才顯示！)
    # -------------------------------------------------------------
    has_sma_crossover = any("SMA 雙均線交叉" in s for s in selected_strats)

    if has_sma_crossover:
        st.subheader("🔥 功能 C：快慢線網格尋優 2D 熱力圖 (尋找穩健參數高原)")
        st.caption("自動測試快線 (5~30) × 慢線 (30~120) 的所有排列組合，以顏色深淺呈現獲利空間，幫你避開過度擬合的孤島陷阱。")

        with st.expander("🚀 點擊展開快慢線參數高原尋優面板", expanded=False):
            c_opt1, c_opt2, c_opt3 = st.columns(3)
            with c_opt1:
                fast_step = st.selectbox("快線步長 (Step)", [1, 2, 3], index=1, help="步長為 2 代表測試 5, 7, 9... 29")
            with c_opt2:
                slow_step = st.selectbox("慢線步長 (Step)", [2, 5, 10], index=1, help="步長為 5 代表測試 30, 35, 40... 120")
            with c_opt3:
                opt_tf = st.selectbox("尋優時間週期", ["日線 (Daily)", "週線 (Weekly)"], index=0)

            if st.button("⚡ 開始跑快慢線 2D 矩陣尋優", key="btn_run_grid_opt"):
                with st.spinner("正在極速計算 200+ 組快慢線組合之收益矩陣..."):
                    heatmap_matrix, flat_results, best_param = GridOptimizer.run_ma_grid(
                        df_target,
                        fast_range=(5, 31, fast_step),
                        slow_range=(30, 121, slow_step),
                        timeframe=opt_tf,
                        initial_capital=initial_capital,
                        monthly_inflow=monthly_inflow,
                        cost_struct=cost_struct
                    )

                st.success(f"🎉 尋優完成！歷史最高報酬參數組合：**快線 {best_param.get('快線')} / 慢線 {best_param.get('慢線')}**，總報酬率達 **{best_param.get('總報酬率 (%)')}%**！")
                st.markdown("💡 **架構師指引**：觀察下方熱力圖，如果最佳參數周圍也是一片深綠色（**寬廣高原**），代表參數非常穩健；如果是孤零零的一個綠點（**孤島**），請不要採用！")

                render_parameter_heatmap(heatmap_matrix, curr_symbol)

                csv_grid = flat_results.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 下載快慢線尋優排列表 (CSV)",
                    data=csv_grid,
                    file_name=f"{target_ticker}_MA_grid_search_results.csv",
                    mime="text/csv",
                    key="btn_download_grid"
                )
