"""分頁 1：單標的 × 多策略 PK 模式 (落實原生幣別標記與暖機期傳遞)。"""
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.cost_model import CostModel, AssetCostStructure
from core.metrics import PerformanceMetrics
from core.portfolio_engine import BacktestEngine
from strategies.registry import StrategyRegistry
from ui.theme import render_metric_card


def render_strategy_pk_tab(
    individual_dict: Dict[str, pd.DataFrame],
    effective_start_date: str,
    initial_capital: float,
    monthly_inflow: float,
    cost_override: Optional[AssetCostStructure] = None
):
    st.header("⚔️ 單標的 × 多策略獲利 PK 模式")
    st.caption(f"以標的原生價格與暖機指標同場較勁，真實回測起點：{effective_start_date}。")

    if not individual_dict:
        st.info("👈 請於側邊欄選取交易標的。")
        return

    target_ticker = st.selectbox("選擇要進行多策略 PK 的標的：", list(individual_dict.keys()))
    df_target = individual_dict[target_ticker]

    # 原生幣別標註
    native_curr = df_target.attrs.get("quote_currency", "USD")
    curr_symbol = "NT$" if native_curr == "TWD" else "$"

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

    with st.expander("⚙️ 展開設定選取策略之參數", expanded=False):
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
            strategy_params_dict[strat_name] = params
            st.markdown("---")

    cost_struct = cost_override if cost_override else CostModel.get_default_costs(target_ticker)

    results_summary = []
    equity_curves = {}

    for strat_name in selected_strats:
        p = strategy_params_dict.get(strat_name, {})
        try:
            res_df = BacktestEngine.run_single_strategy(
                df=df_target,
                strat_name=strat_name,
                strat_params=p,
                initial_capital=initial_capital,
                monthly_inflow=monthly_inflow,
                cost_struct=cost_struct,
                eval_start_date=effective_start_date
            )
            m = PerformanceMetrics.evaluate(res_df, curr_symbol=curr_symbol, initial_capital=initial_capital)
            m["策略名稱"] = strat_name
            results_summary.append(m)
            equity_curves[strat_name] = res_df
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

    st.subheader(f"🏆 獲利冠軍：【{champion_name}】 (原生計價：{curr_symbol})")
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

    st.subheader("📈 策略資產成長曲線對比")
    fig = go.Figure()

    for idx, strat_name in enumerate(selected_strats):
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
