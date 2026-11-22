"""
分頁 1：單標的 × 多策略 PK 模式 (Single Asset Multi-Strategy PK)
架構設計：
- 緊湊控制列 + 可收合參數設定與停損/移動停利風控
- 四個清爽子分頁：
  1. 總覽與成長曲線 (冠軍診斷、核心 KPI、排行榜、對數/線性成長曲線、水下回撤圖)
  2. 月度報酬與風險分析 (年度×月度報酬熱力圖、報酬 vs 最大回撤散佈分析)
  3. 交易帳本與 K 線圖 (逐筆扣款 / 進出場明細、勝率/盈虧比/期望值、K 線買賣箭頭與均線疊加)
  4. 2D 參數高原尋優 (支援 ROI / Sharpe / Calmar 多目標尋優、高原穩健度檢定、一鍵套用最佳參數)
"""
from typing import Dict, Any, Optional, Tuple
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.cost_model import CostModel, AssetCostStructure
from core.indicators import IndicatorEngine
from core.metrics import PerformanceMetrics
from core.portfolio_engine import BacktestEngine
from core.trade_ledger import TradeLedger
from core.grid_optimizer import GridOptimizer
from strategies.registry import StrategyRegistry
from ui.theme import render_metric_card, render_insight_panel
from ui.components_chart import (
    _apply_terminal_layout,
    render_equity_comparison_chart,
    render_drawdown_chart,
    render_monthly_returns_heatmap,
    render_candlestick_with_markers,
    render_parameter_heatmap,
    PALETTE,
)


@st.cache_data(show_spinner=False, ttl=1800)
def _cached_single_strategy_run(
    df_target: pd.DataFrame,
    strat_name: str,
    params_tuple: Tuple[Tuple[str, Any], ...],
    initial_capital: float,
    monthly_inflow: float,
    fee_rate: float,
    tax_rate: float,
    slippage_rate: float,
    fx_series: Optional[pd.Series],
    target_currency: str,
    invest_day_one: bool,
    stop_loss_pct: float,
    trailing_stop_pct: float,
) -> pd.DataFrame:
    cost_struct = AssetCostStructure(fee_rate=fee_rate, tax_rate=tax_rate, slippage_rate=slippage_rate)
    return BacktestEngine.run_single_strategy(
        df=df_target,
        strat_name=strat_name,
        strat_params=dict(params_tuple),
        initial_capital=initial_capital,
        monthly_inflow=monthly_inflow,
        cost_struct=cost_struct,
        fx_series=fx_series,
        target_currency=target_currency,
        invest_day_one=invest_day_one,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
    )


def _render_risk_return_scatter(df_results: pd.DataFrame, champion_name: str):
    fig = go.Figure()
    for idx, row in df_results.iterrows():
        s_name = row["策略名稱"]
        is_champ = s_name == champion_name
        fig.add_trace(
            go.Scatter(
                x=[abs(float(row["_raw_mdd"])) * 100.0],
                y=[float(row["_raw_cagr"]) * 100.0],
                mode="markers+text",
                name=s_name,
                marker=dict(
                    size=16 if is_champ else 11,
                    symbol="star" if is_champ else "circle",
                    color=PALETTE[idx % len(PALETTE)],
                    line=dict(color="#ffffff", width=1.2 if is_champ else 0.6),
                ),
                text=[s_name.split(" ")[0]],
                textposition="top center",
                textfont=dict(size=10, color="#cbd5e1"),
                hovertemplate=(
                    f"{s_name}<br>最大回撤 (MDD): %{{x:.2f}}%<br>"
                    f"年化報酬 (CAGR): %{{y:+.2f}}%<br>"
                    f"夏普比率: {row['夏普比率 (Sharpe)']}<extra></extra>"
                ),
            )
        )

    _apply_terminal_layout(
        fig,
        x_title="最大歷史回撤幅度 |MDD| (%) [越左越抗跌]",
        y_title="年化複合報酬 CAGR (%) [越上越賺]",
        height=380,
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch")


def render_strategy_pk_tab(
    individual_dict: Dict[str, pd.DataFrame],
    curr_symbol: str = "NT$",
    initial_capital: float = 300000.0,
    monthly_inflow: float = 30000.0,
    cost_override: Optional[AssetCostStructure] = None,
    tw_fee_discount: float = 0.5,
    fx_series: Optional[pd.Series] = None,
    target_currency: str = "TWD",
    invest_day_one: bool = False,
):
    if not individual_dict:
        st.info("👈 請在側邊欄選擇標的並載入市場數據。")
        return

    # 若有上輪觸發的一鍵套用最佳參數，於對應參數元件渲染前先行寫入 session_state
    pending_apply = st.session_state.pop("_pending_grid_apply", None)
    if isinstance(pending_apply, dict):
        for k, v in pending_apply.items():
            st.session_state[k] = v

    all_strategies = StrategyRegistry.list_names()
    default_strats = [
        "定期定額 / 買入持有 (DCA / Buy & Hold)",
        "價格突破均線買入/跌破賣出",
        "SMA 雙均線交叉 (黃金交叉)",
        "雙重濾網動能 (均線趨勢 + RSI)",
    ]
    valid_defaults = [s for s in default_strats if s in all_strategies]

    c_sel1, c_sel2 = st.columns([1.1, 2.4])
    with c_sel1:
        target_ticker = st.selectbox(
            "PK 標的",
            list(individual_dict.keys()),
            help="選擇要進行策略對決的單一標的。",
        )
    with c_sel2:
        selected_strats = st.multiselect(
            "參戰策略 (可複選)",
            options=all_strategies,
            default=valid_defaults if valid_defaults else all_strategies[:2],
            help="勾選多個策略同時同場競技。",
        )

    df_target = individual_dict[target_ticker]
    cost_struct = (
        cost_override
        if cost_override
        else CostModel.get_default_costs(target_ticker, tw_fee_discount=tw_fee_discount)
    )
    st.caption(f"適用成本模型 — {CostModel.describe_cost(target_ticker, cost_struct)}")

    if not selected_strats:
        st.warning("請至少勾選一種策略進行評估。")
        return

    # 策略參數設定與風控設定
    strategy_params_dict: Dict[str, Dict[str, Any]] = {}
    with st.expander("策略參數微調與停損 / 移動停利風控", expanded=False):
        c_rk1, c_rk2 = st.columns(2)
        with c_rk1:
            stop_loss_pct = st.number_input(
                "固定停損門檻 (%，0 = 不啟用)",
                min_value=0.0,
                max_value=80.0,
                value=0.0,
                step=1.0,
                key=f"rk_sl_{target_ticker}",
                help="僅適用於擇時策略：收盤跌破買入成本價達此比例時，於隔日開盤停損出場。",
            ) / 100.0
        with c_rk2:
            trailing_stop_pct = st.number_input(
                "移動停利回落門檻 (%，0 = 不啟用)",
                min_value=0.0,
                max_value=80.0,
                value=0.0,
                step=1.0,
                key=f"rk_ts_{target_ticker}",
                help="僅適用於擇時策略：收盤自持倉最高點回落達此比例時，於隔日開盤鎖利出場。",
            ) / 100.0

        st.markdown("---")
        for strat_name in selected_strats:
            strat_cls = StrategyRegistry.get(strat_name)
            specs = strat_cls.get_param_specs()
            st.markdown(f"**{strat_name}** — *{strat_cls.description}*")
            params: Dict[str, Any] = {}
            if specs:
                cols = st.columns(min(len(specs), 4))
                for idx, spec in enumerate(specs):
                    with cols[idx % len(cols)]:
                        key_id = f"{target_ticker}_{strat_cls.strategy_id}_{spec.key}"
                        if spec.param_type == "int":
                            if key_id not in st.session_state:
                                st.session_state[key_id] = int(spec.default)
                            params[spec.key] = st.number_input(
                                spec.label,
                                min_value=int(spec.min_value) if spec.min_value is not None else 1,
                                max_value=int(spec.max_value) if spec.max_value is not None else 500,
                                step=int(spec.step) if spec.step is not None else 1,
                                key=key_id,
                                help=spec.help_text,
                            )
                        elif spec.param_type == "float":
                            if key_id not in st.session_state:
                                st.session_state[key_id] = float(spec.default)
                            params[spec.key] = st.number_input(
                                spec.label,
                                min_value=float(spec.min_value) if spec.min_value is not None else 0.0,
                                max_value=float(spec.max_value) if spec.max_value is not None else 1000000.0,
                                step=float(spec.step) if spec.step is not None else 1.0,
                                key=key_id,
                                help=spec.help_text,
                            )
                        elif spec.param_type == "select":
                            opts = spec.options or []
                            if key_id not in st.session_state:
                                st.session_state[key_id] = spec.default if spec.default in opts else (opts[0] if opts else "")
                            params[spec.key] = st.selectbox(
                                spec.label,
                                options=opts,
                                key=key_id,
                                help=spec.help_text,
                            )
                        elif spec.param_type == "bool":
                            if key_id not in st.session_state:
                                st.session_state[key_id] = bool(spec.default)
                            params[spec.key] = st.checkbox(
                                spec.label,
                                key=key_id,
                                help=spec.help_text,
                            )
            strategy_params_dict[strat_name] = params

    # 執行快取回測
    results_summary = []
    equity_curves: Dict[str, pd.DataFrame] = {}

    for strat_name in selected_strats:
        p = strategy_params_dict.get(strat_name, {})
        params_tuple = tuple(sorted(p.items()))
        try:
            res_df = _cached_single_strategy_run(
                df_target=df_target,
                strat_name=strat_name,
                params_tuple=params_tuple,
                initial_capital=initial_capital,
                monthly_inflow=monthly_inflow,
                fee_rate=cost_struct.fee_rate,
                tax_rate=cost_struct.tax_rate,
                slippage_rate=cost_struct.slippage_rate,
                fx_series=fx_series,
                target_currency=target_currency,
                invest_day_one=invest_day_one,
                stop_loss_pct=stop_loss_pct,
                trailing_stop_pct=trailing_stop_pct,
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

    df_results = (
        pd.DataFrame(results_summary)
        .sort_values(by="_raw_net_profit", ascending=False)
        .reset_index(drop=True)
    )
    champion_row = df_results.iloc[0]
    champion_name = champion_row["策略名稱"]
    champion_profit = float(champion_row["_raw_net_profit"])
    champion_final = float(champion_row["_raw_final"])
    champion_mdd = float(champion_row["_raw_mdd"])

    # 4 個清爽子分頁
    sub_overview, sub_monthly, sub_ledger, sub_grid = st.tabs(
        [
            "📊 總覽與成長曲線",
            "🗓️ 月度報酬與風險散佈",
            "📖 交易帳本與 K 線圖",
            "🔥 2D 參數高原尋優",
        ]
    )

    # --- 子分頁 1：總覽與成長曲線 ---
    with sub_overview:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card(
                "冠軍實賺淨利潤",
                f"{curr_symbol} {champion_profit:+,.0f}",
                f"冠軍：{champion_name}",
                "positive" if champion_profit >= 0 else "negative",
            )
        with c2:
            render_metric_card(
                "冠軍期末總資產",
                f"{curr_symbol} {champion_final:,.0f}",
                f"CAGR {champion_row['CAGR 年化報酬']} | XIRR {champion_row['XIRR 資金加權']}",
                "neutral",
            )
        with c3:
            render_metric_card(
                "總報酬率 / 夏普比率",
                f"{champion_row['_raw_roi']:+.2%}",
                f"Sharpe {champion_row['夏普比率 (Sharpe)']} | Calmar {champion_row['卡瑪比率 (Calmar)']}",
                "positive" if champion_profit >= 0 else "negative",
            )
        with c4:
            render_metric_card(
                "最大歷史回撤 (MDD)",
                f"{champion_mdd:.2%}",
                f"最長套牢 {champion_row['最長套牢天數']}",
                "negative" if champion_mdd < -0.2 else "neutral",
            )

        if "DCA" in champion_name and "Dip" not in champion_name:
            reason_bullets = [
                f"**長期多頭複利勝出**：在此期間內持續買入並持有完整捕捉複利成長，勝過頻繁進出的擇時策略。",
                f"冠軍累計投入本金 {champion_row['累積投入本金']}，年化報酬達 **{champion_row['CAGR 年化報酬']}**。",
            ]
        elif "Dip" in champion_name:
            reason_bullets = [
                f"**逢低抄底策略勝出**：市場出現明顯回檔機會，成功於相對低檔累積低成本籌碼。",
                f"冠軍年化報酬 **{champion_row['CAGR 年化報酬']}**，最大回撤 **{champion_row['最大歷史回撤 (MDD)']}**。",
            ]
        else:
            reason_bullets = [
                f"**擇時避險勝出**：在空頭回檔時及時離場避險，將最大回撤控制在 **{champion_row['最大歷史回撤 (MDD)']}**。",
                f"夏普比率 **{champion_row['夏普比率 (Sharpe)']}**，索提諾比率 **{champion_row['索提諾比率 (Sortino)']}**。",
            ]
        render_insight_panel(f"冠軍策略解讀 — {champion_name}", reason_bullets, tone="success" if champion_profit >= 0 else "warning")

        st.markdown("##### 策略績效完整排行榜")
        display_cols = [
            "策略名稱",
            "實賺獲利金額",
            "最終總資產",
            "累積投入本金",
            "淨獲利報酬率",
            "CAGR 年化報酬",
            "XIRR 資金加權",
            "夏普比率 (Sharpe)",
            "卡瑪比率 (Calmar)",
            "最大歷史回撤 (MDD)",
            "最長套牢天數",
            "累積交易次數",
        ]
        display_df = df_results[display_cols].copy()
        display_df.loc[0, "策略名稱"] = f"👑 {champion_name}"
        st.dataframe(display_df.set_index("策略名稱"), width="stretch")

        c_chart_hdr, c_log_chk = st.columns([3, 1])
        with c_chart_hdr:
            st.markdown("##### 資產成長曲線對比")
        with c_log_chk:
            use_log_scale = st.checkbox("對數座標 (Log Scale)", value=False, key="pk_log_scale")

        render_equity_comparison_chart(
            equity_curves,
            curr_symbol=curr_symbol,
            champion_name=champion_name,
            log_scale=use_log_scale,
        )

        st.markdown("##### 水下歷史回撤對比 (Drawdown)")
        dd_curves = {
            s_name: PerformanceMetrics.calculate_drawdown_series(c_df)
            for s_name, c_df in equity_curves.items()
        }
        render_drawdown_chart(dd_curves, champion_name=champion_name)

    # --- 子分頁 2：月度報酬與風險分析 ---
    with sub_monthly:
        inspect_m_strat = st.selectbox(
            "選擇檢視月度報酬熱力圖之策略",
            selected_strats,
            key="pk_monthly_strat",
        )
        if inspect_m_strat in equity_curves:
            monthly_df = PerformanceMetrics.calculate_monthly_returns(equity_curves[inspect_m_strat])
            render_monthly_returns_heatmap(monthly_df, title=f"{inspect_m_strat} — 各年度月報酬矩陣")

        st.markdown("##### 各策略「風險 vs 年化報酬」分佈象限")
        st.caption("越往左上方（低回撤、高年化報酬）代表策略風險調整後性價比越高。")
        _render_risk_return_scatter(df_results, champion_name)

    # --- 子分頁 3：交易帳本與 K 線圖 ---
    with sub_ledger:
        inspect_strat = st.selectbox(
            "選擇檢視成交紀錄之策略",
            selected_strats,
            key="select_inspect_strat",
        )
        if inspect_strat in equity_curves:
            res_curve = equity_curves[inspect_strat]
            strat_cls = StrategyRegistry.get(inspect_strat)

            if not strat_cls.is_timing:
                buys_df, buy_summary = TradeLedger.extract_buy_records(res_curve, curr_symbol)
                if not buys_df.empty:
                    b_c1, b_c2, b_c3, b_c4 = st.columns(4)
                    with b_c1:
                        render_metric_card("累積扣款次數", buy_summary["累積購買次數"], "成功買入次數", "neutral")
                    with b_c2:
                        render_metric_card("累積持有股數", buy_summary["累積持有股數"], "期末持有總股數", "neutral")
                    with b_c3:
                        render_metric_card("平均買入成本", buy_summary["平均持股成本"], "扣除餘現金之每股成本", "positive")
                    with b_c4:
                        render_metric_card("期末持股市值", buy_summary["期末持股市值"], "按期末收盤價換算", "neutral")

                    st.dataframe(buys_df.set_index("購買序號"), width="stretch")
                    csv_buys = buys_df.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        label=f"📥 下載【{inspect_strat}】購買紀錄 (CSV)",
                        data=csv_buys,
                        file_name=f"{target_ticker}_{strat_cls.strategy_id}_purchase_history.csv",
                        mime="text/csv",
                        key="btn_download_buys",
                    )
                else:
                    st.info("該策略在回測期間內尚無購買紀錄。")
            else:
                trades_df, summary = TradeLedger.extract_trades(
                    res_curve,
                    pos_series=None,
                    cost_fee=cost_struct.fee_rate,
                    cost_tax=cost_struct.tax_rate,
                    cost_slip=cost_struct.slippage_rate,
                    curr_symbol=curr_symbol,
                )

                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                with m_c1:
                    render_metric_card(
                        "策略勝率",
                        summary["勝率"],
                        f"獲利 {summary['獲利筆數']} 筆 / 虧損 {summary['虧損筆數']} 筆",
                        "positive" if summary["_raw_win_rate"] >= 0.5 else "negative",
                    )
                with m_c2:
                    render_metric_card(
                        "盈虧比 (Profit Factor)",
                        summary["盈虧比 (Profit Factor)"],
                        f"單筆期望值 {summary['單筆期望值']}",
                        "positive" if summary["_raw_pf"] >= 1.5 else "neutral",
                    )
                with m_c3:
                    render_metric_card("平均持倉天數", summary["平均持倉天數"], "已平倉平均抱單時間", "neutral")
                with m_c4:
                    render_metric_card(
                        "總交易回合數",
                        f"{summary['總交易次數']} 筆",
                        f"已平倉 {summary['已平倉交易']} 筆",
                        "neutral",
                    )

                # 疊加均線參考
                ma_overlays: Dict[str, pd.Series] = {}
                p_used = strategy_params_dict.get(inspect_strat, {})
                tf_used = p_used.get("timeframe", "日線 (Daily)")
                if strat_cls.strategy_id == "price_sma_breakout":
                    per = int(p_used.get("period", 20))
                    ma_overlays[f"SMA {per}"] = IndicatorEngine.calculate_ma(df_target, per, tf_used)
                elif strat_cls.strategy_id == "sma_crossover":
                    f_p, s_p = int(p_used.get("fast", 20)), int(p_used.get("slow", 60))
                    ma_overlays[f"快線 SMA {f_p}"] = IndicatorEngine.calculate_ma(df_target, f_p, tf_used)
                    ma_overlays[f"慢線 SMA {s_p}"] = IndicatorEngine.calculate_ma(df_target, s_p, tf_used)
                elif strat_cls.strategy_id == "ema_crossover":
                    f_p, s_p = int(p_used.get("fast", 12)), int(p_used.get("slow", 26))
                    ma_overlays[f"快線 EMA {f_p}"] = IndicatorEngine.calculate_ema(df_target, f_p, tf_used)
                    ma_overlays[f"慢線 EMA {s_p}"] = IndicatorEngine.calculate_ema(df_target, s_p, tf_used)
                elif strat_cls.strategy_id == "bollinger_bands":
                    per = int(p_used.get("period", 20))
                    std_k = float(p_used.get("num_std", 2.0))
                    u_b, m_b, l_b = IndicatorEngine.calculate_bollinger(df_target, per, std_k, tf_used)
                    ma_overlays["上軌"] = u_b
                    ma_overlays["中軌"] = m_b
                    ma_overlays["下軌"] = l_b

                st.markdown("##### K 線進出場訊號標記圖 (原生價格)")
                native_curr = str(df_target["quote_currency"].iloc[0]) if "quote_currency" in df_target.columns else "USD"
                native_sym = "NT$" if native_curr == "TWD" else "$"
                render_candlestick_with_markers(
                    res_curve,
                    res_curve["position"],
                    target_ticker,
                    curr_symbol=native_sym,
                    ma_overlays=ma_overlays if ma_overlays else None,
                )

                if not trades_df.empty:
                    st.dataframe(trades_df.set_index("進場日期"), width="stretch")
                    csv_trades = trades_df.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        label=f"📥 下載【{inspect_strat}】逐筆交易明細 (CSV)",
                        data=csv_trades,
                        file_name=f"{target_ticker}_{strat_cls.strategy_id}_trade_log.csv",
                        mime="text/csv",
                        key="btn_download_trades",
                    )

    # --- 子分頁 4：2D 參數高原尋優 ---
    with sub_grid:
        st.caption("使用與主引擎完全相同的除權息、匯率與扣款撮合邏輯進行全排列組合掃描，尋找具備高容錯率的「參數高原」。")
        c_opt0, c_opt_m = st.columns([2, 1])
        with c_opt0:
            opt_strategy = st.selectbox(
                "選擇要執行 2D 尋優的策略",
                options=[
                    "SMA 雙均線交叉 (快線週期 × 慢線週期)",
                    "MACD 柱狀圖/交叉 (EMA快線 × EMA慢線)",
                    "RSI 超買超賣 (超賣買入門檻 × 超買賣出門檻)",
                    "KD 隨機指標 (超賣買入K × 超買賣出K)",
                    "跌破均線且創新低買入 (基準均線 × 加碼間隔天數)",
                ],
                index=0,
                key="pk_grid_opt_strategy",
            )
        with c_opt_m:
            metric_choice = st.selectbox(
                "尋優目標指標",
                options=["總報酬率 (%)", "夏普比率 (Sharpe)", "卡瑪比率 (Calmar)"],
                index=0,
                key="pk_grid_metric_choice",
                help="除了追求最高總報酬外，亦可選擇夏普比率或卡瑪比率找出風險調整後最佳參數。",
            )
        metric_key = {"總報酬率 (%)": "roi", "夏普比率 (Sharpe)": "sharpe", "卡瑪比率 (Calmar)": "calmar"}[metric_choice]

        c_opt1, c_opt2, c_opt3 = st.columns(3)
        if "SMA 雙均線" in opt_strategy:
            with c_opt1:
                fast_step = st.selectbox("快線步長", [1, 2, 3], index=1)
            with c_opt2:
                slow_step = st.selectbox("慢線步長", [2, 5, 10], index=1)
            with c_opt3:
                opt_tf = st.selectbox("計算週期", ["日線 (Daily)", "週線 (Weekly)"], index=0, key="grid_tf_sma")
        elif "MACD" in opt_strategy:
            with c_opt1:
                fast_step = st.selectbox("快線 EMA 步長", [1, 2], index=0)
            with c_opt2:
                slow_step = st.selectbox("慢線 EMA 步長", [1, 2, 5], index=1)
            with c_opt3:
                opt_tf = st.selectbox("計算週期", ["日線 (Daily)", "週線 (Weekly)"], index=0, key="grid_tf_macd")
        elif "RSI" in opt_strategy:
            with c_opt1:
                rsi_p = st.number_input("RSI 基準週期", value=14, min_value=5, max_value=30, step=1)
            with c_opt2:
                step_val = st.selectbox("門檻掃描步長", [1, 2, 5], index=1)
            with c_opt3:
                opt_tf = st.selectbox("計算週期", ["日線 (Daily)", "週線 (Weekly)"], index=0, key="grid_tf_rsi")
        elif "KD" in opt_strategy:
            with c_opt1:
                kd_p = st.number_input("KD 基準週期", value=9, min_value=3, max_value=20, step=1)
            with c_opt2:
                step_val = st.selectbox("K值掃描步長", [1, 2, 5], index=1)
            with c_opt3:
                opt_tf = st.selectbox("計算週期", ["日線 (Daily)", "週線 (Weekly)"], index=0, key="grid_tf_kd")
        else:
            with c_opt1:
                dip_amt = st.number_input("每次加碼買入金額", value=50000.0, min_value=1000.0, step=10000.0)
            with c_opt2:
                sma_step = st.selectbox("均線掃描步長", [5, 10, 20], index=1)
            with c_opt3:
                opt_tf = st.selectbox("計算週期", ["日線 (Daily)", "週線 (Weekly)"], index=0, key="grid_tf_dip")

        if st.button("⚡ 開始執行 2D 矩陣尋優", key="btn_run_generic_grid", type="primary"):
            with st.spinner("正在運算所有參數組合之收益與風險矩陣..."):
                if "SMA 雙均線" in opt_strategy:
                    heatmap_df, flat_df, best_p = GridOptimizer.run_ma_grid(
                        df_target, fast_range=(5, 31, fast_step), slow_range=(30, 121, slow_step),
                        timeframe=opt_tf, initial_capital=initial_capital, monthly_inflow=monthly_inflow,
                        cost_struct=cost_struct, fx_series=fx_series, target_currency=target_currency,
                        metric=metric_key, invest_day_one=invest_day_one,
                    )
                    x_lbl, y_lbl, rev_y = "快線週期", "慢線週期", True
                    apply_info = {
                        "strat_id": "sma_crossover",
                        "updates": {"timeframe": opt_tf, "fast": int(best_p["參數X (快線)"]), "slow": int(best_p["參數Y (慢線)"])},
                    }
                elif "MACD" in opt_strategy:
                    heatmap_df, flat_df, best_p = GridOptimizer.run_macd_grid(
                        df_target, fast_range=(5, 21, fast_step), slow_range=(20, 61, slow_step),
                        timeframe=opt_tf, initial_capital=initial_capital, monthly_inflow=monthly_inflow,
                        cost_struct=cost_struct, fx_series=fx_series, target_currency=target_currency,
                        metric=metric_key, invest_day_one=invest_day_one,
                    )
                    x_lbl, y_lbl, rev_y = "EMA 快線週期", "EMA 慢線週期", True
                    apply_info = {
                        "strat_id": "macd_strategy",
                        "updates": {"timeframe": opt_tf, "fast": int(best_p["參數X (EMA快線)"]), "slow": int(best_p["參數Y (EMA慢線)"])},
                    }
                elif "RSI" in opt_strategy:
                    heatmap_df, flat_df, best_p = GridOptimizer.run_rsi_grid(
                        df_target, period=int(rsi_p), buy_range=(15, 41, step_val), sell_range=(60, 86, step_val),
                        timeframe=opt_tf, initial_capital=initial_capital, monthly_inflow=monthly_inflow,
                        cost_struct=cost_struct, fx_series=fx_series, target_currency=target_currency,
                        metric=metric_key, invest_day_one=invest_day_one,
                    )
                    x_lbl, y_lbl, rev_y = "超賣買入門檻 (RSI)", "超買賣出門檻 (RSI)", False
                    apply_info = {
                        "strat_id": "rsi_strategy",
                        "updates": {"timeframe": opt_tf, "period": int(rsi_p), "buy": float(best_p["參數X (超賣買入)"]), "sell": float(best_p["參數Y (超買賣出)"])},
                    }
                elif "KD" in opt_strategy:
                    heatmap_df, flat_df, best_p = GridOptimizer.run_kd_grid(
                        df_target, period=int(kd_p), buy_range=(15, 36, step_val), sell_range=(65, 86, step_val),
                        timeframe=opt_tf, initial_capital=initial_capital, monthly_inflow=monthly_inflow,
                        cost_struct=cost_struct, fx_series=fx_series, target_currency=target_currency,
                        metric=metric_key, invest_day_one=invest_day_one,
                    )
                    x_lbl, y_lbl, rev_y = "超賣買入門檻 (K值)", "超買賣出門檻 (K值)", False
                    apply_info = {
                        "strat_id": "kd_strategy",
                        "updates": {"timeframe": opt_tf, "period": int(kd_p), "buy_k": float(best_p["參數X (超賣買入K)"]), "sell_k": float(best_p["參數Y (超買賣出K)"])},
                    }
                else:
                    heatmap_df, flat_df, best_p = GridOptimizer.run_dip_grid(
                        df_target, sma_range=(20, 121, sma_step), interval_range=(1, 11, 1),
                        buy_amount=dip_amt, timeframe=opt_tf, initial_capital=initial_capital,
                        monthly_inflow=monthly_inflow, cost_struct=cost_struct,
                        fx_series=fx_series, target_currency=target_currency,
                        metric=metric_key, invest_day_one=invest_day_one,
                    )
                    x_lbl, y_lbl, rev_y = "基準均線週期 (SMA)", "加碼冷卻天數", False
                    apply_info = {
                        "strat_id": "dip_buyer",
                        "updates": {"timeframe": opt_tf, "sma_period": int(best_p["參數X (SMA均線週期)"]), "min_interval": int(best_p["參數Y (抄底間隔天數)"]), "buy_amount": float(dip_amt)},
                    }

                st.session_state["last_grid_result"] = {
                    "ticker": target_ticker,
                    "heatmap_df": heatmap_df,
                    "flat_df": flat_df,
                    "best_p": best_p,
                    "x_lbl": x_lbl,
                    "y_lbl": y_lbl,
                    "rev_y": rev_y,
                    "metric_choice": metric_choice,
                    "opt_strategy": opt_strategy,
                    "apply_info": apply_info,
                }

        grid_state = st.session_state.get("last_grid_result")
        if grid_state and grid_state.get("ticker") == target_ticker and not grid_state["flat_df"].empty:
            best_p = grid_state["best_p"]
            flat_df = grid_state["flat_df"]
            heatmap_df = grid_state["heatmap_df"]
            x_col = best_p.get("x_col", "")
            y_col = best_p.get("y_col", "")
            plateau = best_p.get("plateau_info", {})

            bullets = [
                f"**最佳參數**：`{x_col} = {best_p.get(x_col)}` / `{y_col} = {best_p.get(y_col)}`",
                f"**實賺淨利**：{curr_symbol} {best_p.get('實賺獲利金額', 0):,.0f}（總報酬 {best_p.get('總報酬率 (%)', 0):+.2f}%｜MDD {best_p.get('最大回撤 (%)', 0):.2f}%｜Sharpe {best_p.get('夏普比率 (Sharpe)', 0):.2f}）",
                f"**高原穩健度檢定**：{plateau.get('status', '-')}（3×3 鄰域中位數為最佳值的 {plateau.get('ratio', 0):.0%}）",
            ]
            render_insight_panel("尋優結果摘要", bullets, tone="info")

            if st.button("✨ 一鍵套用最佳參數至主回測", key="btn_apply_best_grid"):
                app_info = grid_state["apply_info"]
                sid = app_info["strat_id"]
                st.session_state["_pending_grid_apply"] = {
                    f"{target_ticker}_{sid}_{p_k}": p_v
                    for p_k, p_v in app_info["updates"].items()
                }
                st.toast("已套用最佳參數，請於「總覽與成長曲線」查看更新結果！", icon="✅")
                st.rerun()

            render_parameter_heatmap(
                heatmap_df,
                curr_symbol=curr_symbol,
                x_title=grid_state["x_lbl"],
                y_title=grid_state["y_lbl"],
                reverse_y=grid_state["rev_y"],
                metric_label=grid_state["metric_choice"],
                best_x=best_p.get(x_col),
                best_y=best_p.get(y_col),
            )

            with st.expander("檢視前 15 名參數組合排列表", expanded=False):
                st.dataframe(flat_df.head(15), width="stretch")

            csv_grid = flat_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 下載完整尋優矩陣結果 (CSV)",
                data=csv_grid,
                file_name=f"{target_ticker}_{grid_state['opt_strategy'].split(' ')[0]}_grid_results.csv",
                mime="text/csv",
                key="btn_download_generic_grid",
            )

    with st.expander("💡 量化回測機制與扣款細節說明", expanded=False):
        st.markdown(
            """
            1. **無未來視撮合**：技術指標均以 T 日收盤之**還原除權息價格 (Native Currency)** 計算，確認訊號後一律於 **T+1 日開盤價** 執行買賣。
            2. **扣款時機**：有設期初本金時於第 1 個交易日立即建倉；若期初本金為 0，勾選左側「首日即扣首筆定投」即可於第 1 天扣款，否則將於次月首個開市日起扣款。
            3. **頻率拆分**：選用「每週」或「每 10 交易日」定投時，系統會自動把每月新注資等比拆分為當月各期金額，確保所有頻率比較在同一總投入本金基準上進行。
            4. **還原與拆股**：採用已經過拆股還原調整之價格序列，並將現金股利直接計入現金池再投資，徹底避免拆股重複乘算造成的虛假獲利。
            """
        )
