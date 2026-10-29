"""
分頁 1：單標的 × 多策略 PK 模式 (Single Asset Multi-Strategy PK - 頂級白話教學與扣款細節版)
包含：
- 💡 新手速查手冊：7 大量化策略性格解密 ＆ 核心指標白話對照 ＆ 扣款撮合細節
- 完整支援 DCA、Dip DCA 與擇時策略之逐筆成交/購買歷史 (Purchase Ledger) ＋ 📥 下載 CSV
- K 線進出場標記圖 (TradingView 質感 🟢買入 / 🔴賣出箭頭)
- 通用 2D 矩陣尋優熱力圖 (支援 SMA, MACD, RSI, KD, Dip DCA 高原尋優)
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
    st.caption("評估同一檔資產在不同交易策略下的「實賺獲利金額」與「抗跌避險能力」，幫你找出最適合該標的的獲利密碼。")

    if not individual_dict:
        st.info("👈 請在左側側邊欄選擇或輸入標的代碼，並點擊加載市場數據。")
        return

    # 💡 新手白話教學與扣款撮合細節展開面板
    with st.expander("💡 第一次使用？點擊展開「3 分鐘量化策略與扣款撮合細節手冊」", expanded=False):
        st.markdown(
            """
            #### ⚠️ 關鍵細節：扣款時機與交易撮合規則（避免對帳誤解）
            1. **開始日期若為 7/1，當天會扣款嗎？**
               * **有設期初本金**（如期初 30 萬）：7/1 首個開盤日會**立即投入 30 萬全額買進**，次月初（8/1）再開始扣每月定投 3 萬。
               * **期初本金設為 0**：第 1 個月（7 月）視為開戶簽約準備期，**7 月不扣款**，第一筆 3 萬元將於**次月首個交易日（8/1）**才扣款買入！若想第 1 天就扣第一筆 3 萬，請直接把 3 萬填在「期初本金」。
            2. **DCA 扣款日不是看日曆，是看「開市日」**：
               * 若扣款日（如 7/1）剛好是週六日或國定假日，系統會**自動順延至下一個有開盤的日子（如 7/3 週一）**以開盤價成交。休市日絕不可能憑空成交。
            3. **T 日收盤確認破線，T+1 日開盤才成交（無未來視）**：
               * 在看盤軟體（如三竹）上肉眼看到「今天跌破均線」，那是收盤時才確定的事。量化系統嚴禁偷看未來，因此系統一律在**次日（T+1）開盤**才真正下單買入，帳本上的成交日期會比破線日晚 1 天，這是完全正常的！
            4. **為什麼均線數值跟一般看盤軟體（如三竹）有微小出入？**
               * 一般券商看盤軟體預設為「未還原日線」（除息時會有跳空向下缺口）。
               * 本系統為了防止除息引發假破線停損，採用「**還原除權息收盤價 (adj_close)**」計算均線，並在「標的原生幣別（美股美元、台股台幣）」下運算，杜絕匯率干擾。

            ---
            #### 🧠 7 大量化策略性格指南（你適合哪一種？）
            * 💰 **定期定額 / 買入持有 (DCA)**：【佛系存股派】不管漲跌，時間到就扣款。最省心、消除人性情緒，牛市時能完整吃到市場長期複利。
            * 🛒 **跌破均線且創新低買入 (Dip DCA)**：【特價掃貨派】平時存錢，只有在價格跌破均線且破新低、市場在特價時才分批加碼（且買入後至少冷卻 3 天防暴跌），平均持股成本通常比無腦 DCA 更低！
            * 🏄 **價格突破均線買入/跌破賣出**：【順勢騎浪派】只要股價站上均線就買入持有；一旦跌破均線立刻獲利了結或空倉避險，擅長躲避大崩盤。
            * ⚔️ **SMA 雙均線交叉 (黃金交叉)**：【波段長波派】短期快線向上穿越長期慢線買進、跌破賣出。過濾短線假破線雜訊，專吃大波段趨勢。
            * 🏹 **RSI 超買超賣**：【低吸高拋派】在市場恐慌超跌時（RSI < 30）逆勢買進撿便宜，在市場貪婪過熱時（RSI > 70）獲利了結。
            * ⚡ **MACD 柱狀圖/交叉**：【動能追隨派】觀察多空力量的消長，快線穿過慢線、動能由負翻正時順勢跟進。
            * 🎯 **KD 隨機指標**：【短線轉折派】擺盪指標經典，在低檔黃金交叉時進場、高檔過熱時離場，適合震盪盤整行情。

            ---
            #### 📊 核心績效指標白話解密（看懂成績單）
            * **實賺獲利總額**：扣掉你掏出的自付本金後，口袋裡**真金白銀淨賺的金額**（總資產 - 自付總本金）。
            * **年化複合成長 (CAGR)**：換算成「每年平均幫你賺幾 %」（如同銀行的年利率概念，方便跨時間長度對比）。
            * **最大歷史回撤 (MDD)**：回測期間**「從歷史最高點摔到最低點跌了幾 %」**。代表你心臟需要承受的最大痛苦程度。
            * **勝率 (Win Rate) vs 盈虧比 (Profit Factor)**：勝率代表贏錢筆數佔比；盈虧比代表「平均賺一次賺的錢 ÷ 平均賠一次賠的錢」。只要盈虧比高（大賺小賠），40% 勝率一樣能賺大錢！
            """
        )

    target_ticker = st.selectbox(
        "選擇要進行多策略 PK 的標的：",
        list(individual_dict.keys()),
        help="選擇你想檢驗的單一資產，系統會讓多種策略同時在這檔標的歷史走勢上對決。"
    )
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
        default=valid_defaults if valid_defaults else all_strategies[:2],
        help="勾選你想同台 PK 的策略，下方會自動輸出排行榜與成長曲線對比。"
    )

    if not selected_strats:
        st.warning("請至少勾選一種策略進行評估。")
        return

    with st.expander("⚙️ 展開微調選取策略之參數 (含計算週期與敏感度)", expanded=False):
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
        render_metric_card("冠軍實賺獲利金額", f"{curr_symbol} {champion_profit:+,.0f}", "實收淨利潤 (已扣成本)", "positive" if champion_profit >= 0 else "negative")
    with c2:
        render_metric_card("冠軍最終總資產", f"{curr_symbol} {champion_final:,.0f}", "期末帳戶總值", "neutral")
    with c3:
        render_metric_card("冠軍總報酬率", f"{df_results.iloc[0]['_raw_roi']:+.2%}", "基於累積自付本金", "positive" if champion_profit >= 0 else "negative")
    with c4:
        render_metric_card("冠軍最大歷史回撤 (MDD)", f"{champion_mdd:.2%}", "策略承受的最大跌幅痛苦", "negative" if champion_mdd < -0.2 else "neutral")

    st.subheader("📋 策略對抗排列表")
    display_cols = [
        "策略名稱", "實賺獲利金額", "最終總資產", "累積投入本金", 
        "淨獲利報酬率", "TWR 總報酬率", "CAGR 年化報酬", "最大歷史回撤 (MDD)", "累積交易次數"
    ]
    display_df = df_results[display_cols].copy()
    display_df.loc[0, "策略名稱"] = f"👑 {champion_name} (冠軍)"
    st.dataframe(display_df.set_index("策略名稱"), use_container_width=True)

    st.subheader("📈 策略資產成長曲線對比 ($)")
    st.caption("觀察各策略在不同歷史時期的資產累積速度。虛線代表你總共掏出來的自付本金。")
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

    # 逐筆成交 / 扣款購買歷史帳本
    st.subheader("📖 策略成交明細帳本 (Trade / Purchase Log)")
    inspect_strat = st.selectbox(
        "選擇要檢視詳細交易/購買紀錄的策略：",
        selected_strats,
        key="select_inspect_strat",
        help="定投策略會顯示逐筆扣款紀錄；擇時買賣策略會顯示每筆進出場勝率與持倉天數。"
    )

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
                render_metric_card("盈虧比 (Profit Factor)", summary["盈虧比 (Profit Factor)"], "總賺金額 ÷ 總賠金額", "positive" if summary["_raw_pf"] >= 1.5 else "neutral")
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

    # 通用型 2D 矩陣尋優面板 (支援 SMA, MACD, RSI, KD, Dip DCA)
    st.markdown("---")
    st.subheader("🔥 策略深度尋優：2D 參數矩陣熱力圖 (尋找穩健參數高原)")
    st.caption("自動測試雙變數的所有排列組合，以顏色深淺呈現獲利空間，避開過度擬合的「孤島陷阱」。")

    with st.expander("🚀 點擊展開「通用 2D 參數高原尋優」控制台", expanded=False):
        st.info(
            """
            🧭 **量化熱力圖判讀守則：**
            * 🟢 **參數高原（推薦採用）**：最佳參數周遭同樣是一大片綠色，代表即便市場節奏微變，策略依然具備高容錯率與穩健獲利能力。
            * 🔴 **孤島陷阱（嚴禁採用）**：僅單一綠點獲利極高，但周圍全是一片暗紅或虧損，代表純屬歷史雜訊巧合，實盤極易失效。
            """
        )

        opt_strategy = st.selectbox(
            "選擇要執行 2D 參數尋優的策略：",
            options=[
                "SMA 雙均線交叉 (快線週期 × 慢線週期)",
                "MACD 柱狀圖/交叉 (EMA快線 × EMA慢線)",
                "RSI 超買超賣 (超賣買入門檻 × 超買賣部門檻)",
                "KD 隨機指標 (超賣買入K × 超買賣出K)",
                "跌破均線且創新低買入 (基準均線 × 加碼間隔天數)"
            ],
            index=0
        )

        c_opt1, c_opt2, c_opt3 = st.columns(3)

        if "SMA 雙均線" in opt_strategy:
            with c_opt1:
                fast_step = st.selectbox("快線步長 (Step)", [1, 2, 3], index=1, help="測試 5, 7, 9 ... 29 週期")
            with c_opt2:
                slow_step = st.selectbox("慢線步長 (Step)", [2, 5, 10], index=1, help="測試 30, 35, 40 ... 120 週期")
            with c_opt3:
                opt_tf = st.selectbox("計算週期", ["日線 (Daily)", "週線 (Weekly)"], index=0)

        elif "MACD" in opt_strategy:
            with c_opt1:
                fast_step = st.selectbox("快線 EMA 步長", [1, 2], index=0, help="測試 5 至 20 週期")
            with c_opt2:
                slow_step = st.selectbox("慢線 EMA 步長", [1, 2, 5], index=1, help="測試 20 至 60 週期")
            with c_opt3:
                opt_tf = st.selectbox("計算週期", ["日線 (Daily)", "週線 (Weekly)"], index=0)

        elif "RSI" in opt_strategy:
            with c_opt1:
                rsi_p = st.number_input("RSI 基準週期", value=14, min_value=5, max_value=30, step=1)
            with c_opt2:
                step_val = st.selectbox("門檻掃描步長", [1, 2, 5], index=1, help="測試超賣(15~40) 與 超買(60~85)")
            with c_opt3:
                opt_tf = st.selectbox("計算週期", ["日線 (Daily)", "週線 (Weekly)"], index=0)

        elif "KD" in opt_strategy:
            with c_opt1:
                kd_p = st.number_input("KD 基準週期", value=9, min_value=3, max_value=20, step=1)
            with c_opt2:
                step_val = st.selectbox("K值掃描步長", [1, 2, 5], index=1, help="測試買入K(15~35) 與 賣出K(65~85)")
            with c_opt3:
                opt_tf = st.selectbox("計算週期", ["日線 (Daily)", "週線 (Weekly)"], index=0)

        elif "跌破均線" in opt_strategy:
            with c_opt1:
                dip_amt = st.number_input("每次加碼買入金額", value=50000.0, step=10000.0)
            with c_opt2:
                sma_step = st.selectbox("均線掃描步長", [5, 10, 20], index=1, help="測試均線 20 至 120 週期")
            with c_opt3:
                opt_tf = st.selectbox("計算週期", ["日線 (Daily)", "週線 (Weekly)"], index=0)

        if st.button("⚡ 開始執行 2D 矩陣尋優運算", key="btn_run_generic_grid"):
            with st.spinner("正在極速運算排列組合之收益矩陣..."):
                if "SMA 雙均線" in opt_strategy:
                    heatmap_df, flat_df, best_p = GridOptimizer.run_ma_grid(
                        df_target, fast_range=(5, 31, fast_step), slow_range=(30, 121, slow_step),
                        timeframe=opt_tf, initial_capital=initial_capital, monthly_inflow=monthly_inflow,
                        cost_struct=cost_struct
                    )
                    x_lbl, y_lbl = "快線週期 (Fast Period)", "慢線週期 (Slow Period)"
                    rev_y = True

                elif "MACD" in opt_strategy:
                    heatmap_df, flat_df, best_p = GridOptimizer.run_macd_grid(
                        df_target, fast_range=(5, 21, fast_step), slow_range=(20, 61, slow_step),
                        timeframe=opt_tf, initial_capital=initial_capital, monthly_inflow=monthly_inflow,
                        cost_struct=cost_struct
                    )
                    x_lbl, y_lbl = "EMA 快線週期", "EMA 慢線週期"
                    rev_y = True

                elif "RSI" in opt_strategy:
                    heatmap_df, flat_df, best_p = GridOptimizer.run_rsi_grid(
                        df_target, period=rsi_p, buy_range=(15, 41, step_val), sell_range=(60, 86, step_val),
                        timeframe=opt_tf, initial_capital=initial_capital, monthly_inflow=monthly_inflow,
                        cost_struct=cost_struct
                    )
                    x_lbl, y_lbl = "超賣買入門檻 (RSI)", "超買賣部門檻 (RSI)"
                    rev_y = False

                elif "KD" in opt_strategy:
                    heatmap_df, flat_df, best_p = GridOptimizer.run_kd_grid(
                        df_target, period=kd_p, buy_range=(15, 36, step_val), sell_range=(65, 86, step_val),
                        timeframe=opt_tf, initial_capital=initial_capital, monthly_inflow=monthly_inflow,
                        cost_struct=cost_struct
                    )
                    x_lbl, y_lbl = "超賣買入門檻 (K值)", "超買賣部門檻 (K值)"
                    rev_y = False

                elif "跌破均線" in opt_strategy:
                    heatmap_df, flat_df, best_p = GridOptimizer.run_dip_grid(
                        df_target, sma_range=(20, 121, sma_step), interval_range=(1, 11, 1),
                        buy_amount=dip_amt, timeframe=opt_tf, initial_capital=initial_capital,
                        monthly_inflow=monthly_inflow, cost_struct=cost_struct
                    )
                    x_lbl, y_lbl = "基準均線週期 (SMA)", "加碼冷卻天數 (Days)"
                    rev_y = False

            if not flat_df.empty:
                param_cols = [c for c in flat_df.columns if c.startswith("參數")]
                best_info = " / ".join([f"{col}: {best_p.get(col)}" for col in param_cols])
                st.success(
                    f"🎉 **尋優運算完成！** 歷史最佳參數組合：**{best_info}**，"
                    f"總報酬率達 **{best_p.get('總報酬率 (%)'):+.2f}%**（實賺 {curr_symbol} {best_p.get('實賺獲利金額'):,.0f}，累計交易 {best_p.get('交易次數')} 次）！"
                )

                st.caption(f"📊 熱力圖（X軸：{x_lbl}，Y軸：{y_lbl}）：顏色越綠代表總報酬率越高")
                render_parameter_heatmap(heatmap_df, curr_symbol, x_title=x_lbl, y_title=y_lbl, reverse_y=rev_y)

                csv_grid = flat_df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label=f"📥 下載【{opt_strategy.split(' ')[0]}】尋優完整排列表 (CSV)",
                    data=csv_grid,
                    file_name=f"{target_ticker}_{opt_strategy.split(' ')[0]}_grid_results.csv",
                    mime="text/csv",
                    key="btn_download_generic_grid"
                )
