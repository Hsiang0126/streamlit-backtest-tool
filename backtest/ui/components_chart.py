"""
高級圖表渲染模組 (Chart Components Module)
統一深色金融終端美學與互動體驗，包含：
1. 資產成長曲線對比圖 (支援線性 / 對數座標切換與時間區間快選)
2. 水下回撤對比圖 (Underwater Drawdown Chart)
3. 年度 × 月度報酬熱力圖 (Monthly Returns Heatmap)
4. TradingView 質感 K 線買賣箭頭標記圖 (支援疊加均線與還原區間對齊)
5. 通用 2D 參數高原尋優熱力圖 (自動標註最佳參數星號)
6. 馬科維茨效率前緣彩虹散佈圖 (含最高夏普、最小波動與使用者目前組合標記)
7. 多資產相關係數熱力圖 (Correlation Heatmap)
8. 多資產動態權重漂移堆疊面積圖 (Weight Drift & Rebalance Markers)
"""
from typing import Optional, Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


PALETTE = [
    "#38bdf8",
    "#22c55e",
    "#f59e0b",
    "#a78bfa",
    "#fb7185",
    "#2dd4bf",
    "#f97316",
    "#60a5fa",
]


def _apply_terminal_layout(
    fig: go.Figure,
    y_title: str = "",
    x_title: str = "",
    height: int = 420,
    hovermode: str = "x unified",
    show_range_buttons: bool = False,
) -> go.Figure:
    xaxis_cfg: Dict[str, Any] = dict(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.10)",
        zeroline=False,
        title=x_title or None,
    )
    if show_range_buttons:
        xaxis_cfg["rangeselector"] = dict(
            bgcolor="#131a29",
            activecolor="#1e293b",
            bordercolor="rgba(148, 163, 184, 0.2)",
            font=dict(color="#e2e8f0", size=11),
            buttons=[
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(step="all", label="ALL"),
            ],
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#111827",
        hovermode=hovermode,
        height=height,
        margin=dict(l=16, r=16, t=36, b=16),
        xaxis=xaxis_cfg,
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.10)",
            zeroline=False,
            title=y_title or None,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        font=dict(color="#cbd5e1", size=12),
    )
    return fig


def render_equity_comparison_chart(
    curves_dict: Dict[str, pd.DataFrame],
    curr_symbol: str = "NT$",
    champion_name: Optional[str] = None,
    log_scale: bool = False,
    height: int = 430,
):
    fig = go.Figure()
    for idx, (name, curve) in enumerate(curves_dict.items()):
        if curve is None or curve.empty or "portfolio_value" not in curve.columns:
            continue
        is_champ = champion_name is not None and name == champion_name
        color = PALETTE[idx % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve["portfolio_value"],
                name=f"👑 {name}" if is_champ else name,
                line=dict(color=color, width=3.0 if is_champ else 1.8),
                hovertemplate="日期: %{x|%Y-%m-%d}<br>總資產: " + curr_symbol + " %{y:,.0f}<extra></extra>",
            )
        )

    first_curve = next((c for c in curves_dict.values() if c is not None and not c.empty), None)
    if first_curve is not None and "total_invested" in first_curve.columns:
        fig.add_trace(
            go.Scatter(
                x=first_curve.index,
                y=first_curve["total_invested"],
                name="累計自付本金",
                line=dict(color="#64748b", dash="dot", width=1.6),
                hovertemplate="日期: %{x|%Y-%m-%d}<br>投入本金: " + curr_symbol + " %{y:,.0f}<extra></extra>",
            )
        )

    _apply_terminal_layout(
        fig,
        y_title=f"資產價值 ({curr_symbol})",
        height=height,
        show_range_buttons=True,
    )
    if log_scale:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch")


def render_drawdown_chart(
    drawdowns_dict: Dict[str, pd.Series],
    champion_name: Optional[str] = None,
    height: int = 340,
):
    fig = go.Figure()
    for idx, (name, dd) in enumerate(drawdowns_dict.items()):
        if dd is None or dd.empty:
            continue
        is_champ = champion_name is not None and name == champion_name
        color = PALETTE[idx % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values * 100.0,
                name=f"👑 {name}" if is_champ else name,
                line=dict(color=color, width=2.4 if is_champ else 1.5),
                fill="tozeroy" if len(drawdowns_dict) <= 2 or is_champ else None,
                fillcolor="rgba(239, 68, 68, 0.12)" if is_champ or len(drawdowns_dict) == 1 else "rgba(56, 189, 248, 0.08)",
                hovertemplate="日期: %{x|%Y-%m-%d}<br>回撤幅度: %{y:.2f}%<extra></extra>",
            )
        )

    _apply_terminal_layout(fig, y_title="歷史回撤 (%)", height=height, show_range_buttons=True)
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, width="stretch")


def render_monthly_returns_heatmap(monthly_df: pd.DataFrame, title: str = ""):
    if monthly_df is None or monthly_df.empty:
        st.info("區間天數較短，尚無完整月度報酬矩陣。")
        return

    z_pct = monthly_df.to_numpy(dtype=float) * 100.0
    text_grid = [
        [f"{v:+.1f}%" if np.isfinite(v) else "" for v in row]
        for row in z_pct
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_pct,
            x=list(monthly_df.columns),
            y=list(monthly_df.index),
            zmid=0.0,
            zmin=-15.0,
            zmax=15.0,
            text=text_grid,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale=[
                [0.0, "#dc2626"],
                [0.35, "#f87171"],
                [0.5, "#1e293b"],
                [0.65, "#4ade80"],
                [1.0, "#16a34a"],
            ],
            colorbar=dict(title="報酬 (%)"),
            hovertemplate="%{y} %{x}<br>報酬率: %{z:+.2f}%<extra></extra>",
        )
    )
    _apply_terminal_layout(fig, height=max(260, 42 * len(monthly_df) + 110), hovermode="closest")
    fig.update_layout(title=dict(text=title, font=dict(size=13)) if title else None)
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(side="top")
    st.plotly_chart(fig, width="stretch")


def render_candlestick_with_markers(
    df: pd.DataFrame,
    pos_series: pd.Series,
    ticker_name: str,
    curr_symbol: str = "NT$",
    ma_overlays: Optional[Dict[str, pd.Series]] = None,
):
    if "user_start_idx" in df.columns:
        plot_df = df[df["user_start_idx"] == True]
        if plot_df.empty:
            plot_df = df
    else:
        plot_df = df

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name=f"{ticker_name} K線",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        )
    )

    if ma_overlays:
        for idx, (ma_name, ma_s) in enumerate(ma_overlays.items()):
            aligned_ma = ma_s.reindex(plot_df.index)
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=aligned_ma,
                    name=ma_name,
                    line=dict(color=PALETTE[(idx + 2) % len(PALETTE)], width=1.4),
                    opacity=0.85,
                )
            )

    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []

    pos_vals = pos_series.reindex(plot_df.index).fillna(0).astype(int).values
    low_vals = plot_df["low"].values
    high_vals = plot_df["high"].values
    dates = plot_df.index

    if len(plot_df) > 0 and pos_vals[0] == 1:
        buy_dates.append(dates[0])
        buy_prices.append(low_vals[0] * 0.985)

    for i in range(1, len(plot_df)):
        if pos_vals[i] == 1 and pos_vals[i - 1] == 0:
            buy_dates.append(dates[i])
            buy_prices.append(low_vals[i] * 0.985)
        elif pos_vals[i] == 0 and pos_vals[i - 1] == 1:
            sell_dates.append(dates[i])
            sell_prices.append(high_vals[i] * 1.015)

    if buy_dates:
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers",
                name="買入進場 (BUY)",
                marker=dict(symbol="triangle-up", size=12, color="#22c55e", line=dict(color="#ffffff", width=0.8)),
                hovertemplate="買進 %{x|%Y-%m-%d}<extra></extra>",
            )
        )

    if sell_dates:
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode="markers",
                name="賣出出場 (SELL)",
                marker=dict(symbol="triangle-down", size=12, color="#ef4444", line=dict(color="#ffffff", width=0.8)),
                hovertemplate="賣出 %{x|%Y-%m-%d}<extra></extra>",
            )
        )

    _apply_terminal_layout(fig, y_title=f"原生價格 ({curr_symbol})", height=460, show_range_buttons=True)
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch")


def render_parameter_heatmap(
    heatmap_df: pd.DataFrame,
    curr_symbol: str = "NT$",
    x_title: str = "參數 X",
    y_title: str = "參數 Y",
    reverse_y: bool = True,
    metric_label: str = "總報酬率 (%)",
    best_x: Optional[Any] = None,
    best_y: Optional[Any] = None,
):
    vals = heatmap_df.values.flatten()
    valid_vals = vals[~np.isnan(vals)]
    max_val = float(np.max(valid_vals)) if len(valid_vals) > 0 else 10.0
    min_val = float(np.min(valid_vals)) if len(valid_vals) > 0 else -10.0
    z_mid = 0.0 if min_val < 0 else (min_val + max_val) / 2.0

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            zmid=z_mid,
            zmin=min_val,
            zmax=max(max_val, min_val + 1e-4),
            colorscale=[
                [0.0, "#dc2626"],
                [0.35, "#f87171"],
                [0.5, "#1e293b"],
                [0.65, "#4ade80"],
                [1.0, "#16a34a"],
            ],
            colorbar=dict(title=metric_label),
            hovertemplate=f"{x_title}: %{{x}}<br>{y_title}: %{{y}}<br>{metric_label}: %{{z:.2f}}<extra></extra>",
        )
    )

    if best_x is not None and best_y is not None:
        fig.add_trace(
            go.Scatter(
                x=[best_x],
                y=[best_y],
                mode="markers+text",
                name="最佳參數點",
                marker=dict(symbol="star", size=16, color="#facc15", line=dict(color="#0b0f19", width=1.5)),
                text=["★ 最佳"],
                textposition="top center",
                textfont=dict(color="#facc15", size=11),
                hoverinfo="skip",
            )
        )

    _apply_terminal_layout(fig, x_title=x_title, y_title=y_title, height=440, hovermode="closest")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed" if reverse_y else True)
    st.plotly_chart(fig, width="stretch")


def render_efficient_frontier(
    frontier_df: pd.DataFrame,
    summary_stats: Dict[str, Any],
    current_point: Optional[Dict[str, float]] = None,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frontier_df["volatility"],
            y=frontier_df["return"],
            mode="markers",
            marker=dict(
                size=5,
                color=frontier_df["sharpe"],
                colorscale="Viridis",
                colorbar=dict(title="夏普比率 (Sharpe)"),
                opacity=0.72,
            ),
            name="隨機模擬投資組合",
            hovertemplate="年化波動度: %{x:.2f}%<br>年化預期報酬: %{y:.2f}%<br>夏普比率: %{marker.color:.2f}<extra></extra>",
        )
    )

    # 標示各單一資產座標
    asset_stats = summary_stats.get("asset_stats", {})
    if asset_stats:
        fig.add_trace(
            go.Scatter(
                x=[v["volatility"] * 100.0 for v in asset_stats.values()],
                y=[v["return"] * 100.0 for v in asset_stats.values()],
                mode="markers+text",
                name="單一資產",
                marker=dict(symbol="circle-open", size=10, color="#cbd5e1", line=dict(width=2)),
                text=list(asset_stats.keys()),
                textposition="top right",
                textfont=dict(color="#94a3b8", size=10),
                hovertemplate="%{text}<br>波動度: %{x:.2f}%<br>年化報酬: %{y:.2f}%<extra></extra>",
            )
        )

    ms = summary_stats.get("max_sharpe", {})
    if ms:
        fig.add_trace(
            go.Scatter(
                x=[ms["volatility"] * 100.0],
                y=[ms["return"] * 100.0],
                mode="markers+text",
                name="最高夏普組合",
                marker=dict(symbol="star", size=17, color="#facc15", line=dict(color="#ffffff", width=1.2)),
                text=["⭐ 最高夏普"],
                textposition="top center",
                textfont=dict(color="#facc15", size=11),
            )
        )

    mv = summary_stats.get("min_vol", {})
    if mv:
        fig.add_trace(
            go.Scatter(
                x=[mv["volatility"] * 100.0],
                y=[mv["return"] * 100.0],
                mode="markers+text",
                name="最小波動組合",
                marker=dict(symbol="diamond", size=13, color="#38bdf8", line=dict(color="#ffffff", width=1.2)),
                text=["🛡️ 最小波動"],
                textposition="bottom center",
                textfont=dict(color="#38bdf8", size=11),
            )
        )

    if current_point:
        fig.add_trace(
            go.Scatter(
                x=[current_point["volatility"] * 100.0],
                y=[current_point["return"] * 100.0],
                mode="markers+text",
                name="目前配置組合",
                marker=dict(symbol="x", size=14, color="#fb7185", line=dict(color="#ffffff", width=2)),
                text=[f"📍 目前組合 (Sharpe {current_point['sharpe']:.2f})"],
                textposition="top center",
                textfont=dict(color="#fb7185", size=11),
            )
        )

    _apply_terminal_layout(
        fig,
        x_title="年化波動度 (%) [越左越穩]",
        y_title="年化預期報酬率 (%) [越上越賺]",
        height=440,
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch")


def render_correlation_heatmap(corr_matrix: pd.DataFrame):
    if corr_matrix is None or corr_matrix.empty:
        st.info("尚無足夠資料計算資產相關性。")
        return

    text_vals = [[f"{val:.2f}" for val in row] for row in corr_matrix.values]
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            zmin=-1.0,
            zmax=1.0,
            zmid=0.0,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorscale=[
                [0.0, "#38bdf8"],
                [0.5, "#1e293b"],
                [1.0, "#ef4444"],
            ],
            colorbar=dict(title="相關係數"),
            hovertemplate="%{y} vs %{x}<br>相關係數: %{z:.3f}<extra></extra>",
        )
    )
    _apply_terminal_layout(fig, height=360, hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")


def render_weight_drift_chart(
    res_df: pd.DataFrame,
    tickers: List[str],
    rebalance_dates: Optional[List[pd.Timestamp]] = None,
    has_cash_reserve: bool = False,
):
    weight_cols = [f"weight_{t}" for t in tickers if f"weight_{t}" in res_df.columns]
    if not weight_cols:
        return

    fig = go.Figure()
    for idx, t in enumerate(tickers):
        col = f"weight_{t}"
        if col in res_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=res_df.index,
                    y=res_df[col] * 100.0,
                    name=t,
                    stackgroup="weights",
                    mode="lines",
                    line=dict(width=0.8, color=PALETTE[idx % len(PALETTE)]),
                    hovertemplate=f"{t}: %{{y:.1f}}%<extra></extra>",
                )
            )

    if has_cash_reserve or (
        "weight_CASH" in res_df.columns and float(res_df["weight_CASH"].max()) > 0.01
    ):
        if "weight_CASH" in res_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=res_df.index,
                    y=res_df["weight_CASH"] * 100.0,
                    name="現金活存儲備",
                    stackgroup="weights",
                    mode="lines",
                    line=dict(width=0.8, color="#64748b"),
                    hovertemplate="現金活存: %{y:.1f}%<extra></extra>",
                )
            )

    if rebalance_dates:
        for dt in rebalance_dates[:40]:
            fig.add_vline(
                x=pd.Timestamp(dt).strftime("%Y-%m-%d"),
                line_width=1,
                line_dash="dot",
                line_color="rgba(250, 204, 21, 0.45)",
            )

    _apply_terminal_layout(fig, y_title="權重佔比 (%)", height=360, show_range_buttons=True)
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    st.plotly_chart(fig, width="stretch")
