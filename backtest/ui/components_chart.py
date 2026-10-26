"""
高級圖表渲染模組 (Chart Components Module - 完整圖表庫)
包含：
1. TradingView 質感 K 線買賣箭頭標記圖
2. 2D 參數高原尋優熱力圖 (zmid=0 鮮明對比)
3. 經典「馬科維茨效率前緣」彩虹散佈圖 (Efficient Frontier)
"""
from typing import Optional, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def render_candlestick_with_markers(
    df: pd.DataFrame,
    pos_series: pd.Series,
    ticker_name: str,
    curr_symbol: str = "NT$"
):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="K線行情",
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149"
    ))

    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []

    pos_vals = pos_series.reindex(df.index).fillna(0).astype(int).values
    low_vals = df["low"].values
    high_vals = df["high"].values
    dates = df.index

    for i in range(1, len(df)):
        if pos_vals[i] == 1 and pos_vals[i - 1] == 0:
            buy_dates.append(dates[i])
            buy_prices.append(low_vals[i] * 0.985)
        elif pos_vals[i] == 0 and pos_vals[i - 1] == 1:
            sell_dates.append(dates[i])
            sell_prices.append(high_vals[i] * 1.015)

    if buy_dates:
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode="markers+text",
            name="買入進場 (BUY)",
            marker=dict(symbol="triangle-up", size=13, color="#3fb950"),
            text=["▲ BUY"] * len(buy_dates),
            textposition="bottom center",
            textfont=dict(color="#3fb950", size=10)
        ))

    if sell_dates:
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode="markers+text",
            name="賣出出場 (SELL)",
            marker=dict(symbol="triangle-down", size=13, color="#f85149"),
            text=["▼ SELL"] * len(sell_dates),
            textposition="top center",
            textfont=dict(color="#f85149", size=10)
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title=f"價格 ({curr_symbol})"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_parameter_heatmap(
    heatmap_df: pd.DataFrame,
    curr_symbol: str = "NT$"
):
    vals = heatmap_df.values.flatten()
    valid_vals = vals[~np.isnan(vals)]
    max_val = float(np.max(valid_vals)) if len(valid_vals) > 0 else 10.0
    min_val = float(np.min(valid_vals)) if len(valid_vals) > 0 else -10.0

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        zmid=0.0,
        zmin=min(min_val, -5.0),
        zmax=max(max_val, 15.0),
        colorscale=[
            [0.0, "#d73a49"],
            [0.4, "#f85149"],
            [0.5, "#21262d"],
            [0.6, "#2ea44f"],
            [1.0, "#3fb950"]
        ],
        colorbar=dict(title="總報酬率 (%)"),
        hovertemplate="快線週期: %{x}<br>慢線週期: %{y}<br>總報酬率: %{z:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="快線週期 (Fast Period)", showgrid=False),
        yaxis=dict(title="慢線週期 (Slow Period)", showgrid=False, autorange="reversed")
    )
    st.plotly_chart(fig, use_container_width=True)


def render_efficient_frontier(
    frontier_df: pd.DataFrame,
    summary_stats: Dict[str, Any]
):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frontier_df["volatility"],
        y=frontier_df["return"],
        mode="markers",
        marker=dict(
            size=5,
            color=frontier_df["sharpe"],
            colorscale="Viridis",
            colorbar=dict(title="夏普比率 (Sharpe)"),
            opacity=0.75
        ),
        name="可能投資組合",
        hovertemplate="年化波動度: %{x:.2f}%<br>年化預期報酬: %{y:.2f}%<br>夏普比率: %{marker.color:.2f}<extra></extra>"
    ))

    ms = summary_stats.get("max_sharpe", {})
    if ms:
        fig.add_trace(go.Scatter(
            x=[ms["volatility"] * 100.0],
            y=[ms["return"] * 100.0],
            mode="markers+text",
            name="最高夏普組合 (性價比冠軍)",
            marker=dict(symbol="star", size=18, color="#f1e05a", line=dict(color="#ffffff", width=1.5)),
            text=["⭐ 最高夏普點"],
            textposition="top center",
            textfont=dict(color="#f1e05a", size=12, family="sans-serif")
        ))

    mv = summary_stats.get("min_vol", {})
    if mv:
        fig.add_trace(go.Scatter(
            x=[mv["volatility"] * 100.0],
            y=[mv["return"] * 100.0],
            mode="markers+text",
            name="最小波動組合 (極致防禦)",
            marker=dict(symbol="diamond", size=14, color="#58a6ff", line=dict(color="#ffffff", width=1.5)),
            text=["🛡️ 最小波動點"],
            textposition="bottom center",
            textfont=dict(color="#58a6ff", size=12, family="sans-serif")
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        hovermode="closest",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=True, gridcolor="#21262d", title="年化波動度 (%) [越左越穩]"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title="年化預期報酬率 (%) [越上越賺]"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
