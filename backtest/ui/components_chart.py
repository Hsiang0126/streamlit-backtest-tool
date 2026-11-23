"""
簡約沉穩淺色圖表渲染模組 (Editorial Chart Components Module)
統一淺色編輯室美學與跨裝置互動體驗，包含：
1. 資產成長曲線對比圖 (支援線性 / 對數座標切換與長區間快選)
2. 水下回撤對比圖 (Underwater Drawdown Chart)
3. 年度 × 月度報酬熱力圖 (Monthly Returns Heatmap)
4. K 線買賣箭頭標記圖 (支援均線疊加與長區間週 K 切換)
5. 通用 2D 參數高原尋優熱力圖 (自動標註最佳參數星號)
6. 馬科維茨效率前緣散佈圖 (含最高夏普、最小波動與使用者目前組合標記)
7. 多資產相關係數熱力圖 (Correlation Heatmap)
8. 多資產動態權重漂移堆疊面積圖 (Weight Drift & Rebalance Markers)
"""
import html
from typing import Optional, Dict, Any, List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


PALETTE = [
    "#1E3A5F",
    "#0F766E",
    "#B45309",
    "#4F46E5",
    "#B91C1C",
    "#2563EB",
    "#4D7C0F",
    "#64748B",
]

PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}
_PLOTLY_CONFIG = PLOTLY_CONFIG


def _apply_terminal_layout(
    fig: go.Figure,
    y_title: str = "",
    x_title: str = "",
    height: int = 420,
    hovermode: str = "x unified",
    show_range_buttons: bool = False,
    show_legend_bottom: bool = True,
) -> go.Figure:
    xaxis_cfg: Dict[str, Any] = dict(
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.04)",
        linecolor="#CBD5E1",
        tickfont=dict(color="#475569", size=11),
        zeroline=False,
        automargin=True,
        title=x_title or None,
    )
    if show_range_buttons:
        xaxis_cfg["rangeselector"] = dict(
            x=0,
            y=1.06,
            xanchor="left",
            yanchor="bottom",
            bgcolor="#F1F4F8",
            activecolor="#DCE3EC",
            bordercolor="#CBD5E1",
            borderwidth=1,
            font=dict(color="#1E293B", size=11),
            buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(step="all", label="ALL"),
            ],
        )

    margin_cfg = dict(
        l=12,
        r=12,
        t=48 if show_range_buttons else 24,
        b=72 if show_legend_bottom else 28,
    )

    layout_kwargs: Dict[str, Any] = dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        hovermode=hovermode,
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#CBD5E1",
            font=dict(color="#0F172A", size=12),
        ),
        height=height,
        margin=margin_cfg,
        xaxis=xaxis_cfg,
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(15, 23, 42, 0.08)",
            zeroline=True,
            zerolinecolor="#CBD5E1",
            tickfont=dict(color="#475569", size=11),
            automargin=True,
            title=y_title or None,
        ),
        font=dict(
            color="#334155",
            size=12,
            family="Inter, Noto Sans TC, -apple-system, BlinkMacSystemFont, sans-serif",
        ),
    )

    if show_legend_bottom:
        layout_kwargs["legend"] = dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#334155"),
        )

    fig.update_layout(**layout_kwargs)
    return fig


def render_equity_comparison_chart(
    curves_dict: Dict[str, pd.DataFrame],
    curr_symbol: str = "NT$",
    champion_name: Optional[str] = None,
    log_scale: bool = False,
    height: int = 440,
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
                name=f"{name}（領先）" if is_champ else name,
                line=dict(color=color, width=2.8 if is_champ else 1.8),
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
                line=dict(color="#94A3B8", dash="dot", width=1.6),
                hovertemplate="日期: %{x|%Y-%m-%d}<br>投入本金: " + curr_symbol + " %{y:,.0f}<extra></extra>",
            )
        )

    _apply_terminal_layout(
        fig,
        y_title=f"資產價值 ({curr_symbol})",
        height=height,
        show_range_buttons=True,
        show_legend_bottom=True,
    )
    if log_scale:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)


def render_drawdown_chart(
    drawdowns_dict: Dict[str, pd.Series],
    champion_name: Optional[str] = None,
    height: int = 350,
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
                name=f"{name}（領先）" if is_champ else name,
                line=dict(color=color, width=2.3 if is_champ else 1.5),
                fill="tozeroy" if len(drawdowns_dict) <= 2 or is_champ else None,
                fillcolor="rgba(185, 28, 28, 0.10)" if is_champ or len(drawdowns_dict) == 1 else "rgba(30, 58, 95, 0.08)",
                hovertemplate="日期: %{x|%Y-%m-%d}<br>回撤幅度: %{y:.2f}%<extra></extra>",
            )
        )

    _apply_terminal_layout(
        fig,
        y_title="歷史回撤 (%)",
        height=height,
        show_range_buttons=True,
        show_legend_bottom=True,
    )
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)


_MONTHLY_COLOR_STOPS = [
    (-15.0, (190, 18, 60)),
    (-6.6, (252, 165, 165)),
    (0.0, (248, 250, 252)),
    (6.6, (134, 239, 172)),
    (15.0, (21, 128, 61)),
]


def _monthly_cell_colors(val: float) -> Tuple[str, str]:
    """依月報酬百分比線性內插背景色，並依 WCAG 相對亮度自動選取高對比前景文字色。"""
    clamped = max(-15.0, min(15.0, float(val)))
    rgb = _MONTHLY_COLOR_STOPS[-1][1]
    for (v0, c0), (v1, c1) in zip(_MONTHLY_COLOR_STOPS, _MONTHLY_COLOR_STOPS[1:]):
        if clamped <= v1:
            t = 0.0 if v1 == v0 else (clamped - v0) / (v1 - v0)
            rgb = tuple(int(round(c0[i] + (c1[i] - c0[i]) * t)) for i in range(3))
            break

    def _lin(ch: int) -> float:
        s = ch / 255.0
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _lin(rgb[0]) + 0.7152 * _lin(rgb[1]) + 0.0722 * _lin(rgb[2])
    fg = "#FFFFFF" if (1.05 / (lum + 0.05)) >= ((lum + 0.05) / 0.058) else "#0F172A"
    bg = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
    return bg, fg


def _build_monthly_heatmap_html(monthly_df: pd.DataFrame) -> str:
    """建構無 4 空格縮排之響應式年度月報酬 HTML 表格（凍結年度欄、手機可橫滑）。"""
    cols = list(monthly_df.columns)
    head_cells = ['<th scope="col" class="hm-year">年度</th>']
    for col in cols:
        cls = ' class="hm-annual"' if str(col) == "全年" else ""
        head_cells.append(f'<th scope="col"{cls}>{html.escape(str(col))}</th>')

    body_rows: List[str] = []
    for yr, row in monthly_df.iterrows():
        yr_str = str(yr)
        cells = [f'<th scope="row" class="hm-year">{html.escape(yr_str)}</th>']
        for col in cols:
            col_str = str(col)
            is_annual = col_str == "全年"
            raw_val = row[col]
            try:
                num_val = float(raw_val)
            except (TypeError, ValueError):
                num_val = float("nan")
            if np.isfinite(num_val):
                pct = num_val * 100.0
                bg, fg = _monthly_cell_colors(pct)
                cls = ' class="hm-annual"' if is_annual else ""
                tip = html.escape(f"{yr_str} {col_str}：{pct:+.2f}%")
                cells.append(
                    f'<td{cls} style="background-color:{bg};color:{fg};" title="{tip}">{pct:+.1f}%</td>'
                )
            else:
                empty_cls = "hm-empty hm-annual" if is_annual else "hm-empty"
                cells.append(f'<td class="{empty_cls}">–</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    legend_html = (
        '<div class="hm-legend">'
        '<span class="hm-legend-scale">'
        "<span>報酬 (%)</span>"
        "<span>-15%</span>"
        '<span class="hm-legend-bar" aria-hidden="true"></span>'
        "<span>+15%</span>"
        "<span>(中立 0%)</span>"
        "</span>"
        '<span class="hm-swipe-hint">← 左右滑動檢視各月 →</span>'
        "</div>"
    )
    table_html = (
        '<div class="heatmap-scroll">'
        '<table class="hm-table">'
        f"<thead><tr>{''.join(head_cells)}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )
    return f'<div class="hm-wrapper">{legend_html}{table_html}</div>'


def render_monthly_returns_heatmap(monthly_df: pd.DataFrame, title: str = ""):
    if monthly_df is None or monthly_df.empty:
        st.info("區間天數較短，尚無完整月度報酬矩陣。")
        return

    if title:
        st.markdown(f"###### {title}")

    st.markdown(_build_monthly_heatmap_html(monthly_df), unsafe_allow_html=True)


def render_candlestick_with_markers(
    df: pd.DataFrame,
    pos_series: pd.Series,
    ticker_name: str,
    curr_symbol: str = "NT$",
    ma_overlays: Optional[Dict[str, pd.Series]] = None,
    resample_weekly: bool = False,
):
    if "user_start_idx" in df.columns:
        plot_df = df[df["user_start_idx"] == True]
        if plot_df.empty:
            plot_df = df
    else:
        plot_df = df

    # 先以日線訊號定位實際買賣日期與價位，確保切換週 K 線顯示時進出場點位不失真
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

    if resample_weekly and len(plot_df) > 10:
        candle_df = (
            plot_df[["open", "high", "low", "close"]]
            .resample("W-FRI")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna(subset=["open", "high", "low", "close"])
        )
    else:
        candle_df = plot_df

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=candle_df.index,
            open=candle_df["open"],
            high=candle_df["high"],
            low=candle_df["low"],
            close=candle_df["close"],
            name=f"{ticker_name} {'週K線' if resample_weekly else 'K線'}",
            increasing_line_color="#15803D",
            decreasing_line_color="#B91C1C",
        )
    )

    if ma_overlays:
        for idx, (ma_name, ma_s) in enumerate(ma_overlays.items()):
            aligned_ma = ma_s.reindex(candle_df.index).ffill() if resample_weekly else ma_s.reindex(candle_df.index)
            fig.add_trace(
                go.Scatter(
                    x=candle_df.index,
                    y=aligned_ma,
                    name=ma_name,
                    line=dict(color=PALETTE[(idx + 2) % len(PALETTE)], width=1.4),
                    opacity=0.85,
                )
            )

    if buy_dates:
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers",
                name="買入進場 (BUY)",
                marker=dict(symbol="triangle-up", size=11, color="#15803D", line=dict(color="#FFFFFF", width=0.9)),
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
                marker=dict(symbol="triangle-down", size=11, color="#B91C1C", line=dict(color="#FFFFFF", width=0.9)),
                hovertemplate="賣出 %{x|%Y-%m-%d}<extra></extra>",
            )
        )

    _apply_terminal_layout(
        fig,
        y_title=f"原生價格 ({curr_symbol})",
        height=470,
        show_range_buttons=True,
        show_legend_bottom=True,
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)


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
                [0.0, "#BE123C"],
                [0.28, "#FCA5A5"],
                [0.5, "#F8FAFC"],
                [0.72, "#86EFAC"],
                [1.0, "#15803D"],
            ],
            colorbar=dict(
                orientation="h",
                y=-0.26,
                yanchor="top",
                x=0.5,
                xanchor="center",
                len=0.92,
                thickness=12,
                title=dict(text=metric_label, side="top", font=dict(size=12)),
                tickfont=dict(size=12),
            ),
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
                marker=dict(symbol="star", size=15, color="#B45309", line=dict(color="#FFFFFF", width=1.4)),
                text=["最佳參數"],
                textposition="top center",
                textfont=dict(color="#92400E", size=12),
                cliponaxis=False,
                hoverinfo="skip",
            )
        )

    _apply_terminal_layout(
        fig,
        x_title=x_title,
        y_title=y_title,
        height=470,
        hovermode="closest",
        show_legend_bottom=False,
    )
    fig.update_layout(margin=dict(l=48, r=14, t=28, b=108))
    fig.update_xaxes(showgrid=False, tickfont=dict(color="#334155", size=12))
    fig.update_yaxes(showgrid=False, autorange="reversed" if reverse_y else True, tickfont=dict(color="#334155", size=12))
    st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)


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
                colorscale=[
                    [0.0, "#CBD5E1"],
                    [0.5, "#3B82F6"],
                    [1.0, "#0F766E"],
                ],
                colorbar=dict(title="夏普比率 (Sharpe)"),
                opacity=0.72,
            ),
            name="隨機模擬投資組合",
            hovertemplate="年化波動度: %{x:.2f}%<br>年化預期報酬: %{y:.2f}%<br>夏普比率: %{marker.color:.2f}<extra></extra>",
        )
    )

    asset_stats = summary_stats.get("asset_stats", {})
    if asset_stats:
        fig.add_trace(
            go.Scatter(
                x=[v["volatility"] * 100.0 for v in asset_stats.values()],
                y=[v["return"] * 100.0 for v in asset_stats.values()],
                mode="markers+text",
                name="單一資產",
                marker=dict(symbol="circle-open", size=9, color="#334155", line=dict(width=1.8)),
                text=list(asset_stats.keys()),
                textposition="top right",
                textfont=dict(color="#475569", size=10),
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
                marker=dict(symbol="star", size=16, color="#B45309", line=dict(color="#FFFFFF", width=1.2)),
                text=["最高夏普"],
                textposition="top center",
                textfont=dict(color="#92400E", size=11),
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
                marker=dict(symbol="diamond", size=12, color="#1E3A5F", line=dict(color="#FFFFFF", width=1.2)),
                text=["最小波動"],
                textposition="bottom center",
                textfont=dict(color="#1E3A5F", size=11),
            )
        )

    if current_point:
        fig.add_trace(
            go.Scatter(
                x=[current_point["volatility"] * 100.0],
                y=[current_point["return"] * 100.0],
                mode="markers+text",
                name="目前配置組合",
                marker=dict(symbol="x", size=13, color="#B91C1C", line=dict(color="#FFFFFF", width=1.8)),
                text=[f"目前組合 (Sharpe {current_point['sharpe']:.2f})"],
                textposition="top center",
                textfont=dict(color="#B91C1C", size=11),
            )
        )

    _apply_terminal_layout(
        fig,
        x_title="年化波動度 (%) [越左越穩]",
        y_title="年化預期報酬率 (%) [越上越賺]",
        height=460,
        hovermode="closest",
        show_legend_bottom=True,
    )
    st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)


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
            textfont=dict(size=13),
            colorscale=[
                [0.0, "#1D4ED8"],
                [0.5, "#F8FAFC"],
                [1.0, "#DC2626"],
            ],
            colorbar=dict(
                orientation="h",
                y=-0.24,
                yanchor="top",
                x=0.5,
                xanchor="center",
                len=0.88,
                thickness=12,
                title=dict(text="相關係數", side="top", font=dict(size=12)),
                tickfont=dict(size=12),
            ),
            hovertemplate="%{y} vs %{x}<br>相關係數: %{z:.3f}<extra></extra>",
        )
    )
    _apply_terminal_layout(fig, height=390, hovermode="closest", show_legend_bottom=False)
    fig.update_layout(margin=dict(l=48, r=14, t=24, b=96))
    fig.update_xaxes(tickfont=dict(color="#334155", size=12))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color="#334155", size=12))
    st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)


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
                    line=dict(width=0.8, color="#94A3B8"),
                    hovertemplate="現金活存: %{y:.1f}%<extra></extra>",
                )
            )

    if rebalance_dates and len(rebalance_dates) <= 60:
        for dt in rebalance_dates:
            fig.add_vline(
                x=pd.Timestamp(dt).strftime("%Y-%m-%d"),
                line_width=1,
                line_dash="dot",
                line_color="rgba(30, 58, 95, 0.35)",
            )
    elif rebalance_dates and len(rebalance_dates) > 60:
        st.caption(f"累計觸發調倉 {len(rebalance_dates)} 次（因長區間調倉次數較多，已隱藏垂直參考線以維持圖表可讀性）。")

    _apply_terminal_layout(
        fig,
        y_title="權重佔比 (%)",
        height=370,
        show_range_buttons=True,
        show_legend_bottom=True,
    )
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)
