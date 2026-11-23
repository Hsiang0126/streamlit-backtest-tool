"""
簡約沉穩淺色編輯室金融主題樣式 (Editorial Minimalist Financial Theme)
提供柔和留白、等寬數字對齊、跨裝置響應式網格與零霓虹干擾的專業分析閱讀體驗。
"""
import html
import re
from typing import Dict, List
import pandas as pd
import streamlit as st


def apply_editorial_theme():
    editorial_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+TC:wght@400;500;600;700&display=swap');

    :root {
        --bg-canvas: #F8F9FB;
        --bg-surface: #FFFFFF;
        --bg-subtle: #F1F4F8;
        --border-subtle: #E2E8F0;
        --border-strong: #CBD5E1;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
        --accent-primary: #1E3A5F;
        --accent-positive: #15803D;
        --accent-negative: #B91C1C;
        --accent-warning: #B45309;
    }

    .stApp {
        background-color: var(--bg-canvas);
        color: var(--text-primary);
        font-family: "Inter", "Noto Sans TC", -apple-system, BlinkMacSystemFont, "PingFang TC", "Microsoft JhengHei", sans-serif;
    }

    section[data-testid="stMain"] [data-testid="stMainBlockContainer"],
    .block-container {
        padding: 3.75rem 2rem 3.5rem 2rem;
        max-width: 1360px;
        box-sizing: border-box;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-subtle);
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5 {
        color: var(--text-primary) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
        letter-spacing: -0.015em;
    }

    /* 編輯室標題區 */
    .editorial-header {
        padding-bottom: 14px;
        margin-bottom: 14px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .editorial-kicker {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 4px;
    }
    .editorial-header h1.editorial-title {
        font-size: clamp(1.38rem, 2.2vw, 1.95rem) !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.025em !important;
        line-height: 1.24 !important;
        margin: 0 0 6px 0 !important;
        padding: 0 !important;
    }
    .editorial-subtitle {
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.5;
        margin: 0;
    }

    /* 頂部環境參數膠囊列 */
    .context-banner {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin: 0 0 18px 0;
    }
    .context-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 0.82rem;
        color: var(--text-primary);
        font-weight: 500;
        font-variant-numeric: tabular-nums;
        line-height: 1.35;
    }
    .context-pill-label {
        color: var(--text-muted);
        font-weight: 500;
        font-size: 0.78rem;
    }

    /* 響應式 KPI 網格與指標卡片 */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 16px 0;
        width: 100%;
        box-sizing: border-box;
    }
    .metric-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 14px 16px;
        min-width: 0;
        box-sizing: border-box;
        transition: border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: var(--border-strong);
    }
    .metric-title {
        font-size: 0.78rem;
        color: var(--text-secondary);
        font-weight: 500;
        letter-spacing: 0.01em;
        margin-bottom: 6px;
    }
    .metric-value-positive,
    .metric-value-negative,
    .metric-value-neutral {
        font-size: clamp(1.12rem, 1.65vw, 1.46rem);
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        line-height: 1.25;
        letter-spacing: -0.02em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-value-positive { color: var(--accent-positive); }
    .metric-value-negative { color: var(--accent-negative); }
    .metric-value-neutral  { color: var(--accent-primary); }

    .metric-sub {
        font-size: 0.76rem;
        color: var(--text-muted);
        margin-top: 6px;
        font-variant-numeric: tabular-nums;
        line-height: 1.4;
    }

    /* 簡潔診斷摘要面板 */
    .insight-panel {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-left: 3px solid var(--accent-primary);
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0 18px 0;
    }
    .insight-panel.tone-success { border-left-color: var(--accent-positive); }
    .insight-panel.tone-warning { border-left-color: var(--accent-warning); }
    .insight-panel-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 6px;
    }
    .insight-panel ul {
        margin: 0;
        padding-left: 18px;
        color: var(--text-secondary);
        font-size: 0.86rem;
        line-height: 1.6;
    }
    .insight-panel strong {
        color: var(--text-primary);
        font-weight: 600;
    }
    .insight-panel code {
        background-color: var(--bg-subtle);
        border: 1px solid var(--border-subtle);
        border-radius: 4px;
        padding: 1px 5px;
        font-size: 0.84em;
        color: var(--accent-primary);
    }

    /* 分頁標籤列（支援手機水平滑動） */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: transparent;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 0;
        overflow-x: auto;
        overflow-y: hidden;
        flex-wrap: nowrap;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.9rem;
        background-color: transparent;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-primary) !important;
        background-color: var(--bg-surface);
        border-bottom: 2px solid var(--accent-primary) !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--accent-primary) !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: var(--border-subtle) !important;
    }

    /* Expander 與按鈕 */
    div[data-testid="stExpander"] {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
    }
    .stButton > button,
    .stDownloadButton > button {
        min-height: 40px;
        border-radius: 8px;
        border: 1px solid var(--border-strong);
        background-color: var(--bg-surface);
        color: var(--text-primary);
        font-weight: 500;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--accent-primary);
        background-color: var(--bg-subtle);
        color: var(--accent-primary);
    }
    .stButton > button[kind="primary"] {
        background-color: var(--accent-primary);
        border-color: var(--accent-primary);
        color: #FFFFFF;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #162C49;
        border-color: #162C49;
        color: #FFFFFF;
    }

    /* 細緻淺色捲軸 */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: var(--bg-subtle); }
    ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

    /* 響應式年度月報酬熱力矩陣 */
    .hm-wrapper {
        margin: 4px 0 16px 0;
    }
    .hm-legend {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 8px;
        font-size: 0.78rem;
        color: var(--text-secondary);
    }
    .hm-legend-scale {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-variant-numeric: tabular-nums;
    }
    .hm-legend-bar {
        display: inline-block;
        width: 112px;
        height: 10px;
        border-radius: 999px;
        border: 1px solid var(--border-subtle);
        background: linear-gradient(90deg, #BE123C 0%, #FCA5A5 28%, #F8FAFC 50%, #86EFAC 72%, #15803D 100%);
    }
    .hm-swipe-hint {
        display: none;
        font-size: 0.74rem;
        color: var(--text-muted);
    }
    .heatmap-scroll {
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        background-color: var(--bg-surface);
    }
    .hm-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-variant-numeric: tabular-nums;
        font-size: 0.8125rem;
        line-height: 1.35;
    }
    .hm-table thead th {
        background-color: var(--bg-subtle);
        color: var(--text-secondary);
        font-weight: 600;
        padding: 8px 6px;
        text-align: center;
        border-bottom: 1px solid var(--border-strong);
        white-space: nowrap;
    }
    .hm-table tbody td {
        padding: 8px 5px;
        text-align: center;
        font-weight: 600;
        border-bottom: 1px solid rgba(226, 232, 240, 0.7);
        border-right: 1px solid rgba(255, 255, 255, 0.45);
        min-width: 56px;
        white-space: nowrap;
    }
    .hm-table tbody tr:last-child td,
    .hm-table tbody tr:last-child th {
        border-bottom: none;
    }
    .hm-table th.hm-year {
        position: sticky;
        left: 0;
        z-index: 2;
        background-color: var(--bg-subtle);
        color: var(--text-primary);
        font-weight: 600;
        padding: 8px 10px;
        text-align: left;
        border-right: 1px solid var(--border-strong);
        border-bottom: 1px solid var(--border-subtle);
        min-width: 56px;
        white-space: nowrap;
    }
    .hm-table thead th.hm-year {
        z-index: 3;
        border-bottom: 1px solid var(--border-strong);
    }
    .hm-table th.hm-annual,
    .hm-table td.hm-annual {
        border-left: 2px solid var(--border-strong);
        min-width: 66px;
        font-weight: 700;
    }
    .hm-table td.hm-empty {
        background-color: var(--bg-surface);
        color: #CBD5E1;
        font-weight: 400;
    }

    /* 固定欄序響應式策略排行榜（防誤拖移與截字） */
    .lb-wrapper {
        margin: 4px 0 18px 0;
    }
    .lb-meta {
        display: none;
    }
    .lb-scroll {
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        background-color: var(--bg-surface);
    }
    .lb-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-variant-numeric: tabular-nums;
        font-size: 0.83rem;
        line-height: 1.4;
    }
    .lb-table thead th {
        background-color: var(--bg-subtle);
        color: var(--text-secondary);
        font-weight: 600;
        padding: 10px 12px;
        border-bottom: 1px solid var(--border-strong);
        white-space: nowrap;
        user-select: none;
    }
    .lb-table tbody th,
    .lb-table tbody td {
        padding: 10px 12px;
        border-bottom: 1px solid var(--border-subtle);
        white-space: nowrap;
        color: var(--text-primary);
    }
    .lb-table tbody tr:last-child th,
    .lb-table tbody tr:last-child td {
        border-bottom: none;
    }
    .lb-table th.lb-num,
    .lb-table td.lb-num {
        text-align: right;
    }
    .lb-table th.lb-sticky {
        position: sticky;
        left: 0;
        z-index: 2;
        text-align: left;
        background-color: var(--bg-surface);
        font-weight: 600;
        border-right: 1px solid var(--border-subtle);
        box-shadow: 2px 0 6px -4px rgba(15, 23, 42, 0.12);
    }
    .lb-table thead th.lb-sticky {
        z-index: 3;
        background-color: var(--bg-subtle);
        border-right: 1px solid var(--border-strong);
    }
    .lb-table tbody tr.lb-champ td {
        background-color: #F2F6FB;
    }
    .lb-table tbody tr.lb-champ th.lb-sticky {
        background-color: #E8F0F9;
        color: var(--accent-primary);
        font-weight: 700;
    }
    .lb-table tbody tr:not(.lb-champ):hover td {
        background-color: #F8FAFC;
    }
    .lb-table td.lb-pos {
        color: var(--accent-positive);
        font-weight: 600;
    }
    .lb-table td.lb-neg {
        color: var(--accent-negative);
        font-weight: 600;
    }

    /* 手機垂直滑動圖表防誤觸放大／軸邊緣拖曳 */
    .js-plotly-plot .plotly .nsewdrag,
    .js-plotly-plot .plotly .draglayer {
        touch-action: pan-y !important;
        cursor: default !important;
    }
    .js-plotly-plot .plotly .ewdrag,
    .js-plotly-plot .plotly .nsdrag,
    .js-plotly-plot .plotly .nwdrag,
    .js-plotly-plot .plotly .nedrag,
    .js-plotly-plot .plotly .swdrag,
    .js-plotly-plot .plotly .sedrag {
        pointer-events: none !important;
    }

    /* 平板與窄視窗 RWD (<= 1120px) */
    @media (max-width: 1120px) {
        .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 1080px) {
        section[data-testid="stMain"] [data-testid="stMainBlockContainer"],
        .block-container {
            padding: 3.75rem 1.25rem 4rem 1.25rem !important;
        }
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.75rem !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 min(100%, 210px) !important;
            min-width: min(100%, 210px) !important;
        }
    }

    /* 手機 RWD (<= 768px / <= 520px) */
    @media (max-width: 768px) {
        section[data-testid="stMain"] [data-testid="stMainBlockContainer"],
        .block-container {
            padding: 3.75rem 0.9rem 5.5rem 0.9rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 14px;
            min-height: 44px;
            font-size: 0.88rem;
        }
        .stButton > button,
        .stDownloadButton > button {
            min-height: 44px;
        }
        button[data-testid="stSidebarCollapseButton"],
        div[data-testid="stSidebarCollapsedControl"] button {
            min-width: 44px;
            min-height: 44px;
        }
        .hm-swipe-hint {
            display: inline;
        }
        .lb-meta {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 6px;
        }
        .js-plotly-plot .modebar-container {
            display: none !important;
        }
    }
    @media (max-width: 520px) {
        .metric-grid {
            grid-template-columns: minmax(0, 1fr);
        }
        .context-pill {
            font-size: 0.78rem;
            padding: 5px 10px;
        }
    }
    </style>
    """
    st.markdown(editorial_css, unsafe_allow_html=True)


def _format_inline_markdown(text: str) -> str:
    """安全轉換行內 Markdown (**粗體** 與 `行內代碼`) 為 HTML，同時防止未跳脫標籤注入。"""
    escaped = html.escape(str(text))
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped


def _build_metric_card_html(title: str, value: str, sub: str = "", state: str = "neutral") -> str:
    val_class = {
        "positive": "metric-value-positive",
        "negative": "metric-value-negative",
        "neutral": "metric-value-neutral",
    }.get(state, "metric-value-neutral")

    safe_title = html.escape(str(title))
    safe_value = html.escape(str(value))
    safe_sub = html.escape(str(sub))

    return (
        '<div class="metric-card">'
        f'<div class="metric-title">{safe_title}</div>'
        f'<div class="{val_class}" title="{safe_value}">{safe_value}</div>'
        f'<div class="metric-sub">{safe_sub}</div>'
        "</div>"
    )


def render_page_header(
    title: str,
    subtitle: str,
    kicker: str = "QUANTITATIVE PORTFOLIO ANALYTICS",
):
    safe_kicker = html.escape(str(kicker))
    safe_title = html.escape(str(title))
    safe_sub = html.escape(str(subtitle))
    header_html = (
        '<header class="editorial-header">'
        f'<div class="editorial-kicker">{safe_kicker}</div>'
        f'<h1 class="editorial-title">{safe_title}</h1>'
        f'<p class="editorial-subtitle">{safe_sub}</p>'
        "</header>"
    )
    st.markdown(header_html, unsafe_allow_html=True)


def render_metric_grid(cards: List[Dict[str, str]]):
    if not cards:
        return
    inner = "".join(
        _build_metric_card_html(
            title=c.get("title", ""),
            value=c.get("value", ""),
            sub=c.get("sub", ""),
            state=c.get("state", "neutral"),
        )
        for c in cards
    )
    st.markdown(f'<div class="metric-grid">{inner}</div>', unsafe_allow_html=True)


def render_context_banner(
    tickers: List[str],
    start_str: str,
    end_str: str,
    capital_summary: str,
    currency_label: str,
):
    ticker_str = " · ".join(tickers[:6]) + (f" (+{len(tickers) - 6})" if len(tickers) > 6 else "")
    items = [
        ("標的", ticker_str),
        ("區間", f"{start_str} ~ {end_str}"),
        ("資金", capital_summary),
        ("幣別", currency_label),
    ]
    parts = [
        f'<span class="context-pill"><span class="context-pill-label">{html.escape(lbl)}</span>'
        f"{html.escape(val)}</span>"
        for lbl, val in items
    ]
    st.markdown(f'<div class="context-banner">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_insight_panel(title: str, bullets: List[str], tone: str = "info"):
    tone_cls = {
        "success": "tone-success",
        "warning": "tone-warning",
        "info": "",
    }.get(tone, "")
    li_html = "".join(f"<li>{_format_inline_markdown(b)}</li>" for b in bullets if b)
    panel_html = (
        f'<div class="insight-panel {tone_cls}">'
        f'<div class="insight-panel-title">{html.escape(title)}</div>'
        f"<ul>{li_html}</ul>"
        "</div>"
    )
    st.markdown(panel_html, unsafe_allow_html=True)


_LEADERBOARD_SIGNED_COLS = {"淨獲利報酬率", "實賺獲利金額", "CAGR 年化報酬", "XIRR 資金加權"}


def _build_leaderboard_html(df: pd.DataFrame) -> str:
    """建構無 4 空格縮排之固定欄序策略排行榜 HTML 表格（凍結首欄不截字、手機可橫滑）。"""
    cols = [str(c) for c in df.columns]
    head_cells = [
        f'<th scope="col" class="{"lb-sticky" if idx == 0 else "lb-num"}">{html.escape(col)}</th>'
        for idx, col in enumerate(cols)
    ]

    body_rows: List[str] = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        cells: List[str] = []
        for col_idx, col in enumerate(cols):
            val_str = str(row.iloc[col_idx])
            safe_val = html.escape(val_str)
            if col_idx == 0:
                cells.append(f'<th scope="row" class="lb-sticky">{safe_val}</th>')
            else:
                cls_parts = ["lb-num"]
                if col in _LEADERBOARD_SIGNED_COLS:
                    if "+" in val_str:
                        cls_parts.append("lb-pos")
                    elif "-" in val_str:
                        cls_parts.append("lb-neg")
                cells.append(f'<td class="{" ".join(cls_parts)}">{safe_val}</td>')
        tr_cls = ' class="lb-champ"' if row_idx == 0 else ""
        body_rows.append(f"<tr{tr_cls}>{''.join(cells)}</tr>")

    return (
        '<div class="lb-wrapper">'
        '<div class="lb-meta"><span class="hm-swipe-hint">← 左右滑動檢視完整指標 →</span></div>'
        '<div class="lb-scroll">'
        '<table class="lb-table">'
        f'<thead><tr>{"".join(head_cells)}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        "</table>"
        "</div>"
        "</div>"
    )


def render_leaderboard_table(df: pd.DataFrame):
    if df is None or df.empty:
        return
    st.markdown(_build_leaderboard_html(df), unsafe_allow_html=True)
