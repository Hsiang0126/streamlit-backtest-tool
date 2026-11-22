"""
深色金融終端 UI 主題樣式 (Modern Quantitative Terminal Theme)
提供簡潔俐落、層次清楚、對比舒適、等寬數字對齊的專業金融視覺體驗。
"""
import html
from typing import List
import streamlit as st


def apply_dark_theme():
    dark_css = """
    <style>
    :root {
        --bg-canvas: #0b0f19;
        --bg-surface: #131a29;
        --bg-elevated: #192234;
        --border-subtle: rgba(148, 163, 184, 0.16);
        --border-accent: rgba(56, 189, 248, 0.35);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --accent-blue: #38bdf8;
        --accent-green: #22c55e;
        --accent-red: #ef4444;
        --accent-amber: #f59e0b;
    }

    .stApp {
        background-color: var(--bg-canvas);
        color: var(--text-primary);
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f1523;
        border-right: 1px solid var(--border-subtle);
    }

    h1, h2, h3, h4 {
        color: var(--text-primary) !important;
        font-weight: 600;
        letter-spacing: -0.015em;
    }

    /* 頂部環境參數摘要橫幅 */
    .context-banner {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px 18px;
        background: linear-gradient(135deg, rgba(19, 26, 41, 0.96), rgba(25, 34, 52, 0.90));
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 10px 16px;
        margin: 6px 0 18px 0;
        font-size: 0.84rem;
        color: var(--text-secondary);
    }
    .context-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: var(--text-primary);
        font-weight: 500;
        font-variant-numeric: tabular-nums;
    }
    .context-pill-label {
        color: var(--text-secondary);
        font-weight: 400;
    }
    .context-divider {
        color: rgba(148, 163, 184, 0.28);
    }

    /* KPI 指標卡片 */
    .metric-card {
        background: linear-gradient(180deg, #141c2c 0%, #101725 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        transition: border-color 0.2s ease, transform 0.15s ease;
    }
    .metric-card:hover {
        border-color: var(--border-accent);
    }
    .metric-title {
        font-size: 0.78rem;
        color: var(--text-secondary);
        font-weight: 500;
        letter-spacing: 0.03em;
        margin-bottom: 6px;
    }
    .metric-value-positive,
    .metric-value-negative,
    .metric-value-neutral {
        font-size: 1.55rem;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .metric-value-positive { color: var(--accent-green); }
    .metric-value-negative { color: var(--accent-red); }
    .metric-value-neutral  { color: var(--accent-blue); }

    .metric-sub {
        font-size: 0.77rem;
        color: var(--text-secondary);
        margin-top: 6px;
        font-variant-numeric: tabular-nums;
    }

    /* 簡潔診斷摘要面板 */
    .insight-panel {
        background: rgba(19, 26, 41, 0.92);
        border: 1px solid var(--border-subtle);
        border-left: 3px solid var(--accent-blue);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
    }
    .insight-panel.tone-success { border-left-color: var(--accent-green); }
    .insight-panel.tone-warning { border-left-color: var(--accent-amber); }
    .insight-panel-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 6px;
    }
    .insight-panel ul {
        margin: 0;
        padding-left: 18px;
        color: #cbd5e1;
        font-size: 0.84rem;
        line-height: 1.55;
    }

    /* 主分頁與子分頁樣式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 9px 18px;
        border-radius: 8px 8px 0 0;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.9rem;
        background-color: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-blue) !important;
        background-color: var(--bg-surface);
        border-bottom: 2px solid var(--accent-blue) !important;
    }

    /* Expander 與 DataFrame 質感微調 */
    div[data-testid="stExpander"] {
        background-color: rgba(19, 26, 41, 0.65);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
    }
    .stButton > button {
        border-radius: 7px;
        border: 1px solid var(--border-subtle);
        font-weight: 500;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--accent-blue);
    }

    /* 細緻深色捲軸 */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #0b0f19; }
    ::-webkit-scrollbar-thumb { background: #273249; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #3b4b6b; }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, sub: str = "", state: str = "neutral"):
    val_class = {
        "positive": "metric-value-positive",
        "negative": "metric-value-negative",
        "neutral": "metric-value-neutral",
    }.get(state, "metric-value-neutral")

    safe_title = html.escape(str(title))
    safe_value = html.escape(str(value))
    safe_sub = html.escape(str(sub))

    card_html = f"""
    <div class="metric-card">
        <div class="metric-title">{safe_title}</div>
        <div class="{val_class}">{safe_value}</div>
        <div class="metric-sub">{safe_sub}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


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
    parts = []
    for idx, (lbl, val) in enumerate(items):
        if idx > 0:
            parts.append('<span class="context-divider">|</span>')
        parts.append(
            f'<span class="context-pill"><span class="context-pill-label">{html.escape(lbl)}</span>'
            f'{html.escape(val)}</span>'
        )
    st.markdown(f'<div class="context-banner">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_insight_panel(title: str, bullets: List[str], tone: str = "info"):
    tone_cls = {
        "success": "tone-success",
        "warning": "tone-warning",
        "info": "",
    }.get(tone, "")
    li_html = "".join(f"<li>{b}</li>" for b in bullets if b)
    panel_html = f"""
    <div class="insight-panel {tone_cls}">
        <div class="insight-panel-title">{html.escape(title)}</div>
        <ul>{li_html}</ul>
    </div>
    """
    st.markdown(panel_html, unsafe_allow_html=True)
