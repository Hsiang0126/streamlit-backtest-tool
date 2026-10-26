"""深色金融終端 UI 主題樣式 (Dark Terminal Theme)。"""
import streamlit as st


def apply_dark_theme():
    dark_css = """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    h1, h2, h3 {
        color: #f0f6fc !important;
        font-weight: 600;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value-positive {
        font-size: 1.65rem;
        font-weight: 700;
        color: #3fb950;
    }
    .metric-value-negative {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f85149;
    }
    .metric-value-neutral {
        font-size: 1.65rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #8b949e;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0e1117;
        border-bottom: 2px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px 6px 0 0;
        color: #8b949e;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        background-color: #161b22;
        border-bottom: 2px solid #58a6ff !important;
    }
    .dataframe {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
    }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, sub: str = "", state: str = "neutral"):
    val_class = {
        "positive": "metric-value-positive",
        "negative": "metric-value-negative",
        "neutral": "metric-value-neutral"
    }.get(state, "metric-value-neutral")

    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="{val_class}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
