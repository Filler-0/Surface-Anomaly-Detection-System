import streamlit as st


def inject_global_styles():
    st.markdown(
        """
        <style>
        /* ===== Base app ===== */
        .stApp {
            background: #f8fafc;
            color: #0f172a;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* ===== Sidebar ===== */
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
            min-width: 260px !important;
            max-width: 260px !important;
        }
        
        section[data-testid="stSidebar"] * {
            color: #0f172a !important;
        }
        
        button[kind="header"][aria-label="Close sidebar"] {
            display: none !important;
        }
        
        button[kind="header"][aria-label="Open sidebar"] {
            display: none !important;
        }

        /* ===== Sidebar ===== */
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
        }

        section[data-testid="stSidebar"] * {
            color: #0f172a !important;
        }

        /* ===== Typography ===== */
        h1, h2, h3 {
            color: #0f172a;
            letter-spacing: -0.02em;
            line-height: 1.15;
        }

        p, div, label, span {
            color: #334155;
        }

        /* ===== Hero ===== */
        .app-hero {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 28px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.25rem;
        }

        .hero-title {
            font-size: 2.6rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0 0 0.4rem 0;
        }

        .hero-subtitle {
            color: #64748b;
            font-size: 1rem;
            line-height: 1.6;
        }

        /* ===== Cards ===== */
        .app-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }

        .section-title {
            color: #0f172a;
            font-size: 1.15rem;
            font-weight: 650;
            margin-bottom: 0.35rem;
        }

        .section-subtitle {
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }

        /* ===== Upload ===== */
        div[data-testid="stFileUploader"] {
            background: #f8fafc;
            border: 1px dashed #cbd5e1;
            border-radius: 16px;
            padding: 10px;
        }

        /* ===== Inputs ===== */
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-baseweb="select"] > div,
        div[data-testid="stDateInput"] input {
            background: #ffffff !important;
            color: #0f172a !important;
            border-radius: 12px !important;
            border: 1px solid #cbd5e1 !important;
        }

        /* ===== Button ===== */
        div[data-testid="stButton"] > button {
            width: 100%;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.85rem 1rem !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            color: #ffffff !important;
            background: #2563eb !important;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.22);
        }
        
        div[data-testid="stButton"] > button p,
        div[data-testid="stButton"] > button span {
            color: #ffffff !important;
        }
        
        div[data-testid="stButton"] > button:hover {
            background: #1d4ed8 !important;
        }

        /* ===== Metrics ===== */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 14px;
        }

        div[data-testid="stMetricLabel"] {
            color: #64748b;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a;
        }

        /* ===== Expander ===== */
        div[data-testid="stExpander"] {
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            background: #ffffff;
            overflow: hidden;
        }

        /* ===== Alerts ===== */
        .stAlert {
            border-radius: 14px;
        }

        /* ===== Dataframe ===== */
        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            overflow: hidden;
            background: white;
        }

        /* ===== Small badges ===== */
        .status-normal {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .status-anomalous {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
            font-weight: 600;
            font-size: 0.85rem;
        }

        /* ===== KPI cards ===== */
        .kpi-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 20px;
            min-height: 132px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        .kpi-label {
            color: #64748b;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 14px;
        }
        
        .kpi-value {
            color: #0f172a;
            font-size: 2.2rem;
            font-weight: 750;
            line-height: 1;
            margin-bottom: 10px;
        }
        
        .kpi-sub {
            color: #94a3b8;
            font-size: 0.86rem;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="app-hero">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def open_card(title: str | None = None, subtitle: str | None = None):
    header_html = ""
    if title:
        header_html += f'<div class="section-title">{title}</div>'
    if subtitle:
        header_html += f'<div class="section-subtitle">{subtitle}</div>'

    st.markdown(f'<div class="app-card">{header_html}', unsafe_allow_html=True)


def close_card():
    st.markdown("</div>", unsafe_allow_html=True)


def render_kpi(label: str, value: str, sub: str = ""):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def verdict_badge(verdict: str):
    verdict_upper = (verdict or "").upper()
    if verdict_upper == "NORMAL":
        st.markdown('<span class="status-normal">NORMAL</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-anomalous">ANOMALOUS</span>', unsafe_allow_html=True)
