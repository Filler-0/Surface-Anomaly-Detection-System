import streamlit as st


def inject_global_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

        .stApp {
            font-family: "Manrope", "Segoe UI", sans-serif;
            background:
                radial-gradient(1200px 500px at 10% -10%, #dbeafe 0%, rgba(219, 234, 254, 0) 60%),
                radial-gradient(900px 420px at 110% 0%, #dcfce7 0%, rgba(220, 252, 231, 0) 58%),
                #f8fafc;
            color: #0f172a;
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.6rem;
            padding-bottom: 2.1rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border-right: 1px solid #e2e8f0;
            min-width: 260px !important;
            max-width: 260px !important;
        }

        section[data-testid="stSidebar"] * {
            color: #0f172a !important;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
            border-radius: 12px;
            margin: 2px 4px;
            padding: 6px 10px !important;
            text-transform: lowercase;
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a::first-letter {
            text-transform: uppercase;
            font-size: 1.25em;
            font-weight: 800;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
            background: #eff6ff;
        }

        h1, h2, h3 {
            color: #0f172a;
            letter-spacing: -0.02em;
            line-height: 1.15;
        }

        p, div, label, span {
            color: #334155;
        }

        .app-hero {
            background:
                linear-gradient(120deg, rgba(59, 130, 246, 0.08) 0%, rgba(16, 185, 129, 0.08) 100%),
                #ffffff;
            border: 1px solid #dbe7f8;
            border-radius: 22px;
            padding: 28px 30px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.15rem;
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 0.4rem 0;
        }

        .hero-subtitle {
            color: #64748b;
            font-size: 1rem;
            line-height: 1.6;
        }

        .app-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }

        .app-card:hover {
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.08);
            transform: translateY(-1px);
        }

        .info-card {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 20px;
            backdrop-filter: blur(3px);
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

        div[data-testid="stFileUploader"] {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px dashed #94a3b8;
            border-radius: 16px;
            padding: 10px;
        }

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

        div[data-testid="stButton"] > button {
            width: 100%;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.85rem 1rem !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            color: #ffffff !important;
            background: linear-gradient(90deg, #2563eb 0%, #0ea5e9 100%) !important;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.24);
        }

        div[data-testid="stButton"] > button p,
        div[data-testid="stButton"] > button span {
            color: #ffffff !important;
        }

        div[data-testid="stButton"] > button:hover {
            filter: brightness(0.95);
        }

        div[data-testid="stButton"] > button[kind="secondary"] {
            color: #1f2937 !important;
            border: 1px solid #cbd5e1 !important;
            background: linear-gradient(90deg, #cbd5e1 0%, #94a3b8 100%) !important;
            box-shadow: 0 8px 18px rgba(51, 65, 85, 0.2);
        }

        div[data-testid="stButton"] > button[kind="secondary"] p,
        div[data-testid="stButton"] > button[kind="secondary"] span {
            color: #1f2937 !important;
        }

        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            filter: brightness(0.98);
        }

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

        div[data-testid="stExpander"] {
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            background: #ffffff;
            overflow: hidden;
        }

        .stAlert {
            border-radius: 14px;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            overflow: hidden;
            background: white;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            border-radius: 12px !important;
            font-weight: 600 !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: #eff6ff !important;
            color: #1d4ed8 !important;
            border: 1px solid #bfdbfe !important;
        }

        div[data-testid="stImage"] img {
            border-radius: 14px;
            border: 1px solid #e2e8f0;
        }

        div[data-testid="stProgressBar"] > div > div > div > div {
            background: linear-gradient(90deg, #22c55e 0%, #14b8a6 100%) !important;
        }

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

        .status-manual {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: #fef3c7;
            color: #92400e;
            border: 1px solid #fde68a;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .status-unsupported {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: #e0e7ff;
            color: #3730a3;
            border: 1px solid #c7d2fe;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .kpi-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #dbe7f8;
            border-radius: 18px;
            padding: 20px;
            min-height: 186px;
            height: 186px;
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.07);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
            overflow: hidden;
        }

        .kpi-label {
            color: #64748b;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 10px;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }

        .kpi-value {
            color: #0f172a;
            font-size: 2.5rem;
            font-weight: 750;
            line-height: 1;
            margin-bottom: 8px;
        }

        .kpi-sub {
            color: #94a3b8;
            font-size: 0.9rem;
            line-height: 1.4;
            overflow-wrap: anywhere;
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
    elif verdict_upper == "ANOMALOUS":
        st.markdown('<span class="status-anomalous">ANOMALOUS</span>', unsafe_allow_html=True)
    elif verdict_upper == "MANUAL_INSPECTION":
        st.markdown('<span class="status-manual">MANUAL_INSPECTION</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-unsupported">UNSUPPORTED_FORMAT</span>', unsafe_allow_html=True)
