import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

# ==============================================================================
# 🎨 COLOR PALETTE & GLOBAL CONFIGURATION
# ==============================================================================
PALETTE = {
    "warm_ivory": "#F7F3EB",
    "soft_pearl": "#FFFCF7",
    "oatmeal": "#EFE7DA",
    "warm_sand": "#E5DCCE",
    "pure_white": "#FFFFFF",
    "deep_espresso": "#332820",
    "warm_taupe": "#75685C",
    "muted_sandstone": "#9C8E82",
    "rich_mocha": "#6F4E37",
    "dark_espresso": "#4B3428",
    "warm_caramel": "#C68B59",
    "muted_sage": "#7F9478",
    "trophy_gold": "#D4AA63",
    "terracotta": "#A75D43",
    "ivory_border": "#E4D9CB",
    "light_divider": "#ECE3D7",
    "chart_grid": "rgba(111, 78, 55, 0.08)"
}

COLORWAY = [
    '#6F4E37', '#C68B59', '#7F9478', '#D4AA63',
    '#4B3428', '#A75D43', '#8D7B68', '#5E7153'
]

st.set_page_config(
    page_title="FIFA Insights | Football Analytics & ML Suite",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# 💅 CUSTOM STYLING (CSS)
# ==============================================================================
CUSTOM_CSS = f"""
<style>
    /* Remove excessive top gap from Streamlit body - push content right up */
    header[data-testid="stHeader"] {{
        display: none !important;
        height: 0px !important;
    }}
    .main .block-container,
    div[data-testid="stAppViewBlockContainer"] {{
        padding-top: 0.35rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }}
    
    /* Global Page Background & Base Text */
    .stApp {{
        background-color: {PALETTE['warm_ivory']} !important;
        color: {PALETTE['deep_espresso']};
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }}
    
    /* Brand Badge on Left */
    .brand-container {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding-top: 2px;
    }}
    .brand-icon-box {{
        background-color: {PALETTE['dark_espresso']};
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        box-shadow: 0 2px 6px rgba(75, 52, 40, 0.2);
    }}
    .brand-title {{
        font-size: 1.2rem;
        font-weight: 800;
        color: {PALETTE['deep_espresso']};
        letter-spacing: 0.3px;
        line-height: 1.1;
    }}
    .brand-subtitle {{
        font-size: 0.76rem;
        color: {PALETTE['warm_taupe']};
        font-weight: 500;
        margin-top: 2px;
    }}

    /* =========================================================================
       🧭 TOP SEGMENTED CONTROL NAVBAR (EXACT MATCH TO REFERENCE IMAGE 2)
       Container: Oatmeal Cream (#EFE7DA) Pill
       Active Tab: Soft Pearl / Pure White (#FFFCF7) Pill with subtle drop shadow
       Inactive Tabs: Completely Transparent with Warm Taupe (#75685C) Text
       NO black background, NO dark BaseWeb boxes, NO red outline, NO dividers
       ========================================================================= */
    div[data-testid="stSegmentedControl"],
    div[data-testid="stButtonGroup"] {{
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        background: transparent !important;
        background-color: transparent !important;
        margin-bottom: 0px !important;
        border: none !important;
    }}

    /* Outer Container Pill - Oatmeal Cream */
    div[data-testid="stSegmentedControl"] > div,
    div[data-testid="stSegmentedControl"] div[data-baseweb="button-group"],
    div[data-baseweb="button-group"]:has(button[kind="segmented"]),
    div[data-testid="stSegmentedControl"] div[role="radiogroup"],
    div[data-testid="stButtonGroup"] > div,
    div[data-testid="stButtonGroup"] div[data-baseweb="button-group"] {{
        background: {PALETTE['oatmeal']} !important;
        background-color: {PALETTE['oatmeal']} !important;
        border: 1px solid {PALETTE['ivory_border']} !important;
        border-radius: 14px !important;
        padding: 4px 6px !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        box-shadow: 0 1px 4px rgba(111, 78, 55, 0.05) !important;
    }}

    /* Inactive Buttons - Transparent Background & Warm Taupe Text */
    div[data-testid="stSegmentedControl"] button,
    div[data-testid="stButtonGroup"] button,
    div[data-baseweb="button-group"] button,
    button[kind="segmented"] {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-left: none !important;
        border-right: none !important;
        border-top: none !important;
        border-bottom: none !important;
        border-radius: 10px !important;
        padding: 7px 18px !important;
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        color: {PALETTE['warm_taupe']} !important;
        box-shadow: none !important;
        outline: none !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
    }}

    /* Inactive Hover */
    div[data-testid="stSegmentedControl"] button:hover:not([aria-checked="true"]):not([data-checked="true"]):not([data-state="on"]),
    div[data-testid="stButtonGroup"] button:hover:not([aria-checked="true"]):not([data-checked="true"]):not([data-state="on"]),
    div[data-baseweb="button-group"] button:hover:not([aria-checked="true"]):not([data-checked="true"]):not([data-state="on"]) {{
        background: rgba(111, 78, 55, 0.06) !important;
        background-color: rgba(111, 78, 55, 0.06) !important;
        color: {PALETTE['deep_espresso']} !important;
        box-shadow: none !important;
        outline: none !important;
        border: none !important;
    }}

    /* Active Selected Pill Tab (Matching Image 2 "Overview") */
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[data-checked="true"],
    div[data-testid="stSegmentedControl"] button[data-state="on"],
    div[data-testid="stSegmentedControl"] button[aria-selected="true"],
    div[data-testid="stButtonGroup"] button[aria-checked="true"],
    div[data-testid="stButtonGroup"] button[data-checked="true"],
    div[data-testid="stButtonGroup"] button[data-state="on"],
    div[data-baseweb="button-group"] button[aria-checked="true"],
    div[data-baseweb="button-group"] button[data-checked="true"],
    div[data-baseweb="button-group"] button[data-state="on"],
    div[data-baseweb="button-group"] button[aria-selected="true"],
    button[kind="segmented"][aria-checked="true"],
    button[kind="segmented"][data-state="on"] {{
        background: {PALETTE['soft_pearl']} !important;
        background-color: {PALETTE['soft_pearl']} !important;
        color: {PALETTE['deep_espresso']} !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(111, 78, 55, 0.12) !important;
        border: none !important;
        outline: none !important;
    }}

    /* Eliminate Red Focus Outline / Ring from BaseWeb */
    div[data-testid="stSegmentedControl"] button:focus,
    div[data-testid="stSegmentedControl"] button:focus-visible,
    div[data-testid="stSegmentedControl"] button:active,
    div[data-testid="stButtonGroup"] button:focus,
    div[data-testid="stButtonGroup"] button:focus-visible,
    div[data-testid="stButtonGroup"] button:active,
    div[data-baseweb="button-group"] button:focus,
    div[data-baseweb="button-group"] button:focus-visible,
    div[data-baseweb="button-group"] button:active,
    button[kind="segmented"]:focus,
    button[kind="segmented"]:focus-visible,
    button[kind="segmented"]:active {{
        outline: none !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(111, 78, 55, 0.12) !important;
    }}

    /* Remove BaseWeb divider lines between buttons */
    div[data-testid="stSegmentedControl"] button::before,
    div[data-testid="stSegmentedControl"] button::after,
    div[data-testid="stButtonGroup"] button::before,
    div[data-testid="stButtonGroup"] button::after,
    div[data-baseweb="button-group"] button::before,
    div[data-baseweb="button-group"] button::after {{
        display: none !important;
        content: none !important;
    }}

    /* Enforce Deep Espresso text for active tab and warm taupe for inactive */
    div[data-testid="stSegmentedControl"] button *,
    div[data-testid="stButtonGroup"] button *,
    div[data-baseweb="button-group"] button *,
    button[kind="segmented"] * {{
        color: inherit !important;
        font-weight: inherit !important;
    }}
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] *,
    div[data-testid="stSegmentedControl"] button[data-checked="true"] *,
    div[data-testid="stSegmentedControl"] button[data-state="on"] *,
    div[data-testid="stButtonGroup"] button[aria-checked="true"] *,
    div[data-testid="stButtonGroup"] button[data-checked="true"] *,
    div[data-testid="stButtonGroup"] button[data-state="on"] *,
    div[data-baseweb="button-group"] button[aria-checked="true"] *,
    div[data-baseweb="button-group"] button[data-checked="true"] *,
    div[data-baseweb="button-group"] button[data-state="on"] * {{
        color: {PALETTE['deep_espresso']} !important;
        font-weight: 700 !important;
    }}
    
    /* Fallback: Hide radio circles if st.radio is used */
    div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child,
    div[data-testid="stRadio"] input[type="radio"],
    div[data-testid="stRadio"] svg {{
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        background-color: {PALETTE['oatmeal']} !important;
        border: 1px solid {PALETTE['ivory_border']} !important;
        border-radius: 14px !important;
        padding: 4px 6px !important;
        display: inline-flex !important;
        gap: 6px !important;
    }}
    div[data-testid="stRadio"] label {{
        background-color: transparent !important;
        border-radius: 10px !important;
        padding: 7px 18px !important;
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        color: {PALETTE['warm_taupe']} !important;
        cursor: pointer !important;
        border: none !important;
    }}
    div[data-testid="stRadio"] label:has(input:checked) {{
        background-color: {PALETTE['soft_pearl']} !important;
        color: {PALETTE['deep_espresso']} !important;
        box-shadow: 0 2px 8px rgba(111, 78, 55, 0.12) !important;
        font-weight: 700 !important;
    }}

    /* Card Containers */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        background-color: {PALETTE['soft_pearl']} !important;
        border: 1px solid {PALETTE['ivory_border']} !important;
        border-radius: 14px !important;
        padding: 20px 22px !important;
        box-shadow: 0 2px 8px rgba(111, 78, 55, 0.03) !important;
        margin-bottom: 12px !important;
    }}

    /* Headings & Text Colors */
    h1, h2, h3, h4 {{
        color: {PALETTE['deep_espresso']} !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px;
    }}
    p, span {{
        color: {PALETTE['warm_taupe']};
    }}

    /* WIDGET TEXT VISIBILITY FIX (High contrast, bold espresso text) */
    label, p[data-testid="stWidgetLabel"], div[data-testid="stWidgetLabel"] p {{
        color: {PALETTE['deep_espresso']} !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }}
    
    /* Selectbox Styling & Visible Text */
    div[data-baseweb="select"] > div {{
        background-color: {PALETTE['pure_white']} !important;
        border: 1px solid {PALETTE['ivory_border']} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] * {{
        color: {PALETTE['deep_espresso']} !important;
        font-weight: 600 !important;
    }}
    div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"] {{
        background-color: {PALETTE['pure_white']} !important;
        border: 1px solid {PALETTE['ivory_border']} !important;
    }}
    div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] li {{
        color: {PALETTE['deep_espresso']} !important;
        background-color: {PALETTE['pure_white']} !important;
        font-weight: 500 !important;
    }}
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="menu"] li:hover {{
        background-color: {PALETTE['oatmeal']} !important;
        color: {PALETTE['rich_mocha']} !important;
    }}

    /* Number Inputs: High Contrast Espresso Text */
    div[data-testid="stNumberInput"] input {{
        color: {PALETTE['deep_espresso']} !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        background-color: {PALETTE['pure_white']} !important;
    }}
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {{
        background-color: {PALETTE['pure_white']} !important;
        border: 1px solid {PALETTE['ivory_border']} !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stNumberInput"] button {{
        background-color: {PALETTE['oatmeal']} !important;
        color: {PALETTE['deep_espresso']} !important;
        border: none !important;
    }}
    div[data-testid="stNumberInput"] button:hover {{
        background-color: {PALETTE['warm_sand']} !important;
        color: {PALETTE['rich_mocha']} !important;
    }}

    /* Sliders Themed to Rich Mocha */
    div[data-testid="stSlider"] div[role="slider"] {{
        background-color: {PALETTE['rich_mocha']} !important;
        border: 2px solid {PALETTE['pure_white']} !important;
        box-shadow: 0 2px 6px rgba(111, 78, 55, 0.3) !important;
    }}
    div[data-testid="stSlider"] [data-baseweb="slider"] div div:first-child {{
        background-color: {PALETTE['rich_mocha']} !important;
    }}
    div[data-testid="stSlider"] [data-baseweb="slider"] div {{
        background-color: {PALETTE['warm_sand']} !important;
    }}
    div[data-testid="stSlider"] [data-baseweb="slider"] div div {{
        background-color: {PALETTE['rich_mocha']} !important;
    }}

    /* Metric Cards */
    .metric-card {{
        background-color: {PALETTE['soft_pearl']};
        border: 1px solid {PALETTE['ivory_border']};
        border-radius: 12px;
        padding: 16px 18px;
        text-align: left;
        box-shadow: 0 2px 8px rgba(111, 78, 55, 0.03);
        margin-bottom: 8px;
    }}
    .metric-title {{
        font-size: 0.80rem;
        text-transform: uppercase;
        color: {PALETTE['muted_sandstone']};
        letter-spacing: 0.6px;
        font-weight: 600;
        margin-bottom: 4px;
    }}
    .metric-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {PALETTE['rich_mocha']};
        margin: 0;
    }}
    .metric-sub {{
        font-size: 0.80rem;
        color: {PALETTE['warm_taupe']};
        margin-top: 3px;
    }}

    /* Action Buttons */
    .stButton > button {{
        background-color: {PALETTE['rich_mocha']} !important;
        color: {PALETTE['pure_white']} !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 24px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }}
    .stButton > button:hover {{
        background-color: {PALETTE['dark_espresso']} !important;
        box-shadow: 0 4px 12px rgba(75, 52, 40, 0.25) !important;
        transform: translateY(-1px);
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper to format Plotly charts with project palette
def format_fig(fig):
    fig.update_layout(
        paper_bgcolor=PALETTE['soft_pearl'],
        plot_bgcolor=PALETTE['soft_pearl'],
        font=dict(family="Segoe UI, sans-serif", color=PALETTE['deep_espresso'], size=11),
        colorway=COLORWAY,
        margin=dict(l=35, r=25, t=35, b=35),
        xaxis=dict(
            gridcolor=PALETTE['chart_grid'],
            zerolinecolor=PALETTE['chart_grid'],
            tickfont=dict(color=PALETTE['warm_taupe'])
        ),
        yaxis=dict(
            gridcolor=PALETTE['chart_grid'],
            zerolinecolor=PALETTE['chart_grid'],
            tickfont=dict(color=PALETTE['warm_taupe'])
        )
    )
    return fig

# ==============================================================================
# 📦 DATA & MODEL LOADER
# ==============================================================================
@st.cache_resource
def load_resources():
    models_dir = Path("models")
    clf_bundle = joblib.load(models_dir / "classification_models.joblib")
    reg_bundle = joblib.load(models_dir / "regression_models.joblib")
    clu_bundle = joblib.load(models_dir / "clustering_models.joblib")
    
    df_raw = pd.read_csv("data/data.csv")
    df_active = df_raw[df_raw['minutes_played'] > 0].reset_index(drop=True)
    return clf_bundle, reg_bundle, clu_bundle, df_active

clf_bundle, reg_bundle, clu_bundle, df_active = load_resources()

# ==============================================================================
# 🧭 TOP NAVIGATION BAR (MATCHING IMAGE 1 DESIGN)
# ==============================================================================
header_col1, header_col2 = st.columns([1.6, 3.4])

with header_col1:
    st.markdown(f"""
    <div class="brand-container">
        <div class="brand-icon-box">⚽</div>
        <div>
            <div class="brand-title">FIFA INSIGHTS</div>
            <div class="brand-subtitle">Football Analytics & Tactical Suite</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    if hasattr(st, "segmented_control"):
        selected_page = st.segmented_control(
            label="TopNavigation",
            options=["Overview", "Classification", "Regression", "Clustering", "Player Scout"],
            default="Overview",
            label_visibility="collapsed"
        )
        if not selected_page:
            selected_page = "Overview"
    else:
        selected_page = st.radio(
            label="TopNavigation",
            options=["Overview", "Classification", "Regression", "Clustering", "Player Scout"],
            horizontal=True,
            label_visibility="collapsed"
        )

st.markdown(f"<div style='height:1px; background-color:{PALETTE['light_divider']}; margin:6px 0 16px 0;'></div>", unsafe_allow_html=True)

# ==============================================================================
# 1. 📌 OVERVIEW PAGE
# ==============================================================================
if selected_page == "Overview":
    st.markdown("""
    <h2>Project Overview & Benchmarks</h2>
    <p style="margin-top:-6px;">Machine learning framework modeling tactical roles, performance scoring, and unsupervised player archetypes.</p>
    """, unsafe_allow_html=True)
    
    # Key KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Match Rows</div>
            <div class="metric-value">31,558</div>
            <div class="metric-sub">54,600 raw appearances</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Unique Players</div>
            <div class="metric-value">1,248</div>
            <div class="metric-sub">Across 4 pitch positions</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Classification Acc</div>
            <div class="metric-value">91.42%</div>
            <div class="metric-sub">Gradient Boosting (F1: 0.914)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Regression R²</div>
            <div class="metric-value">0.9686</div>
            <div class="metric-sub">RMSE: 5.507 | MAE: 3.874</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Clustering Sil</div>
            <div class="metric-value">0.6547</div>
            <div class="metric-sub">PCA 2D Silhouette Score</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    
    # 3 Track Summary Cards
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        with st.container(border=True):
            st.markdown("<h4>🎯 Classification Track</h4>", unsafe_allow_html=True)
            st.caption("Automated pitch position categorization across 10 supervised models.")
            st.markdown(f"<p style='font-size:0.88rem; margin:6px 0;'><b>Top Model:</b> Gradient Boosting (<b>91.42%</b> Acc, <b>0.9912</b> OvR AUC)</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:0.85rem; color:{PALETTE['warm_taupe']}; margin:0;'>Baselines: Logistic Reg (89.17%), SVM (88.75%)</p>", unsafe_allow_html=True)
            
    with col_t2:
        with st.container(border=True):
            st.markdown("<h4>📈 Regression Track</h4>", unsafe_allow_html=True)
            st.caption("Continuous estimation of player match performance score (0–100 scale).")
            st.markdown(f"<p style='font-size:0.88rem; margin:6px 0;'><b>Top Model:</b> Gradient Boosting Regressor (<b>R² = 0.9686</b>)</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:0.85rem; color:{PALETTE['warm_taupe']}; margin:0;'>Linear Baselines: OLS & Ridge (R² = 0.9641)</p>", unsafe_allow_html=True)
            
    with col_t3:
        with st.container(border=True):
            st.markdown("<h4>🧩 Clustering Track</h4>", unsafe_allow_html=True)
            st.caption("Unsupervised player segmentation into 4 tactical archetypes.")
            st.markdown(f"<p style='font-size:0.88rem; margin:6px 0;'><b>Optimized Score:</b> Silhouette: <b>0.6547</b> | DB Index: <b>0.5274</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:0.85rem; color:{PALETTE['warm_taupe']}; margin:0;'>Tactical Purity: 100% Goalkeeper isolation</p>", unsafe_allow_html=True)

    # Clean Curated Comparison Charts
    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        with st.container(border=True):
            st.markdown("<h4>Classification Benchmark Accuracy</h4>", unsafe_allow_html=True)
            st.caption("Comparison of top classification algorithms ranked by multi-class test accuracy.")
            clf_chart_data = pd.DataFrame({
                "Model": ["Gradient Boost (Tuned)", "Gradient Boost", "Logistic Reg", "SVM", "Random Forest", "MLP Neural Net", "Bagging", "Decision Tree", "AdaBoost", "KNN", "Gaussian NB"],
                "Accuracy": [91.42, 90.83, 89.17, 88.75, 88.58, 88.17, 87.50, 86.92, 86.42, 85.83, 85.08]
            })
            fig_clf = px.bar(clf_chart_data, x="Accuracy", y="Model", orientation="h", text="Accuracy", color_discrete_sequence=[PALETTE['rich_mocha']])
            fig_clf.update_traces(texttemplate='%{text:.2f}%', textposition='inside')
            fig_clf.update_layout(yaxis=dict(autorange="reversed"), height=250)
            st.plotly_chart(format_fig(fig_clf), width="stretch")
        
    with chart_c2:
        with st.container(border=True):
            st.markdown("<h4>Regression R² Comparison</h4>", unsafe_allow_html=True)
            st.caption("Comparison of regression models ranked by variance explained (R² score).")
            reg_chart_data = pd.DataFrame({
                "Model": ["Gradient Boost", "Random Forest", "Decision Tree", "SVR", "Linear Reg", "Ridge", "KNN", "Lasso"],
                "R2": [0.9686, 0.9686, 0.9680, 0.9661, 0.9641, 0.9641, 0.9641, 0.9626]
            })
            fig_reg = px.bar(reg_chart_data, x="R2", y="Model", orientation="h", text="R2", color_discrete_sequence=[PALETTE['warm_caramel']])
            fig_reg.update_traces(texttemplate='%{text:.4f}', textposition='inside')
            fig_reg.update_layout(yaxis=dict(autorange="reversed"), height=250)
            st.plotly_chart(format_fig(fig_reg), width="stretch")

# ==============================================================================
# 2. 🎯 CLASSIFICATION PAGE
# ==============================================================================
elif selected_page == "Classification":
    st.markdown("""
    <h2>Tactical Position Classification</h2>
    <p style="margin-top:-6px;">Predict player pitch position using supervised classification models trained on match performance metrics.</p>
    """, unsafe_allow_html=True)
    
    col_pred_in, col_pred_out = st.columns([1.1, 0.9])
    
    with col_pred_in:
        with st.container(border=True):
            st.markdown("<h4>Predict Position</h4>", unsafe_allow_html=True)
            st.caption("Select an algorithm and input player stats to predict primary playing role.")
            
            c_algo = st.selectbox("Algorithm", list(clf_bundle['models'].keys()))
            
            # Quick Presets
            preset = st.radio("Quick Role Presets", ["Custom", "Attacking Striker", "Playmaking Midfielder", "Center Defender", "Goalkeeper"], horizontal=True)
            
            defaults = {
                "goals": 2, "shots": 5, "shots_on_target": 3, "xg": 1.8, "key_passes": 2,
                "passes": 28, "acc": 78.0, "tackles": 1, "interceptions": 0, "clearances": 0,
                "saves": 0, "save_pct": 0.0, "distance": 10.5, "speed": 32.4
            }
            if preset == "Attacking Striker":
                defaults.update({"goals": 2, "shots": 6, "shots_on_target": 4, "xg": 1.9, "key_passes": 2, "passes": 22, "tackles": 0, "clearances": 0, "saves": 0})
            elif preset == "Playmaking Midfielder":
                defaults.update({"goals": 0, "shots": 2, "shots_on_target": 1, "xg": 0.3, "key_passes": 5, "passes": 68, "acc": 89.0, "tackles": 3, "clearances": 1, "saves": 0})
            elif preset == "Center Defender":
                defaults.update({"goals": 0, "shots": 0, "shots_on_target": 0, "xg": 0.0, "key_passes": 0, "passes": 42, "acc": 82.0, "tackles": 5, "interceptions": 4, "clearances": 7, "saves": 0})
            elif preset == "Goalkeeper":
                defaults.update({"goals": 0, "shots": 0, "shots_on_target": 0, "xg": 0.0, "key_passes": 0, "passes": 18, "acc": 65.0, "tackles": 0, "interceptions": 0, "clearances": 2, "saves": 6, "save_pct": 85.0})
                
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                inp_goals = st.number_input("Goals", 0, 10, defaults["goals"])
                inp_shots = st.number_input("Shots", 0, 20, defaults["shots"])
                inp_sot = st.number_input("Shots on Target", 0, 15, defaults["shots_on_target"])
                inp_xg = st.number_input("Expected Goals (xG)", 0.0, 5.0, float(defaults["xg"]), step=0.1)
            with fc2:
                inp_passes = st.number_input("Successful Passes", 0, 150, defaults["passes"])
                inp_kp = st.number_input("Key Passes", 0, 15, defaults["key_passes"])
                inp_acc = st.slider("Pass Accuracy %", 0.0, 100.0, float(defaults["acc"]))
                inp_dist = st.number_input("Distance (km)", 0.0, 18.0, float(defaults["distance"]), step=0.5)
            with fc3:
                inp_tackles = st.number_input("Tackles", 0, 15, defaults["tackles"])
                inp_inter = st.number_input("Interceptions", 0, 15, defaults["interceptions"])
                inp_clear = st.number_input("Clearances", 0, 20, defaults["clearances"])
                inp_saves = st.number_input("Saves (GK)", 0, 20, defaults["saves"])

            btn_classify = st.button("Predict Position", use_container_width=True)
        
    with col_pred_out:
        with st.container(border=True):
            st.markdown("<h4>Prediction Output</h4>", unsafe_allow_html=True)
            
            goal_contrib = inp_goals / (90.0/90.0 + 1e-5)
            def_work = (inp_tackles + inp_inter + inp_clear) / (90.0/90.0 + 1e-5)
            save_pct = defaults["save_pct"]
            top_spd = defaults["speed"]
            off_contrib = inp_goals * 2.0 + inp_sot
            def_contrib = inp_tackles * 1.5 + inp_inter + inp_clear
            def_actions = inp_tackles + inp_inter + inp_clear
            
            input_data = pd.DataFrame([[
                inp_goals, 0, inp_shots, inp_sot, inp_xg,
                inp_kp, inp_passes, inp_acc,
                inp_tackles, inp_inter, inp_clear, def_actions,
                inp_saves, save_pct, inp_dist, top_spd,
                off_contrib, def_contrib, goal_contrib, def_work
            ]], columns=clf_bundle['features'])
            
            model = clf_bundle['models'][c_algo]
            scaled_input = clf_bundle['scaler'].transform(input_data)
            pred_pos = model.predict(scaled_input)[0]
            pred_probs = model.predict_proba(scaled_input)[0]
            
            badge_color = {
                "Forward": PALETTE['terracotta'],
                "Midfielder": PALETTE['warm_caramel'],
                "Defender": PALETTE['muted_sage'],
                "Goalkeeper": PALETTE['rich_mocha']
            }.get(pred_pos, PALETTE['rich_mocha'])
            
            st.markdown(f"""
            <div style="background-color:{PALETTE['oatmeal']}; border-left:5px solid {badge_color}; border-radius:10px; padding:16px 20px; margin-bottom:14px;">
                <div style="font-size:0.80rem; color:{PALETTE['muted_sandstone']}; font-weight:600; text-transform:uppercase;">Predicted Playing Role</div>
                <div style="font-size:2rem; font-weight:800; color:{badge_color}; margin:2px 0;">{pred_pos}</div>
                <div style="font-size:0.85rem; color:{PALETTE['warm_taupe']};">Confidence: <b>{max(pred_probs)*100:.1f}%</b> via {c_algo.split('(')[0].strip()}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption("Class Probability Distribution")
            prob_df = pd.DataFrame({"Position": clf_bundle['classes'], "Probability": pred_probs})
            fig_prob = px.bar(prob_df, x="Probability", y="Position", orientation="h", text="Probability", color="Position",
                              color_discrete_map={
                                  "Forward": PALETTE['terracotta'],
                                  "Midfielder": PALETTE['warm_caramel'],
                                  "Defender": PALETTE['muted_sage'],
                                  "Goalkeeper": PALETTE['rich_mocha']
                              })
            fig_prob.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig_prob.update_layout(showlegend=False, xaxis_range=[0, 1.15], height=170)
            st.plotly_chart(format_fig(fig_prob), width="stretch")
        
    # Feature Importance
    with st.container(border=True):
        st.markdown("<h4>Key Discriminative Predictors</h4>", unsafe_allow_html=True)
        st.caption("Relative feature importance weights driving player position classification.")
        rf_model = clf_bundle['models']['Random Forest (88.6%)']
        imp_df = pd.DataFrame({"Feature": clf_bundle['features'], "Importance": rf_model.feature_importances_}).sort_values(by="Importance", ascending=False).head(8)
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[PALETTE['rich_mocha']])
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=210)
        st.plotly_chart(format_fig(fig_imp), width="stretch")

# ==============================================================================
# 3. 📈 REGRESSION PAGE
# ==============================================================================
elif selected_page == "Regression":
    st.markdown("""
    <h2>Player Performance Score Regression</h2>
    <p style="margin-top:-6px;">Estimate player match performance ratings (0–100 scale) based on physical endurance and tactical output.</p>
    """, unsafe_allow_html=True)
    
    col_reg_in, col_reg_out = st.columns([1.1, 0.9])
    
    with col_reg_in:
        with st.container(border=True):
            st.markdown("<h4>Estimate Match Rating</h4>", unsafe_allow_html=True)
            st.caption("Input match statistics to calculate estimated overall player performance rating.")
            
            r_algo = st.selectbox("Algorithm", list(reg_bundle['models'].keys()))
            
            rc1, rc2 = st.columns(2)
            with rc1:
                r_minutes = st.slider("Minutes Played", 1, 90, 85)
                r_goals = st.number_input("Goals Scored", 0, 5, 1)
                r_assists = st.number_input("Assists", 0, 5, 1)
                r_sot = st.number_input("Shots on Target", 0, 10, 2)
                r_xg = st.number_input("Expected Goals (xG)", 0.0, 4.0, 0.85, step=0.1)
                r_kp = st.number_input("Key Passes Created", 0, 10, 3)
                r_passes = st.number_input("Passes Completed", 0, 120, 45)
                r_acc = st.slider("Pass Accuracy Rate %", 0.0, 100.0, 84.0)
            with rc2:
                r_tackles = st.number_input("Tackles Made", 0, 12, 2)
                r_inter = st.number_input("Interceptions Made", 0, 10, 1)
                r_clear = st.number_input("Clearances Made", 0, 15, 1)
                r_saves = st.number_input("Goalkeeper Saves", 0, 15, 0)
                r_save_pct = st.slider("Save Percentage %", 0.0, 100.0, 0.0)
                r_dist = st.number_input("Distance Covered (km)", 0.0, 16.0, 10.2, step=0.5)
                r_speed = st.number_input("Top Speed (km/h)", 0.0, 36.0, 31.5, step=0.5)
                r_stamina = st.slider("Stamina Score", 0, 100, 78)
                
            btn_reg = st.button("Calculate Performance Rating", use_container_width=True)
        
    with col_reg_out:
        with st.container(border=True):
            st.markdown("<h4>Rating Estimation Output</h4>", unsafe_allow_html=True)
            
            input_reg_df = pd.DataFrame([[
                r_minutes, r_goals, r_assists, r_sot, r_xg,
                r_kp, r_acc, r_passes,
                r_tackles, r_inter, r_clear,
                r_saves, r_save_pct,
                r_dist, r_speed, r_stamina
            ]], columns=reg_bundle['features'])
            
            reg_model = reg_bundle['models'][r_algo]
            reg_scaled = reg_bundle['scaler'].transform(input_reg_df)
            pred_rating = max(0.0, min(100.0, float(reg_model.predict(reg_scaled)[0])))
            
            tier_title = "World Class" if pred_rating >= 80 else ("Starting Quality" if pred_rating >= 65 else "Squad Rotation")
            tier_color = PALETTE['muted_sage'] if pred_rating >= 80 else (PALETTE['warm_caramel'] if pred_rating >= 65 else PALETTE['terracotta'])
            
            st.markdown(f"""
            <div style="background-color:{PALETTE['oatmeal']}; border-left:5px solid {tier_color}; border-radius:10px; padding:16px 20px; margin-bottom:14px;">
                <div style="font-size:0.80rem; color:{PALETTE['muted_sandstone']}; font-weight:600; text-transform:uppercase;">Estimated Performance Score</div>
                <div style="font-size:2.4rem; font-weight:800; color:{tier_color}; margin:2px 0;">{pred_rating:.1f} <span style="font-size:1.0rem; color:{PALETTE['warm_taupe']};">/ 100</span></div>
                <div style="font-size:0.85rem; color:{PALETTE['warm_taupe']};">Tier: <b>{tier_title}</b> via {r_algo.split('(')[0].strip()}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Rating Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred_rating,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': PALETTE['deep_espresso']},
                    'bar': {'color': PALETTE['rich_mocha']},
                    'steps': [
                        {'range': [0, 50], 'color': PALETTE['oatmeal']},
                        {'range': [50, 75], 'color': PALETTE['warm_sand']},
                        {'range': [75, 100], 'color': PALETTE['ivory_border']}
                    ],
                    'threshold': {
                        'line': {'color': PALETTE['terracotta'], 'width': 3},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            fig_gauge.update_layout(height=210, margin=dict(l=20, r=20, t=15, b=15), paper_bgcolor=PALETTE['soft_pearl'])
            st.plotly_chart(fig_gauge, width="stretch")

# ==============================================================================
# 4. 🧩 CLUSTERING PAGE
# ==============================================================================
elif selected_page == "Clustering":
    st.markdown("""
    <h2>Unsupervised Tactical Segmentation</h2>
    <p style="margin-top:-6px;">Discover latent player archetypes directly from multidimensional match metrics without ground-truth labels.</p>
    """, unsafe_allow_html=True)
    
    col_clu_in, col_clu_out = st.columns([1.1, 0.9])
    
    with col_clu_in:
        with st.container(border=True):
            st.markdown("<h4>Discover Tactical Archetype</h4>", unsafe_allow_html=True)
            st.caption("Input player attributes to project their tactical profile into the unsupervised latent space.")
            
            clu_algo = st.selectbox("Clustering Method", ["K-Means (Centroid Partitioning)", "Agglomerative Hierarchical (Ward Linkage)"])
            
            c1, c2 = st.columns(2)
            with c1:
                cl_goals = st.number_input("Goals per match", 0.0, 3.0, 0.5, step=0.1)
                cl_sot = st.number_input("Shots on Target", 0.0, 5.0, 1.2, step=0.1)
                cl_xg = st.number_input("Expected Goals (xG)", 0.0, 2.5, 0.45, step=0.05)
                cl_passes = st.number_input("Passes per match", 0.0, 100.0, 42.0, step=2.0)
                cl_acc = st.slider("Pass Accuracy %", 0.0, 100.0, 81.0)
                cl_kp = st.number_input("Key Passes", 0.0, 6.0, 1.4, step=0.1)
            with c2:
                cl_tackles = st.number_input("Tackles per match", 0.0, 8.0, 2.1, step=0.1)
                cl_inter = st.number_input("Interceptions", 0.0, 6.0, 1.3, step=0.1)
                cl_clear = st.number_input("Clearances", 0.0, 10.0, 1.8, step=0.1)
                cl_saves = st.number_input("Saves (Goalkeeper)", 0.0, 10.0, 0.0, step=0.5)
                cl_dist = st.number_input("Distance Covered (km)", 0.0, 15.0, 10.4, step=0.2)
                cl_spd = st.number_input("Top Speed (km/h)", 0.0, 36.0, 31.8, step=0.2)
                
            btn_clu = st.button("Cluster Tactical Archetype", use_container_width=True)
        
    with col_clu_out:
        with st.container(border=True):
            st.markdown("<h4>Archetype Assignment</h4>", unsafe_allow_html=True)
            
            cl_raw = np.array([[
                cl_goals, 0.2, cl_sot, cl_xg, cl_kp, cl_acc, cl_passes,
                cl_tackles, cl_inter, cl_clear, cl_saves, (80.0 if cl_saves > 0 else 0.0),
                cl_dist, cl_spd, (cl_goals*2.0 + cl_sot), (cl_tackles*1.5 + cl_clear)
            ]])
            
            mean_v = clu_bundle['data_sample'].mean().values
            std_v = clu_bundle['data_sample'].std().values + 1e-5
            cl_scaled_arr = (cl_raw - mean_v) / std_v
            cl_scaled_df = pd.DataFrame(cl_scaled_arr, columns=clu_bundle['features'])
            
            cluster_id = clu_bundle['kmeans'].predict(cl_scaled_df)[0]
            archetype_title = clu_bundle['archetypes'].get(cluster_id, "All-Round Outfield Player")
            
            st.markdown(f"""
            <div style="background-color:{PALETTE['oatmeal']}; border-left:5px solid {PALETTE['rich_mocha']}; border-radius:10px; padding:16px 20px; margin-bottom:14px;">
                <div style="font-size:0.80rem; color:{PALETTE['muted_sandstone']}; font-weight:600; text-transform:uppercase;">Identified Latent Archetype</div>
                <div style="font-size:1.7rem; font-weight:800; color:{PALETTE['deep_espresso']}; margin:2px 0;">{archetype_title}</div>
                <div style="font-size:0.85rem; color:{PALETTE['warm_taupe']};">Assigned to: <b>Cluster #{cluster_id}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            cl_pca_coord = clu_bundle['pca'].transform(cl_scaled_df)[0]
            
            pca_df = pd.DataFrame(clu_bundle['pca_coords'], columns=["PC1", "PC2"]).head(600)
            pca_df['Cluster'] = [clu_bundle['archetypes'].get(cid, f"Cluster {cid}") for cid in clu_bundle['kmeans'].labels_[:600]]
            
            fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", opacity=0.6,
                                 color_discrete_sequence=COLORWAY,
                                 title="2D PCA Latent Space Projection")
            
            fig_pca.add_trace(go.Scatter(
                x=[cl_pca_coord[0]], y=[cl_pca_coord[1]],
                mode="markers+text",
                marker=dict(size=14, color=PALETTE['terracotta'], symbol="star", line=dict(width=2, color="black")),
                name="Your Player",
                text=["Assigned Profile"],
                textposition="top center"
            ))
            fig_pca.update_layout(height=260)
            st.plotly_chart(format_fig(fig_pca), width="stretch")

# ==============================================================================
# 5. 🔍 PLAYER SCOUT
# ==============================================================================
elif selected_page == "Player Scout":
    st.markdown("""
    <h2>Player Tactical Scout & Profiler</h2>
    <p style="margin-top:-6px;">Search any player from the FIFA dataset to review their seasonal match statistics and tactical profile.</p>
    """, unsafe_allow_html=True)
    
    with st.container(border=True):
        player_names = df_active['player_name'].unique().tolist()
        sel_player = st.selectbox("Search & Select Professional Player", sorted(player_names[:300]))
        
        player_matches = df_active[df_active['player_name'] == sel_player]
        player_pos = player_matches['position'].iloc[0]
        avg_minutes = player_matches['minutes_played'].mean()
        avg_rating = player_matches['player_rating'].mean() if 'player_rating' in player_matches.columns else 75.0
        total_goals = player_matches['goals'].sum()
        total_assists = player_matches['assists'].sum()
        
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Primary Position</div>
                <div class="metric-value">{player_pos}</div>
                <div class="metric-sub">{len(player_matches)} Matches Tracked</div>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Season Rating</div>
                <div class="metric-value">{avg_rating:.1f}</div>
                <div class="metric-sub">Average Player Score</div>
            </div>
            """, unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Goal Output</div>
                <div class="metric-value">{total_goals}</div>
                <div class="metric-sub">{player_matches['goals'].mean():.2f} Goals / Match</div>
            </div>
            """, unsafe_allow_html=True)
        with sc4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Playmaking Assists</div>
                <div class="metric-value">{total_assists}</div>
                <div class="metric-sub">{player_matches['assists'].mean():.2f} Assists / Match</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown("<h4>Tactical Radar Fingerprint</h4>", unsafe_allow_html=True)
        st.caption("Standardized attribute distribution relative to dataset positional baselines.")
        
        categories = ['Goals Threat', 'Passing Volume', 'Pass Accuracy', 'Defensive Work', 'Physical Speed', 'Endurance Distance']
        
        r_vals = [
            min(100, float(player_matches['goals'].mean() * 60 + 20)),
            min(100, float(player_matches['successful_passes'].mean() * 1.5 + 20)),
            min(100, float(player_matches['pass_accuracy'].mean())),
            min(100, float(player_matches['tackles'].mean() * 25 + 15)),
            min(100, float(player_matches['top_speed_kmh'].mean() * 2.8)),
            min(100, float(player_matches['distance_covered_km'].mean() * 8.5))
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=categories,
            fill='toself',
            fillcolor='rgba(111, 78, 55, 0.25)',
            line=dict(color=PALETTE['rich_mocha'], width=2),
            name=sel_player
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor=PALETTE['chart_grid']),
                angularaxis=dict(gridcolor=PALETTE['chart_grid'], tickfont=dict(color=PALETTE['deep_espresso']))
            ),
            paper_bgcolor=PALETTE['soft_pearl'],
            height=320,
            margin=dict(l=35, r=35, t=15, b=15)
        )
        st.plotly_chart(fig_radar, width="stretch")
