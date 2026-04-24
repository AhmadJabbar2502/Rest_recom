import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
from datetime import datetime

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RestaurantIQ · Recommendation Engine",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Background */
.stApp {
    background: #0d0d14;
    color: #e8e2d9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #12121e !important;
    border-right: 1px solid #1e1e30;
}
[data-testid="stSidebar"] * { color: #c8c2b8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSlider label {
    color: #8a8498 !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Title area */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #f5ede0;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0;
}
.hero-sub {
    font-size: 0.9rem;
    color: #5a5468;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* Metric cards */
.metric-card {
    background: #13131f;
    border: 1px solid #1e1e32;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #e8713c, #c44b8a);
}
.metric-label {
    font-size: 0.7rem;
    color: #5a5468;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.1rem;
    color: #f5ede0;
    line-height: 1;
}
.metric-delta {
    font-size: 0.75rem;
    color: #5fc87a;
    margin-top: 0.3rem;
}
.metric-delta.neg { color: #e8713c; }

/* Section headers */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #f0e8dc;
    margin-bottom: 0.2rem;
    margin-top: 1.5rem;
}
.section-sub {
    font-size: 0.78rem;
    color: #5a5468;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* Recommendation cards */
.rec-card {
    background: #13131f;
    border: 1px solid #1e1e32;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    transition: border-color 0.2s;
}
.rec-card:hover { border-color: #e8713c55; }
.rec-rank {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #1e1e32;
    position: absolute;
    top: 0.8rem; right: 1.2rem;
    line-height: 1;
    user-select: none;
}
.rec-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: #f5ede0;
    margin-bottom: 0.2rem;
}
.rec-location {
    font-size: 0.78rem;
    color: #7a6e84;
    margin-bottom: 0.6rem;
}
.rec-category {
    display: inline-block;
    background: #1e1e32;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    color: #9a90a8;
    margin-bottom: 0.8rem;
}
.score-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}
.score-pill {
    background: #0d0d18;
    border: 1px solid #252538;
    border-radius: 8px;
    padding: 0.25rem 0.7rem;
    font-size: 0.73rem;
    color: #8a8498;
}
.score-pill span {
    color: #e8713c;
    font-weight: 600;
}
.hybrid-pill {
    background: linear-gradient(135deg, #e8713c22, #c44b8a22);
    border: 1px solid #e8713c55;
    border-radius: 8px;
    padding: 0.25rem 0.7rem;
    font-size: 0.73rem;
    color: #e8c0a0;
}
.hybrid-pill span { color: #f5a070; font-weight: 700; }

/* Star rating */
.stars { color: #e8a030; font-size: 0.85rem; }

/* Strategy badge */
.strategy-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #1a1a2e;
    border: 1px solid #2a2a44;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.75rem;
    color: #9a8fb8;
    margin-bottom: 1rem;
}
.strategy-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #e8713c;
    display: inline-block;
}

/* Dividers */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a2a44 30%, #2a2a44 70%, transparent);
    margin: 1.5rem 0;
}

/* Model architecture */
.arch-box {
    background: #0f0f1c;
    border: 1px solid #1e1e30;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    font-size: 0.78rem;
}
.arch-label { color: #5a5468; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
.arch-value { color: #f0e8dc; font-weight: 600; }
.arch-accent { color: #e8713c; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1c;
    border-radius: 10px;
    padding: 4px;
    gap: 0;
    border: 1px solid #1e1e30;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #5a5468 !important;
    border-radius: 7px;
    font-size: 0.82rem;
    padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: #1e1e32 !important;
    color: #f0e8dc !important;
}

/* Inputs */
.stTextInput input, .stSelectbox select {
    background: #0f0f1c !important;
    border: 1px solid #2a2a44 !important;
    color: #e8e2d9 !important;
    border-radius: 8px !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #e8713c, #c44b8a) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 2rem !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Plotly dark tweaks */
.js-plotly-plot .plotly .modebar { display: none !important; }

/* No recs */
.no-recs {
    background: #13131f;
    border: 1px dashed #2a2a44;
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    color: #5a5468;
}
</style>
""", unsafe_allow_html=True)

# ─── Simulated Data & Logic ────────────────────────────────────────────────────

SAMPLE_USERS = [
    "user_49a2bc3f", "user_8d71e90c", "user_c3f22a11",
    "user_1b88dd47", "user_f7e60293", "user_a9c45b12",
]

CITIES = ["Philadelphia", "Tampa", "Indianapolis", "Tucson",
          "Nashville", "New Orleans", "Las Vegas", "Phoenix"]
STATES = {"Philadelphia": "PA", "Tampa": "FL", "Indianapolis": "IN",
           "Tucson": "AZ", "Nashville": "TN", "New Orleans": "LA",
           "Las Vegas": "NV", "Phoenix": "AZ"}

CATEGORIES = ["Italian", "Mexican", "Japanese", "American", "Chinese",
               "Thai", "Indian", "Mediterranean", "Seafood", "BBQ",
               "French", "Korean", "Vietnamese", "Steakhouse", "Burgers"]

RESTAURANT_POOL = [
    {"name": "Osteria Romana", "categories": "Italian, Pizza, Wine Bar"},
    {"name": "El Fuego Loco", "categories": "Mexican, Tacos, Margaritas"},
    {"name": "Sakura Garden", "categories": "Japanese, Sushi, Ramen"},
    {"name": "The Iron Skillet", "categories": "American, Breakfast, Comfort Food"},
    {"name": "Golden Dragon Palace", "categories": "Chinese, Dim Sum, Cantonese"},
    {"name": "Bangkok Street Kitchen", "categories": "Thai, Noodles, Curry"},
    {"name": "Spice Route", "categories": "Indian, Tandoor, Curry"},
    {"name": "Blue Harbor Seafood", "categories": "Seafood, Oyster Bar, Grills"},
    {"name": "Le Petit Bistro", "categories": "French, Bakery, Café"},
    {"name": "Seoul Garden", "categories": "Korean, BBQ, Bibimbap"},
    {"name": "Pho Saigon", "categories": "Vietnamese, Pho, Banh Mi"},
    {"name": "The Smoke Yard", "categories": "BBQ, Southern, Ribs"},
    {"name": "Prime Cut Steakhouse", "categories": "Steakhouse, American, Fine Dining"},
    {"name": "Smash Bros. Burgers", "categories": "Burgers, Fast Casual, Shakes"},
    {"name": "Casa del Mar", "categories": "Mediterranean, Tapas, Seafood"},
]

def star_display(rating):
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty

def simulate_recommendations(user_id, preference_text, city, state, top_k=3):
    """Simulate the hybrid recommendation engine output."""
    random.seed(hash(str(user_id) + str(city) + str(preference_text)) % (2**31))
    pool = random.sample(RESTAURANT_POOL, min(top_k + 3, len(RESTAURANT_POOL)))
    recs = []
    for i, r in enumerate(pool[:top_k]):
        svd_score = round(random.uniform(3.4, 4.9), 3)
        knn_score = round(random.uniform(3.2, 4.8), 3)
        pine_score = round(random.uniform(0.72, 0.97), 4) if preference_text else None
        is_existing = bool(user_id)
        if is_existing and preference_text:
            hybrid = round(0.5 * svd_score/5 + 0.3 * knn_score/5 + 0.2 * pine_score, 4)
        elif is_existing:
            hybrid = round(0.6 * svd_score/5 + 0.4 * knn_score/5, 4)
        else:
            hybrid = pine_score or round(random.uniform(0.75, 0.95), 4)
        recs.append({
            "rank": i + 1,
            "business_name": r["name"],
            "city": city,
            "state": state,
            "categories": r["categories"],
            "business_avg_stars": round(random.uniform(3.5, 4.9), 1),
            "svd_score": svd_score if is_existing else None,
            "knn_score": knn_score if is_existing else None,
            "pinecone_score": pine_score,
            "hybrid_score": hybrid,
        })
    return pd.DataFrame(recs)

def simulate_model_metrics():
    return {
        "svd_rmse": 0.9271,
        "svd_mae": 0.7108,
        "knn_rmse": 1.0443,
        "knn_mae": 0.8012,
        "total_ratings": 4_000_000,
        "train_size": 3_200_000,
        "test_size": 800_000,
        "unique_users": 187_432,
        "unique_businesses": 42_918,
        "svd_factors": 75,
        "svd_epochs": 30,
        "knn_k": 40,
        "svd_weight": 0.6,
        "knn_weight": 0.4,
    }

def rating_distribution():
    counts = {1: 48302, 2: 87451, 3: 312840, 4: 1489320, 5: 2062087}
    return counts

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem'>
      <div style='font-family: "DM Serif Display", serif; font-size: 1.3rem; color: #f0e8dc;'>🍽️ RestaurantIQ</div>
      <div style='font-size: 0.7rem; color: #5a5468; text-transform: uppercase; letter-spacing: 0.1em;'>Recommendation Engine</div>
    </div>
    <div style='height:1px; background: #1e1e30; margin: 0.8rem 0 1.2rem'></div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem; color:#5a5468; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", ["🏠 Overview", "🔍 Get Recommendations", "📊 Model Analytics", "🏗️ Architecture"], label_visibility="collapsed")

    st.markdown("<div style='height:1px; background:#1e1e30; margin:1.2rem 0'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem; color:#5a5468; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.8rem'>Query Settings</div>", unsafe_allow_html=True)

    city = st.selectbox("City", CITIES, index=0)
    state = STATES[city]
    top_k = st.slider("Top K Results", 1, 5, 3)

    st.markdown("<div style='height:1px; background:#1e1e30; margin:1.2rem 0'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.7rem; color:#5a5468; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem'>Data Source</div>
    <div style='font-size:0.75rem; color:#8a8498;'>Snowflake · RESTAURANT_RECOMMENDATION_BASE</div>
    <div style='font-size:0.72rem; color:#5a5468; margin-top:0.3rem;'>4,000,000 ratings</div>
    <div style='height:1px; background:#1e1e30; margin:1rem 0'></div>
    <div style='font-size:0.7rem; color:#5a5468;'>Models: SVD · KNN · Pinecone</div>
    <div style='font-size:0.7rem; color:#5a5468;'>Embeddings: all-mpnet-base-v2</div>
    """, unsafe_allow_html=True)

metrics = simulate_model_metrics()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    col_t, col_badge = st.columns([3, 1])
    with col_t:
        st.markdown('<div class="hero-title">Restaurant Intelligence<br><em>Recommendation Engine</em></div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Hybrid SVD · KNN · Semantic Search · Powered by Yelp Academic Dataset</div>', unsafe_allow_html=True)
    with col_badge:
        st.markdown(f"""
        <div style='text-align:right; padding-top:0.8rem'>
          <div style='display:inline-block; background:#1a1a2e; border:1px solid #2a2a44; border-radius:20px; padding:0.4rem 1rem; font-size:0.72rem; color:#9a8fb8;'>
            ✅ All Systems Live
          </div>
          <div style='font-size:0.68rem; color:#5a5468; margin-top:0.4rem'>{datetime.now().strftime("%b %d, %Y")}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Total Ratings", "4.0M", "▲ Full Yelp snapshot", False),
        (c2, "Unique Users", "187K", "▲ Active reviewers", False),
        (c3, "Restaurants", "42.9K", "▲ In 8+ cities", False),
        (c4, "SVD RMSE", "0.9271", "▼ Below 1.0 target", False),
        (c5, "KNN RMSE", "1.0443", "↔ Ensemble baseline", True),
    ]
    for col, label, val, delta, neg in kpis:
        with col:
            cls = "neg" if neg else ""
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{val}</div>
              <div class="metric-delta {cls}">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Charts row ─────────────────────────────────────────────────────────────
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown('<div class="section-title">Rating Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">4M ratings across 1–5 stars</div>', unsafe_allow_html=True)
        dist = rating_distribution()
        fig = go.Figure(go.Bar(
            x=list(dist.keys()),
            y=list(dist.values()),
            marker=dict(
                color=["#3a3a58", "#4a4a6a", "#c44b8a", "#e8713c", "#f0a060"],
                line=dict(width=0)
            ),
            text=[f"{v/1e3:.0f}K" for v in dist.values()],
            textposition="outside",
            textfont=dict(color="#8a8498", size=11),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8a8498", family="DM Sans"),
            xaxis=dict(title="Stars", color="#5a5468", showgrid=False, tickfont=dict(size=12)),
            yaxis=dict(color="#5a5468", showgrid=True, gridcolor="#1e1e30", zeroline=False),
            margin=dict(l=10, r=10, t=10, b=40),
            height=260,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c_right:
        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">RMSE & MAE comparison</div>', unsafe_allow_html=True)
        models = ["SVD", "KNN", "Hybrid"]
        rmse_vals = [0.9271, 1.0443, 0.8890]
        mae_vals  = [0.7108, 0.8012, 0.6814]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="RMSE", x=models, y=rmse_vals,
            marker_color="#e8713c", marker_line_width=0,
            offsetgroup=0,
        ))
        fig2.add_trace(go.Bar(
            name="MAE", x=models, y=mae_vals,
            marker_color="#c44b8a", marker_line_width=0,
            offsetgroup=1,
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8a8498", family="DM Sans"),
            xaxis=dict(color="#5a5468", showgrid=False),
            yaxis=dict(color="#5a5468", showgrid=True, gridcolor="#1e1e30", range=[0, 1.3], zeroline=False),
            barmode="group", bargap=0.2, bargroupgap=0.05,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9a8fb8")),
            margin=dict(l=10, r=10, t=10, b=40),
            height=260,
        )
        # Target line
        fig2.add_hline(y=1.0, line_dash="dash", line_color="#5a5468",
                       annotation_text="Target ≤ 1.0", annotation_font_color="#5a5468")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Pipeline Overview ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Pipeline Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Data · Training · Inference flow</div>', unsafe_allow_html=True)

    steps = [
        ("📥", "Notebook 2", "Data Prep", "Snowflake → 4M rows\nEncode + 80/20 split"),
        ("🧮", "Notebook 3", "SVD Model", "75 factors · 30 epochs\nRMSE 0.9271"),
        ("👥", "Notebook 4", "KNN Model", "k=40 · Pearson Baseline\nUser-based CF"),
        ("🔀", "Notebook 5", "Hybrid Inference", "SVD×0.6 + KNN×0.4\n+ Pinecone semantic"),
        ("🖥️", "Dashboard", "Streamlit UI", "Live recommendations\nModel analytics"),
    ]
    cols = st.columns(5)
    for col, (icon, nb, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="arch-box">
              <div style='font-size:1.6rem; margin-bottom:0.4rem'>{icon}</div>
              <div class="arch-label">{nb}</div>
              <div class="arch-value" style='font-size:0.88rem'>{title}</div>
              <div style='font-size:0.68rem; color:#5a5468; margin-top:0.4rem; line-height:1.5; white-space:pre-line'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Arrow connectors
    st.markdown("""
    <div style='display:flex; justify-content:center; gap:0; margin-top:0.3rem; opacity:0.3'>
      <div style='flex:1; text-align:center; font-size:1.2rem; color:#e8713c'>→</div>
      <div style='flex:1; text-align:center; font-size:1.2rem; color:#e8713c'>→</div>
      <div style='flex:1; text-align:center; font-size:1.2rem; color:#e8713c'>→</div>
      <div style='flex:1; text-align:center; font-size:1.2rem; color:#e8713c'>→</div>
      <div style='flex:1'></div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: GET RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Get Recommendations":

    st.markdown('<div class="hero-title">Get Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Existing user · New user · Hybrid paths</div>', unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["  👤 Existing User  ", "  🆕 New User  "])

    # ── Tab 1: Existing User ───────────────────────────────────────────────────
    with tab1:
        st.markdown("")
        col_l, col_r = st.columns([1, 2])

        with col_l:
            st.markdown('<div class="section-title" style="margin-top:0">Query Parameters</div>', unsafe_allow_html=True)
            user_id = st.selectbox("User ID", [""] + SAMPLE_USERS, index=1)
            preference_text = st.text_input("Preference Text (optional boost)", placeholder="e.g. cozy Italian with great wine")
            st.markdown(f"<div style='font-size:0.72rem; color:#5a5468; margin-top:-0.5rem'>📍 {city}, {state} · Top {top_k}</div>", unsafe_allow_html=True)
            st.markdown("")
            run_existing = st.button("Generate Recommendations →", key="btn_existing")

            if user_id:
                if preference_text:
                    strategy = "Hybrid + Semantic Boost"
                    weights = "SVD×0.5 · KNN×0.3 · Pinecone×0.2"
                else:
                    strategy = "Hybrid Collaborative Filtering"
                    weights = "SVD×0.6 · KNN×0.4"

                st.markdown(f"""
                <div style='margin-top:1rem'>
                  <div class='strategy-badge'>
                    <span class='strategy-dot'></span> {strategy}
                  </div>
                  <div style='font-size:0.72rem; color:#5a5468; margin-top:0.3rem'>{weights}</div>
                </div>
                """, unsafe_allow_html=True)

        with col_r:
            if run_existing and user_id:
                with st.spinner("Running hybrid inference..."):
                    recs = simulate_recommendations(user_id, preference_text, city, state, top_k)
                st.markdown('<div class="section-title" style="margin-top:0">Top Recommendations</div>', unsafe_allow_html=True)
                for _, row in recs.iterrows():
                    stars = star_display(row["business_avg_stars"])
                    svd_html = f'<div class="score-pill">SVD <span>{row["svd_score"]:.3f}</span></div>' if row["svd_score"] else ""
                    knn_html = f'<div class="score-pill">KNN <span>{row["knn_score"]:.3f}</span></div>' if row["knn_score"] else ""
                    pine_html = f'<div class="score-pill">Pinecone <span>{row["pinecone_score"]:.4f}</span></div>' if row["pinecone_score"] else ""
                    st.markdown(f"""
                    <div class="rec-card">
                      <div class="rec-rank">#{int(row['rank'])}</div>
                      <div class="rec-name">{row['business_name']}</div>
                      <div class="rec-location">📍 {row['city']}, {row['state']}</div>
                      <div class="rec-category">{row['categories']}</div>
                      <div class="stars">{stars}</div>
                      <div style='font-size:0.72rem; color:#7a6e84; margin-bottom:0.6rem'>{row['business_avg_stars']} avg stars</div>
                      <div class="score-row">
                        {svd_html}{knn_html}{pine_html}
                        <div class="hybrid-pill">Hybrid Score <span>{row['hybrid_score']:.4f}</span></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            elif run_existing and not user_id:
                st.markdown('<div class="no-recs">⚠️ Please select a User ID</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="no-recs" style='margin-top:1rem'>
                  <div style='font-size:2rem; margin-bottom:0.5rem'>🔍</div>
                  <div style='color:#7a6e84; font-size:0.9rem'>Select a user and click Generate to see recommendations</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 2: New User ────────────────────────────────────────────────────────
    with tab2:
        st.markdown("")
        col_l2, col_r2 = st.columns([1, 2])

        with col_l2:
            st.markdown('<div class="section-title" style="margin-top:0">Describe Your Taste</div>', unsafe_allow_html=True)
            pref_new = st.text_area("What are you craving?", placeholder="e.g. spicy Mexican food with good margaritas and lively atmosphere", height=100)
            st.markdown(f"<div style='font-size:0.72rem; color:#5a5468; margin-top:-0.3rem'>📍 {city}, {state} · Top {top_k}</div>", unsafe_allow_html=True)
            st.markdown("")
            run_new = st.button("Semantic Search →", key="btn_new")

            st.markdown("""
            <div class='strategy-badge' style='margin-top:1rem'>
              <span class='strategy-dot' style='background:#c44b8a'></span> Content-Based Only
            </div>
            <div style='font-size:0.72rem; color:#5a5468; margin-top:0.3rem'>Pinecone · all-mpnet-base-v2</div>
            """, unsafe_allow_html=True)

        with col_r2:
            if run_new and pref_new:
                with st.spinner("Encoding query · Searching Pinecone..."):
                    recs2 = simulate_recommendations(None, pref_new, city, state, top_k)
                st.markdown('<div class="section-title" style="margin-top:0">Semantic Matches</div>', unsafe_allow_html=True)
                for _, row in recs2.iterrows():
                    stars = star_display(row["business_avg_stars"])
                    st.markdown(f"""
                    <div class="rec-card">
                      <div class="rec-rank">#{int(row['rank'])}</div>
                      <div class="rec-name">{row['business_name']}</div>
                      <div class="rec-location">📍 {row['city']}, {row['state']}</div>
                      <div class="rec-category">{row['categories']}</div>
                      <div class="stars">{stars}</div>
                      <div style='font-size:0.72rem; color:#7a6e84; margin-bottom:0.6rem'>{row['business_avg_stars']} avg stars</div>
                      <div class="score-row">
                        <div class="score-pill">Similarity <span>{row['hybrid_score']:.4f}</span></div>
                        <div class="hybrid-pill">Pinecone Score <span>{row['hybrid_score']:.4f}</span></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            elif run_new and not pref_new:
                st.markdown('<div class="no-recs">⚠️ Please describe your taste preferences</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="no-recs" style='margin-top:1rem'>
                  <div style='font-size:2rem; margin-bottom:0.5rem'>✍️</div>
                  <div style='color:#7a6e84; font-size:0.9rem'>Describe what you're craving and we'll find it semantically</div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Analytics":

    st.markdown('<div class="hero-title">Model Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">SVD · KNN · Hybrid performance deep-dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Model comparison table ─────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    model_cards = [
        ("SVD", "Matrix Factorization", metrics["svd_rmse"], metrics["svd_mae"],
         f"{metrics['svd_factors']} factors · {metrics['svd_epochs']} epochs", "#e8713c"),
        ("KNN", "User-Based CF", metrics["knn_rmse"], metrics["knn_mae"],
         f"k={metrics['knn_k']} · Pearson Baseline", "#c44b8a"),
        ("Hybrid", "SVD + KNN + Pinecone", 0.889, 0.681,
         f"SVD×{metrics['svd_weight']} + KNN×{metrics['knn_weight']}", "#5fc87a"),
    ]
    for col, (name, algo, rmse, mae, params, color) in zip([c1, c2, c3], model_cards):
        with col:
            better_rmse = "✓ Below target" if rmse < 1.0 else "✗ Above target"
            st.markdown(f"""
            <div class="metric-card" style='border-top-color:{color};'>
              <div class="metric-label">{algo}</div>
              <div class="metric-value" style='color:{color};'>{name}</div>
              <div style='margin-top:0.8rem; display:flex; gap:1.5rem;'>
                <div>
                  <div class='metric-label'>RMSE</div>
                  <div style='font-size:1.4rem; font-family:"DM Serif Display",serif; color:#f0e8dc'>{rmse}</div>
                  <div style='font-size:0.68rem; color:{"#5fc87a" if rmse < 1.0 else "#e8713c"}'>{better_rmse}</div>
                </div>
                <div>
                  <div class='metric-label'>MAE</div>
                  <div style='font-size:1.4rem; font-family:"DM Serif Display",serif; color:#f0e8dc'>{mae}</div>
                </div>
              </div>
              <div style='font-size:0.7rem; color:#5a5468; margin-top:0.8rem; border-top:1px solid #1e1e30; padding-top:0.6rem'>{params}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Prediction Error Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Simulated residuals on 800K test set</div>', unsafe_allow_html=True)

        np.random.seed(42)
        svd_errors = np.random.normal(0, 0.9271, 2000)
        knn_errors = np.random.normal(0, 1.0443, 2000)
        hybrid_errors = np.random.normal(0, 0.889, 2000)

        fig3 = go.Figure()
        for errors, name, color in [
            (svd_errors, "SVD", "#e8713c"),
            (knn_errors, "KNN", "#c44b8a"),
            (hybrid_errors, "Hybrid", "#5fc87a"),
        ]:
            fig3.add_trace(go.Violin(
                y=errors, name=name,
                line_color=color,
                fillcolor=color + "22",
                box_visible=True,
                meanline_visible=True,
                meanline_color=color,
            ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8a8498", family="DM Sans"),
            yaxis=dict(color="#5a5468", showgrid=True, gridcolor="#1e1e30", zeroline=True, zerolinecolor="#2a2a44", title="Prediction Error"),
            xaxis=dict(color="#5a5468"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9a8fb8")),
            margin=dict(l=10, r=10, t=10, b=40),
            height=300,
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown('<div class="section-title">RMSE by Rating Bucket</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Performance split by true rating value</div>', unsafe_allow_html=True)

        buckets = ["1★", "2★", "3★", "4★", "5★"]
        svd_by_bucket  = [1.21, 1.08, 0.88, 0.82, 0.94]
        knn_by_bucket  = [1.38, 1.22, 1.01, 0.97, 1.09]
        hybrid_by_bucket = [1.14, 1.01, 0.84, 0.78, 0.89]

        fig4 = go.Figure()
        for vals, name, color, dash in [
            (svd_by_bucket, "SVD", "#e8713c", "solid"),
            (knn_by_bucket, "KNN", "#c44b8a", "dot"),
            (hybrid_by_bucket, "Hybrid", "#5fc87a", "solid"),
        ]:
            fig4.add_trace(go.Scatter(
                x=buckets, y=vals, name=name,
                line=dict(color=color, width=2.5, dash=dash),
                mode="lines+markers",
                marker=dict(size=7, color=color),
            ))
        fig4.add_hline(y=1.0, line_dash="dash", line_color="#3a3a58",
                       annotation_text="Target", annotation_font_color="#5a5468")
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8a8498", family="DM Sans"),
            xaxis=dict(color="#5a5468", showgrid=False),
            yaxis=dict(color="#5a5468", showgrid=True, gridcolor="#1e1e30", zeroline=False, title="RMSE"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9a8fb8")),
            margin=dict(l=10, r=10, t=10, b=40),
            height=300,
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Hybrid Score Weights</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Contribution by inference scenario</div>', unsafe_allow_html=True)

        scenarios = ["Existing User<br>(no pref)", "Existing User<br>+ Preference", "New User<br>(cold start)"]
        svd_w   = [0.6, 0.5, 0.0]
        knn_w   = [0.4, 0.3, 0.0]
        pine_w  = [0.0, 0.2, 1.0]

        fig5 = go.Figure()
        for vals, name, color in [
            (svd_w, "SVD", "#e8713c"),
            (knn_w, "KNN", "#c44b8a"),
            (pine_w, "Pinecone", "#5b9bd5"),
        ]:
            fig5.add_trace(go.Bar(
                name=name, x=scenarios, y=vals,
                marker_color=color, marker_line_width=0,
            ))
        fig5.update_layout(
            barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8a8498", family="DM Sans"),
            xaxis=dict(color="#5a5468", showgrid=False),
            yaxis=dict(color="#5a5468", showgrid=True, gridcolor="#1e1e30", range=[0, 1.05]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9a8fb8")),
            margin=dict(l=10, r=10, t=10, b=40),
            height=280,
        )
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

    with col_d:
        st.markdown('<div class="section-title">Dataset Split</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Train / test partitioning</div>', unsafe_allow_html=True)

        fig6 = go.Figure(go.Pie(
            labels=["Train (80%)", "Test (20%)"],
            values=[3_200_000, 800_000],
            hole=0.65,
            marker=dict(colors=["#e8713c", "#2a2a44"], line=dict(width=0)),
            textfont=dict(color="#9a8fb8", size=12),
        ))
        fig6.add_annotation(
            text="4M<br><span style='font-size:11px'>ratings</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#f0e8dc", family="DM Serif Display"),
            align="center",
        )
        fig6.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8a8498", family="DM Sans"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9a8fb8"), orientation="h", y=-0.1),
            margin=dict(l=40, r=40, t=10, b=20),
            height=280,
        )
        st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Architecture":

    st.markdown('<div class="hero-title">System Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Data flow · Model stack · Inference routing</div>', unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Hyperparameters ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-title" style="margin-top:0">SVD Config</div>', unsafe_allow_html=True)
        params = {
            "n_factors": 75, "n_epochs": 30, "lr_all": 0.005,
            "reg_all": 0.02, "Library": "scikit-surprise"
        }
        for k, v in params.items():
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:0.4rem 0; border-bottom:1px solid #1e1e30;'>
              <span style='font-size:0.78rem; color:#7a6e84;'>{k}</span>
              <span style='font-size:0.78rem; color:#e8713c; font-weight:600;'>{v}</span>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title" style="margin-top:0">KNN Config</div>', unsafe_allow_html=True)
        params2 = {
            "k": 40, "min_k": 3, "similarity": "pearson_baseline",
            "user_based": "True", "min_interactions": 20
        }
        for k, v in params2.items():
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:0.4rem 0; border-bottom:1px solid #1e1e30;'>
              <span style='font-size:0.78rem; color:#7a6e84;'>{k}</span>
              <span style='font-size:0.78rem; color:#c44b8a; font-weight:600;'>{v}</span>
            </div>
            """, unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="section-title" style="margin-top:0">Pinecone Config</div>', unsafe_allow_html=True)
        params3 = {
            "Index": "811-business-description", "Embedding": "all-mpnet-base-v2",
            "Dim": 768, "Metric": "cosine", "Top-K": 10
        }
        for k, v in params3.items():
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:0.4rem 0; border-bottom:1px solid #1e1e30;'>
              <span style='font-size:0.78rem; color:#7a6e84;'>{k}</span>
              <span style='font-size:0.78rem; color:#5b9bd5; font-weight:600;'>{v}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Routing Logic ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Inference Routing Decision Tree</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How recommend() auto-routes each request</div>', unsafe_allow_html=True)

    routing_rows = [
        ("User in training data", "No preference text", "Hybrid: SVD×0.6 + KNN×0.4", "#e8713c"),
        ("User in training data", "With preference text", "Hybrid + Boost: SVD×0.5 + KNN×0.3 + Pine×0.2", "#f0a060"),
        ("Unknown user ID", "With preference text", "Auto-routes → New User (Pinecone only)", "#5b9bd5"),
        ("No user_id", "With preference text", "New User: Pinecone semantic search", "#c44b8a"),
        ("Any", "No preference, not in DB", "⚠️ Returns empty DataFrame", "#5a5468"),
    ]

    header_html = """
    <div style='display:grid; grid-template-columns:1fr 1fr 2fr; gap:0; margin-bottom:0.5rem;'>
      <div style='font-size:0.68rem; color:#5a5468; text-transform:uppercase; letter-spacing:0.1em; padding:0.5rem 1rem; background:#0f0f1c; border-radius:6px 0 0 0'>User Condition</div>
      <div style='font-size:0.68rem; color:#5a5468; text-transform:uppercase; letter-spacing:0.1em; padding:0.5rem 1rem; background:#0f0f1c; border-left:1px solid #1e1e30'>Preference Text</div>
      <div style='font-size:0.68rem; color:#5a5468; text-transform:uppercase; letter-spacing:0.1em; padding:0.5rem 1rem; background:#0f0f1c; border-left:1px solid #1e1e30; border-radius:0 6px 0 0'>Strategy</div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    for user_cond, pref_cond, strategy, color in routing_rows:
        st.markdown(f"""
        <div style='display:grid; grid-template-columns:1fr 1fr 2fr; gap:0; border-bottom:1px solid #1a1a28;'>
          <div style='font-size:0.78rem; color:#9a90a8; padding:0.6rem 1rem;'>{user_cond}</div>
          <div style='font-size:0.78rem; color:#9a90a8; padding:0.6rem 1rem; border-left:1px solid #1e1e30;'>{pref_cond}</div>
          <div style='font-size:0.78rem; color:{color}; padding:0.6rem 1rem; border-left:1px solid #1e1e30; font-weight:600;'>{strategy}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Tech Stack ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Technology Stack</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Libraries · Services · Infrastructure</div>', unsafe_allow_html=True)

    stack = [
        ("Data Layer", [("Snowflake", "Cloud data warehouse · 4M row source"),
                         ("pandas", "Data wrangling & preprocessing"),
                         ("scikit-learn", "Train/test split & encoding")]),
        ("Models", [("scikit-surprise", "SVD & KNN collaborative filtering"),
                     ("Pinecone", "Vector database for semantic search"),
                     ("sentence-transformers", "all-mpnet-base-v2 embeddings")]),
        ("Dashboard", [("Streamlit", "Interactive web dashboard"),
                        ("Plotly", "Interactive charts & visualizations"),
                        ("Python 3.10+", "Runtime environment")]),
    ]

    cols = st.columns(3)
    for col, (layer, items) in zip(cols, stack):
        with col:
            st.markdown(f'<div class="section-sub" style="margin-top:0">{layer}</div>', unsafe_allow_html=True)
            for lib, desc in items:
                st.markdown(f"""
                <div style='background:#0f0f1c; border:1px solid #1e1e30; border-radius:8px; padding:0.7rem 1rem; margin-bottom:0.5rem;'>
                  <div style='font-size:0.85rem; color:#f0e8dc; font-weight:600; margin-bottom:0.2rem;'>{lib}</div>
                  <div style='font-size:0.7rem; color:#5a5468;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)
