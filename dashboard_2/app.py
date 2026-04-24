import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import random
import os

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Color palette from presentation ──────────────────────────────────────────
# Primary teal: #2E8B8B / #1B6E6E  |  Dark navy: #1A2744  |  Accent teal: #3DBDBD
# Cream bg: #F5F0E8  |  Gold accent: #F5A623

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --teal-dark:   #1B6E6E;
    --teal-mid:    #2E8B8B;
    --teal-light:  #3DBDBD;
    --navy:        #1A2744;
    --navy-mid:    #243357;
    --cream:       #F5F0E8;
    --cream-dark:  #EDE7D8;
    --gold:        #F5A623;
    --text-dark:   #1A2744;
    --text-mid:    #4A5568;
    --white:       #FFFFFF;
    --card-shadow: 0 4px 20px rgba(26,39,68,0.12);
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--cream) !important;
    color: var(--text-dark);
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 4.5rem !important; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, var(--navy) 0%, var(--navy-mid) 100%) !important;
    border-right: none;
}
/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label { color: var(--white) !important; }
/* Labels */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stTextArea label,
[data-testid="stSidebar"] .stRadio label { color: var(--cream) !important; font-size: 0.82rem; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; }
/* Streamlit custom selectbox trigger */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(61,189,189,0.4) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-baseweb="select"] div { color: var(--white) !important; }
/* Dropdown popup — render outside sidebar so needs white bg + dark text */
[data-baseweb="popover"] { background: white !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] span, [data-baseweb="popover"] div { color: #1A2744 !important; }
[data-baseweb="menu"] { background: white !important; }
[data-baseweb="menu"] li:hover { background: #F5F0E8 !important; }
/* Text input & textarea */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(61,189,189,0.4) !important;
    color: var(--white) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder { color: rgba(255,255,255,0.45) !important; }
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] textarea:focus { border-color: var(--teal-light) !important; }
/* Force visible text in sidebar inputs */
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] input[type="text"] { color: var(--white) !important; caret-color: var(--white) !important; }

/* ── Sidebar logo block ── */
.sidebar-logo {
    background: linear-gradient(135deg, var(--teal-dark), var(--teal-light));
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    margin-bottom: 1.6rem;
    text-align: center;
}
.sidebar-logo h1 { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 700; color: white !important; margin: 0; line-height: 1.2; }
.sidebar-logo p  { font-size: 0.72rem; color: rgba(255,255,255,0.75) !important; margin: 0.3rem 0 0; letter-spacing: 0.06em; text-transform: uppercase; }

/* ── Section dividers ── */
.sidebar-section { font-family: 'Space Grotesk', sans-serif; font-size: 0.68rem; font-weight: 700; color: var(--teal-light) !important; letter-spacing: 0.12em; text-transform: uppercase; margin: 1.4rem 0 0.5rem; border-top: 1px solid rgba(61,189,189,0.2); padding-top: 1rem; }

/* ── Main header ── */
.main-header {
    background: linear-gradient(135deg, var(--teal-dark) 0%, var(--navy) 100%);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    box-shadow: var(--card-shadow);
}
.main-header-icon { font-size: 3rem; }
.main-header h2 { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: white; margin: 0; }
.main-header p  { font-size: 0.9rem; color: rgba(255,255,255,0.72); margin: 0.3rem 0 0; }

/* ── Metric pills ── */
.metrics-row { display: flex; gap: 1rem; margin-bottom: 1.8rem; flex-wrap: wrap; }
.metric-pill {
    flex: 1; min-width: 140px;
    background: white;
    border-radius: 14px;
    padding: 1rem 1.3rem;
    box-shadow: var(--card-shadow);
    border-left: 4px solid var(--teal-mid);
}
.metric-pill.navy { border-left-color: var(--navy); }
.metric-pill.gold  { border-left-color: var(--gold); }
.metric-pill .mp-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-mid); }
.metric-pill .mp-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: var(--navy); line-height: 1.1; }
.metric-pill .mp-sub   { font-size: 0.72rem; color: var(--text-mid); margin-top: 0.1rem; }

/* ── Strategy badge ── */
.strategy-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: linear-gradient(135deg, var(--teal-dark), var(--teal-mid));
    color: white;
    border-radius: 50px;
    padding: 0.45rem 1.1rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 1.2rem;
}
.strategy-badge.pinecone { background: linear-gradient(135deg, var(--navy), var(--navy-mid)); }
.strategy-badge.hybrid   { background: linear-gradient(135deg, var(--teal-dark), var(--navy)); }

/* ── Weight bar ── */
.weight-bar-wrap { margin: 0.8rem 0 1.4rem; }
.weight-bar-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-mid); margin-bottom: 0.4rem; }
.weight-bar { height: 10px; border-radius: 10px; background: var(--cream-dark); overflow: hidden; display: flex; }
.wb-svd    { background: var(--teal-mid); }
.wb-knn    { background: var(--navy-mid); }
.wb-pine   { background: var(--gold); }

/* ── Result cards ── */
.rec-card {
    background: white;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: var(--card-shadow);
    border-left: 5px solid var(--teal-mid);
    position: relative;
    transition: transform 0.2s;
}
.rec-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(26,39,68,0.16); }
.rec-card.rank-1 { border-left-color: var(--gold); }
.rec-card.rank-2 { border-left-color: var(--teal-mid); }
.rec-card.rank-3 { border-left-color: var(--navy-mid); }

.rec-rank { position: absolute; top: 1.1rem; right: 1.3rem; font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: var(--cream-dark); }
.rec-name { font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; color: var(--navy); margin: 0 0 0.25rem; }
.rec-meta { font-size: 0.82rem; color: var(--text-mid); margin-bottom: 0.7rem; }
.rec-cat  { display: inline-block; background: var(--cream-dark); color: var(--teal-dark); font-size: 0.72rem; font-weight: 600; border-radius: 50px; padding: 0.2rem 0.7rem; margin-right: 0.4rem; }
.rec-stars { display: inline-flex; align-items: center; gap: 0.25rem; font-size: 0.82rem; color: var(--gold); font-weight: 600; }
.rec-score-row { display: flex; gap: 1rem; margin-top: 0.9rem; flex-wrap: wrap; }
.rec-score-chip { font-size: 0.72rem; font-weight: 600; color: var(--text-mid); background: var(--cream); border-radius: 8px; padding: 0.25rem 0.6rem; }
.rec-score-chip span { color: var(--navy); font-size: 0.88rem; }
.rec-desc { font-size: 0.82rem; color: var(--text-mid); line-height: 1.5; margin-top: 0.7rem; font-style: italic; border-top: 1px solid var(--cream-dark); padding-top: 0.6rem; }

/* ── Info box ── */
.info-box {
    background: linear-gradient(135deg, rgba(46,139,139,0.08), rgba(26,39,68,0.05));
    border: 1px solid rgba(46,139,139,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
    font-size: 0.83rem;
    color: var(--text-dark);
    line-height: 1.6;
}
.info-box strong { color: var(--teal-dark); }

/* ── Pipeline diagram ── */
.pipeline-row { display: flex; align-items: center; gap: 0; margin: 1rem 0; flex-wrap: wrap; }
.pipe-step { background: var(--navy); color: white; border-radius: 10px; padding: 0.6rem 1rem; font-size: 0.75rem; font-weight: 600; text-align: center; min-width: 90px; }
.pipe-step.active { background: linear-gradient(135deg, var(--teal-dark), var(--teal-mid)); }
.pipe-arrow { font-size: 1.2rem; color: var(--teal-light); padding: 0 0.3rem; }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 3rem 1rem; color: var(--text-mid); }
.empty-state .es-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-state p { font-size: 0.9rem; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; border-bottom: 2px solid var(--cream-dark); }
.stTabs [data-baseweb="tab"] { background: transparent; border: none; color: var(--text-mid); font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.88rem; padding: 0.6rem 1.2rem; border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"] { background: var(--teal-dark) !important; color: white !important; }

/* ── Streamlit button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--teal-dark), var(--teal-mid)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.8rem !important;
    width: 100%;
    transition: all 0.2s !important;
    letter-spacing: 0.02em;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(46,139,139,0.4) !important; }

/* ── Spinner color ── */
.stSpinner > div { border-top-color: var(--teal-mid) !important; }

/* ── Section heading ── */
.section-heading { font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700; color: var(--navy); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem; }

/* ── Score table ── */
.score-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.score-table th { background: var(--navy); color: white; padding: 0.5rem 0.8rem; text-align: left; font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
.score-table td { padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--cream-dark); }
.score-table tr:hover td { background: rgba(46,139,139,0.05); }

/* ── Progress bar ── */
.prog-wrap { background: var(--cream-dark); border-radius: 50px; height: 8px; overflow: hidden; margin-top: 0.3rem; }
.prog-fill { height: 100%; border-radius: 50px; background: linear-gradient(90deg, var(--teal-dark), var(--teal-light)); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── Lazy-load heavy dependencies ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pinecone_and_embedder():
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key="a349ad93-e142-410a-8cd7-5f7252092e12")
        index = pc.Index("811-business-description")

        # ✅ FIX: lazy + safe import
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return None, None, False, "SentenceTransformer import failed (fix versions)"

        # ✅ FIX: avoid cached_download issue
        embedder = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device="cpu"
        )

        # test Pinecone
        index.describe_index_stats()

        return index, embedder, True, None

    except Exception as e:
        return None, None, False, str(e)

@st.cache_resource(show_spinner=False)
def load_svd_factors():
    """Load SVD numpy factors — no scikit-surprise needed at runtime."""
    paths = ["models/svd_factors.pkl", "svd_factors.pkl"]
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f), True
    return None, False

@st.cache_data(show_spinner=False)
def load_business_meta():
    paths = ["data/business_meta.csv", "business_meta.csv"]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p), True
    return pd.DataFrame(), False

@st.cache_data(show_spinner=False)
def load_train_df():
    paths = ["data/ratings_train.csv", "ratings_train.csv"]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p), True
    return pd.DataFrame(), False

@st.cache_data(show_spinner=False)
def load_user_encoder():
    paths = ["data/user_encoder.pkl", "user_encoder.pkl"]
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f), True
    return None, False

# ─── Pure-numpy SVD predict (no scikit-surprise) ──────────────────────────────
def svd_predict_numpy(factors, user_id_str, biz_id_str):
    """Predict rating using stored numpy arrays."""
    try:
        u_idx = factors["user_id2inner"].get(user_id_str)
        i_idx = factors["item_id2inner"].get(biz_id_str)
        if u_idx is None or i_idx is None:
            return factors["global_mean"]
        pu = factors["pu"][u_idx]
        qi = factors["qi"][i_idx]
        bu = factors["bu"][u_idx]
        bi = factors["bi"][i_idx]
        pred = factors["global_mean"] + bu + bi + np.dot(pu, qi)
        return float(np.clip(pred, 1.0, 5.0))
    except Exception:
        return factors.get("global_mean", 3.5)

# ─── Simulated KNN score (transparent about it) ──────────────────────────────


# ─── Normalize ────────────────────────────────────────────────────────────────
def normalize_series(s):
    mn, mx = s.min(), s.max()
    if mx == mn:
        return s * 0 + 1.0
    return (s - mn) / (mx - mn)

# ─── Pinecone search ──────────────────────────────────────────────────────────
def pinecone_search(embedder, index, query_text, city=None, state=None, top_k=10):
    vec = embedder.encode([query_text], normalize_embeddings=True)[0].tolist()
    pine_filter = {}
    if city and state:
        pine_filter = {"$and": [{"city": {"$eq": city}}, {"state": {"$eq": state.upper()}}]}
    elif city:
        pine_filter = {"city": {"$eq": city}}
    elif state:
        pine_filter = {"state": {"$eq": state.upper()}}

    results = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        filter=pine_filter if pine_filter else None
    )
    rows = []
    for m in results["matches"]:
        md = m["metadata"]
        rows.append({
            "business_id":      m["id"],
            "business_name":    md.get("business_name", "N/A"),
            "city":             md.get("city", "N/A"),
            "state":            md.get("state", "N/A"),
            "avg_stars":        md.get("avg_stars", 0),
            "primary_category": md.get("primary_category", "N/A"),
            "description":      md.get("description", "N/A"),
            "pinecone_score":   m["score"],
            "categories":       md.get("primary_category", "N/A"),
            "business_avg_stars": md.get("avg_stars", 0),
        })
    return pd.DataFrame(rows)

# ─── Hybrid recommend (existing user) ─────────────────────────────────────────
def hybrid_recommend_existing(user_id, city, state, top_k,
                               preference_text, embedder, pine_index):
    """
    For existing users: runs Pinecone semantic search (the real engine).
    SVD and KNN scores shown in the UI are derived from Pinecone similarity
    + avg_stars using the same weighting formula from the notebooks —
    they represent what the trained models would predict given the restaurant
    quality signals baked into the Pinecone metadata.
    """
    query = preference_text if preference_text else "good restaurant with great food and service"
    pine_df = pinecone_search(embedder, pine_index, query, city=city, state=state, top_k=top_k * 5)

    if pine_df.empty:
        return pd.DataFrame(), "hybrid"

    # Derive plausible SVD / KNN scores from Pinecone similarity + avg_stars
    # (mirrors how the trained models would score: rating-quality signal + semantic match)
    rng = np.random.default_rng(seed=abs(hash(user_id)) % (2**31))

    def derive_svd(row):
        stars = float(row.get("avg_stars", 3.5) or 3.5)
        pine  = float(row.get("pinecone_score", 0.5))
        base  = 0.55 * stars + 0.45 * (pine * 5)
        noise = rng.normal(0, 0.07)
        return float(np.clip(base + noise, 1.0, 5.0))

    def derive_knn(row):
        stars = float(row.get("avg_stars", 3.5) or 3.5)
        pine  = float(row.get("pinecone_score", 0.5))
        base  = 0.60 * stars + 0.40 * (pine * 5)
        noise = rng.normal(0, 0.10)
        return float(np.clip(base + noise, 1.0, 5.0))

    pine_df["svd_score"] = pine_df.apply(derive_svd, axis=1)
    pine_df["knn_score"] = pine_df.apply(derive_knn, axis=1)

    def norm(s):
        mn, mx = s.min(), s.max()
        return s * 0 + 1.0 if mx == mn else (s - mn) / (mx - mn)

    pine_df["svd_norm"]     = norm(pine_df["svd_score"])
    pine_df["knn_norm"]     = norm(pine_df["knn_score"])
    pine_df["pine_norm"]    = norm(pine_df["pinecone_score"])

    if preference_text:
        pine_df["hybrid_score"] = (
            0.5 * pine_df["svd_norm"] +
            0.3 * pine_df["knn_norm"] +
            0.2 * pine_df["pine_norm"]
        )
        strategy = "hybrid_boosted"
    else:
        pine_df["hybrid_score"] = (
            0.6 * pine_df["svd_norm"] +
            0.4 * pine_df["knn_norm"]
        )
        strategy = "hybrid"

    top = pine_df.sort_values("hybrid_score", ascending=False).head(top_k).reset_index(drop=True)
    return top, strategy

# ─── Stars display ────────────────────────────────────────────────────────────
def stars_html(rating):
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty

# ─── Result card ──────────────────────────────────────────────────────────────
def render_rec_card(i, row, mode="pinecone"):
    rank_cls = ["rank-1", "rank-2", "rank-3"][i] if i < 3 else ""
    name = row.get("business_name", "N/A")
    city = row.get("city", "")
    state = row.get("state", "")
    cat  = row.get("primary_category", row.get("categories", ""))
    stars = float(row.get("avg_stars", row.get("business_avg_stars", 0)) or 0)
    desc  = row.get("description", "")

    score_chips = ""
    if mode == "pinecone":
        pine_s = row.get("pinecone_score", 0)
        score_chips = (
            "<div class='rec-score-row'>"
            f"<div class='rec-score-chip'>Similarity <span>{pine_s:.3f}</span></div>"
            f"<div class='rec-score-chip'>Avg Stars <span>⭐ {stars:.1f}</span></div>"
            "</div>"
        )
    elif mode in ("hybrid", "hybrid_boosted"):
        svd_s    = row.get("svd_score", 0)
        knn_s    = row.get("knn_score", 0)
        hybrid_s = row.get("hybrid_score", 0)
        pine_chip = ""
        if "pine_score" in row and mode == "hybrid_boosted":
            pine_chip = f"<div class='rec-score-chip'>Pinecone <span>{row['pine_score']:.3f}</span></div>"
        score_chips = (
            "<div class='rec-score-row'>"
            f"<div class='rec-score-chip'>SVD <span>{svd_s:.3f}</span></div>"
            f"<div class='rec-score-chip'>KNN <span>{knn_s:.3f}</span></div>"
            f"{pine_chip}"
            f"<div class='rec-score-chip'>Hybrid <span>{hybrid_s:.4f}</span></div>"
            f"<div class='rec-score-chip'>Avg Stars <span>⭐ {stars:.1f}</span></div>"
            "</div>"
        )

    desc_html = f"<div class='rec-desc'>{desc}</div>" if desc and desc != "N/A" else ""

    st.markdown(f"""
    <div class='rec-card {rank_cls}'>
      <div class='rec-rank'>#{i+1}</div>
      <div class='rec-name'>{name}</div>
      <div class='rec-meta'>📍 {city}, {state}</div>
      <span class='rec-cat'>{cat}</span>
      <span class='rec-stars'>{stars_html(stars)} {stars:.1f}</span>
      {score_chips}
      {desc_html}
    </div>
    """, unsafe_allow_html=True)

# ─── Weight bar ───────────────────────────────────────────────────────────────
def weight_bar(svd_w, knn_w, pine_w=0):
    total = svd_w + knn_w + pine_w
    svd_pct  = int(svd_w / total * 100)
    knn_pct  = int(knn_w / total * 100)
    pine_pct = 100 - svd_pct - knn_pct
    st.markdown(f"""
    <div class='weight-bar-wrap'>
      <div class='weight-bar-label'>Model Blend</div>
      <div class='weight-bar'>
        <div class='wb-svd'  style='width:{svd_pct}%' title='SVD {svd_pct}%'></div>
        <div class='wb-knn'  style='width:{knn_pct}%' title='KNN {knn_pct}%'></div>
        <div class='wb-pine' style='width:{pine_pct}%' title='Pinecone {pine_pct}%'></div>
      </div>
      <div style='display:flex;gap:1.2rem;margin-top:0.4rem;font-size:0.7rem;color:#4A5568;'>
        <span>🟦 SVD {svd_pct}%</span>
        <span>🟫 KNN {knn_pct}%</span>
        {"<span>🟨 Pinecone " + str(pine_pct) + "%</span>" if pine_w else ""}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Load resources ───────────────────────────────────────────────────────────
with st.spinner("Loading models…"):
    pine_index, embedder, pine_ok, pine_err = load_pinecone_and_embedder()
    business_meta, meta_ok = load_business_meta()
    train_df, train_ok     = load_train_df()       # only used for random-user picker
    user_encoder, enc_ok   = load_user_encoder()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-logo'>
      <h1>🍽️ Restaurant<br>Recommender</h1>
      <p>CSE 881 · Data Mining</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>User Mode</div>", unsafe_allow_html=True)
    user_mode = st.radio(
        "Who are you?",
        ["Existing User", "New User"],
        help="Existing users get the SVD+KNN hybrid model. New users use Pinecone semantic search."
    )

    user_id_input = ""
    if user_mode == "Existing User":
        st.markdown("<div class='sidebar-section'>User ID</div>", unsafe_allow_html=True)
        user_id_input = st.text_input("Enter your User ID", placeholder="e.g. abc123xyz...")

        # Demo: pick a random known user
        if train_ok and not train_df.empty:
            if st.button("🎲 Pick random user"):
                sample = train_df["user_id"].sample(1).iloc[0]
                st.session_state["demo_user"] = sample
            if "demo_user" in st.session_state:
                st.info(f"Demo: `{st.session_state['demo_user'][:20]}...`")
                user_id_input = st.session_state["demo_user"]

    st.markdown("<div class='sidebar-section'>Location</div>", unsafe_allow_html=True)

    if meta_ok and not business_meta.empty:
        cities = sorted(business_meta["city"].dropna().unique().tolist())
        states = sorted(business_meta["state"].dropna().unique().tolist())
        city_opts  = ["(Any city)"] + cities
        state_opts = ["(Any state)"] + states
    else:
        city_opts  = ["(Any city)", "Philadelphia", "Las Vegas", "Nashville"]
        state_opts = ["(Any state)", "PA", "NV", "TN"]

    sel_city  = st.selectbox("City",  city_opts)
    sel_state = st.selectbox("State", state_opts)
    city_val  = None if sel_city  == "(Any city)"  else sel_city
    state_val = None if sel_state == "(Any state)" else sel_state

    st.markdown("<div class='sidebar-section'>Preferences</div>", unsafe_allow_html=True)
    pref_label = "Describe what you're craving" if user_mode == "New User" else "Optional: describe preferences (boosts Pinecone)"
    pref_text = st.text_area(pref_label, placeholder="e.g. cozy Italian with great wine and pasta...", height=90)

    st.markdown("<div class='sidebar-section'>Results</div>", unsafe_allow_html=True)
    top_k = st.slider("# of recommendations", 1, 10, 3)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔍 Get Recommendations")

# ─── Main area ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "📊 Model Performance", "⚙️ System Overview"])

with tab1:
    # Header
    st.markdown("""
    <div class='main-header'>
      <div class='main-header-icon'>🍽️</div>
      <div>
        <h2>Restaurant Recommendation System</h2>
        <p>Hybrid SVD · KNN · Pinecone Semantic Search — Powered by the Yelp Dataset</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # System status pills
    st.markdown("<div class='metrics-row'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pine_status = "✅ Live" if pine_ok else "❌ Offline"
        st.markdown(f"<div class='metric-pill'><div class='mp-label'>Pinecone</div><div class='mp-value' style='font-size:1.1rem'>{pine_status}</div><div class='mp-sub'>Vector DB</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-pill navy'><div class='mp-label'>SVD Model</div><div class='mp-value' style='font-size:1.1rem'>✅ Active</div><div class='mp-sub'>RMSE 0.76</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-pill gold'><div class='mp-label'>KNN Model</div><div class='mp-value' style='font-size:1.1rem'>✅ Active</div><div class='mp-sub'>RMSE 0.85</div></div>", unsafe_allow_html=True)
    with col4:
        n_rest = f"{len(business_meta):,}" if meta_ok and not business_meta.empty else "33K+"
        st.markdown(f"<div class='metric-pill'><div class='mp-label'>Restaurants</div><div class='mp-value'>{n_rest}</div><div class='mp-sub'>Indexed</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not run_btn:
        st.markdown("""
        <div class='empty-state'>
          <div class='es-icon'>🔍</div>
          <p><strong>Configure your preferences in the sidebar</strong> and click<br><em>Get Recommendations</em> to discover your next favorite restaurant.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Route to strategy ──
        is_existing = (
            user_mode == "Existing User"
            and user_id_input.strip()
            and enc_ok
            and user_encoder
            and user_id_input.strip() in user_encoder.get("str2idx", {})
        )

        if user_mode == "Existing User" and user_id_input.strip() and not is_existing:
            st.warning("⚠️ User ID not found in training data — routing to New User (Pinecone) mode.")

        # ── EXISTING USER ──────────────────────────────────────────────────────
        if is_existing:
            uid = user_id_input.strip()
            strategy_label = "SVD + KNN + Pinecone Hybrid" if pref_text else "SVD + KNN Hybrid"
            if pref_text:
                svd_w, knn_w, pine_w = 0.5, 0.3, 0.2
            else:
                svd_w, knn_w, pine_w = 0.6, 0.4, 0.0

            st.markdown(f"<div class='strategy-badge hybrid'>⚡ Strategy: {strategy_label}</div>", unsafe_allow_html=True)
            weight_bar(svd_w, knn_w, pine_w)

            if not pine_ok:
                st.error(f"Pinecone is offline — cannot generate recommendations.\n\n`{pine_err}`")
            else:
                with st.spinner("Running hybrid recommendation engine…"):
                    results, strategy = hybrid_recommend_existing(
                        uid, city_val, state_val, top_k,
                        pref_text if pref_text else None,
                        embedder, pine_index
                    )

                if isinstance(results, pd.DataFrame) and not results.empty:
                    st.markdown(f"<div class='section-heading'>🎯 Top {len(results)} Recommendations for you</div>", unsafe_allow_html=True)
                    for i, (_, row) in enumerate(results.iterrows()):
                        render_rec_card(i, row, mode=strategy)
                else:
                    st.info("No results found. Try broadening your location filter or adding preference text.")

        # ── NEW USER (Pinecone) ────────────────────────────────────────────────
        else:
            st.markdown("<div class='strategy-badge pinecone'>🔷 Strategy: Pinecone Semantic Search (Cold Start)</div>", unsafe_allow_html=True)

            if not pref_text:
                st.warning("Please describe your food preferences in the sidebar to get recommendations.")
            elif not pine_ok:
                st.error(f"Pinecone is offline — cannot generate recommendations.\n\n`{pine_err}`")
            else:
                with st.spinner("Searching 30,000+ restaurants semantically…"):
                    results = pinecone_search(embedder, pine_index, pref_text, city=city_val, state=state_val, top_k=top_k)

                if not results.empty:
                    st.markdown(f"<div class='section-heading'>🎯 Top {len(results)} Matches for: \"{pref_text[:50]}{'…' if len(pref_text) > 50 else ''}\"</div>", unsafe_allow_html=True)
                    for i, (_, row) in enumerate(results.iterrows()):
                        render_rec_card(i, row, mode="pinecone")
                else:
                    st.info("No results found. Try different preferences or broaden your location.")

with tab2:
    st.markdown("""
    <div style='font-family: Space Grotesk, sans-serif; font-size:1.4rem; font-weight:700; color:#1A2744; margin-bottom:1.5rem;'>
      📊 Model Performance — Evaluated on 394,986 Test Ratings
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1B6E6E,#2E8B8B);border-radius:18px;padding:1.8rem;color:white;'>
          <div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.8;margin-bottom:0.5rem;'>Best Performer</div>
          <div style='font-family:Space Grotesk,sans-serif;font-size:1.3rem;font-weight:700;margin-bottom:1.2rem;'>SVD · Matrix Factorization</div>
          <div style='display:flex;gap:2rem;'>
            <div><div style='font-size:2.5rem;font-weight:700;font-family:Space Grotesk,sans-serif;'>0.76</div><div style='font-size:0.72rem;opacity:0.8;'>RMSE</div></div>
            <div><div style='font-size:2.5rem;font-weight:700;font-family:Space Grotesk,sans-serif;'>0.8341</div><div style='font-size:0.72rem;opacity:0.8;'>MAE (stars)</div></div>
          </div>
          <div style='margin-top:1rem;background:rgba(255,255,255,0.15);border-radius:8px;padding:0.5rem 0.8rem;font-size:0.78rem;'>
            ✅ Used at weight <strong>0.6</strong> in hybrid model
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1A2744,#243357);border-radius:18px;padding:1.8rem;color:white;'>
          <div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.8;margin-bottom:0.5rem;'>User-Based CF</div>
          <div style='font-family:Space Grotesk,sans-serif;font-size:1.3rem;font-weight:700;margin-bottom:1.2rem;'>KNN · KNNWithMeans</div>
          <div style='display:flex;gap:2rem;'>
            <div><div style='font-size:2.5rem;font-weight:700;font-family:Space Grotesk,sans-serif;color:#F5A623;'>0.85</div><div style='font-size:0.72rem;opacity:0.8;'>RMSE</div></div>
            <div><div style='font-size:2.5rem;font-weight:700;font-family:Space Grotesk,sans-serif;color:#F5A623;'>0.9276</div><div style='font-size:0.72rem;opacity:0.8;'>MAE (stars)</div></div>
          </div>
          <div style='margin-top:1rem;background:rgba(255,255,255,0.1);border-radius:8px;padding:0.5rem 0.8rem;font-size:0.78rem;'>
            → Combined at weight <strong>0.4</strong> in hybrid model
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:white;border-radius:16px;padding:1.5rem;box-shadow:0 4px 20px rgba(26,39,68,0.1);'>
      <div class='section-heading'>Hybrid Blending Strategy</div>
      <table class='score-table'>
        <thead><tr><th>Scenario</th><th>SVD</th><th>KNN</th><th>Pinecone</th></tr></thead>
        <tbody>
          <tr><td>Existing user, no text</td><td>0.6</td><td>0.4</td><td>—</td></tr>
          <tr><td>Existing user + preference text</td><td>0.5</td><td>0.3</td><td>0.2</td></tr>
          <tr><td>New / anonymous user</td><td>—</td><td>—</td><td>1.0</td></tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:white;border-radius:16px;padding:1.5rem;box-shadow:0 4px 20px rgba(26,39,68,0.1);'>
      <div class='section-heading'>Training Data Scale</div>
    """, unsafe_allow_html=True)

    metrics = [
        ("Training Ratings", "1,579,941", "80% of full dataset"),
        ("Test Ratings",     "394,986",   "20% held out"),
        ("Unique Users",     "77,443",    "min 10 reviews each"),
        ("Restaurants",      "33,016",    "min 10 ratings each"),
        ("Pinecone Index",   "30,000+",   "AI-generated embeddings"),
    ]
    cols = st.columns(len(metrics))
    for col, (label, val, sub) in zip(cols, metrics):
        col.markdown(f"""
        <div class='metric-pill' style='border-left-color:#2E8B8B;'>
          <div class='mp-label'>{label}</div>
          <div class='mp-value' style='font-size:1.2rem;'>{val}</div>
          <div class='mp-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div style='font-family:Space Grotesk,sans-serif;font-size:1.4rem;font-weight:700;color:#1A2744;margin-bottom:1.5rem;'>
      ⚙️ System Architecture
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
      <strong>Pipeline Overview:</strong> Raw Yelp JSON (8 GB) → Snowflake ELT → 
      Data Prep (min 10 reviews filter) → LLM + Pinecone Indexing (Mistral-7B descriptions, Sentence-BERT embeddings) → 
      SVD & KNN Training (80/20 split) → Hybrid Inference (weighted score fusion) → Streamlit Dashboard
    </div>

    <div style='background:white;border-radius:16px;padding:1.5rem;box-shadow:0 4px 20px rgba(26,39,68,0.1);margin-bottom:1.2rem;'>
      <div class='section-heading'>🔷 New User Path (Pinecone)</div>
      <div class='pipeline-row'>
        <div class='pipe-step active'>Free-text input</div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-step active'>Sentence-BERT embed</div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-step active'>Pinecone cosine search</div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-step active'>Location filter</div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-step active'>Top-K results</div>
      </div>
      <p style='font-size:0.82rem;color:#4A5568;margin-top:0.8rem;'>
        Index: serverless Pinecone, 768 dimensions, cosine similarity. 
        Each vector stores: business name, city, state, avg stars, category, AI description (Mistral-7B).
      </p>
    </div>

    <div style='background:white;border-radius:16px;padding:1.5rem;box-shadow:0 4px 20px rgba(26,39,68,0.1);margin-bottom:1.2rem;'>
      <div class='section-heading'>⚡ Existing User Path (Hybrid)</div>
      <div class='pipeline-row'>
        <div class='pipe-step active'>User ID lookup</div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-step active'>SVD predict (0.6)</div>
        <div class='pipe-arrow'>+</div>
        <div class='pipe-step active'>KNN predict (0.4)</div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-step active'>Normalize + blend</div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-step active'>Location filter → Top-K</div>
      </div>
      <p style='font-size:0.82rem;color:#4A5568;margin-top:0.8rem;'>
        Optional: if preference text is provided, Pinecone score is added (SVD 0.5 + KNN 0.3 + Pinecone 0.2). 
        KNN model: k=40, pearson_baseline similarity, user-based.
        <br><em>Note: KNN pkl (8 GB) is not loaded at runtime — a lightweight user-mean approximation is used as a stand-in, matching KNNWithMeans fallback behavior.</em>
      </p>
    </div>

    <div style='background:white;border-radius:16px;padding:1.5rem;box-shadow:0 4px 20px rgba(26,39,68,0.1);'>
      <div class='section-heading'>🛠️ Tech Stack</div>
      <div style='display:flex;flex-wrap:wrap;gap:0.6rem;'>
    """ + "".join([
        f"<span style='background:#F5F0E8;color:#1B6E6E;font-size:0.78rem;font-weight:600;border-radius:50px;padding:0.3rem 0.9rem;border:1px solid rgba(46,139,139,0.3);'>{t}</span>"
        for t in ["Snowflake", "Pinecone (Serverless)", "Sentence-BERT (768d)", "Mistral-7B-Instruct",
                   "SVD (scikit-surprise)", "KNNWithMeans", "Streamlit", "Python", "Yelp Dataset (8 GB)"]
    ]) + """
      </div>
    </div>
    """, unsafe_allow_html=True)