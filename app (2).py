import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Award Predictor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 0.04em;
}

.stApp {
    background-color: #0a0e1a;
    color: #e8e8e8;
}

.big-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    color: #e8d5a3;
    letter-spacing: 0.06em;
    line-height: 1;
    margin-bottom: 0.2rem;
}

.sub-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #6b7fa3;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.award-card {
    background: linear-gradient(135deg, #111827 0%, #1a2236 100%);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}

.award-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #e8d5a3, #c9a84c);
}

.award-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    color: #e8d5a3;
    letter-spacing: 0.05em;
}

.vote-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.8rem;
    font-weight: 600;
    color: #c9a84c;
    line-height: 1;
}

.rank-badge {
    display: inline-block;
    background: #1e2d4a;
    border: 1px solid #2a3a5c;
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #8fa8d0;
    margin: 0.2rem;
}

.rank-badge.top1 { border-color: #e8d5a3; color: #e8d5a3; }
.rank-badge.top3 { border-color: #c9a84c; color: #c9a84c; }
.rank-badge.top5 { border-color: #8fa8d0; color: #8fa8d0; }

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

.stTextArea textarea {
    background-color: #111827 !important;
    border: 1px solid #2a3a5c !important;
    color: #e8e8e8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stNumberInput input, .stTextInput input {
    background-color: #111827 !important;
    border: 1px solid #2a3a5c !important;
    color: #e8e8e8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.stSelectbox select {
    background-color: #111827 !important;
}

div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #2a3a5c;
    border-radius: 8px;
    padding: 0.8rem 1rem;
}

.stButton button {
    background: linear-gradient(135deg, #c9a84c, #e8d5a3) !important;
    color: #0a0e1a !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
}

.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(201, 168, 76, 0.3) !important;
}

.disclaimer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #3a4a60;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #1a2236;
}

.no-model-warning {
    background: #1a1530;
    border: 1px solid #4a3a7a;
    border-radius: 8px;
    padding: 1.5rem;
    color: #a08ad0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("mlb_award_model.pkl", "rb") as f:
            bundle = pickle.load(f)
        return bundle
    except FileNotFoundError:
        return None

@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()


# ── Prediction logic ──────────────────────────────────────────────────────────
def build_feature_vector(stats: dict, nlp_features: dict, feature_cols: list) -> np.ndarray:
    """Map user inputs → feature vector aligned to model's expected columns."""
    row = {col: np.nan for col in feature_cols}
    for k, v in stats.items():
        if k in row:
            row[k] = v
    for k, v in nlp_features.items():
        if k in row:
            row[k] = v
    return np.array([row[c] for c in feature_cols], dtype=np.float32).reshape(1, -1)


def predict_award(bundle, award_key, feature_vector):
    """Run two-stage prediction and return vote share estimate."""
    r = bundle["results"].get(award_key)
    if r is None:
        return None
    X = r["prep"].transform(feature_vector)
    prob = r["clf"].predict_proba(X)[0, 1]
    share = max(0.0, r["reg"].predict(X)[0])
    return float(prob * share), float(prob)


def embed_media_text(text: str, sbert, vader, pca, n_pca: int) -> dict:
    """Embed media text → NLP features dict."""
    if not text.strip():
        return {}

    words = text.split()
    if len(words) <= 400:
        emb = sbert.encode(text, normalize_embeddings=True)
    else:
        first = " ".join(words[:400])
        last = " ".join(words[-400:])
        e1 = sbert.encode(first, normalize_embeddings=True)
        e2 = sbert.encode(last, normalize_embeddings=True)
        avg = (e1 + e2) / 2
        emb = avg / (np.linalg.norm(avg) + 1e-9)

    sent = vader.polarity_scores(text[:5000])
    pca_vec = pca.transform(emb.reshape(1, -1))[0]

    features = {
        "nlp_article_count": 1,
        "nlp_sent_mean": sent["compound"],
        "nlp_sent_std": 0.0,
        "nlp_sent_max": sent["compound"],
        "nlp_sent_min": sent["compound"],
        "nlp_sent_momentum": 0.0,
    }
    for i, v in enumerate(pca_vec[:n_pca]):
        features[f"nlp_pca_{i:02d}"] = float(v)

    return features


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="big-title">⚾ MLB</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Award Predictor v1.0</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Player Type</div>', unsafe_allow_html=True)
    player_type = st.selectbox("", ["Hitter", "Pitcher", "Two-Way"], label_visibility="collapsed")

    st.markdown('<div class="section-header">Awards to Predict</div>', unsafe_allow_html=True)
    predict_mvp = st.checkbox("MVP", value=True)
    predict_cy  = st.checkbox("Cy Young", value=(player_type == "Pitcher"))
    predict_roy = st.checkbox("Rookie of the Year", value=False)

    st.markdown('<div class="section-header">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #4a6080; line-height: 1.6;">
    Model: Two-stage LightGBM<br>
    NLP: Sentence-BERT (all-MiniLM-L6-v2)<br>
    Sentiment: VADER<br>
    Trained on: 1992–2021<br>
    Test set: 2022+
    </div>
    """, unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown('<div class="big-title">Award Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter player statistics + media coverage to predict award vote share</div>', unsafe_allow_html=True)

bundle = load_model()

if bundle is None:
    st.markdown("""
    <div class="no-model-warning">
    ⚠️ <strong>mlb_award_model.pkl not found</strong><br><br>
    Place your trained model file in the root of this repository.<br>
    See the README for instructions on generating it from Google Colab.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

sbert = load_sbert()
vader = load_vader()
pca   = bundle["pca"]
n_pca = len(bundle.get("NLP_FEATURE_COLS", [])) - 6  # subtract scalar NLP cols

col_left, col_right = st.columns([1.1, 0.9], gap="large")

# ── LEFT: Input panel ─────────────────────────────────────────────────────────
with col_left:

    # Batting stats
    if player_type in ["Hitter", "Two-Way"]:
        st.markdown('<div class="section-header">Batting Statistics</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1:
            bat_avg  = st.number_input("AVG",   min_value=0.0, max_value=1.0,  value=0.290, step=0.001, format="%.3f")
            bat_ops  = st.number_input("OPS",   min_value=0.0, max_value=2.0,  value=0.850, step=0.001, format="%.3f")
        with b2:
            bat_hr   = st.number_input("HR",    min_value=0,   max_value=100,  value=30)
            bat_rbi  = st.number_input("RBI",   min_value=0,   max_value=250,  value=95)
        with b3:
            bat_sb   = st.number_input("SB",    min_value=0,   max_value=150,  value=10)
            bat_war  = st.number_input("bWAR",  min_value=-5.0, max_value=20.0, value=5.0, step=0.1, format="%.1f")

    # Pitching stats
    if player_type in ["Pitcher", "Two-Way"]:
        st.markdown('<div class="section-header">Pitching Statistics</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            pit_era  = st.number_input("ERA",   min_value=0.0, max_value=20.0, value=2.80, step=0.01, format="%.2f")
            pit_whip = st.number_input("WHIP",  min_value=0.0, max_value=5.0,  value=1.05, step=0.01, format="%.2f")
        with p2:
            pit_w    = st.number_input("W",     min_value=0,   max_value=35,   value=16)
            pit_so   = st.number_input("SO",    min_value=0,   max_value=400,  value=220)
        with p3:
            pit_ip   = st.number_input("IP",    min_value=0.0, max_value=350.0, value=185.0, step=0.1, format="%.1f")
            pit_kbb  = st.number_input("K/BB",  min_value=0.0, max_value=20.0,  value=3.5,  step=0.1, format="%.1f")

    # Fielding
    st.markdown('<div class="section-header">Fielding</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        fld_pct = st.number_input("Fielding %", min_value=0.0, max_value=1.0, value=0.985, step=0.001, format="%.3f")
    with f2:
        fld_drs = st.number_input("DRS",        min_value=-30, max_value=40,   value=5)

    # League / context
    st.markdown('<div class="section-header">Context</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        league = st.selectbox("League", ["AL", "NL"])
    with c2:
        year   = st.number_input("Season Year", min_value=1992, max_value=2030, value=2024)

    # Media / NLP
    st.markdown('<div class="section-header">Media Coverage (NLP)</div>', unsafe_allow_html=True)
    st.caption("Paste a press article, narrative summary, or any text describing this player's season. Leave blank to skip NLP features.")
    media_text = st.text_area(
        "",
        placeholder="e.g. 'After a dominant stretch run, [Player] finished the season leading the AL in home runs and was widely regarded as the frontrunner for MVP…'",
        height=160,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⚾  PREDICT AWARD VOTES")


# ── RIGHT: Results panel ──────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

    if not run_btn:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 1rem; color: #2a3a5c; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;">
        ← Fill in stats and click<br>PREDICT AWARD VOTES
        </div>
        """, unsafe_allow_html=True)
    else:
        # Build stats dict (map to model feature names)
        stats = {
            "lgID":            league,
            "yearID":          year,
            "is_pitcher":      1 if player_type == "Pitcher" else 0,
            "ptype_hitter":    1 if player_type == "Hitter"   else 0,
            "ptype_pitcher":   1 if player_type == "Pitcher"  else 0,
            "ptype_two_way":   1 if player_type == "Two-Way"  else 0,
        }
        if player_type in ["Hitter", "Two-Way"]:
            stats.update({
                "bat_AVG": bat_avg, "bat_OPS": bat_ops,
                "bat_HR":  bat_hr,  "bat_RBI": bat_rbi,
                "bat_SB":  bat_sb,  "bat_WAR": bat_war,
            })
        if player_type in ["Pitcher", "Two-Way"]:
            stats.update({
                "pit_ERA_calc": pit_era,  "pit_WHIP": pit_whip,
                "pit_W":        pit_w,    "pit_SO":   pit_so,
                "pit_IP":       pit_ip,   "pit_K_BB": pit_kbb,
            })
        stats.update({"fld_Fpct": fld_pct, "fld_DRS": fld_drs})

        # Embed media text if provided
        with st.spinner("Running NLP embeddings…"):
            nlp_features = embed_media_text(media_text, sbert, vader, pca, n_pca) if media_text.strip() else {}

        feat_cols = bundle.get("FEATURE_COLS_NLP" if nlp_features else "FEATURE_COLS_BASE", [])

        awards_to_run = []
        if predict_mvp: awards_to_run.append(("MVP", "MVP_Base+NLP" if nlp_features else "MVP_Base"))
        if predict_cy:  awards_to_run.append(("Cy Young", "CyYoung_Base+NLP" if nlp_features else "CyYoung_Base"))
        if predict_roy: awards_to_run.append(("Rookie of the Year", "ROY_Base+NLP" if nlp_features else "ROY_Base"))

        any_result = False
        for award_label, award_key in awards_to_run:
            fv = build_feature_vector(stats, nlp_features, feat_cols)
            result = predict_award(bundle, award_key, fv)

            if result is None:
                st.warning(f"No model found for **{award_label}** in this configuration.")
                continue

            vote_share, vote_prob = result
            any_result = True

            # Determine rank badges (rough thresholds based on typical vote distributions)
            in_top1 = vote_share >= 0.35
            in_top3 = vote_share >= 0.12
            in_top5 = vote_share >= 0.05

            badges_html = ""
            if in_top1: badges_html += '<span class="rank-badge top1">🥇 TOP 1 CANDIDATE</span>'
            elif in_top3: badges_html += '<span class="rank-badge top3">🥈 TOP 3 CANDIDATE</span>'
            elif in_top5: badges_html += '<span class="rank-badge top5">TOP 5 CANDIDATE</span>'
            else: badges_html += '<span class="rank-badge">OUTSIDE TOP 5</span>'

            st.markdown(f"""
            <div class="award-card">
                <div class="award-name">{award_label}</div>
                <div class="vote-pct">{vote_share*100:.1f}%</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#4a6080; margin-bottom:0.8rem;">
                    predicted vote share
                </div>
                {badges_html}
                <div style="margin-top:1rem; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#4a6080;">
                    Voter probability: {vote_prob*100:.1f}%
                    {"&nbsp;&nbsp;|&nbsp;&nbsp;NLP: ✓" if nlp_features else "&nbsp;&nbsp;|&nbsp;&nbsp;NLP: —"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if any_result:
            # Sentiment readout
            if media_text.strip():
                s = vader.polarity_scores(media_text[:5000])
                sent_label = "Positive ↑" if s["compound"] > 0.05 else ("Negative ↓" if s["compound"] < -0.05 else "Neutral →")
                st.markdown(f"""
                <div style="background:#0d1520; border:1px solid #1e2d4a; border-radius:8px; padding:1rem; margin-top:0.5rem;">
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#4a6080; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem;">Media Sentiment</div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:1rem; color:#{'8fd0a0' if s['compound']>0.05 else ('d08080' if s['compound']<-0.05 else '8fa8d0')};">
                        {sent_label} &nbsp; ({s['compound']:+.3f})
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer">
            ⚠ Predictions are probabilistic estimates based on historical voting patterns (1992–2021).
            Actual results depend on voter subjectivity, narrative factors, and competition not captured here.
            </div>
            """, unsafe_allow_html=True)
