"""
============================================================
STEP 10 — STREAMLIT WEB APP  ("FakeShield")
============================================================
Module: app/streamlit_app.py

Run with:
    streamlit run app/streamlit_app.py

Layout
------
10a. Title + sidebar (model info / dataset stats / about).
10b. Input form: title, body, subject dropdown, date picker, submit.
10c. Results: color-coded risk gauge, risk %, prediction badge,
     credibility pre-screen score.
10d. Explanation tabs: LIME text, SHAP metadata bar chart, full report.
10e. Batch analysis: CSV upload, color-coded preview, download button.
10f. Footer: performance stats + disclaimer.
"""

from __future__ import annotations

import io
import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make src/ importable when run via `streamlit run app/streamlit_app.py`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_fake_news_model.pt")

st.set_page_config(page_title="FakeShield — AI Misinformation Detector",
                   page_icon="🔍", layout="wide")

# ------------------------------------------------------------------ #
# Custom styling (frontend-design: distinctive, editorial/forensic look)
# ------------------------------------------------------------------ #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important;
                 letter-spacing: -0.02em; }
    .stApp { background:
        radial-gradient(circle at 20% 0%, #11161f 0%, #0a0d12 60%); }
    .badge { display:inline-block; padding:0.55rem 1.3rem;
             border-radius:999px; font-weight:700; font-size:1.15rem;
             font-family:'Space Grotesk',sans-serif; }
    .badge-fake { background:#3a0d12; color:#ff6b81;
                  border:1px solid #ff6b81; }
    .badge-real { background:#0c2e1a; color:#3ddc84;
                  border:1px solid #3ddc84; }
    .metric-card { background:#141923; border:1px solid #232a36;
                   border-radius:14px; padding:1.1rem 1.3rem; }
    .word-fake { background:#4a1620; color:#ff8a9b; padding:1px 5px;
                 border-radius:4px; }
    .word-real { background:#10331f; color:#5ee9a0; padding:1px 5px;
                 border-radius:4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

SUBJECT_OPTIONS = [
    "politicsNews", "worldnews", "News", "politics",
    "Government News", "left-news", "US_News", "Middle-east",
]


# ------------------------------------------------------------------ #
# Lazy model loader (cached across reruns)
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner="Loading FakeShield engine ...")
def _load_engine():
    """Import the heavy prediction modules once and cache them.

    Returns:
        The ``predict`` module, or ``None`` if the model is missing.
    """
    if not os.path.exists(BEST_MODEL_PATH):
        return None
    import predict  # noqa: WPS433 - intentional lazy import
    return predict


def _gauge(score: float) -> go.Figure:
    """Build a color-coded Plotly risk gauge (Step 10c).

    Args:
        score: Misinformation risk score in ``[0, 1]``.

    Returns:
        A Plotly figure.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={"suffix": "%", "font": {"size": 46}},
        title={"text": "Misinformation Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#e8eef7"},
            "steps": [
                {"range": [0, 30], "color": "#1d6b3f"},
                {"range": [30, 60], "color": "#b9952b"},
                {"range": [60, 85], "color": "#c2622a"},
                {"range": [85, 100], "color": "#a4242f"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "value": score * 100,
            },
        },
    ))
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#e8eef7"})
    return fig


def _render_lime_html(words: list) -> str:
    """Render LIME word weights as color-highlighted HTML (Step 10d).

    Args:
        words: List of ``{"word", "weight"}`` dicts.

    Returns:
        An HTML string.
    """
    spans = []
    for item in words:
        if "word" not in item:
            continue
        cls = "word-fake" if item["weight"] > 0 else "word-real"
        spans.append(
            f"<span class='{cls}'>{item['word']} "
            f"({item['weight']:+.3f})</span>"
        )
    return " ".join(spans) if spans else "<i>No LIME words available.</i>"


# ================================================================== #
# Sidebar (Step 10a)
# ================================================================== #
with st.sidebar:
    st.header("🔍 FakeShield")
    st.caption("AI Misinformation Detector")
    st.divider()
    st.subheader("Model Info")
    st.markdown(
        "- **Architecture:** Multi-input NN\n"
        "- **Text branch:** BERT (bert-base-uncased) [CLS]\n"
        "- **Metadata branch:** 23 engineered features\n"
        "- **Fusion:** 160 → 64 → 1 (sigmoid)"
    )
    st.subheader("Dataset")
    st.markdown(
        "- Fake.csv — 23,502 articles\n"
        "- True.csv — 21,417 articles\n"
        "- Labels: 1 = Fake, 0 = True"
    )
    st.subheader("About")
    st.markdown(
        "FakeShield combines deep contextual text understanding with "
        "stylometric & temporal metadata signals, then explains every "
        "decision using SHAP + LIME."
    )

st.title("🔍 FakeShield — AI Misinformation Detector")

engine = _load_engine()
if engine is None:
    st.warning(
        "⚠️ No trained model found at `models/best_fake_news_model.pt`. "
        "Run the training pipeline first:\n\n"
        "```bash\npython src/data_preprocessing.py\n"
        "python src/train.py\n```"
    )

# ================================================================== #
# Single-article analysis (Steps 10b–10d)
# ================================================================== #
left, right = st.columns([1, 1.15], gap="large")

with left:
    st.subheader("📝 Analyze an Article")
    in_title = st.text_input("Article Title", "")
    in_text = st.text_area("Article Body Text", "", height=260)
    in_subject = st.selectbox("Subject / Category", SUBJECT_OPTIONS)
    in_date = st.date_input("Publication Date")
    submit = st.button("🔍 Analyze Article", type="primary",
                        use_container_width=True)

with right:
    st.subheader("📊 Results")
    if submit:
        if engine is None:
            st.error("Model not available — train it first.")
        elif not (in_title.strip() or in_text.strip()):
            st.error("Please enter at least a title or body text.")
        else:
            with st.spinner("Running multi-input analysis ..."):
                report = engine.predict_single_article(
                    in_title, in_text, in_subject, str(in_date))

            score = report["misinformation_risk_score"]
            st.plotly_chart(_gauge(score), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                if report["prediction"] == "FAKE":
                    st.markdown(
                        "<span class='badge badge-fake'>🚨 FAKE NEWS"
                        "</span>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<span class='badge badge-real'>✅ REAL NEWS"
                        "</span>", unsafe_allow_html=True)
            with c2:
                st.metric("Risk Tier", report["risk_tier"])

            st.metric("Credibility Pre-Screen Score",
                      f"{report['credibility_risk_score']:.4f}")

            tab1, tab2, tab3 = st.tabs([
                "📝 Text Analysis (LIME)",
                "📊 Metadata Signals (SHAP)",
                "📋 Full Report",
            ])
            with tab1:
                st.markdown("Red = pushes toward **FAKE**, "
                            "Green = pushes toward **REAL**.")
                st.markdown(
                    _render_lime_html(report["top_suspicious_words"]),
                    unsafe_allow_html=True)
            with tab2:
                meta = [m for m in report["top_metadata_signals"]
                        if "feature" in m]
                if meta:
                    mdf = pd.DataFrame(meta)
                    fig = go.Figure(go.Bar(
                        x=mdf["shap_value"], y=mdf["feature"],
                        orientation="h", marker_color="#5b9bd5"))
                    fig.update_layout(
                        height=320, paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#e8eef7"},
                        title="Top 5 Metadata Feature Contributions")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("SHAP signals unavailable.")
            with tab3:
                st.json(report)
    else:
        st.info("Enter an article on the left and click "
                "**Analyze Article**.")

st.divider()

# ================================================================== #
# Batch analysis (Step 10e)
# ================================================================== #
st.subheader("📁 Batch Analysis")
uploaded = st.file_uploader(
    "Upload a CSV for batch analysis "
    "(columns: title, text, subject, date)", type=["csv"])

if uploaded is not None and engine is not None:
    tmp_path = os.path.join(PROJECT_ROOT, "data", "_uploaded_batch.csv")
    with open(tmp_path, "wb") as fh:
        fh.write(uploaded.getbuffer())
    with st.spinner("Scoring uploaded articles ..."):
        results = engine.batch_predict(tmp_path)

    def _hl(row: pd.Series):
        """Row-wise color by risk tier."""
        color = {
            "LOW": "#10331f", "MEDIUM": "#3a3416",
            "HIGH": "#4a2a14", "CRITICAL": "#4a1620",
        }.get(row.get("risk_tier", ""), "")
        return [f"background-color: {color}"] * len(row)

    st.dataframe(results.head(50).style.apply(_hl, axis=1),
                 use_container_width=True)
    buff = io.StringIO()
    results.to_csv(buff, index=False)
    st.download_button("⬇️ Download Results as CSV",
                       buff.getvalue(),
                       file_name="fakeshield_results.csv",
                       mime="text/csv")
elif uploaded is not None:
    st.error("Model not available — train it first.")

# ================================================================== #
# Footer (Step 10f)
# ================================================================== #
st.divider()
comp_csv = os.path.join(PLOTS_DIR, "model_comparison.csv")
if os.path.exists(comp_csv):
    st.markdown("**Model Performance (from evaluation):**")
    st.dataframe(pd.read_csv(comp_csv), use_container_width=True)
else:
    st.caption("Model performance stats appear here after running "
               "`python src/evaluate.py`.")
st.caption(
    "⚠️ This tool is for educational purposes. "
    "Always verify news with trusted sources."
)
