"""
FRONTEND - interactive web app for the project.

    pip install streamlit
    streamlit run app.py          -> opens in your browser at http://localhost:8501

Tabs: Try it live · Community insights · Model comparison · Data explorer
Uses the saved data, models and charts - no API key needed (optional live Claude if ANTHROPIC_API_KEY is set).
"""
import os
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from step2_clean_reddit import clean_for_llm, clean_for_nlp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="US–India Trade Deal Sentiment", page_icon="📊", layout="wide")
COLORS = {"positive": "#2A78D6", "negative": "#E34948", "neutral": "#8A8984"}
FIG = ROOT / "outputs" / "figures"


# ---------------------------------------------------------------- loading (cached)
@st.cache_resource
def load_models():
    m = {name: joblib.load(ROOT / f"models/{f}.joblib") for name, f in [
        ("TF-IDF + Naive Bayes", "naive_bayes"),
        ("TF-IDF + Logistic Regression", "logistic_regression"),
        ("TF-IDF + Linear SVM", "linear_svm")]}
    return m, SentimentIntensityAnalyzer()


@st.cache_data
def load_data():
    clean = pd.read_csv(ROOT / "data/clean.csv")
    labels = pd.read_csv(ROOT / "data/llm_labels.csv")
    df = clean.merge(labels, on="doc_id", how="left")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    comp = pd.read_csv(ROOT / "outputs/model_comparison.csv")
    return df, comp


MODELS, VADER = load_models()
DF, COMP = load_data()


def badge(label):
    c = COLORS.get(label, "#555")
    return f"<span style='background:{c};color:white;padding:4px 12px;border-radius:12px;font-weight:600'>{label}</span>"


def ask_claude(text):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        from step3_llm_sentiment import call_llm
        out = call_llm(anthropic.Anthropic(), [(0, clean_for_llm(text))])
        return out[0] if out else None
    except Exception as e:
        return {"sentiment": "error", "reason": str(e)}


# ---------------------------------------------------------------- header
st.title("How does Reddit feel about the US–India trade deal?")
st.caption("LLM (Claude) vs classical NLP (TF-IDF, VADER) · Shreesha Reddy · M.S. Data Science, ASU")

opinions = DF[DF["llm_post_type"] == "opinion"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clean Reddit items", f"{len(DF):,}", "from 2,005 collected")
c2.metric("Opinions that are negative", f"{(opinions.llm_sentiment == 'negative').mean():.0%}")
c3.metric("LLM score (macro-F1)", f"{COMP.iloc[0].macro_f1:.2f}")
c4.metric("Best NLP score (macro-F1)", f"{COMP.iloc[1:].macro_f1.max():.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Try it live", "💬 Community insights", "🏆 Model comparison", "📂 Data explorer"])

# ---------------------------------------------------------------- tab 1: live demo
with tab1:
    st.subheader("Type any comment — every model labels it")
    examples = ["Great deal, farmers will love starving 😂",
                "This is the epitome of bending the knee.",
                "India and the US signed a framework agreement today; tariffs cut to 18%.",
                "Proud that our government did not bend the knee. Good negotiation."]
    pick = st.selectbox("Pick an example, or write your own below", ["(write my own)"] + examples)
    text = st.text_area("Comment", value="" if pick == "(write my own)" else pick, height=90)

    if st.button("Analyse", type="primary") and text.strip():
        nlp_text = clean_for_nlp(text)
        st.write("**What TF-IDF sees after cleaning:**", f"`{nlp_text or '(empty)'}`")
        score = VADER.polarity_scores(clean_for_llm(text))["compound"]
        v = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
        rows = [("VADER (word dictionary)", v, f"score {score:+.2f}")]
        rows += [(name, m.predict([nlp_text])[0], "learned from 871 LLM-labelled comments")
                 for name, m in MODELS.items()]
        for name, label, note in rows:
            a, b, c = st.columns([2, 1, 3])
            a.write(f"**{name}**"); b.markdown(badge(label), unsafe_allow_html=True); c.caption(note)

        st.divider()
        llm = ask_claude(text)
        a, b, c = st.columns([2, 1, 3])
        a.write("**LLM (Claude)**")
        if llm:
            b.markdown(badge(llm.get("sentiment", "?")), unsafe_allow_html=True)
            c.caption(f"topic: {llm.get('topic', '-')} · reason: {llm.get('reason', '-')}")
        else:
            b.write("—")
            c.caption("No API key set. Show the LLM live by pasting the project prompt into Claude in your browser, "
                      "or set ANTHROPIC_API_KEY before running the app.")

    with st.expander("How does each model decide?"):
        st.markdown("""
- **VADER** – looks up each word in a ready-made happy/sad dictionary and adds the scores. No training. Fooled by sarcasm.
- **Naive Bayes** – how often each word appears in positive / negative / neutral comments (like a spam filter).
- **Logistic Regression** – gives every word a weight, adds them up, turns it into a probability.
- **Linear SVM** – draws the best dividing line between the classes.
- **LLM (Claude)** – reads the whole comment and understands meaning, sarcasm and Hinglish (zero-shot, temperature 0).
""")

# ---------------------------------------------------------------- tab 2: insights
with tab2:
    st.subheader("What the community is talking about")
    a, b = st.columns(2)
    a.image(str(FIG / "llm_sentiment_overall.png"), caption="Overall sentiment (LLM labels)")
    b.image(str(FIG / "llm_net_sentiment_by_topic.png"), caption="Net sentiment by topic = % positive − % negative")
    st.image(str(FIG / "llm_sentiment_over_time.png"), caption="Sentiment over time — spikes in Feb 2026 and Sep 2026")
    st.markdown("""
**Key insights**
- About **two-thirds of opinions are negative**; only about 1 in 10 is positive.
- Anger is **political, not economic**: domestic politics −78 vs tariff terms −39.
- Main themes: **surrender** ("bending the knee"), **distrust of Trump / delay fatigue**, **farmers & dairy**,
  **Russian oil**, **mockery of politicians**.
- The positive voices are mostly **government defenders and national pride** — a polarised debate.
""")

# ---------------------------------------------------------------- tab 3: comparison
with tab3:
    st.subheader("Same 200 held-out comments, every model")
    show = COMP[["model", "macro_f1", "ci_low", "ci_high", "accuracy", "f1_positive"]].copy()
    show.columns = ["Model", "Macro-F1", "95% CI low", "95% CI high", "Accuracy", "F1 (positive)"]
    st.dataframe(show.style.format({c: "{:.2f}" for c in show.columns[1:]}), hide_index=True, use_container_width=True)
    st.info("**Macro-F1** scores every class equally, so a model can't win by always guessing 'negative'. "
            "Early result: test labels come from an independent 2nd LLM pass; human labels will replace them.")
    a, b = st.columns(2)
    a.image(str(FIG / "model_comparison.png"), caption="Macro-F1 with 95% confidence intervals")
    b.image(str(FIG / "tfidf_top_terms.png"), caption="What TF-IDF learned (top words)")
    st.image(str(FIG / "confusion_matrices.png"), caption="Confusion matrices: where each model goes wrong")

# ---------------------------------------------------------------- tab 4: data explorer
with tab4:
    st.subheader("Browse the 1,071 cleaned Reddit items")
    f1, f2, f3, f4 = st.columns(4)
    sent = f1.multiselect("Sentiment", ["negative", "neutral", "positive"], default=["negative", "neutral", "positive"])
    topics = sorted(DF["llm_topic"].dropna().unique())
    top = f2.multiselect("Topic", topics, default=topics)
    kind = f3.multiselect("Type", ["post", "comment"], default=["post", "comment"])
    q = f4.text_input("Search text")
    view = DF[DF.llm_sentiment.isin(sent) & DF.llm_topic.isin(top) & DF.doc_type.isin(kind)]
    if q:
        view = view[view.text_llm.str.contains(q, case=False, na=False)]
    st.caption(f"{len(view):,} items")
    st.dataframe(view[["created_at", "community", "doc_type", "llm_sentiment", "llm_topic", "text_llm"]]
                 .rename(columns={"text_llm": "text", "llm_sentiment": "sentiment", "llm_topic": "topic"})
                 .sort_values("created_at", ascending=False),
                 hide_index=True, use_container_width=True, height=450)
