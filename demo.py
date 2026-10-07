"""
LIVE DEMO - type any comment, see what every model thinks.

    python demo.py                      # interactive: type comments, 'q' to quit
    python demo.py "Great deal, farmers will love starving 😂"   # one comment

Uses the models already trained in Step 5 (models/*.joblib) and VADER.
If ANTHROPIC_API_KEY is set, it also asks Claude (the LLM) live, using the same prompt as Step 3.
"""
import os
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from step2_clean_reddit import clean_for_llm, clean_for_nlp          # same cleaning as the project
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODELS = {name: joblib.load(ROOT / f"models/{file}.joblib") for name, file in [
    ("Naive Bayes", "naive_bayes"),
    ("Logistic Regression", "logistic_regression"),
    ("Linear SVM", "linear_svm"),
]}
vader = SentimentIntensityAnalyzer()

SAMPLES = [
    "Great deal, farmers will love starving 😂",
    "This is the epitome of bending the knee.",
    "India and the US signed a framework agreement today; tariffs cut to 18%.",
    "Proud that our government did not bend the knee. Good negotiation.",
]
# Claude's answers for the 4 samples (saved, so the demo works without an API key)
SAVED_LLM = {
    SAMPLES[0]: "negative  (saved Claude answer: sarcasm about farmers)",
    SAMPLES[1]: "negative  (saved Claude answer: accuses India of surrendering)",
    SAMPLES[2]: "neutral   (saved Claude answer: factual news)",
    SAMPLES[3]: "positive  (saved Claude answer: praises the government)",
}


def vader_label(text, threshold=0.05):
    c = vader.polarity_scores(text)["compound"]
    return ("positive" if c >= threshold else "negative" if c <= -threshold else "neutral"), c


def ask_llm(text):
    """Optional: live Claude call with the project's prompt (needs ANTHROPIC_API_KEY)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        from step3_llm_sentiment import call_llm
        out = call_llm(anthropic.Anthropic(), [(0, clean_for_llm(text))])
        return f"{out[0]['sentiment']}  ({out[0]['topic']}; reason: {out[0]['reason']})" if out else "no answer"
    except Exception as e:
        return f"error: {e}"


def predict(text):
    print("\n" + "=" * 70)
    print("COMMENT:", text)
    nlp_text = clean_for_nlp(text)
    print("Cleaned for TF-IDF:", nlp_text or "(empty)")
    print("-" * 70)
    v, score = vader_label(clean_for_llm(text))
    print(f"{'VADER (dictionary)':26} -> {v:9} (score {score:+.2f})")
    for name, model in MODELS.items():
        print(f"{'TF-IDF + ' + name:26} -> {model.predict([nlp_text])[0]}")
    llm = ask_llm(text) or SAVED_LLM.get(text)
    print(f"{'LLM (Claude)':26} -> {llm if llm else '(set ANTHROPIC_API_KEY to call Claude live)'}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(" ".join(sys.argv[1:]))
    else:
        print("Sample comments first:")
        for s in SAMPLES:
            predict(s)
        print("\nNow type your own comment (or 'q' to quit).")
        while True:
            t = input("\n> ").strip()
            if t.lower() in {"q", "quit", "exit"}:
                break
            if t:
                predict(t)
