# US–India Trade Deal: Reddit Sentiment — LLM vs Classical NLP

**Question:** How does the Reddit community feel about the US–India trade deal, what are they talking
about, and which model measures it best — an LLM, or classical NLP (TF-IDF / VADER)?

**Answer (short):** Reddit is ~2/3 negative. Anger targets domestic politics ("Modi surrendered", Adani)
and geopolitics (distrust of Trump) more than tariff numbers. The **LLM (zero-shot)** is the best model
(macro-F1 0.84 vs ≤0.39 for classical models*), mainly because it understands sarcasm, Hinglish and context.
\*provisional — gold labels are an independent LLM pass until human labels are added (see Limitations).

---

## Architecture

```
                 ┌──────────────────────────── DATA LAYER ─────────────────────────────┐
  Reddit  ──►  STEP 1 Collect            ──►  STEP 2 Clean & preprocess
 (79 subs)     PRAW (free) / Apify          • bot/deleted/mod removal
               4 queries · 12 months        • thread relevance: keywords + human review
               posts + top-15 comments      • <4 words, near-duplicates
               2,005 items                   • 2 text versions:  text_llm (light) │ text_nlp (heavy)
                                             1,071 items                │                 │
                 └──────────────────────────────────────────────────────┼─────────────────┼──┘
                                                                        ▼                 ▼
                 ┌──────────── MODEL LAYER ────────────────────────────────────────────────┐
                 │ STEP 3  LLM (Claude, zero-shot prompt)        STEP 5  Classical NLP       │
                 │   sentiment · topic · news/opinion   ──labels──►  TF-IDF (1-2 grams)      │
                 │   = insights + "teacher" labels      (871 train)   ├ Naive Bayes          │
                 │                                                   ├ Logistic Regression   │
                 │                                                   └ Linear SVM            │
                 │                                                VADER (lexicon, no training)│
                 └──────────────────────────────┬─────────────────────────┬─────────────────┘
                                                ▼                         ▼
                 ┌──────────── EVALUATION LAYER ──────────────────────────────────────────┐
                 │ STEP 4  Gold test set: 200 held-out items, blind, stratified            │
                 │ STEP 6  Compare: macro-F1 (+95% bootstrap CI), accuracy, per-class F1,   │
                 │         confusion matrices, speed, cost  ──►  best model + insights     │
                 └─────────────────────────────────────────────────────────────────────────┘
```

## Codebase

| File | Step | What it does |
|---|---|---|
| `src/step1_collect_reddit.py` | 1 | Search Reddit (PRAW free or Apify), save raw JSON + flat table |
| `src/step2_clean_reddit.py` | 2 | Filters + relevance review + two cleaned text versions |
| `src/step3_llm_sentiment.py` | 3 | Zero-shot LLM labelling via Claude API (prompt inside) |
| `src/step3b_llm_results.py` | 3 | LLM results → charts & tables |
| `src/step4_make_gold_sheet.py` | 4 | Draw 200 test items → blind labelling spreadsheet |
| `src/step5_nlp_models.py` | 5 | VADER + TF-IDF models, 5-fold CV grid search, top words |
| `src/step5b_nlp_charts.py` | 5 | TF-IDF "what people talk about" chart |
| `src/step6_compare_models.py` | 6 | Score every model on the gold set, pick the best |
| `run_all.py` | – | Runs the whole pipeline in order |

```
data/        raw/ reddit_raw.json · reddit_raw.csv · clean.csv · llm_labels.csv
             thread_review.csv (relevance audit) · gold_sample_ids.csv · cleaning_log.csv
models/      fitted TF-IDF pipelines (.joblib)
outputs/     figures/*.png · model_comparison.csv · nlp_cv_results.csv · tfidf_top_terms.csv
             gold_labelling_sheet.xlsx (fill this in!)
archive/     earlier X+Reddit version (project later narrowed to Reddit only)
```

## Run it

```bash
pip install -r requirements.txt
python run_all.py                                            # uses saved data
python run_all.py --gold outputs/gold_labelling_sheet.xlsx   # after you label the 200 items
python run_all.py --scrape praw                              # re-collect (needs REDDIT_CLIENT_ID/SECRET)
export ANTHROPIC_API_KEY=... && python src/step3_llm_sentiment.py   # re-run LLM labelling via API
```

## Key design decisions
- **Reddit only** – free official API (reproducible), 93% opinions vs ~50% on X.
- **Comments kept** – 92% of the data; opinions live in replies, posts are mostly news.
- **Hybrid relevance filter** – keywords + manual review of all 155 threads (audit trail in `thread_review.csv`).
- **Two text versions** – LLMs read natural text; TF-IDF needs normalised tokens. Negations kept.
- **Zero-shot LLM** – no labelled data needed; handles sarcasm & Hinglish.
- **LLM as teacher for TF-IDF** (knowledge distillation) – no human training labels available.
- **Blind, stratified, held-out gold set** – avoids circular evaluation and anchoring bias.
- **Macro-F1 + bootstrap CI** – fair to the rare positive class; tells real differences from luck.

## Limitations
1. Gold labels are currently an independent 2nd LLM pass, not human → LLM score is an upper bound.
2. Only 35 positive training examples for TF-IDF models (class imbalance).
3. Upvote scores unavailable from the scraper used; data concentrated in Feb 2026.
4. Reddit users ≠ Indian public (younger, urban, English-speaking).

## Future work
Human labels from 2 annotators (Cohen's kappa) · fine-tuned BERT/RoBERTa · compare a second LLM (GPT) ·
repeat LLM runs for consistency · upvote-weighted sentiment via PRAW.
