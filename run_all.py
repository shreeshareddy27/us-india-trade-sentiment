"""
Run the whole pipeline in order (skips scraping by default - uses data/raw/reddit_raw.json).

  python run_all.py                       # steps 1-6, gold = fallback LLM 2nd pass
  python run_all.py --gold outputs/gold_labelling_sheet.xlsx   # steps 1-6 with HUMAN gold labels
  python run_all.py --scrape praw         # re-collect Reddit for free first (needs Reddit API keys)
"""
import argparse, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("--gold"); ap.add_argument("--scrape", choices=["praw", "apify"])
a = ap.parse_args()
py = sys.executable
steps = [
    [py, "src/step1_collect_reddit.py"] + (["--method", a.scrape] if a.scrape else ["--skip-scrape"]),
    [py, "src/step2_clean_reddit.py"],
    # Step 3 (LLM labelling) is NOT re-run automatically: it costs API money.
    # Run `python src/step3_llm_sentiment.py` yourself if you have ANTHROPIC_API_KEY.
    [py, "src/step3b_llm_results.py"],
    [py, "src/step5_nlp_models.py"],
    [py, "src/step5b_nlp_charts.py"],
    [py, "src/step6_compare_models.py"] + (["--gold", a.gold] if a.gold else []),
]
for s in steps:
    print("\n" + "=" * 70 + f"\n>>> {' '.join(s[1:])}\n" + "=" * 70)
    subprocess.run(s, check=True)
print("\nDone. Charts in outputs/figures, results in outputs/model_comparison.csv")
