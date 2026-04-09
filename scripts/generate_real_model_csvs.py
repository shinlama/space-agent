from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.config import ALL_FACTORS, GOOGLE_REVIEW_SAMPLE_CSV
from modules.model_loader import load_models
from modules.preprocess import load_data
from modules.score import calculate_final_research_metrics, calculate_place_scores
from modules.sentiment import run_sentiment_analysis


def main() -> None:
    print("Loading review data...")
    df_reviews = load_data(GOOGLE_REVIEW_SAMPLE_CSV, cache_version="real-csv-v1")
    print(f"Loaded {len(df_reviews):,} reviews from {GOOGLE_REVIEW_SAMPLE_CSV.name}")

    print("Loading Sentence-BERT and sentiment models...")
    sbert_model, sentiment_pipeline, sentiment_model_name = load_models()

    if type(sbert_model).__name__ != "SentenceTransformer":
        raise RuntimeError(
            "Sentence-BERT fallback encoder is active. Aborting because a real model run was requested."
        )
    if sentiment_model_name == "heuristic-rule-based-fallback":
        raise RuntimeError(
            "Heuristic sentiment fallback is active. Aborting because a real model run was requested."
        )

    print(f"Sentence-BERT: {type(sbert_model).__name__}")
    print(f"Sentiment model: {sentiment_model_name}")

    print("Running review-level sentiment analysis...")
    df_reviews_with_sentiment, df_avg_sentiment = run_sentiment_analysis(
        df_reviews,
        sentiment_pipeline,
        sentiment_model_name,
    )

    print("Running placeness scoring...")
    df_place_scores, df_review_scores = calculate_place_scores(
        df_reviews,
        sbert_model,
        sentiment_pipeline,
        ALL_FACTORS,
        sentiment_model_name=sentiment_model_name,
    )

    print("Calculating final cafe-level research metrics...")
    df_final_metrics = calculate_final_research_metrics(
        df_place_scores,
        list(ALL_FACTORS.keys()),
        len(df_reviews),
    )

    outputs = {
        "review_placeness_scores_real.csv": df_review_scores,
        "placeness_final_research_metrics_real.csv": df_final_metrics,
        "reviews_with_sentiment_real.csv": df_reviews_with_sentiment,
        "cafe_avg_sentiment_real.csv": df_avg_sentiment,
    }

    for filename, df in outputs.items():
        output_path = PROJECT_ROOT / filename
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {filename} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
