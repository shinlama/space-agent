from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.config import ALL_FACTORS, GOOGLE_REVIEW_SAMPLE_CSV, SIMILARITY_THRESHOLD
from modules.model_loader import get_embedding_model_name, load_models
from modules.preprocess import (
    is_metadata_only,
    is_numeric_only,
    load_data,
    truncate_text_for_bert,
)
from modules.score import calculate_final_research_metrics, split_into_sentences
from modules.sentiment import process_sentiment_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate real-model placeness CSVs.")
    parser.add_argument("--start-cafe", type=int, default=0, help="0-based inclusive cafe index")
    parser.add_argument("--end-cafe", type=int, default=None, help="0-based exclusive cafe index")
    parser.add_argument("--output-suffix", type=str, default="", help="Optional suffix appended before .csv")
    parser.add_argument("--sentiment-batch-size", type=int, default=16, help="Batch size for sentiment inference")
    parser.add_argument("--encode-batch-size", type=int, default=128, help="Batch size for Sentence-BERT encoding")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="jhgan/ko-sroberta-multitask",
        help="Sentence-BERT model id to require for generation",
    )
    parser.add_argument(
        "--sentiment-model",
        type=str,
        default="cringepnh/koelectra-korean-sentiment",
        help="Sentiment model id to require for generation",
    )
    return parser.parse_args()


def _to_numpy(embeddings):
    if hasattr(embeddings, "cpu"):
        return embeddings.cpu().numpy()
    return np.asarray(embeddings)


def _batched_encode(model, texts: List[str], batch_size: int = 128) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        emb = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        chunks.append(_to_numpy(emb))
    return np.vstack(chunks) if chunks else np.empty((0, 0))


def _predict_sentiment_scores(
    texts: List[str],
    sentiment_pipeline,
    sentiment_model_name: str,
    batch_size: int = 32,
) -> List[float]:
    scores = [0.5] * len(texts)
    model_inputs = []

    for idx, text in enumerate(texts):
        value = (text or "").strip()
        if is_numeric_only(value):
            try:
                rating_value = float(value)
                scores[idx] = 0.9 if rating_value >= 4.0 else 0.5 if rating_value >= 3.0 else 0.1
            except ValueError:
                scores[idx] = 0.5
            continue

        if is_metadata_only(value):
            scores[idx] = 0.5
            continue

        model_inputs.append((idx, truncate_text_for_bert(value)))

    for start in range(0, len(model_inputs), batch_size):
        batch = model_inputs[start : start + batch_size]
        batch_texts = [text for _, text in batch]
        results = sentiment_pipeline(batch_texts, truncation=True, max_length=512)
        for (original_idx, _), result in zip(batch, results):
            _, score = process_sentiment_result(result, sentiment_model_name)
            scores[original_idx] = float(score)

    return scores


def _require_real_models(
    sbert_model,
    sentiment_model_name: str,
    expected_embedding_model: str,
    expected_sentiment_model: str,
) -> None:
    if type(sbert_model).__name__ != "SentenceTransformer":
        raise RuntimeError("Sentence-BERT fallback encoder is active.")
    loaded_embedding_name = get_embedding_model_name(sbert_model)
    if loaded_embedding_name != expected_embedding_model:
        raise RuntimeError(
            f"Requested embedding model '{expected_embedding_model}' but loaded '{loaded_embedding_name}'."
        )
    if sentiment_model_name == "heuristic-rule-based-fallback":
        raise RuntimeError("Heuristic sentiment fallback is active.")
    if sentiment_model_name != expected_sentiment_model:
        raise RuntimeError(
            f"Requested sentiment model '{expected_sentiment_model}' but loaded '{sentiment_model_name}'."
        )


def _filename_with_suffix(filename: str, suffix: str) -> str:
    if not suffix:
        return filename
    base = filename[:-4] if filename.endswith(".csv") else filename
    return f"{base}_{suffix}.csv"


def main() -> None:
    args = _parse_args()

    print("Loading review data...")
    df_reviews = load_data(GOOGLE_REVIEW_SAMPLE_CSV, cache_version="real-fast-v1")
    print(f"Loaded {len(df_reviews):,} reviews")

    all_cafe_names = list(dict.fromkeys(df_reviews["cafe_name"].tolist()))
    start_cafe = max(0, args.start_cafe)
    end_cafe = args.end_cafe if args.end_cafe is not None else len(all_cafe_names)
    end_cafe = min(end_cafe, len(all_cafe_names))

    if start_cafe >= end_cafe:
        raise ValueError(f"Invalid cafe slice: start={start_cafe}, end={end_cafe}")

    selected_cafes = set(all_cafe_names[start_cafe:end_cafe])
    df_reviews = df_reviews[df_reviews["cafe_name"].isin(selected_cafes)].copy()
    suffix = args.output_suffix.strip()
    if not suffix and (start_cafe != 0 or end_cafe != len(all_cafe_names)):
        suffix = f"part_{start_cafe:04d}_{end_cafe:04d}"

    print(
        "Selected cafes: "
        f"{start_cafe:,}..{end_cafe - 1:,} / {len(all_cafe_names):,} "
        f"({df_reviews['cafe_name'].nunique():,} cafes, {len(df_reviews):,} reviews)"
    )

    print("Loading real models...")
    sbert_model, sentiment_pipeline, sentiment_model_name = load_models(
        args.embedding_model,
        args.sentiment_model,
    )
    _require_real_models(
        sbert_model,
        sentiment_model_name,
        expected_embedding_model=args.embedding_model,
        expected_sentiment_model=args.sentiment_model,
    )
    print(f"Sentence-BERT: {get_embedding_model_name(sbert_model)}")
    print(f"Sentiment model: {sentiment_model_name}")

    factor_names = list(ALL_FACTORS.keys())
    factor_sentences = list(ALL_FACTORS.values())
    factor_embeddings = _batched_encode(
        sbert_model,
        factor_sentences,
        batch_size=args.encode_batch_size,
    )

    print("Running review-level sentiment analysis...")
    review_texts_all = df_reviews["review_text"].astype(str).tolist()
    review_sentiment_scores = _predict_sentiment_scores(
        review_texts_all,
        sentiment_pipeline,
        sentiment_model_name,
        batch_size=args.sentiment_batch_size,
    )

    df_reviews_with_sentiment = df_reviews.copy()
    df_reviews_with_sentiment["sentiment_score"] = review_sentiment_scores
    df_reviews_with_sentiment["sentiment_label"] = [
        "POSITIVE" if score >= 0.6 else "NEGATIVE" if score <= 0.4 else "NEUTRAL"
        for score in review_sentiment_scores
    ]
    df_avg_sentiment = (
        df_reviews_with_sentiment.groupby("cafe_name", as_index=False)["sentiment_score"]
        .mean()
        .rename(columns={"sentiment_score": "avg_review_sentiment_score"})
    )

    print("Running optimized placeness scoring...")
    cafe_rows: List[Dict] = []
    review_rows: List[Dict] = []
    cafe_groups = list(df_reviews.groupby("cafe_name", sort=False))

    for cafe_idx, (cafe_name, group) in enumerate(cafe_groups, start=1):
        if cafe_idx % 25 == 0 or cafe_idx == 1:
            print(f"Processing cafe {cafe_idx:,}/{len(cafe_groups):,}: {cafe_name}")

        review_texts = group["review_text"].astype(str).tolist()
        review_indices = group.index.tolist()
        n_reviews = len(review_texts)

        all_sentences: List[str] = []
        sentence_to_review: List[int] = []
        for local_idx, review_text in enumerate(review_texts):
            sentences = split_into_sentences(review_text)
            all_sentences.extend(sentences)
            sentence_to_review.extend([local_idx] * len(sentences))

        cafe_scores = {"cafe_name": cafe_name}
        review_factor_scores: Dict[int, Dict[str, float]] = {idx: {} for idx in range(n_reviews)}

        if all_sentences:
            sentence_embeddings = _batched_encode(
                sbert_model,
                all_sentences,
                batch_size=args.encode_batch_size,
            )
            sentence_similarity = cosine_similarity(sentence_embeddings, factor_embeddings)

            relevant_sentence_indices = sorted(
                {idx for idx in range(len(all_sentences)) if np.any(sentence_similarity[idx] >= SIMILARITY_THRESHOLD)}
            )
            sentence_sentiment_map: Dict[int, float] = {}

            if relevant_sentence_indices:
                relevant_sentences = [all_sentences[idx] for idx in relevant_sentence_indices]
                relevant_scores = _predict_sentiment_scores(
                    relevant_sentences,
                    sentiment_pipeline,
                    sentiment_model_name,
                    batch_size=args.sentiment_batch_size,
                )
                sentence_sentiment_map = {
                    sent_idx: score
                    for sent_idx, score in zip(relevant_sentence_indices, relevant_scores)
                }

            factor_review_score_lists: Dict[str, Dict[int, List[float]]] = {
                factor_name: {idx: [] for idx in range(n_reviews)} for factor_name in factor_names
            }

            for sent_idx, local_review_idx in enumerate(sentence_to_review):
                sent_score = sentence_sentiment_map.get(sent_idx)
                if sent_score is None:
                    continue
                for factor_idx, factor_name in enumerate(factor_names):
                    if sentence_similarity[sent_idx, factor_idx] >= SIMILARITY_THRESHOLD:
                        factor_review_score_lists[factor_name][local_review_idx].append(sent_score)

            for factor_name in factor_names:
                review_level_scores = []
                mention_count = 0
                for local_review_idx in range(n_reviews):
                    scores = factor_review_score_lists[factor_name][local_review_idx]
                    if scores:
                        avg_score = float(np.mean(scores))
                        review_factor_scores[local_review_idx][factor_name] = avg_score
                        review_level_scores.append(avg_score)
                        mention_count += 1

                cafe_scores[f"점수_{factor_name}"] = float(np.mean(review_level_scores)) if review_level_scores else 0.5
                cafe_scores[f"리뷰수_{factor_name}"] = mention_count
        else:
            for factor_name in factor_names:
                cafe_scores[f"점수_{factor_name}"] = 0.5
                cafe_scores[f"리뷰수_{factor_name}"] = 0

        review_embeddings = _batched_encode(
            sbert_model,
            review_texts,
            batch_size=args.encode_batch_size,
        )
        review_similarity = cosine_similarity(review_embeddings, factor_embeddings)

        for local_idx, (global_idx, review_text) in enumerate(zip(review_indices, review_texts)):
            row = {
                "review_index": global_idx,
                "cafe_name": cafe_name,
                "review_text": review_text,
            }
            for factor_idx, factor_name in enumerate(factor_names):
                row[f"{factor_name}_유사도"] = float(review_similarity[local_idx, factor_idx])
                row[f"{factor_name}_점수"] = review_factor_scores[local_idx].get(factor_name, np.nan)
            review_rows.append(row)

        cafe_rows.append(cafe_scores)

    df_place_scores = pd.DataFrame(cafe_rows)
    df_review_scores = pd.DataFrame(review_rows)
    df_final_metrics = calculate_final_research_metrics(
        df_place_scores,
        factor_names,
        len(df_reviews),
    )

    outputs = {
        "review_placeness_scores_real.csv": df_review_scores,
        "placeness_final_research_metrics_real.csv": df_final_metrics,
        "reviews_with_sentiment_real.csv": df_reviews_with_sentiment,
        "cafe_avg_sentiment_real.csv": df_avg_sentiment,
    }

    for filename, df in outputs.items():
        output_name = _filename_with_suffix(filename, suffix)
        output_path = PROJECT_ROOT / output_name
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {output_name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
