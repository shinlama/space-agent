from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.research_scoring import (
    DEFAULT_DEMO_OUTPUT_DIR,
    DEFAULT_MAPPING_CSV,
    calculate_scores,
)


DEPLOY_EVIDENCE_COLUMNS = [
    "review_index",
    "place_id",
    "source_cafe_name",
    "district",
    "neighborhood",
    "address",
    "review_text",
    "factor",
    "evidence",
    "cafe_name",
    "sentiment_key",
    "sentiment_label",
    "sentiment_value",
    "factor_category",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Streamlit-ready placeness score datasets from LLM mappings."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DEMO_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scored_evidence, factor_scores, place_scores = calculate_scores(input_path)

    evidence_path = output_dir / "scored_evidence.parquet"
    factor_path = output_dir / "factor_scores.parquet"
    place_path = output_dir / "place_scores.parquet"
    factor_csv_path = output_dir / "factor_scores.csv"
    place_csv_path = output_dir / "place_scores.csv"

    deploy_evidence = scored_evidence[DEPLOY_EVIDENCE_COLUMNS].copy()
    deploy_evidence.to_parquet(
        evidence_path,
        index=False,
        compression="zstd",
        row_group_size=20_000,
    )
    factor_scores.to_parquet(factor_path, index=False, compression="zstd")
    place_scores.to_parquet(place_path, index=False, compression="zstd")
    factor_scores.to_csv(factor_csv_path, index=False, encoding="utf-8-sig")
    place_scores.to_csv(place_csv_path, index=False, encoding="utf-8-sig")

    mention_share_sums = factor_scores.groupby("place_id")["mention_share"].sum()
    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "model": (
            str(scored_evidence["model"].dropna().iloc[0])
            if "model" in scored_evidence and scored_evidence["model"].notna().any()
            else ""
        ),
        "mapping_rows": int(len(scored_evidence)),
        "mapped_reviews": int(scored_evidence["review_index"].nunique()),
        "mapped_places": int(scored_evidence["place_id"].nunique()),
        "factor_score_rows": int(len(factor_scores)),
        "place_score_rows": int(len(place_scores)),
        "factor_score_min": float(factor_scores["factor_score"].min()),
        "factor_score_max": float(factor_scores["factor_score"].max()),
        "placeness_score_min": float(place_scores["placeness_score"].min()),
        "placeness_score_max": float(place_scores["placeness_score"].max()),
        "mention_share_sum_min": float(mention_share_sums.min()),
        "mention_share_sum_max": float(mention_share_sums.max()),
        "factor_counts": {
            str(key): int(value)
            for key, value in scored_evidence["factor"].value_counts().items()
        },
        "sentiment_counts": {
            str(key): int(value)
            for key, value in scored_evidence["sentiment_key"].value_counts().items()
        },
        "outputs": {
            "scored_evidence_parquet": str(evidence_path),
            "factor_scores_parquet": str(factor_path),
            "place_scores_parquet": str(place_path),
            "factor_scores_csv": str(factor_csv_path),
            "place_scores_csv": str(place_csv_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
