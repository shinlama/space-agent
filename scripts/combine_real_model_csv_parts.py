from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine chunked real-model CSV outputs.")
    parser.add_argument(
        "--suffixes",
        nargs="+",
        required=True,
        help="Chunk suffixes such as part_0000_0900 part_0900_1799",
    )
    return parser.parse_args()


def _combine_one(base_filename: str, suffixes: list[str]) -> None:
    frames = []
    for suffix in suffixes:
        part_name = f"{base_filename[:-4]}_{suffix}.csv"
        part_path = PROJECT_ROOT / part_name
        if not part_path.exists():
            raise FileNotFoundError(f"Missing chunk file: {part_path}")
        frames.append(pd.read_csv(part_path, encoding="utf-8-sig"))

    combined = pd.concat(frames, ignore_index=True)

    if "review_index" in combined.columns:
        combined = combined.sort_values(["review_index"]).reset_index(drop=True)
    elif "cafe_name" in combined.columns:
        combined = combined.sort_values(["cafe_name"]).reset_index(drop=True)

    output_path = PROJECT_ROOT / base_filename
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {output_path.name}: {len(combined):,} rows")


def main() -> None:
    args = _parse_args()
    suffixes = args.suffixes

    for base_filename in [
        "review_placeness_scores_real.csv",
        "placeness_final_research_metrics_real.csv",
        "reviews_with_sentiment_real.csv",
        "cafe_avg_sentiment_real.csv",
    ]:
        _combine_one(base_filename, suffixes)


if __name__ == "__main__":
    main()
