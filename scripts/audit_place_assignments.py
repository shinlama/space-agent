from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from revalidate_google_places import decode_place_slug, name_similarity, normalize_name


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "google_reviews_full_corrected_textclean_v4.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "place_assignment_audit.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit source-place to Google-place assignments before LLM mapping."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    return parser.parse_args()


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    configure_console_encoding()
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "상가업소번호",
        "상호명",
        "시군구명",
        "행정동명",
        "도로명주소",
        "place_id",
    ]
    reviews = pd.read_csv(
        args.input,
        encoding="utf-8-sig",
        usecols=columns,
        on_bad_lines="skip",
    )
    review_counts = reviews.groupby("상가업소번호").size().rename("review_count")
    places = reviews.drop_duplicates("상가업소번호").copy()
    places = places.merge(review_counts, on="상가업소번호", how="left")
    places["google_place_name"] = places["place_id"].map(decode_place_slug)
    places["name_similarity"] = places.apply(
        lambda row: name_similarity(row["상호명"], row["google_place_name"]),
        axis=1,
    )
    places["substring_match"] = places.apply(
        lambda row: (
            normalize_name(row["상호명"])
            in normalize_name(row["google_place_name"])
            or normalize_name(row["google_place_name"])
            in normalize_name(row["상호명"])
        ),
        axis=1,
    )
    places["target_assignment_count"] = places.groupby("place_id")[
        "상가업소번호"
    ].transform("nunique")

    ranked = places.sort_values(
        ["place_id", "substring_match", "name_similarity", "review_count"],
        ascending=[True, False, False, False],
    )
    best_ids = set(ranked.drop_duplicates("place_id")["상가업소번호"])
    places["best_assignment_for_target"] = places["상가업소번호"].isin(best_ids)
    places["low_name_similarity"] = (
        places["name_similarity"] < args.similarity_threshold
    ) & ~places["substring_match"]
    places["duplicate_target_non_best"] = (
        places["target_assignment_count"] > 1
    ) & ~places["best_assignment_for_target"]
    places["eligible_conservative"] = (
        places["best_assignment_for_target"] & ~places["low_name_similarity"]
    )
    places["audit_reason"] = places.apply(
        lambda row: "; ".join(
            reason
            for condition, reason in (
                (row["low_name_similarity"], "low_name_similarity"),
                (row["duplicate_target_non_best"], "duplicate_google_target"),
            )
            if condition
        )
        or "eligible_conservative",
        axis=1,
    )

    places = places.sort_values(
        ["eligible_conservative", "target_assignment_count", "name_similarity"],
        ascending=[True, False, True],
    )
    places.to_csv(args.output, index=False, encoding="utf-8-sig")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "similarity_threshold": args.similarity_threshold,
        "review_rows": int(len(reviews)),
        "source_places": int(len(places)),
        "unique_google_targets": int(places["place_id"].nunique()),
        "shared_google_targets": int(
            places.loc[places["target_assignment_count"] > 1, "place_id"].nunique()
        ),
        "source_places_in_shared_targets": int(
            (places["target_assignment_count"] > 1).sum()
        ),
        "low_name_similarity_places": int(places["low_name_similarity"].sum()),
        "duplicate_target_non_best_places": int(
            places["duplicate_target_non_best"].sum()
        ),
        "eligible_conservative_places": int(places["eligible_conservative"].sum()),
        "eligible_conservative_reviews": int(
            places.loc[places["eligible_conservative"], "review_count"].sum()
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
