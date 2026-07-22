from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from revalidate_google_places import (
    NAME,
    ROAD_ADDRESS,
    STORE_ID,
    haversine_m,
    name_similarity,
)
from revalidate_google_places_http import name_identity_match, road_address_key


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify revalidated Google place links for retention or recrawl."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "google_place_revalidation_all_v5.csv",
    )
    parser.add_argument(
        "--override-results",
        type=Path,
        action="append",
        default=[],
        help="Higher-quality result CSV whose store IDs replace base results.",
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "google_reviews_full_corrected_textclean_v4.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "place_revalidation_classified_v5.csv",
    )
    return parser.parse_args()


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def cafe_category(value: object) -> bool:
    text = str(value or "").lower()
    tokens = (
        "커피",
        "카페",
        "제과",
        "베이커리",
        "빵",
        "디저트",
        "도넛",
        "아이스크림",
        "초콜릿",
        "찻집",
        "tea house",
        "coffee",
        "cafe",
        "bakery",
        "dessert",
    )
    return any(token in text for token in tokens)


def load_results(base_path: Path, override_paths: list[Path]) -> pd.DataFrame:
    base = pd.read_csv(base_path, encoding="utf-8-sig", dtype=str).fillna("")
    base = base.drop_duplicates(STORE_ID, keep="last")
    for override_path in override_paths:
        override = pd.read_csv(
            override_path, encoding="utf-8-sig", dtype=str
        ).fillna("")
        override = override.drop_duplicates(STORE_ID, keep="last")
        base = base[~base[STORE_ID].isin(set(override[STORE_ID]))]
        base = pd.concat([base, override], ignore_index=True)
    return base


def main() -> None:
    configure_console_encoding()
    args = parse_args()
    args.results = args.results.resolve()
    args.reviews = args.reviews.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frame = load_results(args.results, [path.resolve() for path in args.override_results])
    review_columns = [STORE_ID, "place_id", "lat", "lng"]
    current = pd.read_csv(
        args.reviews,
        encoding="utf-8-sig",
        usecols=review_columns,
        dtype={STORE_ID: str},
        on_bad_lines="skip",
    ).drop_duplicates(STORE_ID)
    current = current.rename(
        columns={
            "place_id": "current_place_slug",
            "lat": "current_google_lat",
            "lng": "current_google_lng",
        }
    )
    frame = frame.merge(current, on=STORE_ID, how="left")

    numeric_columns = (
        "source_lat",
        "source_lng",
        "corrected_google_lat",
        "corrected_google_lng",
        "corrected_distance_m",
        "corrected_name_similarity",
        "corrected_address_similarity",
        "current_google_lat",
        "current_google_lng",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["category_match"] = frame["corrected_google_categories"].map(
        cafe_category
    )
    frame["source_road_key"] = frame[ROAD_ADDRESS].map(road_address_key)
    frame["corrected_road_key"] = frame["corrected_google_address"].map(
        road_address_key
    )
    frame["road_number_match"] = (
        frame["source_road_key"].notna()
        & frame["corrected_road_key"].notna()
        & frame["source_road_key"].eq(frame["corrected_road_key"])
    )
    frame["source_identity_match"] = frame.apply(
        lambda row: name_identity_match(
            row[NAME], row["corrected_google_name"], row["road_number_match"]
        ),
        axis=1,
    )
    frame["existing_corrected_name_similarity"] = frame.apply(
        lambda row: name_similarity(
            row["existing_place_name"], row["corrected_google_name"]
        ),
        axis=1,
    )
    frame["existing_corrected_distance_m"] = frame.apply(
        lambda row: haversine_m(
            row["current_google_lat"],
            row["current_google_lng"],
            row["corrected_google_lat"],
            row["corrected_google_lng"],
        ),
        axis=1,
    )
    frame["current_target_assignment_count"] = frame.groupby(
        "current_place_slug"
    )[STORE_ID].transform("nunique")

    frame["candidate_plausible"] = (
        frame["category_match"]
        & frame["corrected_distance_m"].le(200)
        & (
            frame["source_identity_match"]
            | (
                frame["corrected_name_similarity"].ge(0.50)
                & frame["corrected_address_similarity"].ge(0.50)
            )
            | (
                frame["corrected_distance_m"].le(50)
                & frame["corrected_name_similarity"].ge(0.40)
            )
        )
    )
    frame["existing_link_corroborated"] = (
        frame["candidate_plausible"]
        & frame["current_place_slug"].fillna("").ne("")
        & frame["current_target_assignment_count"].eq(1)
        & (
            frame["existing_corrected_name_similarity"].ge(0.70)
            | (
                frame["existing_corrected_distance_m"].le(100)
                & frame["existing_corrected_name_similarity"].ge(0.50)
            )
        )
    )

    ranked = frame.sort_values(
        [
            "corrected_google_url",
            "candidate_plausible",
            "source_identity_match",
            "corrected_distance_m",
            "corrected_name_similarity",
        ],
        ascending=[True, False, False, True, False],
    )
    valid_url = ranked["corrected_google_url"].ne("")
    best_ids = set(
        ranked[valid_url].drop_duplicates("corrected_google_url")[STORE_ID]
    )
    frame["best_assignment_for_corrected_target"] = frame[STORE_ID].isin(best_ids)
    frame["corrected_target_assignment_count"] = frame.groupby(
        "corrected_google_url"
    )[STORE_ID].transform("nunique")

    frame["final_action"] = "manual_review"
    best = frame["best_assignment_for_corrected_target"]
    frame.loc[best & frame["existing_link_corroborated"], "final_action"] = (
        "keep_existing"
    )
    frame.loc[
        best
        & frame["candidate_plausible"]
        & ~frame["existing_link_corroborated"],
        "final_action",
    ] = "recrawl_corrected"
    frame.loc[
        frame["corrected_distance_m"].le(200)
        & frame["source_identity_match"]
        & ~frame["category_match"],
        "final_action",
    ] = "exclude_non_cafe"
    frame.loc[
        frame["candidate_plausible"] & ~best,
        "final_action",
    ] = "exclude_duplicate_target"

    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary = {
        "base_results": str(args.results),
        "override_results": [str(path.resolve()) for path in args.override_results],
        "reviews": str(args.reviews),
        "output": str(args.output),
        "places": int(len(frame)),
        "actions": {
            key: int(value)
            for key, value in frame["final_action"].value_counts().items()
        },
        "candidate_plausible": int(frame["candidate_plausible"].sum()),
        "existing_link_corroborated": int(
            frame["existing_link_corroborated"].sum()
        ),
        "category_match": int(frame["category_match"].sum()),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
