from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from clean_review_text_rows import classify_nontext_rows


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a conservative review dataset from verified place links."
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "google_reviews_full_corrected_textclean_v4.csv"
        ),
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "place_revalidation_classified_v5.csv",
    )
    parser.add_argument(
        "--recrawled",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "google_reviews_recrawled_corrected_v4.csv",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "google_reviews_recrawled_corrected_v4_progress.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "google_reviews_validated_v5.csv",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    configure_console_encoding()
    args = parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    classification = pd.read_csv(
        args.classification, encoding="utf-8-sig", dtype=str
    ).fillna("")
    store_column = classification.columns[0]
    keep_ids = set(
        classification.loc[
            classification["final_action"].eq("keep_existing"), store_column
        ]
    )
    recrawl_ids = set(
        classification.loc[
            classification["final_action"].eq("recrawl_corrected"), store_column
        ]
    )

    progress = pd.read_csv(args.progress, encoding="utf-8-sig", dtype=str).fillna("")
    if "checked_at" in progress.columns:
        progress["_checked_at"] = pd.to_datetime(
            progress["checked_at"], errors="coerce"
        )
        progress = progress.sort_values("_checked_at", kind="stable").drop(
            columns="_checked_at"
        )
    progress = progress.drop_duplicates(store_column, keep="last")
    completed_ids = set(
        progress.loc[progress["status"].eq("completed"), store_column]
    ) & recrawl_ids

    current_columns = list(
        pd.read_csv(args.current, encoding="utf-8-sig", nrows=0).columns
    )
    recrawled = pd.read_csv(args.recrawled, encoding="utf-8-sig", dtype=str)
    recrawled = recrawled[recrawled[store_column].isin(completed_ids)].copy()
    recrawled = recrawled.dropna(subset=["리뷰"])
    recrawled["리뷰"] = recrawled["리뷰"].astype(str).str.strip()
    recrawled = recrawled[recrawled["리뷰"].ne("")]
    dedupe_columns = [store_column]
    if "review_id" in recrawled.columns:
        dedupe_columns.append("review_id")
    else:
        dedupe_columns.extend(["작성자", "리뷰"])
    recrawled = recrawled.drop_duplicates(dedupe_columns, keep="last")
    replacement = recrawled.reindex(columns=current_columns)

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    current_rows = 0
    kept_current_rows = 0
    first = True
    for chunk in pd.read_csv(
        args.current,
        encoding="utf-8-sig",
        dtype=str,
        chunksize=args.chunksize,
        on_bad_lines="skip",
    ):
        current_rows += len(chunk)
        kept = chunk[chunk[store_column].isin(keep_ids)]
        kept_current_rows += len(kept)
        kept.to_csv(
            temporary,
            mode="w" if first else "a",
            header=first,
            index=False,
            encoding="utf-8-sig" if first else "utf-8",
        )
        first = False

    replacement.to_csv(
        temporary,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8",
    )
    os.replace(temporary, args.output)

    merged = pd.read_csv(args.output, encoding="utf-8-sig", dtype=str).fillna("")
    nontext_reason = classify_nontext_rows(merged)
    nontext_rows_removed = int(nontext_reason.ne("").sum())
    merged = merged.loc[nontext_reason.eq("")].copy()
    final_dedupe_columns = [store_column, "작성자", "리뷰"]
    duplicate_rows_removed = int(
        merged.duplicated(final_dedupe_columns, keep="last").sum()
    )
    merged = merged.drop_duplicates(final_dedupe_columns, keep="last")
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")

    output_rows = len(merged)
    output_places = int(merged[store_column].nunique())
    summary = {
        "current": str(args.current.resolve()),
        "classification": str(args.classification.resolve()),
        "recrawled": str(args.recrawled.resolve()),
        "progress": str(args.progress.resolve()),
        "output": str(args.output),
        "input_rows": current_rows,
        "kept_existing_places": len(keep_ids),
        "kept_existing_rows": kept_current_rows,
        "recrawl_target_places": len(recrawl_ids),
        "recrawl_completed_places": len(completed_ids),
        "recrawl_places_with_text": int(replacement[store_column].nunique()),
        "recrawled_text_rows": len(replacement),
        "nontext_rows_removed_after_merge": nontext_rows_removed,
        "duplicate_rows_removed_after_merge": duplicate_rows_removed,
        "output_places": output_places,
        "output_rows": output_rows,
        "excluded_actions": {
            key: int(value)
            for key, value in classification.loc[
                ~classification["final_action"].isin(
                    ["keep_existing", "recrawl_corrected"]
                ),
                "final_action",
            ].value_counts().items()
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
