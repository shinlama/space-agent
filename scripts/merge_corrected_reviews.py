from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


DEFAULT_RAW = Path(r"F:\JW\space-agent\google_reviews_full_min3_max2000.csv")
DEFAULT_REVALIDATION = Path(
    r"F:\JW\space-agent\outputs\google_place_revalidation_http_v4.csv"
)
DEFAULT_RECRAWLED = Path(
    r"F:\JW\space-agent\outputs\google_reviews_recrawled_corrected_v4.csv"
)
DEFAULT_PROGRESS = Path(
    r"F:\JW\space-agent\outputs\google_reviews_recrawled_corrected_v4_progress.csv"
)
DEFAULT_OUTPUT = Path(r"F:\JW\space-agent\google_reviews_full_corrected_v4.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge high-confidence recrawled reviews without modifying the raw file."
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--revalidation", type=Path, default=DEFAULT_REVALIDATION)
    parser.add_argument("--recrawled", type=Path, default=DEFAULT_RECRAWLED)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


def latest_progress(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    store_col = frame.columns[0]
    if "checked_at" in frame.columns:
        frame["_checked_at"] = pd.to_datetime(frame["checked_at"], errors="coerce")
        frame = frame.sort_values("_checked_at", kind="stable").drop(
            columns="_checked_at"
        )
    return frame.drop_duplicates(store_col, keep="last")


def main() -> int:
    args = parse_args()
    revalidation = pd.read_csv(
        args.revalidation, encoding="utf-8-sig", dtype=str
    ).fillna("")
    store_col = revalidation.columns[0]
    recrawl_targets = set(
        revalidation.loc[
            revalidation["correction_action"].eq("recrawl_corrected"), store_col
        ].astype(str)
    )
    stale_targets = set(
        revalidation.loc[
            revalidation["correction_action"].eq("exclude_stale_or_replaced"),
            store_col,
        ].astype(str)
    )
    progress = latest_progress(args.progress)
    completed_ids = set(
        progress.loc[progress["status"].eq("completed"), store_col].astype(str)
    )
    failed_ids = recrawl_targets - completed_ids

    raw_columns = list(
        pd.read_csv(args.raw, encoding="utf-8-sig", nrows=0).columns
    )
    recrawled = pd.read_csv(args.recrawled, encoding="utf-8-sig", dtype=str)
    recrawled = recrawled[recrawled[store_col].astype(str).isin(completed_ids)].copy()
    recrawled = recrawled.drop_duplicates([store_col, "review_id"], keep="last")
    replacement = recrawled.reindex(columns=raw_columns)

    remove_ids = recrawl_targets | stale_targets
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    if temp_output.exists():
        temp_output.unlink()

    raw_rows = 0
    kept_rows = 0
    first = True
    for chunk in pd.read_csv(
        args.raw,
        encoding="utf-8-sig",
        dtype=str,
        chunksize=args.chunksize,
    ):
        raw_rows += len(chunk)
        kept = chunk[~chunk[store_col].astype(str).isin(remove_ids)]
        kept_rows += len(kept)
        kept.to_csv(
            temp_output,
            mode="w" if first else "a",
            header=first,
            index=False,
            encoding="utf-8-sig" if first else "utf-8",
        )
        first = False

    replacement.to_csv(
        temp_output,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8",
    )
    os.replace(temp_output, args.output)

    summary = {
        "raw_file": str(args.raw),
        "output_file": str(args.output),
        "raw_rows": raw_rows,
        "raw_rows_removed": raw_rows - kept_rows,
        "recrawl_target_places": len(recrawl_targets),
        "recrawl_completed_places": len(completed_ids & recrawl_targets),
        "recrawl_failed_places": len(failed_ids),
        "stale_or_replaced_places_removed": len(stale_targets),
        "recrawled_text_rows_added": len(replacement),
        "output_rows": kept_rows + len(replacement),
        "manual_review_places_retained": int(
            revalidation["correction_action"].eq("manual_review").sum()
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
