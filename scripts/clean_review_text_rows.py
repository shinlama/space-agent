from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(r"F:\JW\space-agent\google_reviews_full_corrected_v4.csv")
DEFAULT_OUTPUT = Path(
    r"F:\JW\space-agent\google_reviews_full_corrected_textclean_v4.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove Google Maps profile/card metadata stored as review text."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


def classify_nontext_rows(frame: pd.DataFrame) -> pd.Series:
    author = frame["작성자"].fillna("").astype(str).str.strip()
    review = frame["리뷰"].fillna("").astype(str).str.strip()

    reason = pd.Series("", index=frame.index, dtype="object")
    blank = review.eq("")
    reason.loc[blank] = "blank_review"

    # Google Maps icon glyphs are private-use Unicode characters. They occur in
    # full review-card text, but not in the actual review body element.
    private_use_icon = review.str.contains(r"[\ue000-\uf8ff]", regex=True)
    reason.loc[reason.eq("") & private_use_icon] = "review_card_metadata"

    author_only = author.ne("") & review.eq(author)
    reason.loc[reason.eq("") & author_only] = "author_only"

    starts_with_author = pd.Series(
        (bool(name) and text.startswith(name) for name, text in zip(author, review)),
        index=frame.index,
        dtype=bool,
    )
    profile_marker = review.str.contains(
        r"지역\s*가이드|리뷰\s*[\d,]+개|사진\s*[\d,]+장",
        regex=True,
    )
    reason.loc[
        reason.eq("") & starts_with_author & profile_marker
    ] = "author_profile_metadata"

    metadata_only = review.str.fullmatch(
        r"(?:지역\s*가이드|리뷰\s*[\d,]+개|사진\s*[\d,]+장|[·.\s])+",
        na=False,
    )
    reason.loc[reason.eq("") & metadata_only] = "profile_metadata_only"
    return reason


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    if temp_output.exists():
        temp_output.unlink()

    input_rows = 0
    output_rows = 0
    removed_by_reason: dict[str, int] = {}
    first = True
    for chunk in pd.read_csv(
        args.input,
        encoding="utf-8-sig",
        dtype=str,
        chunksize=args.chunksize,
    ):
        reasons = classify_nontext_rows(chunk)
        keep = reasons.eq("")
        cleaned = chunk.loc[keep]
        input_rows += len(chunk)
        output_rows += len(cleaned)
        for reason, count in reasons[~keep].value_counts().items():
            removed_by_reason[str(reason)] = removed_by_reason.get(str(reason), 0) + int(
                count
            )
        cleaned.to_csv(
            temp_output,
            mode="w" if first else "a",
            header=first,
            index=False,
            encoding="utf-8-sig" if first else "utf-8",
        )
        first = False

    os.replace(temp_output, args.output)
    summary = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "input_rows": input_rows,
        "removed_rows": input_rows - output_rows,
        "output_rows": output_rows,
        "removed_by_reason": removed_by_reason,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
