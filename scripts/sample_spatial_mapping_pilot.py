from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from filter_spatial_review_candidates import FACTOR_CUES, match_factor_cues


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a factor-balanced pilot sample for contextual mapping."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "google_reviews_validated_v5.csv",
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=None,
        help=(
            "Optional place classification. If supplied, only keep_existing rows "
            "are sampled; useful before the final merged dataset is ready."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "spatial_mapping_pilot100_v5.csv",
    )
    parser.add_argument("--per-factor", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260721)
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

    keep_ids: set[str] | None = None
    store_column = "상가업소번호"
    if args.classification is not None:
        classification = pd.read_csv(
            args.classification, encoding="utf-8-sig", dtype=str
        ).fillna("")
        keep_ids = set(
            classification.loc[
                classification["final_action"].eq("keep_existing"), store_column
            ]
        )

    factor_pools: dict[str, list[pd.Series]] = {factor: [] for factor in FACTOR_CUES}
    max_pool = max(100, args.per_factor * 20)
    random_state = args.seed
    input_rows = 0
    eligible_rows = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            args.input,
            encoding="utf-8-sig",
            dtype=str,
            chunksize=args.chunksize,
            on_bad_lines="skip",
        )
    ):
        input_rows += len(chunk)
        if keep_ids is not None:
            chunk = chunk[chunk[store_column].isin(keep_ids)]
        if chunk.empty:
            continue
        review_text = chunk["리뷰"].fillna("").astype(str).str.strip()
        chunk = chunk[review_text.str.len().ge(4)].copy()
        if chunk.empty:
            continue
        eligible_rows += len(chunk)

        # Sampling before cue expansion keeps memory bounded while retaining a
        # broad pool from every input chunk.
        sampled = chunk.sample(
            n=min(len(chunk), max_pool * 4),
            random_state=random_state + chunk_no,
        )
        for _, row in sampled.iterrows():
            factors, terms = match_factor_cues(str(row["리뷰"]))
            if not factors:
                continue
            enriched = row.copy()
            enriched["prefilter_factors"] = "|".join(factors)
            enriched["prefilter_terms"] = "|".join(terms)
            for factor in factors:
                pool = factor_pools[factor]
                pool.append(enriched)
                if len(pool) > max_pool:
                    # Deterministic bounded reservoir replacement.
                    pool.pop((chunk_no + len(pool) + random_state) % len(pool))

    selected: list[pd.Series] = []
    used_indices: set[tuple[str, str]] = set()
    used_places: set[str] = set()
    factor_selected: dict[str, int] = {}
    for factor_no, factor in enumerate(FACTOR_CUES):
        pool_df = pd.DataFrame(factor_pools[factor])
        if pool_df.empty:
            factor_selected[factor] = 0
            continue
        pool_df = pool_df.sample(frac=1, random_state=args.seed + factor_no)
        count = 0
        for _, row in pool_df.iterrows():
            place_id = str(row.get(store_column, ""))
            identity = (place_id, str(row.get("리뷰", "")))
            if identity in used_indices or place_id in used_places:
                continue
            selected.append(row)
            used_indices.add(identity)
            used_places.add(place_id)
            count += 1
            if count >= args.per_factor:
                break
        factor_selected[factor] = count

    pilot = pd.DataFrame(selected).reset_index(drop=True)
    pilot.insert(0, "pilot_review_index", range(len(pilot)))
    pilot.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary = {
        "input": str(args.input.resolve()),
        "classification": (
            str(args.classification.resolve()) if args.classification else None
        ),
        "output": str(args.output),
        "input_rows": input_rows,
        "eligible_rows": eligible_rows,
        "pilot_rows": len(pilot),
        "pilot_places": pilot[store_column].nunique() if not pilot.empty else 0,
        "target_per_factor": args.per_factor,
        "factor_selected": factor_selected,
        "selection_note": (
            "prefilter factors balance the pilot only; they are not ground-truth labels"
        ),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
