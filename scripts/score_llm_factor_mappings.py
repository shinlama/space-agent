from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from modules.research_scoring import (
    DEFAULT_MAPPING_CSV,
    PROJECT_ROOT,
    calculate_scores,
    write_score_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score LLM placeness-factor mapping results with "
            "positive=1, neutral/mixed=0.5, negative=0."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_MAPPING_CSV,
        help="Input sentence-level LLM mapping CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "demo_outputs",
        help="Directory where scored CSV files will be written.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="llm",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored_evidence, factor_scores, place_scores = calculate_scores(args.input)
    paths = write_score_outputs(
        scored_evidence=scored_evidence,
        factor_scores=factor_scores,
        place_scores=place_scores,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )

    print(f"Input: {args.input}")
    print(f"Scored evidence rows: {len(scored_evidence):,}")
    print(f"Cafe-factor rows: {len(factor_scores):,}")
    print(f"Place rows: {len(place_scores):,}")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
