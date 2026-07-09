from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.research_scoring import DEFAULT_MAPPING_CSV, calculate_scores
from modules.validation_analysis import (
    build_all_factor_extreme_cases,
    build_cluster_feature_matrix,
    evaluate_cluster_range,
    representative_places_for_cluster,
    run_place_clustering,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate validation-analysis tables for the placeness quantification study."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_MAPPING_CSV,
        help="LLM factor mapping sentence CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "demo_outputs" / "validation_analysis",
        help="Directory to write validation CSV outputs.",
    )
    parser.add_argument("--min-total-mentions", type=int, default=10)
    parser.add_argument("--min-factor-mentions", type=int, default=3)
    parser.add_argument("--n-cases", type=int, default=5)
    parser.add_argument("--score-column", choices=["weighted_score", "factor_score"], default="weighted_score")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--n-clusters", type=int, default=None)
    return parser.parse_args()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote {path} ({len(df):,} rows)")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scored_evidence, factor_scores, place_scores = calculate_scores(args.input)

    extreme_cases = build_all_factor_extreme_cases(
        scored_evidence,
        factor_scores,
        place_scores,
        min_total_mentions=args.min_total_mentions,
        min_factor_mentions=args.min_factor_mentions,
        n_cases=args.n_cases,
    )
    write_csv(extreme_cases, args.output_dir / "factor_extreme_cases.csv")

    feature_matrix = build_cluster_feature_matrix(
        factor_scores,
        place_scores,
        score_column=args.score_column,
        min_total_mentions=args.min_total_mentions,
    )
    write_csv(feature_matrix.reset_index(), args.output_dir / "cluster_feature_matrix.csv")

    k_eval = evaluate_cluster_range(feature_matrix, k_min=args.k_min, k_max=args.k_max)
    write_csv(k_eval, args.output_dir / "cluster_k_evaluation.csv")

    if k_eval.empty:
        print("not enough places for clustering")
        return

    n_clusters = args.n_clusters
    if n_clusters is None:
        n_clusters = int(k_eval.sort_values("silhouette_score", ascending=False).iloc[0]["k"])

    cluster_result = run_place_clustering(feature_matrix, place_scores, n_clusters=n_clusters)
    write_csv(cluster_result.clustered_places, args.output_dir / "clustered_places.csv")
    write_csv(cluster_result.cluster_profiles, args.output_dir / "cluster_profiles.csv")
    write_csv(cluster_result.cluster_summary, args.output_dir / "cluster_summary.csv")

    representatives = []
    for cluster_id in cluster_result.cluster_summary["cluster"].tolist():
        representatives.append(
            representative_places_for_cluster(
                cluster_result,
                factor_scores,
                scored_evidence,
                cluster=int(cluster_id),
                limit=10,
            )
        )
    if representatives:
        write_csv(
            pd.concat(representatives, ignore_index=True),
            args.output_dir / "cluster_representative_places.csv",
        )


if __name__ == "__main__":
    main()
