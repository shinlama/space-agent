from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from modules.research_scoring import FACTOR_ORDER


@dataclass(frozen=True)
class ClusterResult:
    clustered_places: pd.DataFrame
    cluster_profiles: pd.DataFrame
    cluster_summary: pd.DataFrame


def _join_examples(values: pd.Series, limit: int = 3) -> str:
    examples: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text.lower() != "nan" and text not in examples:
            examples.append(text)
        if len(examples) >= limit:
            break
    return " | ".join(examples)


def _representative_evidence(
    scored_evidence: pd.DataFrame,
    cafe_name: str,
    factor: str,
    prefer_positive: bool,
    limit: int = 3,
) -> str:
    rows = scored_evidence[
        (scored_evidence["cafe_name"] == cafe_name)
        & (scored_evidence["factor"] == factor)
    ].copy()
    if rows.empty:
        return ""

    rows["_evidence_length"] = rows["evidence"].fillna("").astype(str).str.len()
    sort_columns = ["sentiment_value"]
    ascending = [not prefer_positive]
    if "confidence" in rows.columns and rows["confidence"].notna().any():
        sort_columns.append("confidence")
        ascending.append(False)
    sort_columns.append("_evidence_length")
    ascending.append(False)
    rows = rows.sort_values(sort_columns, ascending=ascending)
    return _join_examples(rows["evidence"], limit=limit)


def get_factor_extreme_cases(
    scored_evidence: pd.DataFrame,
    factor_scores: pd.DataFrame,
    place_scores: pd.DataFrame,
    factor: str,
    min_total_mentions: int = 10,
    min_factor_mentions: int = 3,
    n_cases: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top and bottom places for a factor using the factor score."""
    if factor_scores.empty:
        return pd.DataFrame(), pd.DataFrame()

    place_meta = place_scores[
        [
            "cafe_name",
            "placeness_score",
            "mean_factor_score",
            "mapped_evidence_count",
            "mentioned_factor_count",
        ]
    ].copy()
    rows = (
        factor_scores[factor_scores["factor"] == factor]
        .merge(place_meta, on="cafe_name", how="left")
        .copy()
    )
    rows = rows[
        (rows["mention_count"] >= min_factor_mentions)
        & (rows["mapped_evidence_count"] >= min_total_mentions)
    ].copy()
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    sort_cols = ["factor_score", "mention_count", "weighted_score", "mapped_evidence_count"]
    top = rows.sort_values(sort_cols, ascending=[False, False, False, False]).head(n_cases).copy()
    bottom = rows.sort_values(sort_cols, ascending=[True, False, True, False]).head(n_cases).copy()

    top["case_type"] = "상위"
    bottom["case_type"] = "하위"
    top["representative_evidence"] = top["cafe_name"].apply(
        lambda name: _representative_evidence(scored_evidence, name, factor, prefer_positive=True)
    )
    bottom["representative_evidence"] = bottom["cafe_name"].apply(
        lambda name: _representative_evidence(scored_evidence, name, factor, prefer_positive=False)
    )
    return top.reset_index(drop=True), bottom.reset_index(drop=True)


def build_all_factor_extreme_cases(
    scored_evidence: pd.DataFrame,
    factor_scores: pd.DataFrame,
    place_scores: pd.DataFrame,
    min_total_mentions: int = 10,
    min_factor_mentions: int = 3,
    n_cases: int = 5,
) -> pd.DataFrame:
    cases: list[pd.DataFrame] = []
    for factor in FACTOR_ORDER:
        top, bottom = get_factor_extreme_cases(
            scored_evidence,
            factor_scores,
            place_scores,
            factor=factor,
            min_total_mentions=min_total_mentions,
            min_factor_mentions=min_factor_mentions,
            n_cases=n_cases,
        )
        if not top.empty:
            cases.append(top)
        if not bottom.empty:
            cases.append(bottom)
    if not cases:
        return pd.DataFrame()
    return pd.concat(cases, ignore_index=True)


def build_cluster_feature_matrix(
    factor_scores: pd.DataFrame,
    place_scores: pd.DataFrame,
    score_column: str = "weighted_score",
    min_total_mentions: int = 10,
) -> pd.DataFrame:
    if score_column not in factor_scores.columns:
        raise ValueError(f"Unknown score column: {score_column}")

    valid_places = place_scores[
        place_scores["mapped_evidence_count"] >= min_total_mentions
    ]["cafe_name"]
    matrix = factor_scores.pivot_table(
        index="cafe_name",
        columns="factor",
        values=score_column,
        aggfunc="mean",
        fill_value=0.0,
    )
    matrix = matrix.reindex(columns=FACTOR_ORDER, fill_value=0.0)
    matrix = matrix.loc[matrix.index.isin(valid_places)].copy()
    matrix = matrix.sort_index()
    return matrix


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return (values - mean) / std


def _pairwise_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = values[:, None, :] - centers[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _fit_kmeans(
    values: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = np.inf
    n_samples = values.shape[0]

    for _ in range(n_init):
        initial_idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centers = values[initial_idx].copy()
        labels = np.full(n_samples, -1, dtype=int)

        for _iteration in range(max_iter):
            distances = _pairwise_distances(values, centers)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            new_centers = centers.copy()
            for cluster_id in range(n_clusters):
                members = values[labels == cluster_id]
                if len(members):
                    new_centers[cluster_id] = members.mean(axis=0)
                else:
                    farthest_idx = np.argmax(distances.min(axis=1))
                    new_centers[cluster_id] = values[farthest_idx]
            centers = new_centers

        distances = _pairwise_distances(values, centers)
        inertia = float(np.sum(np.min(distances, axis=1) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    if best_labels is None or best_centers is None:
        raise RuntimeError("K-means clustering failed to converge.")
    return best_labels, best_centers, best_inertia


def _silhouette_score(values: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(values):
        return float("nan")

    distances = _pairwise_distances(values, values)
    scores: list[float] = []
    for idx, label in enumerate(labels):
        same_mask = labels == label
        same_mask[idx] = False
        if same_mask.any():
            a_score = float(distances[idx, same_mask].mean())
        else:
            a_score = 0.0

        b_score = np.inf
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_mask = labels == other_label
            if other_mask.any():
                b_score = min(b_score, float(distances[idx, other_mask].mean()))

        denominator = max(a_score, b_score)
        if denominator == 0 or not np.isfinite(denominator):
            scores.append(0.0)
        else:
            scores.append((b_score - a_score) / denominator)
    return float(np.mean(scores))


def _pca_2d(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0)
    try:
        u_matrix, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((len(values), 2))

    components = u_matrix[:, :2] * singular_values[:2]
    if components.shape[1] == 1:
        components = np.column_stack([components[:, 0], np.zeros(len(values))])
    return components[:, :2]


def evaluate_cluster_range(
    feature_matrix: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    if len(feature_matrix) < 3:
        return pd.DataFrame(columns=["k", "silhouette_score", "inertia"])

    max_k = min(k_max, len(feature_matrix) - 1)
    if max_k < k_min:
        return pd.DataFrame(columns=["k", "silhouette_score", "inertia"])

    x_scaled = _standardize(feature_matrix.values.astype(float))
    rows: list[dict[str, float | int]] = []
    for k in range(k_min, max_k + 1):
        labels, _centers, inertia = _fit_kmeans(x_scaled, k, random_state=random_state, n_init=20)
        rows.append(
            {
                "k": k,
                "silhouette_score": _silhouette_score(x_scaled, labels),
                "inertia": inertia,
            }
        )
    return pd.DataFrame(rows)


def _cluster_name(profile_row: pd.Series) -> str:
    positive = profile_row.sort_values(ascending=False).head(2)
    negative = profile_row.sort_values(ascending=True).head(1)
    if positive.iloc[0] <= 0:
        return f"{negative.index[0]} 취약형"
    if negative.iloc[0] < -0.02:
        return f"{positive.index[0]}·{positive.index[1]} 중심 / {negative.index[0]} 취약형"
    return f"{positive.index[0]}·{positive.index[1]} 중심형"


def run_place_clustering(
    feature_matrix: pd.DataFrame,
    place_scores: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> ClusterResult:
    if feature_matrix.empty:
        empty = pd.DataFrame()
        return ClusterResult(empty, empty, empty)
    if n_clusters < 2 or n_clusters >= len(feature_matrix):
        raise ValueError("n_clusters must be between 2 and the number of places - 1.")

    x_scaled = _standardize(feature_matrix.values.astype(float))
    labels, centers, _inertia = _fit_kmeans(x_scaled, n_clusters, random_state=random_state, n_init=20)
    distances = np.linalg.norm(x_scaled - centers[labels], axis=1)

    clustered = pd.DataFrame(
        {
            "cafe_name": feature_matrix.index,
            "cluster": labels.astype(int),
            "distance_to_center": distances.astype(float),
        }
    )

    if feature_matrix.shape[1] >= 2 and len(feature_matrix) >= 3:
        coords = _pca_2d(x_scaled)
        clustered["pc1"] = coords[:, 0]
        clustered["pc2"] = coords[:, 1]
    else:
        clustered["pc1"] = 0.0
        clustered["pc2"] = 0.0

    clustered = clustered.merge(place_scores, on="cafe_name", how="left")
    profile_source = feature_matrix.copy()
    profile_source["cluster"] = labels
    profiles = profile_source.groupby("cluster")[FACTOR_ORDER].mean().reset_index()
    names = {
        int(row["cluster"]): _cluster_name(row[FACTOR_ORDER])
        for _, row in profiles.iterrows()
    }
    clustered["cluster_name"] = clustered["cluster"].map(names)
    profiles["cluster_name"] = profiles["cluster"].map(names)
    clustered["cluster_label"] = clustered.apply(
        lambda row: f"{int(row['cluster'])}. {row['cluster_name']}",
        axis=1,
    )
    profiles["cluster_label"] = profiles.apply(
        lambda row: f"{int(row['cluster'])}. {row['cluster_name']}",
        axis=1,
    )

    summary_rows: list[dict[str, object]] = []
    for _, profile_row in profiles.iterrows():
        cluster_id = int(profile_row["cluster"])
        cluster_places = clustered[clustered["cluster"] == cluster_id]
        ordered = profile_row[FACTOR_ORDER].sort_values(ascending=False)
        low_ordered = profile_row[FACTOR_ORDER].sort_values(ascending=True)
        summary_rows.append(
            {
                "cluster": cluster_id,
                "cluster_name": names[cluster_id],
                "cluster_label": f"{cluster_id}. {names[cluster_id]}",
                "place_count": int(len(cluster_places)),
                "avg_placeness_score": float(cluster_places["placeness_score"].mean()),
                "avg_mapped_evidence_count": float(cluster_places["mapped_evidence_count"].mean()),
                "top_factors": ", ".join(ordered.head(3).index.tolist()),
                "low_factors": ", ".join(low_ordered.head(2).index.tolist()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("cluster").reset_index(drop=True)
    return ClusterResult(
        clustered_places=clustered.sort_values(["cluster", "distance_to_center"]).reset_index(drop=True),
        cluster_profiles=profiles.sort_values("cluster").reset_index(drop=True),
        cluster_summary=summary,
    )


def representative_places_for_cluster(
    cluster_result: ClusterResult,
    factor_scores: pd.DataFrame,
    scored_evidence: pd.DataFrame,
    cluster: int,
    limit: int = 10,
) -> pd.DataFrame:
    places = (
        cluster_result.clustered_places[cluster_result.clustered_places["cluster"] == cluster]
        .sort_values("distance_to_center")
        .head(limit)
        .copy()
    )
    if places.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for place in places.itertuples():
        scores = factor_scores[factor_scores["cafe_name"] == place.cafe_name].copy()
        if scores.empty:
            dominant_factor = ""
            dominant_score = np.nan
            evidence = ""
        else:
            dominant = scores.sort_values(
                ["weighted_score", "mention_count"],
                ascending=[False, False],
            ).iloc[0]
            dominant_factor = str(dominant["factor"])
            dominant_score = float(dominant["weighted_score"])
            evidence = _representative_evidence(
                scored_evidence,
                str(place.cafe_name),
                dominant_factor,
                prefer_positive=dominant_score >= 0,
                limit=2,
            )
        rows.append(
            {
                "cluster": int(place.cluster),
                "cluster_name": place.cluster_name,
                "cafe_name": place.cafe_name,
                "distance_to_center": float(place.distance_to_center),
                "placeness_score": float(place.placeness_score),
                "mapped_evidence_count": int(place.mapped_evidence_count),
                "dominant_factor": dominant_factor,
                "dominant_weighted_score": dominant_score,
                "representative_evidence": evidence,
            }
        )
    return pd.DataFrame(rows)
