from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAPPING_CSV = PROJECT_ROOT / "llm_factor_mapping_sentences_full.csv"

FACTOR_CATEGORIES: dict[str, list[str]] = {
    "물리적 특성": ["심미성", "개방성", "감각적 경험", "접근성", "쾌적성"],
    "활동적 특성": ["활동성", "상호작용성"],
    "의미적 특성": ["상징성", "기억 및 선호", "지역 정체성"],
}

FACTOR_ORDER = [
    factor
    for factors in FACTOR_CATEGORIES.values()
    for factor in factors
]

FACTOR_CATEGORY = {
    factor: category
    for category, factors in FACTOR_CATEGORIES.items()
    for factor in factors
}

FACTOR_DETAILS: dict[str, dict[str, object]] = {
    "심미성": {
        "english": "Aesthetics",
        "definition": "공간의 인테리어, 디자인, 마감, 색채, 분위기에서 느껴지는 시각적 아름다움과 미적 품질",
        "criteria": ["인테리어 디자인", "시각적 아름다움", "분위기 연출", "사진 찍기 좋은 공간"],
        "examples": ["인테리어가 감성적이다", "공간이 예쁘다", "분위기가 세련됐다"],
    },
    "개방성": {
        "english": "Openness",
        "definition": "공간 배치와 시야 구성이 확장감, 시각적 연결성, 낮은 폐쇄감을 제공하는 정도",
        "criteria": ["공간 배치", "시야 연속성", "채광", "외부 조망", "좌석 간격"],
        "examples": ["공간이 넓고 답답하지 않다", "창이 커서 개방감이 좋다", "좌석 간격이 넓다"],
    },
    "감각적 경험": {
        "english": "Sensory",
        "definition": "조명, 음악, 향, 질감 등 오감을 자극하는 공간 경험의 정도",
        "criteria": ["조명 환경", "음향 환경", "향 요소", "가구 및 소품 질감"],
        "examples": ["커피 향이 좋다", "음악과 분위기가 잘 어울린다", "조명이 감각적이다"],
    },
    "접근성": {
        "english": "Accessibility",
        "definition": "대중교통, 주차, 보행, 위치 조건 등 공간에 쉽게 도달하고 이용할 수 있는 정도",
        "criteria": ["대중교통 접근성", "보행 접근성", "주차 편의성", "찾기 쉬운 위치"],
        "examples": ["역에서 가깝다", "주차가 편하다", "찾아가기 어렵다"],
    },
    "쾌적성": {
        "english": "Comfort",
        "definition": "청결, 온습도, 공기, 채광, 혼잡도 등 머무르는 동안 느끼는 물리적 편안함",
        "criteria": ["청결도", "온열 환경", "공기 및 환기", "채광", "관리 상태"],
        "examples": ["실내가 깨끗하다", "오래 있어도 편하다", "공기가 쾌적하다"],
    },
    "활동성": {
        "english": "Activity",
        "definition": "휴식, 업무, 대화, 모임, 식사 등 다양한 행위를 수용하는 정도",
        "criteria": ["행위 수용성", "공간 활용도", "체류 형태 다양성", "좌석 구성"],
        "examples": ["작업하기 좋다", "모임하기 좋다", "쉬어가기 좋다"],
    },
    "상호작용성": {
        "english": "Sociability",
        "definition": "이용자 간 소통, 직원과의 교류, 커뮤니티 또는 프로그램 참여가 일어나는 정도",
        "criteria": ["사회적 교류", "직원 응대", "체험 프로그램", "커뮤니티 형성"],
        "examples": ["직원이 친절하다", "대화하기 편한 분위기다", "프로그램에 참여할 수 있다"],
    },
    "상징성": {
        "english": "Symbolism",
        "definition": "다른 장소와 구별되는 콘셉트, 브랜딩, 디자인 개성으로 형성되는 장소의 정체성",
        "criteria": ["브랜드 이미지", "디자인 개성", "차별화된 콘셉트", "상징적 스타일"],
        "examples": ["이곳만의 개성이 뚜렷하다", "콘셉트가 확실하다", "독특한 카페다"],
    },
    "기억 및 선호": {
        "english": "Preference",
        "definition": "감성적 경험, 취향 적합성, 재방문 의향, 추천 의향으로 형성되는 선호와 기억",
        "criteria": ["정서적 만족감", "취향 적합성", "기억 형성", "재방문 의향"],
        "examples": ["다시 방문하고 싶다", "취향에 맞다", "기억에 남는다"],
    },
    "지역 정체성": {
        "english": "Local Identity",
        "definition": "지역의 역사, 문화, 경관, 동네 맥락이 공간 경험에 반영되는 정도",
        "criteria": ["지역 맥락 반영", "역사성", "지역 자원 활용", "주변 환경과의 조화"],
        "examples": ["동네 분위기가 느껴진다", "지역 특색이 반영됐다", "주변 환경과 잘 어울린다"],
    },
}

SENTIMENT_VALUES = {
    "positive": 1.0,
    "neutral": 0.0,
    "mixed": 0.0,
    "negative": -1.0,
}

SENTIMENT_LABELS = {
    "positive": "긍정",
    "neutral": "중립",
    "mixed": "혼합",
    "negative": "부정",
}


def read_csv_safely(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949", **kwargs)


def load_mapping_sentences(path: str | Path = DEFAULT_MAPPING_CSV) -> pd.DataFrame:
    df = read_csv_safely(path)
    required = {
        "review_index",
        "cafe_name",
        "review_text",
        "sentence_id",
        "sentence",
        "factor",
        "confidence",
        "evidence",
        "reason",
        "sentiment_hint",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Required columns are missing from {path}: {missing}")

    df = df.copy()
    df["factor"] = df["factor"].astype(str).str.strip()
    df = df[df["factor"].isin(FACTOR_ORDER)].copy()
    df["review_index"] = pd.to_numeric(df["review_index"], errors="coerce").astype("Int64")
    df["sentence_id"] = pd.to_numeric(df["sentence_id"], errors="coerce").astype("Int64")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df.reset_index(drop=True)


def score_mapped_evidence(mapping_df: pd.DataFrame) -> pd.DataFrame:
    scored = mapping_df.copy()
    scored["sentiment_key"] = (
        scored["sentiment_hint"]
        .fillna("neutral")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    scored.loc[~scored["sentiment_key"].isin(SENTIMENT_VALUES), "sentiment_key"] = "neutral"
    scored["sentiment_label"] = scored["sentiment_key"].map(SENTIMENT_LABELS)
    scored["sentiment_value"] = scored["sentiment_key"].map(SENTIMENT_VALUES).astype(float)
    scored["factor_category"] = scored["factor"].map(FACTOR_CATEGORY)
    scored["evidence"] = scored["evidence"].fillna(scored["sentence"])
    scored["evidence"] = scored["evidence"].fillna("").astype(str).str.strip()
    scored["sentence"] = scored["sentence"].fillna("").astype(str).str.strip()
    scored["reason"] = scored["reason"].fillna("").astype(str).str.strip()
    scored["review_text"] = scored["review_text"].fillna("").astype(str).str.strip()
    scored["cafe_name"] = scored["cafe_name"].fillna("이름 없는 장소").astype(str).str.strip()
    return scored


def _count_sentiment(series: pd.Series, key: str) -> int:
    return int((series == key).sum())


def compute_factor_scores(scored_evidence: pd.DataFrame) -> pd.DataFrame:
    if scored_evidence.empty:
        return pd.DataFrame()

    grouped = scored_evidence.groupby(["cafe_name", "factor"], dropna=False)
    factor_scores = grouped.agg(
        factor_category=("factor_category", "first"),
        mention_count=("sentiment_value", "size"),
        factor_score=("sentiment_value", "mean"),
        avg_confidence=("confidence", "mean"),
        positive_count=("sentiment_key", lambda s: _count_sentiment(s, "positive")),
        neutral_count=("sentiment_key", lambda s: _count_sentiment(s, "neutral")),
        mixed_count=("sentiment_key", lambda s: _count_sentiment(s, "mixed")),
        negative_count=("sentiment_key", lambda s: _count_sentiment(s, "negative")),
        evidence_examples=("evidence", lambda s: " | ".join(_unique_nonempty(s, limit=4))),
    ).reset_index()

    total_mentions = (
        factor_scores.groupby("cafe_name")["mention_count"]
        .sum()
        .rename("total_mention_count")
        .reset_index()
    )
    factor_scores = factor_scores.merge(total_mentions, on="cafe_name", how="left")
    factor_scores["mention_share"] = (
        factor_scores["mention_count"] / factor_scores["total_mention_count"]
    )
    factor_scores["weighted_score"] = (
        factor_scores["factor_score"] * factor_scores["mention_share"]
    )
    factor_scores["factor_order"] = factor_scores["factor"].map({factor: i for i, factor in enumerate(FACTOR_ORDER)})
    return factor_scores.sort_values(["cafe_name", "factor_order"]).reset_index(drop=True)


def compute_place_scores(factor_scores: pd.DataFrame) -> pd.DataFrame:
    if factor_scores.empty:
        return pd.DataFrame()

    place_scores = factor_scores.groupby("cafe_name", dropna=False).agg(
        placeness_score=("weighted_score", "sum"),
        mapped_evidence_count=("mention_count", "sum"),
        mentioned_factor_count=("factor", "nunique"),
        mean_factor_score=("factor_score", "mean"),
        positive_count=("positive_count", "sum"),
        neutral_count=("neutral_count", "sum"),
        mixed_count=("mixed_count", "sum"),
        negative_count=("negative_count", "sum"),
        avg_confidence=("avg_confidence", "mean"),
    ).reset_index()
    place_scores["positive_ratio"] = place_scores["positive_count"] / place_scores["mapped_evidence_count"]
    place_scores["negative_ratio"] = place_scores["negative_count"] / place_scores["mapped_evidence_count"]

    top_factors = (
        factor_scores.sort_values(["cafe_name", "mention_count", "factor_score"], ascending=[True, False, False])
        .groupby("cafe_name")["factor"]
        .first()
        .rename("most_mentioned_factor")
        .reset_index()
    )
    place_scores = place_scores.merge(top_factors, on="cafe_name", how="left")
    return place_scores.sort_values("placeness_score", ascending=False).reset_index(drop=True)


def complete_factor_table(factor_scores: pd.DataFrame, cafe_name: str) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "factor": FACTOR_ORDER,
            "factor_category": [FACTOR_CATEGORY[factor] for factor in FACTOR_ORDER],
            "factor_order": list(range(len(FACTOR_ORDER))),
        }
    )
    selected = factor_scores[factor_scores["cafe_name"] == cafe_name].copy()
    table = base.merge(
        selected.drop(columns=["factor_category", "factor_order"], errors="ignore"),
        on="factor",
        how="left",
    )
    count_cols = [
        "mention_count",
        "positive_count",
        "neutral_count",
        "mixed_count",
        "negative_count",
        "total_mention_count",
    ]
    for col in count_cols:
        if col in table:
            table[col] = table[col].fillna(0).astype(int)
    for col in ["factor_score", "avg_confidence", "mention_share", "weighted_score"]:
        if col in table:
            table[col] = table[col].astype(float)
    table["cafe_name"] = cafe_name
    return table.sort_values("factor_order").reset_index(drop=True)


def calculate_scores(
    input_csv: str | Path = DEFAULT_MAPPING_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping_df = load_mapping_sentences(input_csv)
    scored_evidence = score_mapped_evidence(mapping_df)
    factor_scores = compute_factor_scores(scored_evidence)
    place_scores = compute_place_scores(factor_scores)
    return scored_evidence, factor_scores, place_scores


def write_score_outputs(
    scored_evidence: pd.DataFrame,
    factor_scores: pd.DataFrame,
    place_scores: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "llm",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "scored_evidence": output_dir / f"{prefix}_scored_evidence.csv",
        "factor_scores": output_dir / f"{prefix}_factor_scores_by_cafe.csv",
        "place_scores": output_dir / f"{prefix}_placeness_scores_by_cafe.csv",
    }
    scored_evidence.to_csv(paths["scored_evidence"], index=False, encoding="utf-8-sig")
    factor_scores.to_csv(paths["factor_scores"], index=False, encoding="utf-8-sig")
    place_scores.to_csv(paths["place_scores"], index=False, encoding="utf-8-sig")
    return paths


def _unique_nonempty(values: Iterable[object], limit: int = 4) -> list[str]:
    seen: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text.lower() != "nan" and text not in seen:
            seen.append(text)
        if len(seen) >= limit:
            break
    return seen
