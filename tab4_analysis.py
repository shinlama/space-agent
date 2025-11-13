from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster

ALPHA = 0.75
BETA = 0.25

KEYWORD_BOOSTS: Dict[str, List[str]] = {
    "고유성": ["독특", "유니크", "차별", "컨셉", "테마", "특색", "개성", "특별한", "아이덴티티", "유일", "독창"],
    "문화적 맥락": ["전통", "역사", "년", "오래", "옛", "고풍", "문화", "배경", "스토리", "세월", "내력", "유서", "레트로", "빈티지", "클래식", "앤티크", "과거", "옛날"],
    "지역 정체성": ["지역", "동네", "마을", "근처", "주변", "명소", "랜드마크", "상징", "대표", "신촌", "홍대", "강남", "이태원", "연남", "성수", "을지로", "익선동", "북촌", "삼청동", "종로", "명동"],
    "기억/경험": ["추억", "감동", "인상", "특별", "잊을 수", "기억", "회상", "경험", "느낌"],
    "심미성": ["예쁘", "아름", "멋지", "세련", "야경", "뷰", "인테리어", "디자인", "조명", "아늑", "분위기", "감성"],
    "감각적 경험": ["음악", "향", "냄새", "질감", "맛", "오감", "감각", "소리", "촉감"],
    "쾌적성": ["청결", "깨끗", "밝", "통풍", "화장실", "위생", "정돈", "쾌적"],
    "접근성": ["가깝", "접근", "역", "정류장", "도보", "편리"],
    "활동성": ["대화", "업무", "작업", "회의", "공부", "활동", "모임", "스터디"],
    "사회성": ["친절", "서비스", "교류", "소통", "친근", "인사", "배려"],
    "형태성": ["넓", "공간", "구조", "배치", "개방", "동선", "층", "룸"],
}

ACCESSIBILITY_PATTERNS: Sequence[re.Pattern] = [
    re.compile(r"도보\s*(?:로\s*)?(\d+)\s*분", re.IGNORECASE),
    re.compile(r"(\d+)\s*분\s*도보", re.IGNORECASE),
    re.compile(r"도보\s*로\s*(\d+)\s*분\s*(?:이면|만에|걸림|걸려|가능)", re.IGNORECASE),
    re.compile(r"(\d+)\s*분\s*거리", re.IGNORECASE),
    re.compile(r"(?:지하철역|역|버스정류장|정류장)(?:에서|까지)\s*(\d+)\s*분", re.IGNORECASE),
    re.compile(r"(\d+)\s*분\s*(?:이면|만에|걸림|걸려)\s*(?:갈\s*수|도착|가능)", re.IGNORECASE),
]


def _clone_score_template(template: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Dict[str, Optional[float]]]:
    """Deep copy score template to avoid mutation."""
    return json.loads(json.dumps(template))


def _apply_keyword_boosts(review_text: str, factor_sentiments: Dict[str, List[float]]) -> None:
    for factor, keywords in KEYWORD_BOOSTS.items():
        matched = [kw for kw in keywords if kw in review_text]
        if matched:
            boosted_score = min(0.75 + 0.05 * len(matched), 0.95)
            for _ in range(3):
                factor_sentiments[factor].append(boosted_score)


def _apply_accessibility_boost(review_text: str, factor_sentiments: Dict[str, List[float]]) -> None:
    best_score = 0.0
    for pattern in ACCESSIBILITY_PATTERNS:
        for match in pattern.finditer(review_text):
            try:
                minutes = int(match.group(1))
            except (ValueError, IndexError):
                continue
            if minutes <= 5:
                boost = 0.95
            elif minutes <= 10:
                boost = 0.90
            elif minutes <= 15:
                boost = 0.85
            else:
                boost = 0.80
            best_score = max(best_score, boost)
    if best_score > 0:
        for _ in range(5):
            factor_sentiments["접근성"].append(best_score)


def _compute_factor_sentiments(
    review_units: List[str],
    sentiment_model: Callable[[List[str]], List[float]],
    embed_model,
    category_embeddings: Dict[str, np.ndarray],
) -> Tuple[Dict[str, List[float]], List[float]]:
    factor_sentiments: Dict[str, List[float]] = {key: [] for key in category_embeddings.keys()}
    if not review_units:
        return factor_sentiments, []

    sentiment_scores = sentiment_model(review_units)
    unit_embs = embed_model.encode(review_units, normalize_embeddings=True)
    subcat_list = list(category_embeddings.keys())
    factor_mat = np.stack([category_embeddings[name] for name in subcat_list], axis=0)
    sim_mat = np.matmul(unit_embs, factor_mat.T)

    for i, unit in enumerate(review_units):
        raw_sent = float(sentiment_scores[i]) if i < len(sentiment_scores) else 0.5
        sent_adj = np.clip((raw_sent - 0.2) / 0.6, 0, 1)
        sims = sim_mat[i]
        for j, sim in enumerate(sims):
            sim_adj = np.clip((float(sim) - 0.2) / 0.4, 0, 1)
            if sim_adj > 0:
                factor_name = subcat_list[j]
                combined = ALPHA * sim_adj + BETA * sent_adj
                score_scaled = 1 / (1 + np.exp(-1.6 * (combined - 0.3)))
                factor_sentiments[factor_name].append(float(score_scaled))
    return factor_sentiments, [float(score) for score in sentiment_scores]


def _llm_adjust_scores(
    llm_client,
    factor_definitions: Dict[str, Dict[str, str]],
    raw_scores: Dict[str, Dict[str, float]],
    review_samples: Sequence[str],
    delta_limit: float = 0.5,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    if llm_client is None:
        return raw_scores, []

    sample_reviews = "\n".join(list(review_samples)[:3])
    prompt = f"""
당신은 장소성 평가 감사자입니다.
입력된 점수는 SBERT + 감성 회귀모델로 산출된 값입니다.
각 요인별 점수의 타당성을 **요인의 정의에 따라** 정확히 검토하세요.

## 요인 정의 (반드시 참고)
{json.dumps(factor_definitions, ensure_ascii=False, indent=2)}

## 현재 점수
{json.dumps(raw_scores, ensure_ascii=False, indent=2)}

## 리뷰 내용
{sample_reviews}

## 검토 규칙
1. 각 요인의 정의와 키워드를 **정확히** 확인하세요.
   예: "감각적 경험"은 음악, 향기, 질감 등 오감 자극 / "문화적 맥락"은 역사, 전통, 지역 배경
2. 리뷰에서 해당 요인 정의에 맞는 언급이 있는데 점수가 낮거나, 언급이 없는데 점수가 높으면 delta 제안
3. delta는 -{delta_limit:.1f} ~ +{delta_limit:.1f} 범위
4. 근거는 한 문장으로만 작성

## 출력 형식 (JSON만)
{{
  "corrections": [
    {{"factor": "쾌적성", "delta": 0.15, "reason": "청결, 화장실, 충전시설 긍정 언급 많음"}},
    {{"factor": "감각적 경험", "delta": 0.12, "reason": "디저트 맛과 다양성 강조"}}
  ]
}}

보정 불필요 시: {{"corrections": []}}
"""

    try:
        resp = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500,
        )
        correction_result = json.loads(resp.choices[0].message.content)
        corrections = correction_result.get("corrections", [])
    except Exception:
        return raw_scores, []

    corrected_scores = json.loads(json.dumps(raw_scores))
    correction_log: List[Dict[str, Any]] = []

    for correction in corrections:
        if not isinstance(correction, dict):
            continue
        factor_name = correction.get("factor")
        try:
            delta = float(correction.get("delta", 0))
        except (TypeError, ValueError):
            continue
        reason = correction.get("reason", "")

        found = False
        for main_cat, subcats in corrected_scores.items():
            if factor_name in subcats:
                old_val = subcats[factor_name]
                new_val = np.clip(old_val + delta, 0.20, 1.0)
                corrected_scores[main_cat][factor_name] = float(new_val)
                correction_log.append(
                    {
                        "factor": factor_name,
                        "original": round(old_val, 2),
                        "adjusted": round(float(new_val), 2),
                        "delta": round(delta, 2),
                        "reason": reason,
                    }
                )
                found = True
                break
        if not found:
            continue

    return corrected_scores, correction_log


def evaluate_reviews_for_place(
    review_texts: Sequence[str],
    sentiment_model: Callable[[List[str]], List[float]],
    embed_model,
    category_embeddings: Dict[str, np.ndarray],
    score_template: Dict[str, Dict[str, Optional[float]]],
    semantic_split_fn: Callable[[str], List[str]],
    llm_client=None,
    factor_definitions: Optional[Dict[str, Dict[str, str]]] = None,
    llm_delta_limit: float = 0.5,
) -> Optional[Tuple[Dict[str, Dict[str, float]], float, int, List[Dict[str, Any]]]]:
    cleaned_reviews = [str(text).strip() for text in review_texts if str(text).strip()]
    if not cleaned_reviews:
        return None

    review_text = "\n".join(cleaned_reviews)
    review_units = semantic_split_fn(review_text)
    if not review_units:
        return None

    factor_sentiments, sentiment_scores = _compute_factor_sentiments(
        review_units, sentiment_model, embed_model, category_embeddings
    )
    _apply_keyword_boosts(review_text, factor_sentiments)
    _apply_accessibility_boost(review_text, factor_sentiments)

    scores = _clone_score_template(score_template)
    all_values: List[float] = [val for vals in factor_sentiments.values() for val in vals]

    if all_values:
        vmin, vmax = float(np.min(all_values)), float(np.max(all_values))
    else:
        vmin, vmax = 0.5, 0.5

    for main_cat, subcats in scores.items():
        for subcat in subcats.keys():
            vals = factor_sentiments.get(subcat, [])
            if vals and vmax > vmin:
                raw = float(np.mean(vals))
                normed = 0.20 + 0.80 * ((raw - vmin) / (vmax - vmin + 1e-8))
                scores[main_cat][subcat] = float(np.clip(normed, 0.20, 1.0))
            elif vals:
                scores[main_cat][subcat] = float(np.clip(vals[0], 0.20, 1.0))
            else:
                scores[main_cat][subcat] = 0.5

    corrected_scores, correction_log = _llm_adjust_scores(
        llm_client=llm_client,
        factor_definitions=factor_definitions or {},
        raw_scores=scores,
        review_samples=cleaned_reviews,
        delta_limit=llm_delta_limit,
    ) if llm_client and factor_definitions else (scores, [])

    flat_scores = [score for sub in corrected_scores.values() for score in sub.values() if score is not None]
    mean_score = float(np.mean(flat_scores)) if flat_scores else 0.5
    mean_sentiment = float(np.mean(sentiment_scores)) if sentiment_scores else np.nan
    return corrected_scores, mean_score, len(review_units), correction_log, mean_sentiment


def analyze_review_groups(
    review_df: pd.DataFrame,
    group_cols: Sequence[str],
    sentiment_model: Callable[[List[str]], List[float]],
    embed_model,
    category_embeddings: Dict[str, np.ndarray],
    score_template: Dict[str, Dict[str, Optional[float]]],
    semantic_split_fn: Callable[[str], List[str]],
    llm_client=None,
    factor_definitions: Optional[Dict[str, Dict[str, str]]] = None,
    llm_delta_limit: float = 0.5,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> pd.DataFrame:
    results: List[Dict[str, object]] = []
    factor_defs = factor_definitions
    if llm_client is not None and factor_defs is None:
        try:
            with open("factors.json", "r", encoding="utf-8") as f:
                factor_defs = json.load(f)
        except Exception:
            factor_defs = None

    grouped_iter = review_df.groupby(list(group_cols), dropna=False)
    total_groups = grouped_iter.ngroups
    for idx, (group_key, group) in enumerate(grouped_iter, start=1):
        if progress_callback:
            context = " / ".join(str(k) for k in group_key if k)
            progress_callback(idx, total_groups, context)
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        key_dict = {col: val for col, val in zip(group_cols, group_key)}
        review_texts = group.get("리뷰", pd.Series(dtype=str)).tolist()

        evaluation = evaluate_reviews_for_place(
            review_texts=review_texts,
            sentiment_model=sentiment_model,
            embed_model=embed_model,
            category_embeddings=category_embeddings,
            score_template=score_template,
            semantic_split_fn=semantic_split_fn,
            llm_client=llm_client,
            factor_definitions=factor_defs,
            llm_delta_limit=llm_delta_limit,
        )
        if evaluation is None:
            continue

        scores, mean_score, unit_count, correction_log, mean_sentiment = evaluation
        result: Dict[str, object] = dict(key_dict)
        result["리뷰수"] = int(sum(1 for text in review_texts if str(text).strip()))
        result["리뷰문장수"] = unit_count

        if "평점" in group.columns and not group["평점"].dropna().empty:
            result["평균평점"] = float(group["평점"].dropna().mean())
        else:
            result["평균평점"] = np.nan

        result["평균감성점수"] = mean_sentiment
        result["평균장소성점수"] = mean_score
        result["scores"] = scores
        result["리뷰통합"] = "\n".join([str(text).strip() for text in review_texts if str(text).strip()])
        result["corrections"] = correction_log

        if "lat" in group.columns and "lng" in group.columns:
            result["lat"] = group["lat"].iloc[0]
            result["lng"] = group["lng"].iloc[0]

        results.append(result)

    columns = list(group_cols) + [
        "리뷰수",
        "리뷰문장수",
        "평균평점",
        "평균감성점수",
        "평균장소성점수",
        "scores",
        "리뷰통합",
    ]
    if results and "lat" in results[0]:
        columns += ["lat", "lng"]

    return pd.DataFrame(results, columns=columns) if results else pd.DataFrame(columns=columns)


def build_placeness_map(
    df: pd.DataFrame,
    value_col: str = "평균장소성점수",
    center: Tuple[float, float] = (37.5665, 126.9780),
    zoom_start: int = 11,
) -> folium.Map:
    if df.empty or "lat" not in df.columns or "lng" not in df.columns:
        raise ValueError("지도 생성을 위해서는 lat/lng 컬럼과 데이터가 필요합니다.")

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="cartodbpositron")
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        name = row.get("상호명", "이름 없음")
        district = row.get("시군구명", "")
        score = row.get(value_col, np.nan)
        popup_lines = [
            f"<b>{name}</b>",
            f"구: {district}",
            f"감성 점수: {score:.3f}" if not pd.isna(score) else "감성 점수: N/A",
        ]
        if "평균평점" in row and not pd.isna(row["평균평점"]):
            popup_lines.append(f"Google 평점 평균: {row['평균평점']:.2f}")
        popup_html = "<br>".join(popup_lines)

        folium.Marker(
            location=[row["lat"], row["lng"]],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=name,
        ).add_to(marker_cluster)

    heat_data = [
        [row["lat"], row["lng"], row[value_col]]
        for _, row in df.iterrows()
        if value_col in row and not pd.isna(row[value_col])
    ]
    if heat_data:
        HeatMap(heat_data, radius=15, blur=20, max_zoom=12).add_to(m)

    return m


__all__ = [
    "analyze_review_groups",
    "evaluate_reviews_for_place",
    "build_placeness_map",
]

