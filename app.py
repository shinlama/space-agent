from __future__ import annotations

import html
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_REVIEW_CSV = PROJECT_ROOT / "google_reviews_scraped_cleaned.csv"
SAMPLE_PLACE_CSV = PROJECT_ROOT / "서울시_상권_카페빵_표본.csv"
SCORING_CACHE_VERSION = "raw_score_minmax_v1"

from modules.research_scoring import (
    DEFAULT_MAPPING_CSV,
    FACTOR_CATEGORIES,
    FACTOR_DETAILS,
    FACTOR_ORDER,
    complete_factor_table,
    calculate_scores,
)

RECOMMENDATION_PRESETS = {
    "분위기 좋은 곳": ["심미성", "감각적 경험", "쾌적성"],
    "작업/공부하기 좋은 곳": ["쾌적성", "개방성", "접근성", "활동성"],
    "친구와 대화하기 좋은 곳": ["활동성", "개방성", "쾌적성"],
    "방문이 편한 곳": ["접근성", "개방성", "쾌적성"],
    "전체적으로 균형 잡힌 곳": FACTOR_ORDER,
}


st.set_page_config(
    page_title="장소성 정량화 연구 데모",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner="매핑 결과와 점수 계산 결과를 불러오는 중입니다.")
def load_demo_data(
    input_csv: str,
    scoring_version: str = SCORING_CACHE_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ = scoring_version
    return calculate_scores(Path(input_csv))


@st.cache_data(show_spinner=False)
def load_source_data_summary() -> dict[str, int | None]:
    summary: dict[str, int | None] = {
        "sample_place_count": None,
        "collected_review_count": None,
        "collected_place_count": None,
    }

    if SAMPLE_PLACE_CSV.exists():
        for encoding in ("utf-8-sig", "cp949"):
            try:
                sample = pd.read_csv(SAMPLE_PLACE_CSV, encoding=encoding)
                summary["sample_place_count"] = len(sample)
                break
            except (UnicodeDecodeError, ValueError):
                continue

    if RAW_REVIEW_CSV.exists():
        for encoding in ("utf-8-sig", "cp949"):
            try:
                raw_reviews = pd.read_csv(RAW_REVIEW_CSV, encoding=encoding, usecols=["상호명"])
                summary["collected_review_count"] = len(raw_reviews)
                summary["collected_place_count"] = raw_reviews["상호명"].nunique(dropna=True)
                break
            except (UnicodeDecodeError, ValueError):
                continue

    return summary


def format_score(value: float | int | None) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.3f}"


def format_signed_score(value: float | int | None) -> str:
    if pd.isna(value):
        return "-"
    numeric = float(value)
    if numeric > 0:
        return f"+{numeric:.0f}"
    return f"{numeric:.0f}"


def format_percent(value: float | int | None) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def format_count(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}"


def format_count_unit(value: float | int | None, unit: str) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}{unit}"


def sentiment_badge(label: str) -> str:
    classes = {
        "긍정": "positive",
        "중립": "neutral",
        "혼합": "mixed",
        "부정": "negative",
    }
    class_name = classes.get(label, "neutral")
    return f'<span class="badge {class_name}">{html.escape(label)}</span>'


def compact_with_index(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    indexes: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        chars.append(char)
        indexes.append(index)
    return "".join(chars), indexes


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def find_exact_span(review_text: str, evidence: str, used_spans: list[tuple[int, int]]) -> tuple[int, int] | None:
    start = review_text.find(evidence)
    while start != -1:
        span = (start, start + len(evidence))
        if not any(spans_overlap(span, used_span) for used_span in used_spans):
            return span
        start = review_text.find(evidence, start + 1)
    return None


def find_compact_span(review_text: str, evidence: str, used_spans: list[tuple[int, int]]) -> tuple[int, int] | None:
    compact_review, review_indexes = compact_with_index(review_text)
    compact_evidence = normalized_text(evidence)
    start = compact_review.find(compact_evidence)
    while start != -1:
        end = start + len(compact_evidence) - 1
        span = (review_indexes[start], review_indexes[end] + 1)
        if not any(spans_overlap(span, used_span) for used_span in used_spans):
            return span
        start = compact_review.find(compact_evidence, start + 1)
    return None


def trim_fuzzy_span(review_text: str, evidence: str, span: tuple[int, int]) -> tuple[int, int]:
    start, end = span
    candidate = review_text[start:end]
    compact_candidate, candidate_indexes = compact_with_index(candidate)
    evidence_normalized = normalized_text(evidence)
    matching_blocks = [
        block
        for block in SequenceMatcher(None, evidence_normalized, compact_candidate).get_matching_blocks()
        if block.size >= 2
    ]
    if not matching_blocks:
        return span

    compact_start = min(block.b for block in matching_blocks)
    compact_end = max(block.b + block.size for block in matching_blocks)
    if compact_end <= compact_start:
        return span

    trimmed_start = start + candidate_indexes[compact_start]
    trimmed_end = start + candidate_indexes[compact_end - 1] + 1
    token_start = trimmed_start
    while token_start > 0 and not review_text[token_start - 1].isspace():
        token_start -= 1
    token_end = trimmed_start
    while token_end < len(review_text) and not review_text[token_end].isspace():
        token_end += 1
    if token_start < trimmed_start >= token_end - 1:
        next_start = token_end
        while next_start < trimmed_end and review_text[next_start].isspace():
            next_start += 1
        if next_start < trimmed_end:
            trimmed_start = next_start

    if trimmed_end - trimmed_start < 4:
        return span
    return trimmed_start, trimmed_end


def find_fuzzy_span(review_text: str, evidence: str, used_spans: list[tuple[int, int]]) -> tuple[int, int] | None:
    tokens = list(re.finditer(r"\S+", review_text))
    evidence_tokens = re.findall(r"\S+", evidence)
    if not tokens or not evidence_tokens:
        return None

    target_len = len(evidence_tokens)
    min_len = max(1, target_len - 3)
    max_len = min(len(tokens), target_len + 3)
    evidence_normalized = normalized_text(evidence)
    best_span: tuple[int, int] | None = None
    best_score = 0.0

    for window_len in range(min_len, max_len + 1):
        for start_index in range(0, len(tokens) - window_len + 1):
            start = tokens[start_index].start()
            end = tokens[start_index + window_len - 1].end()
            candidate_span = (start, end)
            if any(spans_overlap(candidate_span, used_span) for used_span in used_spans):
                continue
            candidate = normalized_text(review_text[start:end])
            score = SequenceMatcher(None, evidence_normalized, candidate).ratio()
            if score > best_score:
                best_score = score
                best_span = candidate_span

    if best_score < 0.68:
        return None
    return trim_fuzzy_span(review_text, evidence, best_span)


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def highlight_evidence(review_text: str, evidence_rows: pd.DataFrame) -> str:
    evidences = (
        evidence_rows["evidence"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values(key=lambda s: s.str.len(), ascending=False)
    )

    spans: list[tuple[int, int]] = []
    for evidence in evidences:
        if not evidence:
            continue
        span = (
            find_exact_span(review_text, evidence, spans)
            or find_compact_span(review_text, evidence, spans)
            or find_fuzzy_span(review_text, evidence, spans)
        )
        if span is not None:
            spans.append(span)

    parts: list[str] = []
    cursor = 0
    for start, end in merge_spans(spans):
        parts.append(html.escape(review_text[cursor:start]))
        parts.append(f"<mark>{html.escape(review_text[start:end])}</mark>")
        cursor = end
    parts.append(html.escape(review_text[cursor:]))
    return f'<div class="review-box">{"".join(parts)}</div>'


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        .research-caption {
            color: #526071;
            font-size: 0.95rem;
            line-height: 1.55;
        }
        .metric-note {
            color: #667085;
            font-size: 0.82rem;
            margin-top: -0.65rem;
        }
        .factor-card {
            border: 1px solid #d8e3ee;
            border-radius: 8px;
            padding: 1rem;
            background: #fbfdff;
            min-height: 152px;
        }
        .factor-card h4 {
            margin: 0 0 0.35rem 0;
            font-size: 1.05rem;
        }
        .factor-card p {
            margin: 0.25rem 0;
            color: #475467;
            line-height: 1.45;
        }
        .review-box {
            border-left: 5px solid #2f6f9f;
            background: #f6f9fc;
            border-radius: 6px;
            padding: 1rem 1.15rem;
            font-size: 1.05rem;
            line-height: 1.8;
            color: #1f2937;
        }
        mark {
            background: #fff1a8;
            color: #111827;
            border-radius: 4px;
            padding: 0.05rem 0.18rem;
        }
        .badge {
            display: inline-block;
            min-width: 42px;
            text-align: center;
            border-radius: 999px;
            padding: 0.15rem 0.55rem;
            font-size: 0.82rem;
            font-weight: 700;
        }
        .positive { background: #e9f8ef; color: #0f7a3d; }
        .neutral { background: #eef2f6; color: #475467; }
        .mixed { background: #fff6db; color: #966300; }
        .negative { background: #fdecec; color: #b42318; }
        .formula-box {
            border: 1px solid #d7dde5;
            border-radius: 8px;
            padding: 1rem;
            background: #ffffff;
            color: #27364a;
            line-height: 1.65;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_factor_system() -> None:
    st.subheader("1. 선행연구 기반 장소성 요인 평가 체계")
    st.markdown(
        '<p class="research-caption">장소성을 물리적 특성, 활동적 특성, 의미적 특성의 3차원으로 보고, 상업공간 리뷰에서 관찰 가능한 10개 요인으로 재구성한 체계입니다.</p>',
        unsafe_allow_html=True,
    )

    category_cols = st.columns(3)
    for col, (category, factors) in zip(category_cols, FACTOR_CATEGORIES.items()):
        with col:
            st.markdown(f"#### {category}")
            for factor in factors:
                detail = FACTOR_DETAILS[factor]
                criteria = ", ".join(detail["criteria"])
                st.markdown(
                    f"""
                    <div class="factor-card">
                      <h4>{factor} <span style="color:#667085;font-weight:500;">({detail['english']})</span></h4>
                      <p>{detail['definition']}</p>
                      <p><b>판별 기준</b>: {criteria}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("")

    st.markdown("#### 요인별 리뷰 표현 예시")
    selected_factor = st.selectbox("요인을 선택하세요", FACTOR_ORDER, key="factor_detail")
    detail = FACTOR_DETAILS[selected_factor]
    st.write(f"**정의**: {detail['definition']}")
    st.write(f"**판별 기준**: {', '.join(detail['criteria'])}")
    st.write(f"**리뷰 표현 예시**: {', '.join(detail['examples'])}")


def render_mapping_results(scored_evidence: pd.DataFrame, cafe_name: str) -> None:
    st.subheader("2. 리뷰에서 장소성 요인 매핑 결과")
    cafe_rows = scored_evidence[scored_evidence["cafe_name"] == cafe_name].copy()
    if cafe_rows.empty:
        st.warning("선택한 장소의 매핑 결과가 없습니다.")
        return

    review_options = (
        cafe_rows.groupby("review_index")
        .agg(
            review_text=("review_text", "first"),
            mapping_count=("factor", "size"),
        )
        .sort_values(["mapping_count", "review_index"], ascending=[False, True])
        .reset_index()
    )
    review_options["label"] = review_options.apply(
        lambda row: f"review_index {int(row['review_index'])} · 매핑 {int(row['mapping_count'])}개 · {row['review_text'][:54]}",
        axis=1,
    )
    selected_label = st.selectbox("리뷰 선택", review_options["label"].tolist())
    selected_review_index = int(review_options.loc[review_options["label"] == selected_label, "review_index"].iloc[0])
    review_rows = cafe_rows[cafe_rows["review_index"] == selected_review_index].copy()
    review_text = review_rows["review_text"].iloc[0]

    st.markdown("##### 리뷰")
    st.markdown(highlight_evidence(review_text, review_rows), unsafe_allow_html=True)

    display = review_rows[
        ["evidence", "factor", "sentiment_label", "sentiment_value", "reason"]
    ].copy()
    display = display.rename(
        columns={
            "evidence": "근거 구절",
            "factor": "매핑 요인",
            "sentiment_label": "감성 방향",
            "sentiment_value": "구절별 점수",
            "reason": "매핑 근거",
        }
    )
    display["구절별 점수"] = display["구절별 점수"].map(format_signed_score)

    st.markdown("##### 매핑 결과")
    st.dataframe(display, use_container_width=True, hide_index=True)

    badge_html = " ".join(
        sentiment_badge(label)
        for label in review_rows["sentiment_label"].tolist()
    )
    st.markdown(
        f'<div class="research-caption">이 리뷰에서는 총 <b>{len(review_rows)}</b>개의 장소성 근거 구절이 추출되었습니다. {badge_html}</div>',
        unsafe_allow_html=True,
    )


def render_score_results(
    scored_evidence: pd.DataFrame,
    factor_scores: pd.DataFrame,
    place_scores: pd.DataFrame,
    cafe_name: str,
) -> None:
    st.subheader("3. 매핑된 요인의 점수화 계산 결과")

    selected_place = place_scores[place_scores["cafe_name"] == cafe_name].iloc[0]
    equal_weight_score_100 = selected_place["mean_factor_score"] * 100
    mention_weighted_score_100 = selected_place["placeness_score_100"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("동일가중 평균", f"{equal_weight_score_100:.1f} / 100")
    c2.metric("언급비중 반영 점수", f"{mention_weighted_score_100:.1f} / 100")
    c3.metric("요인 근거 구절 수", f"{int(selected_place['mapped_evidence_count']):,}개")
    c4.metric("언급된 요인 수", f"{int(selected_place['mentioned_factor_count'])} / 10")
    st.caption(f"부정 근거 비율: {format_percent(selected_place['negative_ratio'])}")

    st.markdown(
        """
        <div class="formula-box">
        <b>구절별 점수</b>: 긍정 = +1, 중립/혼합 = 0, 부정 = -1<br>
        <b>요인별 원점수</b> = (긍정 구절 수 - 부정 구절 수) / 해당 요인에 매핑된 전체 구절 수<br>
        <b>정규화된 장소성 요인 점수</b> = (요인별 원점수 + 1) / 2 <span style="color:#667085;">Min-Max Scaling</span><br>
        <b>요인별 언급 비중</b> = 해당 요인에 매핑된 구절 수 / 해당 장소의 전체 장소성 근거 구절 수<br>
        <b>동일가중 평균</b> = Σ(정규화된 장소성 요인 점수) / 언급된 요인 수<br>
        <b>언급비중 반영 점수</b> = Σ(정규화된 장소성 요인 점수 × 요인별 언급 비중)
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    table = complete_factor_table(factor_scores, cafe_name)
    plot_df = table.copy()
    plot_df["정규화된 장소성 요인 점수"] = plot_df["factor_score"].fillna(0)
    plot_df["언급 비중"] = plot_df["mention_share"].fillna(0)

    chart = px.bar(
        plot_df,
        x="factor",
        y="정규화된 장소성 요인 점수",
        color="factor_category",
        text=plot_df["정규화된 장소성 요인 점수"].map(lambda v: f"{v:.2f}" if v > 0 else ""),
        category_orders={"factor": FACTOR_ORDER},
        labels={"factor": "장소성 요인", "factor_category": "구분"},
        height=390,
    )
    chart.update_layout(yaxis_range=[0, 1], legend_title_text="구분", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(chart, use_container_width=True)

    display = table[
        [
            "factor_category",
            "factor",
            "positive_count",
            "neutral_count",
            "mixed_count",
            "negative_count",
            "mention_count",
            "raw_factor_score",
            "factor_score",
            "mention_share",
            "weighted_score",
            "evidence_examples",
        ]
    ].copy()
    display = display.rename(
        columns={
            "factor_category": "구분",
            "factor": "요인",
            "positive_count": "긍정",
            "neutral_count": "중립",
            "mixed_count": "혼합",
            "negative_count": "부정",
            "mention_count": "근거 수",
            "raw_factor_score": "요인별 원점수",
            "factor_score": "정규화된 장소성 요인 점수",
            "mention_share": "언급 비중",
            "weighted_score": "가중 점수",
            "evidence_examples": "근거 예시",
        }
    )
    for col in ["요인별 원점수", "정규화된 장소성 요인 점수", "가중 점수"]:
        display[col] = display[col].map(format_score)
    display["언급 비중"] = display["언급 비중"].map(format_percent)
    display["근거 예시"] = display["근거 예시"].fillna("")
    st.dataframe(display, use_container_width=True, hide_index=True)

    selected_evidence = scored_evidence[scored_evidence["cafe_name"] == cafe_name]
    st.download_button(
        "선택 장소의 구절별 점수 데이터 다운로드",
        data=selected_evidence.drop(columns=["confidence"], errors="ignore").to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"{cafe_name}_scored_evidence.csv",
        mime="text/csv",
    )


def render_place_comparison(place_scores: pd.DataFrame) -> None:
    st.subheader("4. 장소별 결과 비교")
    min_count = st.slider(
        "최소 장소성 근거 구절 수",
        min_value=1,
        max_value=int(place_scores["mapped_evidence_count"].max()),
        value=10,
        step=1,
    )
    filtered = place_scores[place_scores["mapped_evidence_count"] >= min_count].copy()
    filtered["equal_weight_score_100"] = filtered["mean_factor_score"] * 100

    chart = px.scatter(
        filtered,
        x="mapped_evidence_count",
        y="placeness_score_100",
        color="mentioned_factor_count",
        hover_data=["cafe_name", "equal_weight_score_100", "most_mentioned_factor", "positive_ratio", "negative_ratio"],
        labels={
            "mapped_evidence_count": "장소성 근거 구절 수",
            "placeness_score_100": "언급비중 반영 점수",
            "equal_weight_score_100": "동일가중 평균",
            "mentioned_factor_count": "언급 요인 수",
        },
        height=420,
    )
    chart.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(chart, use_container_width=True)

    display = filtered[
        [
            "cafe_name",
            "equal_weight_score_100",
            "placeness_score_100",
            "mapped_evidence_count",
            "mentioned_factor_count",
            "most_mentioned_factor",
            "positive_ratio",
            "negative_ratio",
        ]
    ].head(200).copy()
    display = display.rename(
        columns={
            "cafe_name": "장소명",
            "equal_weight_score_100": "동일가중 평균",
            "placeness_score_100": "언급비중 반영 점수",
            "mapped_evidence_count": "근거 구절 수",
            "mentioned_factor_count": "언급 요인 수",
            "most_mentioned_factor": "최다 언급 요인",
            "positive_ratio": "긍정 비율",
            "negative_ratio": "부정 비율",
        }
    )
    display["동일가중 평균"] = display["동일가중 평균"].map(lambda v: f"{v:.1f}")
    display["언급비중 반영 점수"] = display["언급비중 반영 점수"].map(lambda v: f"{v:.1f}")
    display["긍정 비율"] = display["긍정 비율"].map(format_percent)
    display["부정 비율"] = display["부정 비율"].map(format_percent)
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_personalized_recommendation(
    scored_evidence: pd.DataFrame,
    factor_scores: pd.DataFrame,
    place_scores: pd.DataFrame,
) -> None:
    st.subheader("5. 개인화 추천")
    st.markdown(
        """
        <div class="formula-box">
        <b>개인화 추천 점수</b> = 사용자가 선택한 장소성 요인의 정규화된 장소성 요인 점수 평균 × 100<br>
        추천 결과는 새로운 모델을 학습한 것이 아니라, 앞 단계에서 산출한 장소성 요인별 점수를 사용자 선호에 맞춰 다시 정렬한 결과입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        preset_name = st.selectbox("추천 목적", list(RECOMMENDATION_PRESETS.keys()))
    with c2:
        selected_factors = st.multiselect(
            "중요하게 볼 장소성 요인",
            FACTOR_ORDER,
            default=RECOMMENDATION_PRESETS[preset_name],
            key=f"recommendation_factors_{preset_name}",
        )

    if not selected_factors:
        st.info("추천에 반영할 장소성 요인을 하나 이상 선택하세요.")
        return

    selected_scores = factor_scores[factor_scores["factor"].isin(selected_factors)].copy()
    if selected_scores.empty:
        st.warning("선택한 요인에 해당하는 점수 데이터가 없습니다.")
        return

    min_count = st.slider(
        "최소 장소성 근거 구절 수",
        min_value=1,
        max_value=int(place_scores["mapped_evidence_count"].max()),
        value=min(10, int(place_scores["mapped_evidence_count"].max())),
        step=1,
        key="recommendation_min_evidence",
    )

    recommendations = (
        selected_scores.groupby("cafe_name")
        .agg(
            personalized_score=("factor_score", "mean"),
            reflected_factor_count=("factor", "nunique"),
        )
        .reset_index()
    )
    factor_names = (
        selected_scores.groupby("cafe_name")["factor"]
        .apply(lambda values: ", ".join([factor for factor in FACTOR_ORDER if factor in set(values)]))
        .rename("reflected_factors")
        .reset_index()
    )
    top_factor = (
        selected_scores.sort_values(["cafe_name", "factor_score", "mention_count"], ascending=[True, False, False])
        .drop_duplicates("cafe_name")[["cafe_name", "factor"]]
        .rename(columns={"factor": "top_preference_factor"})
    )
    recommendations = (
        recommendations.merge(factor_names, on="cafe_name", how="left")
        .merge(top_factor, on="cafe_name", how="left")
        .merge(
            place_scores[
                [
                    "cafe_name",
                    "mapped_evidence_count",
                    "mentioned_factor_count",
                    "placeness_score_100",
                ]
            ],
            on="cafe_name",
            how="left",
        )
    )
    recommendations["personalized_score_100"] = recommendations["personalized_score"] * 100
    recommendations = recommendations[recommendations["mapped_evidence_count"] >= min_count].sort_values(
        ["personalized_score_100", "mapped_evidence_count"],
        ascending=[False, False],
    )

    if recommendations.empty:
        st.warning("현재 조건에 맞는 추천 장소가 없습니다. 최소 근거 구절 수를 낮춰보세요.")
        return

    st.markdown("##### 추천 결과")
    display = recommendations.head(20)[
        [
            "cafe_name",
            "personalized_score_100",
            "top_preference_factor",
            "reflected_factors",
            "mapped_evidence_count",
            "placeness_score_100",
        ]
    ].copy()
    display.insert(0, "rank", range(1, len(display) + 1))
    display = display.rename(
        columns={
            "rank": "순위",
            "cafe_name": "장소명",
            "personalized_score_100": "개인화 추천 점수",
            "top_preference_factor": "주요 추천 요인",
            "reflected_factors": "반영된 선호 요인",
            "mapped_evidence_count": "근거 구절 수",
            "placeness_score_100": "언급비중 반영 점수",
        }
    )
    display["개인화 추천 점수"] = display["개인화 추천 점수"].map(lambda value: f"{value:.1f}")
    display["언급비중 반영 점수"] = display["언급비중 반영 점수"].map(lambda value: f"{value:.1f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("##### 추천 근거")
    recommended_places = recommendations.head(20)["cafe_name"].tolist()
    selected_place = st.selectbox("추천 근거를 볼 장소", recommended_places, key="recommendation_place")
    selected_row = recommendations[recommendations["cafe_name"] == selected_place].iloc[0]
    st.metric("개인화 추천 점수", f"{selected_row['personalized_score_100']:.1f} / 100")

    place_factor_scores = selected_scores[selected_scores["cafe_name"] == selected_place].sort_values(
        ["factor_score", "mention_count"],
        ascending=[False, False],
    )
    factor_summary = ", ".join(
        f"{row.factor} {row.factor_score * 100:.1f}점"
        for row in place_factor_scores.itertuples()
    )
    st.write(
        f"선택한 선호 요인 중 **{selected_row['top_preference_factor']}**이 가장 높게 나타났습니다. "
        f"반영된 요인 점수는 {factor_summary}입니다."
    )

    evidence_factors = place_factor_scores["factor"].head(3).tolist()
    evidence_rows = scored_evidence[
        (scored_evidence["cafe_name"] == selected_place)
        & (scored_evidence["factor"].isin(evidence_factors))
    ].copy()
    evidence_rows = evidence_rows.sort_values(["sentiment_value", "factor"], ascending=[False, True]).head(8)
    evidence_display = evidence_rows[["factor", "sentiment_label", "evidence", "reason"]].rename(
        columns={
            "factor": "요인",
            "sentiment_label": "감성 방향",
            "evidence": "추천 근거 구절",
            "reason": "매핑 근거",
        }
    )
    st.dataframe(evidence_display, use_container_width=True, hide_index=True)


def main() -> None:
    inject_css()
    st.title("공간 리뷰 텍스트 기반 장소성 정량화")

    if not DEFAULT_MAPPING_CSV.exists():
        st.error(f"매핑 결과 CSV를 찾을 수 없습니다: {DEFAULT_MAPPING_CSV}")
        return

    scored_evidence, factor_scores, place_scores = load_demo_data(
        str(DEFAULT_MAPPING_CSV),
        SCORING_CACHE_VERSION,
    )
    source_summary = load_source_data_summary()

    with st.sidebar:
        st.header("데이터 흐름")
        st.caption("공공 상권정보 표본에서 카페를 선정하고, Google Maps 리뷰에서 장소성 관련 구절을 매핑했습니다.")

        st.markdown("**1. 분석 대상 표본**")
        st.metric("서울시 카페 표본", format_count_unit(source_summary["sample_place_count"], "개"))

        st.markdown("**2. 리뷰 데이터**")
        st.metric("수집 리뷰", format_count_unit(source_summary["collected_review_count"], "건"))
        st.caption(f"리뷰 보유 장소: {format_count_unit(source_summary['collected_place_count'], '개')}")

        st.markdown("**3. 장소성 매핑 데이터**")
        st.metric("장소성 관련 리뷰", format_count_unit(scored_evidence["review_index"].nunique(), "건"))
        st.metric("매핑 근거 구절", format_count_unit(len(scored_evidence), "개"))
        st.caption(f"분석 가능 장소: {format_count_unit(place_scores['cafe_name'].nunique(), '개')}")

        st.divider()
        st.header("장소 선택")
        cafe_options = (
            place_scores.sort_values(["mapped_evidence_count", "placeness_score"], ascending=[False, False])
            ["cafe_name"]
            .tolist()
        )
        cafe_name = st.selectbox("장소 선택", cafe_options)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "평가 체계",
            "리뷰 매핑",
            "점수 계산",
            "장소 비교",
            "개인화 추천",
        ]
    )
    with tab1:
        render_factor_system()
    with tab2:
        render_mapping_results(scored_evidence, cafe_name)
    with tab3:
        render_score_results(scored_evidence, factor_scores, place_scores, cafe_name)
    with tab4:
        render_place_comparison(place_scores)
    with tab5:
        render_personalized_recommendation(scored_evidence, factor_scores, place_scores)


if __name__ == "__main__":
    main()
