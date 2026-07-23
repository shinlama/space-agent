from __future__ import annotations

import html
import json
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

FULL_REVIEW_SUMMARY_JSON = PROJECT_ROOT / "data" / "google_reviews_clear_mismatch_v6.summary.json"
CANDIDATE_SUMMARY_JSON = PROJECT_ROOT / "data" / "spatial_review_candidates_v6.summary.json"
SCORING_CACHE_VERSION = "gpt54nano_signed_score_v6_district_selector_cleanup_20260723"

from modules.research_scoring import (
    DEFAULT_FACTOR_SCORES_PARQUET,
    DEFAULT_PLACE_SCORES_PARQUET,
    DEFAULT_SCORED_EVIDENCE_PARQUET,
    FACTOR_CATEGORIES,
    FACTOR_DETAILS,
    FACTOR_ORDER,
    complete_factor_table,
)
from modules.validation_analysis import (
    build_cluster_feature_matrix,
    evaluate_cluster_range,
    get_factor_extreme_cases,
    representative_places_for_cluster,
    run_place_clustering,
)

RECOMMENDATION_PRESETS = {
    "분위기 좋은 곳": ["심미성", "감각적 경험", "쾌적성"],
    "작업/공부하기 좋은 곳": ["쾌적성", "개방성", "접근성", "활동성"],
    "친구와 대화하기 좋은 곳": ["활동성", "개방성", "쾌적성"],
    "방문이 편한 곳": ["접근성", "개방성", "쾌적성"],
    "전체적으로 균형 잡힌 곳": FACTOR_ORDER,
}
FACTOR_CATEGORY_COLORS = {
    "물리적 특성": "#5CB7F2",
    "활동적 특성": "#8BD646",
    "의미적 특성": "#FFB52E",
}
FACTOR_CATEGORY = {
    factor: category
    for category, factors in FACTOR_CATEGORIES.items()
    for factor in factors
}


st.set_page_config(
    page_title="장소성 정량화 연구 데모",
    page_icon="",
    layout="wide",
)


@st.cache_resource(show_spinner="새 매핑 결과와 점수 계산 결과를 불러오는 중입니다.")
def load_demo_data(
    evidence_parquet: str,
    factor_parquet: str,
    place_parquet: str,
    scoring_version: str = SCORING_CACHE_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ = scoring_version
    return (
        pd.read_parquet(evidence_parquet),
        pd.read_parquet(factor_parquet),
        pd.read_parquet(place_parquet),
    )


@st.cache_data(show_spinner=False)
def load_source_data_summary() -> dict[str, int | None]:
    summary: dict[str, int | None] = {
        "sample_place_count": None,
        "collected_review_count": None,
        "collected_place_count": None,
        "candidate_review_count": None,
        "candidate_place_count": None,
    }

    if FULL_REVIEW_SUMMARY_JSON.exists():
        full_summary = json.loads(FULL_REVIEW_SUMMARY_JSON.read_text(encoding="utf-8"))
        summary["sample_place_count"] = int(full_summary["output_places"])
        summary["collected_review_count"] = int(full_summary["output_rows"])
        summary["collected_place_count"] = int(full_summary["output_places"])

    if CANDIDATE_SUMMARY_JSON.exists():
        candidate_summary = json.loads(CANDIDATE_SUMMARY_JSON.read_text(encoding="utf-8"))
        summary["candidate_review_count"] = int(candidate_summary["candidate_rows"])
        summary["candidate_place_count"] = int(candidate_summary["candidate_places"])

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


def format_signed_decimal(value: float | int | None) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):+.3f}"


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
        .sidebar-focus {
            border: 1px solid #d8e3ee;
            border-radius: 12px;
            padding: 0.95rem 0.9rem 0.85rem;
            margin: 0.8rem 0 1rem;
            background: #ffffff;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        .sidebar-focus-title {
            margin: 0 0 0.65rem;
            color: #1f2937;
            font-weight: 800;
            font-size: 0.98rem;
            letter-spacing: 0;
        }
        .sidebar-focus-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.72rem;
        }
        .sidebar-focus-label {
            color: #667085;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.12rem;
        }
        .sidebar-focus-value {
            color: #111827;
            font-size: 1.72rem;
            font-weight: 600;
            line-height: 1.05;
        }
        .sidebar-focus-value .unit {
            font-size: 0.9rem;
            font-weight: 600;
            margin-left: 0.08rem;
            color: #344054;
        }
        .sidebar-flow-note {
            color: #667085;
            font-size: 0.78rem;
            line-height: 1.45;
            margin: 0.35rem 0 0.75rem;
        }
        .sidebar-step-title {
            color: #1f2937;
            font-size: 0.88rem;
            font-weight: 800;
            margin: 0.95rem 0 0.45rem;
        }
        .sidebar-secondary-list {
            display: grid;
            gap: 0.42rem;
            margin-bottom: 0.3rem;
        }
        .sidebar-secondary-row {
            display: flex;
            justify-content: space-between;
            gap: 0.65rem;
            align-items: baseline;
            border-bottom: 1px solid #e9eef5;
            padding-bottom: 0.36rem;
        }
        .sidebar-secondary-label {
            color: #667085;
            font-size: 0.78rem;
            line-height: 1.3;
        }
        .sidebar-secondary-value {
            color: #344054;
            font-size: 0.92rem;
            font-weight: 800;
            white-space: nowrap;
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
        ["evidence", "factor", "sentiment_label", "sentiment_value"]
    ].copy()
    display = display.rename(
        columns={
            "evidence": "근거 구절",
            "factor": "매핑 요인",
            "sentiment_label": "감성 방향",
            "sentiment_value": "구절별 점수",
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
    equal_weight_score = selected_place["mean_factor_score"]
    mention_weighted_score = selected_place["placeness_score"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("동일가중 평균", format_signed_decimal(equal_weight_score))
    c2.metric("언급비중 반영 점수", format_signed_decimal(mention_weighted_score))
    c3.metric("요인 근거 구절 수", f"{int(selected_place['mapped_evidence_count']):,}개")
    c4.metric("언급된 요인 수", f"{int(selected_place['mentioned_factor_count'])} / 10")
    st.caption(f"부정 근거 비율: {format_percent(selected_place['negative_ratio'])}")

    st.markdown(
        """
        <div class="formula-box">
        <b>구절별 점수</b>: 긍정 = +1, 중립/혼합 = 0, 부정 = -1<br>
        <b>장소성 요인 점수</b> = (긍정 구절 수 - 부정 구절 수) / 해당 요인에 매핑된 전체 구절 수<br>
        <b>요인별 언급 비중</b> = 해당 요인에 매핑된 구절 수 / 해당 장소의 전체 장소성 근거 구절 수<br>
        <b>동일가중 평균</b> = Σ(장소성 요인 점수) / 언급된 요인 수<br>
        <b>언급비중 반영 점수</b> = Σ(장소성 요인 점수 × 요인별 언급 비중)<br>
        <span style="color:#667085;">점수 범위: -1은 부정적 평가, 0은 중립/혼합, +1은 긍정적 평가를 의미합니다.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    table = complete_factor_table(factor_scores, cafe_name)
    plot_df = table.copy()
    plot_df["장소성 요인 점수"] = plot_df["factor_score"].fillna(0)
    plot_df["언급 비중"] = plot_df["mention_share"].fillna(0)

    chart = px.bar(
        plot_df,
        x="factor",
        y="장소성 요인 점수",
        color="factor_category",
        color_discrete_map=FACTOR_CATEGORY_COLORS,
        text=plot_df["장소성 요인 점수"].map(lambda v: f"{v:+.2f}" if pd.notna(v) and v != 0 else ""),
        category_orders={"factor": FACTOR_ORDER},
        labels={"factor": "장소성 요인", "factor_category": "구분"},
        height=390,
    )
    chart.update_traces(
        marker_line_color="rgba(255,255,255,0.75)",
        marker_line_width=1,
        textfont_color="#344054",
        textposition="auto",
    )
    chart.update_layout(
        yaxis_range=[-1, 1],
        legend_title_text="구분",
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#667085"),
        yaxis=dict(gridcolor="#E7ECF3", zeroline=False),
        xaxis=dict(tickfont=dict(color="#667085")),
    )
    chart.add_hline(y=0, line_width=1.2, line_dash="dash", line_color="#AAB4C0")
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
            "factor_score": "장소성 요인 점수",
            "mention_share": "언급 비중",
            "weighted_score": "가중 점수",
            "evidence_examples": "근거 예시",
        }
    )
    for col in ["장소성 요인 점수", "가중 점수"]:
        display[col] = display[col].map(format_signed_decimal)
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

    chart = px.scatter(
        filtered,
        x="mapped_evidence_count",
        y="placeness_score",
        color="mentioned_factor_count",
        hover_data=["cafe_name", "mean_factor_score", "most_mentioned_factor", "positive_ratio", "negative_ratio"],
        labels={
            "mapped_evidence_count": "장소성 근거 구절 수",
            "placeness_score": "언급비중 반영 점수",
            "mean_factor_score": "동일가중 평균",
            "mentioned_factor_count": "언급 요인 수",
        },
        height=420,
    )
    chart.update_layout(yaxis_range=[-1, 1], margin=dict(l=10, r=10, t=20, b=10))
    chart.add_hline(y=0, line_width=1, line_dash="dash", line_color="#98a2b3")
    st.plotly_chart(chart, use_container_width=True)

    display = filtered[
        [
            "cafe_name",
            "mean_factor_score",
            "placeness_score",
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
            "mean_factor_score": "동일가중 평균",
            "placeness_score": "언급비중 반영 점수",
            "mapped_evidence_count": "근거 구절 수",
            "mentioned_factor_count": "언급 요인 수",
            "most_mentioned_factor": "최다 언급 요인",
            "positive_ratio": "긍정 비율",
            "negative_ratio": "부정 비율",
        }
    )
    display["동일가중 평균"] = display["동일가중 평균"].map(format_signed_decimal)
    display["언급비중 반영 점수"] = display["언급비중 반영 점수"].map(format_signed_decimal)
    display["긍정 비율"] = display["긍정 비율"].map(format_percent)
    display["부정 비율"] = display["부정 비율"].map(format_percent)
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_personalized_recommendation(
    scored_evidence: pd.DataFrame,
    factor_scores: pd.DataFrame,
    place_scores: pd.DataFrame,
) -> None:
    st.subheader("4. 개인화 추천")
    st.markdown(
        """
        <div class="formula-box">
        <b>개인화 추천 점수</b> = 사용자가 선택한 장소성 요인의 장소성 요인 점수 평균<br>
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
                    "placeness_score",
                ]
            ],
            on="cafe_name",
            how="left",
        )
    )
    recommendations = recommendations[recommendations["mapped_evidence_count"] >= min_count].sort_values(
        ["personalized_score", "mapped_evidence_count"],
        ascending=[False, False],
    )

    if recommendations.empty:
        st.warning("현재 조건에 맞는 추천 장소가 없습니다. 최소 근거 구절 수를 낮춰보세요.")
        return

    st.markdown("##### 추천 결과")
    display = recommendations.head(20)[
        [
            "cafe_name",
            "personalized_score",
            "top_preference_factor",
            "reflected_factors",
            "mapped_evidence_count",
            "placeness_score",
        ]
    ].copy()
    display.insert(0, "rank", range(1, len(display) + 1))
    display = display.rename(
        columns={
            "rank": "순위",
            "cafe_name": "장소명",
            "personalized_score": "개인화 추천 점수",
            "top_preference_factor": "주요 추천 요인",
            "reflected_factors": "반영된 선호 요인",
            "mapped_evidence_count": "근거 구절 수",
            "placeness_score": "언급비중 반영 점수",
        }
    )
    display["개인화 추천 점수"] = display["개인화 추천 점수"].map(format_signed_decimal)
    display["언급비중 반영 점수"] = display["언급비중 반영 점수"].map(format_signed_decimal)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("##### 추천 근거")
    recommended_places = recommendations.head(20)["cafe_name"].tolist()
    selected_place = st.selectbox("추천 근거를 볼 장소", recommended_places, key="recommendation_place")
    selected_row = recommendations[recommendations["cafe_name"] == selected_place].iloc[0]
    st.metric("개인화 추천 점수", format_signed_decimal(selected_row["personalized_score"]))

    place_factor_scores = selected_scores[selected_scores["cafe_name"] == selected_place].sort_values(
        ["factor_score", "mention_count"],
        ascending=[False, False],
    )
    factor_summary = ", ".join(
        f"{row.factor} {format_signed_decimal(row.factor_score)}"
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
    evidence_display = evidence_rows[["factor", "sentiment_label", "evidence"]].rename(
        columns={
            "factor": "요인",
            "sentiment_label": "감성 방향",
            "evidence": "추천 근거 구절",
        }
    )
    st.dataframe(evidence_display, use_container_width=True, hide_index=True)


def render_validation_analysis(
    scored_evidence: pd.DataFrame,
    factor_scores: pd.DataFrame,
    place_scores: pd.DataFrame,
) -> None:
    st.subheader("5. 검증 분석")
    st.markdown(
        """
        <div class="formula-box">
        <b>검증 관점</b><br>
        1) 요인별 언급비중 반영 점수가 높은 장소와 낮은 장소의 실제 리뷰 근거가 서로 다르게 나타나는지 확인합니다.<br>
        2) 10개 장소성 요인 점수로 장소를 유형화하여, 정량화 결과가 해석 가능한 장소성 유형을 만드는지 확인합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    case_tab, cluster_tab = st.tabs(["요인별 상·하위 사례", "장소성 유형화"])

    with case_tab:
        st.markdown("##### 요인별 상·하위 사례 비교")
        st.caption("장소성 요인 점수를 기준으로 요인별 상위/하위 장소를 추출하고, 언급 비중과 실제 매핑 근거 구절을 함께 확인합니다.")

        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
        with c1:
            selected_factor = st.selectbox("검토할 장소성 요인", FACTOR_ORDER, key="validation_factor")
        with c2:
            min_total_mentions = st.slider(
                "최소 전체 근거 수",
                min_value=1,
                max_value=max(1, int(place_scores["mapped_evidence_count"].max())),
                value=min(10, max(1, int(place_scores["mapped_evidence_count"].max()))),
                step=1,
                key="validation_min_total_mentions",
            )
        with c3:
            max_factor_mentions = max(1, int(factor_scores["mention_count"].max()))
            min_factor_mentions = st.slider(
                "최소 요인 근거 수",
                min_value=1,
                max_value=max_factor_mentions,
                value=min(3, max_factor_mentions),
                step=1,
                key="validation_min_factor_mentions",
            )
        with c4:
            n_cases = st.slider(
                "사례 수",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                key="validation_n_cases",
            )

        top_cases, bottom_cases = get_factor_extreme_cases(
            scored_evidence,
            factor_scores,
            place_scores,
            factor=selected_factor,
            min_total_mentions=min_total_mentions,
            min_factor_mentions=min_factor_mentions,
            n_cases=n_cases,
        )
        combined_cases = pd.concat([top_cases, bottom_cases], ignore_index=True)
        if combined_cases.empty:
            st.warning("현재 조건에 맞는 상·하위 사례가 없습니다. 최소 근거 수를 낮춰보세요.")
        else:
            display = combined_cases[
                [
                    "case_type",
                    "cafe_name",
                    "factor_score",
                    "mention_share",
                    "weighted_score",
                    "mention_count",
                    "mapped_evidence_count",
                    "positive_count",
                    "neutral_count",
                    "mixed_count",
                    "negative_count",
                    "representative_evidence",
                ]
            ].copy()
            display = display.rename(
                columns={
                    "case_type": "구분",
                    "cafe_name": "장소명",
                    "factor_score": "장소성 요인 점수",
                    "mention_share": "언급 비중",
                    "weighted_score": "언급비중 반영 점수",
                    "mention_count": "요인 근거 수",
                    "mapped_evidence_count": "전체 근거 수",
                    "positive_count": "긍정",
                    "neutral_count": "중립",
                    "mixed_count": "혼합",
                    "negative_count": "부정",
                    "representative_evidence": "대표 근거 구절",
                }
            )
            display["장소성 요인 점수"] = display["장소성 요인 점수"].map(format_signed_decimal)
            display["언급 비중"] = display["언급 비중"].map(format_percent)
            display["언급비중 반영 점수"] = display["언급비중 반영 점수"].map(format_signed_decimal)
            st.dataframe(display, use_container_width=True, hide_index=True)

            st.markdown("##### 사례별 근거 구절 확인")
            case_options = [
                f"{row.case_type} · {row.cafe_name}"
                for row in combined_cases.itertuples()
            ]
            selected_case_label = st.selectbox("근거를 볼 사례", case_options, key="validation_case_detail")
            selected_case = combined_cases.iloc[case_options.index(selected_case_label)]
            evidence_rows = scored_evidence[
                (scored_evidence["cafe_name"] == selected_case["cafe_name"])
                & (scored_evidence["factor"] == selected_factor)
            ].copy()
            ascending = selected_case["case_type"] == "하위"
            evidence_rows = evidence_rows.sort_values(
                "sentiment_value",
                ascending=ascending,
            )
            evidence_display = evidence_rows[
                ["sentiment_label", "sentiment_value", "evidence", "review_text"]
            ].rename(
                columns={
                    "sentiment_label": "평가 방향",
                    "sentiment_value": "구절별 점수",
                    "evidence": "근거 구절",
                    "review_text": "리뷰",
                }
            )
            evidence_display["구절별 점수"] = evidence_display["구절별 점수"].map(format_signed_score)
            st.dataframe(evidence_display, use_container_width=True, hide_index=True)

    with cluster_tab:
        st.markdown("##### 장소성 점수 기반 유형화")
        st.caption("각 장소를 10개 장소성 요인 점수 벡터로 표현하고, 유사한 장소성 특성을 가진 장소끼리 군집화합니다.")

        c1, c2 = st.columns([1, 1])
        with c1:
            score_option = st.radio(
                "군집분석 기준",
                ["언급비중 반영 점수", "장소성 요인 점수"],
                horizontal=True,
                key="cluster_score_option",
            )
        with c2:
            cluster_min_mentions = st.slider(
                "군집분석 최소 전체 근거 수",
                min_value=1,
                max_value=max(1, int(place_scores["mapped_evidence_count"].max())),
                value=min(10, max(1, int(place_scores["mapped_evidence_count"].max()))),
                step=1,
                key="cluster_min_mentions",
            )

        score_column = "weighted_score" if score_option == "언급비중 반영 점수" else "factor_score"
        feature_matrix = build_cluster_feature_matrix(
            factor_scores,
            place_scores,
            score_column=score_column,
            min_total_mentions=cluster_min_mentions,
        )
        if len(feature_matrix) < 3:
            st.warning("군집분석을 수행할 장소 수가 부족합니다. 최소 전체 근거 수를 낮춰보세요.")
            return

        k_eval = evaluate_cluster_range(feature_matrix, k_min=2, k_max=8)
        if k_eval.empty:
            st.warning("군집 수를 평가하기에 충분한 데이터가 없습니다.")
            return

        best_k = int(k_eval.sort_values("silhouette_score", ascending=False).iloc[0]["k"])
        best_score = float(k_eval.sort_values("silhouette_score", ascending=False).iloc[0]["silhouette_score"])
        c1, c2, c3 = st.columns(3)
        c1.metric("군집분석 대상 장소", format_count_unit(len(feature_matrix), "개"))
        c2.metric("추천 군집 수", f"{best_k}개")
        c3.metric("최고 Silhouette", f"{best_score:.3f}")

        line_chart = px.line(
            k_eval,
            x="k",
            y="silhouette_score",
            markers=True,
            labels={"k": "군집 수", "silhouette_score": "Silhouette score"},
            height=300,
        )
        line_chart.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(line_chart, use_container_width=True)

        selected_k = st.slider(
            "적용할 군집 수",
            min_value=int(k_eval["k"].min()),
            max_value=int(k_eval["k"].max()),
            value=best_k,
            step=1,
            key="selected_cluster_k",
        )
        cluster_result = run_place_clustering(
            feature_matrix,
            place_scores,
            n_clusters=selected_k,
        )

        scatter = px.scatter(
            cluster_result.clustered_places,
            x="pc1",
            y="pc2",
            color="cluster_label",
            hover_data=["cafe_name", "placeness_score", "mapped_evidence_count", "most_mentioned_factor"],
            labels={"pc1": "PC1", "pc2": "PC2", "cluster_label": "장소성 유형"},
            height=460,
        )
        scatter.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(scatter, use_container_width=True)

        summary = cluster_result.cluster_summary.copy()
        summary_display = summary.rename(
            columns={
                "cluster_label": "군집",
                "place_count": "장소 수",
                "avg_placeness_score": "평균 언급비중 반영 점수",
                "avg_mapped_evidence_count": "평균 근거 수",
                "top_factors": "상위 요인",
                "low_factors": "하위 요인",
            }
        )[
            ["군집", "장소 수", "평균 언급비중 반영 점수", "평균 근거 수", "상위 요인", "하위 요인"]
        ]
        summary_display["평균 언급비중 반영 점수"] = summary_display["평균 언급비중 반영 점수"].map(format_signed_decimal)
        summary_display["평균 근거 수"] = summary_display["평균 근거 수"].map(lambda value: f"{float(value):.1f}")
        st.dataframe(summary_display, use_container_width=True, hide_index=True)

        selected_cluster_label = st.selectbox(
            "상세히 볼 장소성 유형",
            summary["cluster_label"].tolist(),
            key="selected_cluster_label",
        )
        selected_cluster = int(summary[summary["cluster_label"] == selected_cluster_label].iloc[0]["cluster"])

        profile = cluster_result.cluster_profiles[
            cluster_result.cluster_profiles["cluster"] == selected_cluster
        ][FACTOR_ORDER].T.reset_index()
        profile.columns = ["factor", "mean_score"]
        profile["factor_category"] = profile["factor"].map(FACTOR_CATEGORY)
        profile_chart = px.bar(
            profile,
            x="factor",
            y="mean_score",
            color="factor_category",
            color_discrete_map=FACTOR_CATEGORY_COLORS,
            labels={"factor": "장소성 요인", "mean_score": "군집 평균 점수", "factor_category": "구분"},
            height=360,
        )
        profile_chart.update_layout(yaxis_range=[-1, 1], margin=dict(l=10, r=10, t=20, b=10))
        profile_chart.add_hline(y=0, line_width=1, line_dash="dash", line_color="#98a2b3")
        st.plotly_chart(profile_chart, use_container_width=True)

        representatives = representative_places_for_cluster(
            cluster_result,
            factor_scores,
            scored_evidence,
            cluster=selected_cluster,
            limit=10,
        )
        rep_display = representatives.rename(
            columns={
                "cafe_name": "대표 장소",
                "distance_to_center": "군집 중심 거리",
                "placeness_score": "언급비중 반영 점수",
                "mapped_evidence_count": "근거 수",
                "dominant_factor": "대표 요인",
                "dominant_weighted_score": "대표 요인 가중 점수",
                "representative_evidence": "대표 근거 구절",
            }
        )[
            ["대표 장소", "군집 중심 거리", "언급비중 반영 점수", "근거 수", "대표 요인", "대표 요인 가중 점수", "대표 근거 구절"]
        ]
        rep_display["군집 중심 거리"] = rep_display["군집 중심 거리"].map(lambda value: f"{float(value):.3f}")
        rep_display["언급비중 반영 점수"] = rep_display["언급비중 반영 점수"].map(format_signed_decimal)
        rep_display["대표 요인 가중 점수"] = rep_display["대표 요인 가중 점수"].map(format_signed_decimal)
        st.dataframe(rep_display, use_container_width=True, hide_index=True)


def main() -> None:
    inject_css()
    st.title("공간 리뷰 텍스트 기반 장소성 정량화")

    demo_paths = [
        DEFAULT_SCORED_EVIDENCE_PARQUET,
        DEFAULT_FACTOR_SCORES_PARQUET,
        DEFAULT_PLACE_SCORES_PARQUET,
    ]
    missing_paths = [path for path in demo_paths if not path.exists()]
    if missing_paths:
        st.error(
            "앱용 점수 데이터가 없습니다. "
            "`python scripts/build_demo_scoring_data.py`를 먼저 실행하세요.\n\n"
            + "\n".join(str(path) for path in missing_paths)
        )
        return

    scored_evidence, factor_scores, place_scores = load_demo_data(
        str(DEFAULT_SCORED_EVIDENCE_PARQUET),
        str(DEFAULT_FACTOR_SCORES_PARQUET),
        str(DEFAULT_PLACE_SCORES_PARQUET),
        SCORING_CACHE_VERSION,
    )
    source_summary = load_source_data_summary()

    with st.sidebar:
        analysis_review_count = scored_evidence["review_index"].nunique()
        analysis_place_count = place_scores["cafe_name"].nunique()
        evidence_count = len(scored_evidence)

        st.header("데이터 흐름")
        st.caption("서울시 카페 리뷰에서 장소성 후보를 선별하고, 리뷰 구절을 10개 장소성 요인과 평가 방향에 매핑했습니다.")
        st.markdown(
            f"""
            <div class="sidebar-focus">
                <p class="sidebar-focus-title">분석 대상</p>
                <div class="sidebar-focus-grid">
                    <div>
                        <div class="sidebar-focus-label">장소</div>
                        <div class="sidebar-focus-value">{format_count(analysis_place_count)}<span class="unit">곳</span></div>
                    </div>
                    <div>
                        <div class="sidebar-focus-label">장소성 관련 리뷰</div>
                        <div class="sidebar-focus-value">{format_count(analysis_review_count)}<span class="unit">건</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p class="sidebar-flow-note">
            계산 기준: 긍정 +1, 중립/혼합 0, 부정 -1<br>
            요인 점수는 -1에서 +1 범위로 산출합니다.
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<p class="sidebar-step-title">1. 리뷰 데이터</p>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sidebar-secondary-list">
                <div class="sidebar-secondary-row">
                    <span class="sidebar-secondary-label">리뷰 보유 장소</span>
                    <span class="sidebar-secondary-value">{format_count_unit(source_summary["sample_place_count"], "개")}</span>
                </div>
                <div class="sidebar-secondary-row">
                    <span class="sidebar-secondary-label">전체 리뷰</span>
                    <span class="sidebar-secondary-value">{format_count_unit(source_summary["collected_review_count"], "건")}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<p class="sidebar-step-title">2. 장소성 매핑 데이터</p>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sidebar-secondary-list">
                <div class="sidebar-secondary-row">
                    <span class="sidebar-secondary-label">매핑 근거 구절</span>
                    <span class="sidebar-secondary-value">{format_count_unit(evidence_count, "개")}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.header("장소 선택")
        district_counts = place_scores.groupby("district")["place_id"].nunique()
        district_options = sorted(
            district_counts.index.astype(str).tolist()
        )
        default_district = "마포구" if "마포구" in district_options else district_options[0]
        selected_district = st.selectbox(
            "행정구 선택",
            district_options,
            index=district_options.index(default_district),
            format_func=lambda district: f"{district} ({int(district_counts[district]):,}곳)",
            key="sidebar_district",
        )

        district_places = place_scores[
            place_scores["district"].eq(selected_district)
        ].copy()
        district_places = district_places.sort_values(
            ["mapped_evidence_count", "mentioned_factor_count", "cafe_name"],
            ascending=[False, False, True],
        )
        place_lookup = district_places.set_index("place_id", drop=False)
        place_ids = district_places["place_id"].tolist()

        def format_place_option(place_id: str) -> str:
            row = place_lookup.loc[place_id]
            return (
                f"{row['source_cafe_name']} · {row['neighborhood']} "
                f"(근거 {int(row['mapped_evidence_count']):,}개)"
            )

        selected_place_id = st.selectbox(
            "장소 선택",
            place_ids,
            format_func=format_place_option,
            key="sidebar_place",
        )
        cafe_name = str(place_lookup.loc[selected_place_id, "cafe_name"])

    tab1, tab2, tab3, tab4, tab6 = st.tabs(
        [
            "평가 체계",
            "리뷰 매핑",
            "점수 계산",
            "개인화 추천",
            "검증 분석",
        ]
    )
    with tab1:
        render_factor_system()
    with tab2:
        render_mapping_results(scored_evidence, cafe_name)
    with tab3:
        render_score_results(scored_evidence, factor_scores, place_scores, cafe_name)
    with tab4:
        render_personalized_recommendation(scored_evidence, factor_scores, place_scores)
    with tab6:
        render_validation_analysis(scored_evidence, factor_scores, place_scores)


if __name__ == "__main__":
    main()
