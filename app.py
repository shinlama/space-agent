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
SCORING_CACHE_VERSION = "gpt54nano_factor_mean_korean_20261014"

from modules.research_scoring import (
    DEFAULT_FACTOR_SCORES_PARQUET,
    DEFAULT_PLACE_SCORES_PARQUET,
    DEFAULT_SCORED_EVIDENCE_PARQUET,
    FACTOR_CATEGORIES,
    FACTOR_DETAILS,
    FACTOR_ORDER,
    complete_factor_table,
)
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


@st.cache_resource(show_spinner="장소성 점수 계산 결과를 불러오는 중입니다.")
def load_demo_data(
    factor_parquet: str,
    place_parquet: str,
    scoring_version: str = SCORING_CACHE_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ = scoring_version
    return (
        pd.read_parquet(factor_parquet).drop(
            columns=["mention_share", "weighted_score"], errors="ignore"
        ),
        pd.read_parquet(place_parquet).drop(
            columns=["placeness_score", "mean_factor_score"], errors="ignore"
        ),
    )


@st.cache_data(show_spinner=False, max_entries=64)
def load_scored_evidence(
    evidence_parquet: str,
    cafe_names: tuple[str, ...],
    factors: tuple[str, ...] = (),
    scoring_version: str = SCORING_CACHE_VERSION,
) -> pd.DataFrame:
    _ = scoring_version
    if not cafe_names:
        return pd.DataFrame()

    filters: list[tuple[str, str, object]] = [
        ("cafe_name", "in", list(cafe_names)),
    ]
    if factors:
        filters.append(("factor", "in", list(factors)))
    return pd.read_parquet(evidence_parquet, filters=filters)


@st.cache_data(show_spinner=False)
def load_scoring_summary() -> dict[str, object]:
    summary_path = DEFAULT_SCORED_EVIDENCE_PARQUET.parent / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


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
        @media (max-width: 600px) {
            h1 {
                font-size: 2rem !important;
                line-height: 1.35 !important;
                word-break: keep-all;
            }
            h2 {
                font-size: 1.35rem !important;
                line-height: 1.4 !important;
                word-break: keep-all;
            }
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
        lambda row: f"리뷰 {int(row['review_index'])} · 매핑 {int(row['mapping_count'])}개 · {row['review_text'][:54]}",
        axis=1,
    )
    selected_label = st.selectbox("리뷰 선택", review_options["label"].tolist())
    selected_review_index = int(review_options.loc[review_options["label"] == selected_label, "review_index"].iloc[0])
    review_rows = cafe_rows[cafe_rows["review_index"] == selected_review_index].copy()
    review_text = review_rows["review_text"].iloc[0]

    st.markdown("##### 리뷰")
    st.markdown(highlight_evidence(review_text, review_rows), unsafe_allow_html=True)

    display = review_rows[
        ["evidence", "factor"]
    ].copy()
    display = display.rename(
        columns={
            "evidence": "근거 구절",
            "factor": "매핑 요인",
        }
    )
    st.markdown("##### 매핑 결과")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown(
        f'<div class="research-caption">이 리뷰에서는 총 <b>{len(review_rows)}</b>개의 구절이 장소성 요인에 매핑되었습니다.</div>',
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
    selected_evidence = scored_evidence[scored_evidence["cafe_name"] == cafe_name]
    c1, c2, c3 = st.columns(3)
    c1.metric("매핑된 리뷰 수", format_count(selected_evidence["review_index"].nunique()))
    c2.metric("매핑된 구절 수", f"{int(selected_place['mapped_evidence_count']):,}개")
    c3.metric("매핑된 요인 수", f"{int(selected_place['mentioned_factor_count'])} / 10")

    st.markdown(
        """
        <div class="formula-box">
        <b>구절별 점수</b>: 긍정 = +1, 중립/혼합 = 0, 부정 = -1<br>
        <b>장소성 요인 점수</b> = 해당 요인에 매핑된 구절 점수의 평균
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    table = complete_factor_table(factor_scores, cafe_name)
    plot_df = table.copy()
    plot_df["장소성 요인 점수"] = plot_df["factor_score"]

    chart = px.bar(
        plot_df,
        x="factor",
        y="장소성 요인 점수",
        color="factor_category",
        color_discrete_map=FACTOR_CATEGORY_COLORS,
        text=plot_df["장소성 요인 점수"].map(lambda v: f"{v:+.2f}" if pd.notna(v) else ""),
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
            "evidence_examples": "근거 예시",
        }
    )
    display["장소성 요인 점수"] = display["장소성 요인 점수"].map(format_signed_decimal)
    display["근거 예시"] = display["근거 예시"].fillna("")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("#### 요인별 구절 평가")
    selected_factor = st.selectbox("요인 선택", FACTOR_ORDER, key="scoring_factor")
    factor_evidence = selected_evidence[selected_evidence["factor"] == selected_factor]
    if factor_evidence.empty:
        st.info("해당 요인에 매핑된 구절이 없어 점수를 산출하지 않았습니다.")
    else:
        phrase_display = factor_evidence[
            ["evidence", "sentiment_label", "sentiment_value"]
        ].rename(columns={
            "evidence": "근거 구절",
            "sentiment_label": "평가 방향",
            "sentiment_value": "구절별 점수",
        })
        phrase_display["구절별 점수"] = phrase_display["구절별 점수"].map(format_signed_score)
        st.dataframe(phrase_display, use_container_width=True, hide_index=True)

    st.download_button(
        "선택 장소의 요인별 점수 데이터 다운로드",
        data=display.to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"{cafe_name}_factor_scores.csv",
        mime="text/csv",
    )
    st.download_button(
        "선택 장소의 구절별 점수 데이터 다운로드",
        data=selected_evidence.drop(columns=["confidence"], errors="ignore").to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"{cafe_name}_scored_evidence.csv",
        mime="text/csv",
    )


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

    factor_scores, place_scores = load_demo_data(
        str(DEFAULT_FACTOR_SCORES_PARQUET),
        str(DEFAULT_PLACE_SCORES_PARQUET),
        SCORING_CACHE_VERSION,
    )
    source_summary = load_source_data_summary()
    scoring_summary = load_scoring_summary()

    with st.sidebar:
        analysis_review_count = int(scoring_summary.get("mapped_reviews", 0))
        analysis_place_count = place_scores["cafe_name"].nunique()
        evidence_count = int(scoring_summary.get("mapping_rows", 0))

        st.header("데이터 흐름")
        st.caption("서울시 카페 리뷰에서 장소성 후보를 선별하고, 리뷰 구절을 10개 장소성 요인과 평가 방향에 매핑했습니다.")
        st.caption(f"매핑 모델: {scoring_summary.get('model', '확인 불가')} (v6)")
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

    selected_evidence = load_scored_evidence(
        str(DEFAULT_SCORED_EVIDENCE_PARQUET),
        (cafe_name,),
        (),
        SCORING_CACHE_VERSION,
    )

    selected_view = st.radio(
        "분석 단계",
        options=["평가 체계", "리뷰 매핑", "점수 계산"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="main_view",
    )
    if selected_view == "평가 체계":
        render_factor_system()
    elif selected_view == "리뷰 매핑":
        render_mapping_results(selected_evidence, cafe_name)
    elif selected_view == "점수 계산":
        render_score_results(selected_evidence, factor_scores, place_scores, cafe_name)


if __name__ == "__main__":
    main()
