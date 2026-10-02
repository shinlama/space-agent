from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FULL_REVIEW_SUMMARY_JSON = PROJECT_ROOT / "data" / "google_reviews_clear_mismatch_v6.summary.json"
SCORING_CACHE_VERSION = "gpt54nano_signed_score_v6_thesis_ui_20261002"

from modules.research_scoring import (  # noqa: E402
    DEFAULT_FACTOR_SCORES_PARQUET,
    DEFAULT_PLACE_SCORES_PARQUET,
    DEFAULT_SCORED_EVIDENCE_PARQUET,
    FACTOR_ORDER,
    complete_factor_table,
)


FACTOR_LABELS = {
    "심미성": "Aesthetics",
    "개방성": "Openness",
    "감각적 경험": "Sensory Experience",
    "접근성": "Accessibility",
    "쾌적성": "Comfort",
    "활동성": "Activity",
    "상호작용성": "Sociability",
    "상징성": "Symbolism",
    "기억 및 선호": "Memory and Preference",
    "지역 정체성": "Local Identity",
}

CHART_FACTOR_LABELS = {
    "심미성": "Aesthetics",
    "개방성": "Openness",
    "감각적 경험": "Sensory<br>Experience",
    "접근성": "Accessibility",
    "쾌적성": "Comfort",
    "활동성": "Activity",
    "상호작용성": "Sociability",
    "상징성": "Symbolism",
    "기억 및 선호": "Memory and<br>Preference",
    "지역 정체성": "Local<br>Identity",
}

CATEGORY_LABELS = {
    "물리적 특성": "Physical",
    "활동적 특성": "Activity-related",
    "의미적 특성": "Meaning-related",
}

CATEGORY_COLORS = {
    "Physical": "#4E9FDB",
    "Activity-related": "#7CC443",
    "Meaning-related": "#F3AA2B",
}

DISTRICT_LABELS = {
    "강남구": "Gangnam-gu",
    "강동구": "Gangdong-gu",
    "강북구": "Gangbuk-gu",
    "강서구": "Gangseo-gu",
    "관악구": "Gwanak-gu",
    "광진구": "Gwangjin-gu",
    "구로구": "Guro-gu",
    "금천구": "Geumcheon-gu",
    "노원구": "Nowon-gu",
    "도봉구": "Dobong-gu",
    "동대문구": "Dongdaemun-gu",
    "동작구": "Dongjak-gu",
    "마포구": "Mapo-gu",
    "서대문구": "Seodaemun-gu",
    "서초구": "Seocho-gu",
    "성동구": "Seongdong-gu",
    "성북구": "Seongbuk-gu",
    "송파구": "Songpa-gu",
    "양천구": "Yangcheon-gu",
    "영등포구": "Yeongdeungpo-gu",
    "용산구": "Yongsan-gu",
    "은평구": "Eunpyeong-gu",
    "종로구": "Jongno-gu",
    "중구": "Jung-gu",
    "중랑구": "Jungnang-gu",
}


st.set_page_config(
    page_title="Placeness Quantification",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    },
)


@st.cache_resource(show_spinner="Loading calculated results...")
def load_score_data(
    factor_parquet: str,
    place_parquet: str,
    scoring_version: str = SCORING_CACHE_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ = scoring_version
    return pd.read_parquet(factor_parquet), pd.read_parquet(place_parquet)


@st.cache_data(show_spinner=False, max_entries=64)
def load_place_evidence(
    evidence_parquet: str,
    cafe_name: str,
    scoring_version: str = SCORING_CACHE_VERSION,
) -> pd.DataFrame:
    _ = scoring_version
    return pd.read_parquet(
        evidence_parquet,
        filters=[("cafe_name", "==", cafe_name)],
    )


@st.cache_data(show_spinner=False)
def load_scoring_summary() -> dict[str, object]:
    summary_path = DEFAULT_SCORED_EVIDENCE_PARQUET.parent / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_source_summary() -> dict[str, int | None]:
    summary = {"places": None, "reviews": None}
    if not FULL_REVIEW_SUMMARY_JSON.exists():
        return summary

    source = json.loads(FULL_REVIEW_SUMMARY_JSON.read_text(encoding="utf-8"))
    summary["places"] = int(source["output_places"])
    summary["reviews"] = int(source["output_rows"])
    return summary


def format_count(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}"


def format_score(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):+.2f}"


def format_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #667085;
            --line: #DCE3EB;
            --soft: #F5F7FA;
            --navy: #173A63;
        }

        html, body, [class*="css"] {
            font-family: Arial, "Helvetica Neue", sans-serif;
            color: var(--ink);
        }

        .block-container {
            max-width: 1540px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: #F4F6F9;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.25rem;
        }

        [data-testid="stMetric"] {
            border-left: 2px solid #C7D2DF;
            padding-left: 0.9rem;
        }

        [data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.84rem;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 1.75rem;
        }

        .page-subtitle {
            color: var(--muted);
            font-size: 1rem;
            margin: -0.45rem 0 1.15rem;
        }

        .case-heading {
            border-top: 1px solid var(--line);
            border-bottom: 1px solid var(--line);
            padding: 0.85rem 0 0.8rem;
            margin-bottom: 1.05rem;
        }

        .case-kicker {
            color: var(--navy);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .case-title {
            color: var(--ink);
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.25;
            margin-top: 0.2rem;
        }

        .case-meta {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.18rem;
        }

        .sidebar-section {
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 0.35rem 0 0.55rem;
        }

        .data-group {
            border-top: 1px solid #D4DCE6;
            padding-top: 0.55rem;
            margin-top: 0.45rem;
        }

        .data-group-title {
            color: #344054;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .data-row {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.2rem 0;
            color: var(--muted);
            font-size: 0.8rem;
        }

        .data-row strong {
            color: var(--ink);
            font-weight: 700;
            white-space: nowrap;
        }

        .score-note {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.5;
            margin: 0.15rem 0 0.9rem;
        }

        .category-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            align-items: center;
            color: var(--muted);
            font-size: 0.82rem;
            margin: 0.15rem 0 0.2rem;
        }

        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.38rem;
        }

        .legend-swatch {
            width: 0.72rem;
            height: 0.72rem;
            display: inline-block;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 4px;
            box-shadow: none;
        }

        #MainMenu,
        footer,
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        .stDeployButton {
            display: none !important;
        }

        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
        }

        h1, h2, h3, p {
            letter-spacing: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_summary(
    source_summary: dict[str, int | None],
    scoring_summary: dict[str, object],
    place_scores: pd.DataFrame,
    factor_scores: pd.DataFrame,
) -> None:
    mapped_places = int(scoring_summary.get("mapped_places", place_scores["place_id"].nunique()))
    mapped_reviews = scoring_summary.get("mapped_reviews")
    evidence_phrases = int(
        scoring_summary.get("mapping_rows", factor_scores["mention_count"].sum())
    )

    st.markdown('<div class="sidebar-section">Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="data-group">
            <div class="data-group-title">Review dataset</div>
            <div class="data-row"><span>Places</span><strong>{format_count(source_summary['places'])}</strong></div>
            <div class="data-row"><span>Reviews</span><strong>{format_count(source_summary['reviews'])}</strong></div>
        </div>
        <div class="data-group">
            <div class="data-group-title">Mapped data</div>
            <div class="data-row"><span>Places</span><strong>{format_count(mapped_places)}</strong></div>
            <div class="data-row"><span>Reviews</span><strong>{format_count(mapped_reviews)}</strong></div>
            <div class="data-row"><span>Evidence phrases</span><strong>{format_count(evidence_phrases)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def select_place(place_scores: pd.DataFrame) -> tuple[str, str, str, pd.Series]:
    st.markdown(
        '<div class="sidebar-section" style="margin-top:1.4rem;">Place selection</div>',
        unsafe_allow_html=True,
    )

    district_counts = place_scores.groupby("district")["place_id"].nunique()
    district_options = sorted(
        district_counts.index.astype(str).tolist(),
        key=lambda value: DISTRICT_LABELS.get(value, value),
    )
    default_district = "마포구" if "마포구" in district_options else district_options[0]
    district = st.selectbox(
        "District",
        district_options,
        index=district_options.index(default_district),
        format_func=lambda value: (
            f"{DISTRICT_LABELS.get(value, value)} ({int(district_counts[value]):,} places)"
        ),
    )

    district_places = place_scores[place_scores["district"].eq(district)].copy()
    district_places = district_places.sort_values(
        ["mapped_evidence_count", "mentioned_factor_count", "place_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    district_places["anonymous_name"] = [
        f"Example Café {index:03d}" for index in range(1, len(district_places) + 1)
    ]
    lookup = district_places.set_index("place_id", drop=False)

    def format_option(place_id: str) -> str:
        row = lookup.loc[place_id]
        return f"{row['anonymous_name']} · {int(row['mapped_evidence_count']):,} evidence phrases"

    selected_id = st.selectbox(
        "Place",
        district_places["place_id"].tolist(),
        format_func=format_option,
    )
    selected = lookup.loc[selected_id]
    return (
        str(selected["cafe_name"]),
        str(selected["anonymous_name"]),
        DISTRICT_LABELS.get(district, district),
        selected,
    )


def prepare_factor_table(factor_scores: pd.DataFrame, cafe_name: str) -> pd.DataFrame:
    table = complete_factor_table(factor_scores, cafe_name).copy()
    table["Factor"] = table["factor"].map(FACTOR_LABELS)
    table["Chart factor"] = table["factor"].map(CHART_FACTOR_LABELS)
    table["Category"] = table["factor_category"].map(CATEGORY_LABELS)
    table["Factor score"] = table["factor_score"]
    table["Mention proportion"] = table["mention_share"].fillna(0)
    table["Neutral / Mixed"] = table["neutral_count"] + table["mixed_count"]
    return table


def style_chart(figure, y_title: str, y_range: list[float] | None = None) -> None:
    figure.update_traces(
        marker_line_color="rgba(255,255,255,0.9)",
        marker_line_width=0.8,
        textposition="outside",
        cliponaxis=False,
        hovertemplate=None,
    )
    figure.update_layout(
        height=430,
        showlegend=False,
        margin=dict(l=12, r=12, t=55, b=24),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, Helvetica Neue, sans-serif", color="#4B5565", size=12),
        title=dict(font=dict(size=19, color="#172033"), x=0.01, xanchor="left"),
        xaxis=dict(
            title=None,
            categoryorder="array",
            categoryarray=[CHART_FACTOR_LABELS[factor] for factor in FACTOR_ORDER],
            tickfont=dict(size=11, color="#4B5565"),
            showgrid=False,
            linecolor="#C8D1DC",
        ),
        yaxis=dict(
            title=y_title,
            range=y_range,
            gridcolor="#E6EBF1",
            zeroline=False,
            linecolor="#C8D1DC",
        ),
    )


def render_charts(table: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="category-legend">
            <span class="legend-item"><span class="legend-swatch" style="background:#4E9FDB;"></span>Physical</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#7CC443;"></span>Activity-related</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#F3AA2B;"></span>Meaning-related</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_chart = px.bar(
        table,
        x="Chart factor",
        y="Factor score",
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        text=table["Factor score"].map(format_score),
        title="Factor Scores",
    )
    style_chart(score_chart, "Factor score", [-1.08, 1.08])
    score_chart.add_hline(y=0, line_width=1.2, line_color="#7C8796")

    mention_max = max(0.1, float(table["Mention proportion"].max()) * 1.32)
    mention_chart = px.bar(
        table,
        x="Chart factor",
        y="Mention proportion",
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        text=table["Mention proportion"].map(format_percent),
        title="Mention Proportions",
    )
    style_chart(mention_chart, "Mention proportion", [0, mention_max])
    mention_chart.update_yaxes(tickformat=".0%")

    st.plotly_chart(score_chart, use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(mention_chart, use_container_width=True, config={"displayModeBar": False})


def render_summary_table(table: pd.DataFrame) -> None:
    st.subheader("Factor-level Summary")
    display = table[
        [
            "Category",
            "Factor",
            "mention_count",
            "positive_count",
            "Neutral / Mixed",
            "negative_count",
            "Factor score",
            "Mention proportion",
        ]
    ].copy()
    display = display.rename(
        columns={
            "mention_count": "Evidence phrases",
            "positive_count": "Positive",
            "negative_count": "Negative",
        }
    )
    display["Factor score"] = display["Factor score"].map(format_score)
    display["Mention proportion"] = display["Mention proportion"].map(format_percent)
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=390,
    )


def main() -> None:
    inject_css()

    required_paths = [
        DEFAULT_SCORED_EVIDENCE_PARQUET,
        DEFAULT_FACTOR_SCORES_PARQUET,
        DEFAULT_PLACE_SCORES_PARQUET,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        st.error(
            "Calculated data files are missing. Run "
            "`python scripts/build_demo_scoring_data.py` before starting the app.\n\n"
            + "\n".join(str(path) for path in missing_paths)
        )
        return

    factor_scores, place_scores = load_score_data(
        str(DEFAULT_FACTOR_SCORES_PARQUET),
        str(DEFAULT_PLACE_SCORES_PARQUET),
    )
    source_summary = load_source_summary()
    scoring_summary = load_scoring_summary()

    with st.sidebar:
        render_dataset_summary(source_summary, scoring_summary, place_scores, factor_scores)
        cafe_name, anonymous_name, district_label, selected_place = select_place(place_scores)

    place_evidence = load_place_evidence(str(DEFAULT_SCORED_EVIDENCE_PARQUET), cafe_name)
    review_count = place_evidence["review_index"].nunique()
    evidence_count = int(selected_place["mapped_evidence_count"])
    mentioned_factor_count = int(selected_place["mentioned_factor_count"])
    most_mentioned_factor = FACTOR_LABELS.get(
        str(selected_place["most_mentioned_factor"]),
        str(selected_place["most_mentioned_factor"]),
    )

    st.title("Placeness Quantification from Spatial Review Text")
    st.markdown(
        '<p class="page-subtitle">Place-level factor scores and mention proportions derived from user reviews</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="case-heading">
            <div class="case-kicker">Selected place</div>
            <div class="case-title">{anonymous_name}</div>
            <div class="case-meta">{district_label} · Place name anonymized for presentation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Mapped reviews", format_count(review_count))
    metric_cols[1].metric("Evidence phrases", format_count(evidence_count))
    metric_cols[2].metric("Factors mentioned", f"{mentioned_factor_count} / 10")
    metric_cols[3].metric("Most mentioned factor", most_mentioned_factor)
    st.markdown(
        """
        <p class="score-note">
        Factor score shows the direction of evaluation from -1 (negative) to +1 (positive).
        Mention proportion shows the share of evidence phrases assigned to each factor.
        </p>
        """,
        unsafe_allow_html=True,
    )

    table = prepare_factor_table(factor_scores, cafe_name)
    render_charts(table)

    with st.expander("Calculation definitions"):
        st.latex(r"S_{p,f}=\frac{N^{+}_{p,f}-N^{-}_{p,f}}{N_{p,f}}")
        st.caption(
            "Factor score (S): positive evidence phrases receive +1, neutral or mixed phrases receive 0, "
            "and negative phrases receive -1."
        )
        st.latex(r"M_{p,f}=\frac{N_{p,f}}{\sum_{j=1}^{10}N_{p,j}}")
        st.caption(
            "Mention proportion (M): the number of evidence phrases assigned to a factor divided by all "
            "placeness evidence phrases for the selected place."
        )

    render_summary_table(table)


if __name__ == "__main__":
    main()
