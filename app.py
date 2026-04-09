"""
Streamlit app entrypoint for the placeness demo.
This version runs in viewer mode and reads precomputed local CSV files.
"""
import sys
import warnings
from pathlib import Path

import streamlit as st


project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.config import GOOGLE_REVIEW_SAMPLE_CSV, PLACENESS_METRICS_CSV_CANDIDATES
from modules.preprocess import load_data
from modules.ui import (
    preload_result_csvs_to_session_state,
    render_cafe_factor_analysis,
    render_cafe_recommendation,
    render_data_preview,
    render_detailed_results,
    render_multimodal_space_demo,
    render_placeness_calculation,
    render_sentiment_analysis_saved,
)


st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")


def init_session_state():
    if "df_review_scores" not in st.session_state:
        st.session_state.df_review_scores = None
    if "df_reviews_with_sentiment" not in st.session_state:
        st.session_state.df_reviews_with_sentiment = None
    if "df_final_metrics" not in st.session_state:
        st.session_state.df_final_metrics = None
    if "df_avg_sentiment" not in st.session_state:
        st.session_state.df_avg_sentiment = None
    if "df_place_scores" not in st.session_state:
        st.session_state.df_place_scores = None
    if "precomputed_results_loaded" not in st.session_state:
        st.session_state.precomputed_results_loaded = False
    if "precomputed_metrics_csv" not in st.session_state:
        st.session_state.precomputed_metrics_csv = None
    if "precomputed_review_scores_csv" not in st.session_state:
        st.session_state.precomputed_review_scores_csv = None
    if "precomputed_reviews_with_sentiment_csv" not in st.session_state:
        st.session_state.precomputed_reviews_with_sentiment_csv = None
    if "precomputed_avg_sentiment_csv" not in st.session_state:
        st.session_state.precomputed_avg_sentiment_csv = None
    st.session_state.pop("preview_sentiment_result", None)
    if "multimodal_vlm_result" not in st.session_state:
        st.session_state.multimodal_vlm_result = None
    if "multimodal_vlm_error" not in st.session_state:
        st.session_state.multimodal_vlm_error = None
    if "multimodal_vlm_fingerprint" not in st.session_state:
        st.session_state.multimodal_vlm_fingerprint = None


def main():
    st.title("텍스트 리뷰 데이터 기반 공간 정량화 연구")

    init_session_state()
    file_path = GOOGLE_REVIEW_SAMPLE_CSV

    if not file_path.exists():
        st.error(f"리뷰 데이터 파일을 찾을 수 없습니다: {file_path.name}")
        st.info(f"예상 경로: {file_path}")
        return

    try:
        df_reviews = load_data(file_path, cache_version="viewer-v1")
    except Exception as e:
        st.error(f"리뷰 데이터 로드 중 오류가 발생했습니다: {e}")
        return

    if df_reviews.empty:
        st.warning("로드된 리뷰 데이터가 없습니다.")
        return

    preload_result_csvs_to_session_state()

    if st.session_state.df_final_metrics is None:
        st.warning(
            "저장된 결과 CSV를 찾지 못했습니다. "
            f"프로젝트 루트에 {', '.join(PLACENESS_METRICS_CSV_CANDIDATES)} 중 하나가 필요합니다."
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "카페 추천",
            "카페별 요인 분석",
            "멀티모달 데모",
            "결과 데이터",
            "원본 리뷰 데이터",
        ]
    )

    with tab1:
        render_cafe_recommendation()

    with tab2:
        render_cafe_factor_analysis()

    with tab3:
        render_multimodal_space_demo(df_reviews)

    with tab4:
        render_placeness_calculation(df_reviews, None, None, None)
        render_sentiment_analysis_saved(df_reviews, None, None)
        render_detailed_results()

    with tab5:
        render_data_preview(file_path, None, None, tab_suffix="_raw")


if __name__ == "__main__":
    main()
