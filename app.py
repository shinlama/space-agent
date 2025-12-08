"""
장소성 기반 공간 정량 평가 시스템 (LLM & BERT)
엔트리 포인트
"""
import streamlit as st
import warnings
from pathlib import Path

from modules.config import GOOGLE_REVIEW_SAMPLE_CSV, ALL_FACTORS
from modules.model_loader import load_models
from modules.preprocess import load_data
from modules.ui import (
    render_data_preview,
    render_placeness_calculation,
    render_sentiment_analysis,
    render_detailed_results
)

# Streamlit 페이지 설정 (wide 모드로 전체 너비 사용)
st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")


def init_session_state():
    """세션 상태 초기화"""
    if 'df_review_scores' not in st.session_state:
        st.session_state.df_review_scores = None
    if 'df_reviews_with_sentiment' not in st.session_state:
        st.session_state.df_reviews_with_sentiment = None
    if 'df_final_metrics' not in st.session_state:
        st.session_state.df_final_metrics = None
    if 'df_avg_sentiment' not in st.session_state:
        st.session_state.df_avg_sentiment = None
    if 'df_place_scores' not in st.session_state:
        st.session_state.df_place_scores = None
    if 'preview_sentiment_result' not in st.session_state:
        st.session_state.preview_sentiment_result = None


def main():
    """메인 함수"""
    st.title("장소성 기반 공간 정량 평가 시스템")
    st.markdown("---")
    
    # 세션 상태 초기화
    init_session_state()
    
    # 파일 경로 설정
    file_path = GOOGLE_REVIEW_SAMPLE_CSV
    
    # 1. 모델 로드
    with st.spinner("🤖 AI 모델 로드 중... (처음 실행 시 다운로드로 인해 시간이 걸릴 수 있습니다)"):
        try:
            sbert_model, sentiment_pipeline, sentiment_model_name = load_models()
            st.success("✅ 모델 로드 완료!")
        except Exception as e:
            st.error(f"❌ 모델 로드 실패: {e}")
            st.info("💡 해결 방법:\n"
                   "- 인터넷 연결을 확인해주세요 (모델 다운로드 필요)\n"
                   "- 잠시 후 다시 시도해주세요\n"
                   "- Streamlit Cloud의 경우 메모리 제한으로 인해 일부 모델이 로드되지 않을 수 있습니다")
            st.stop()
    
    # 2. 데이터 로드
    if not file_path.exists():
        st.error(f"⚠️ 에러: 리뷰 데이터 파일 '{file_path.name}'를 찾을 수 없습니다. 파일을 확인해주세요.")
        st.info(f"예상 경로: {file_path}")
        return
    
    try:
        df_reviews = load_data(file_path)
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return
    
    if df_reviews.empty:
        st.warning("로드된 리뷰 데이터가 없습니다.")
        return
    
    # 데이터 통계 표시
    unique_cafes = df_reviews['cafe_name'].nunique()
    reviews_per_cafe = df_reviews.groupby('cafe_name').size()
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("카페 수", f"{unique_cafes:,}개")
    with col2:
        st.metric("평균 리뷰 수/카페", f"{reviews_per_cafe.mean():.1f}개")
    with col3:
        st.metric("최대 리뷰 수/카페", f"{reviews_per_cafe.max()}개")
    
    # 3. 데이터 미리보기
    render_data_preview(file_path, sentiment_pipeline, sentiment_model_name)
    
    # 4. 장소성 요인 점수 계산
    render_placeness_calculation(df_reviews, sbert_model, sentiment_pipeline, sentiment_model_name)
    
    # 5. 개별 리뷰 감성 분석
    render_sentiment_analysis(df_reviews, sentiment_pipeline, sentiment_model_name)
    
    # 6. 리뷰별 상세 결과
    render_detailed_results()


if __name__ == "__main__":
    main()
