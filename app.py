"""
장소성 기반 공간 정량 평가 시스템 (LLM & BERT)
엔트리 포인트
"""
import sys
import streamlit as st
import warnings
from pathlib import Path

# Streamlit Cloud에서 모듈 경로 문제 해결
# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.config import GOOGLE_REVIEW_SAMPLE_CSV, ALL_FACTORS
from modules.model_loader import load_models
from modules.preprocess import load_data
from modules.ui import (
    render_data_preview,
    render_placeness_calculation,
    render_sentiment_analysis,
    render_detailed_results,
    render_cafe_recommendation,
    render_multimodal_space_demo,
)

# render_cafe_factor_analysis는 선택적 import (배포 환경 호환성)
try:
    from modules.ui import render_cafe_factor_analysis
except ImportError:
    # 함수가 없는 경우 대체 함수 정의
    def render_cafe_factor_analysis():
        st.error("⚠️ 카페별 요인 분석 기능을 사용할 수 없습니다. modules/ui.py 파일을 확인해주세요.")
        st.info("이 기능은 placeness_final_research_metrics (3).csv 파일이 필요합니다.")

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
    if 'multimodal_vlm_result' not in st.session_state:
        st.session_state.multimodal_vlm_result = None
    if 'multimodal_vlm_error' not in st.session_state:
        st.session_state.multimodal_vlm_error = None
    if 'multimodal_vlm_fingerprint' not in st.session_state:
        st.session_state.multimodal_vlm_fingerprint = None


def main():
    """메인 함수"""
    st.title("텍스트 리뷰 데이터 기반 공간 정량화 도구")
    
    # 세션 상태 초기화
    init_session_state()
    
    # 파일 경로 설정
    file_path = GOOGLE_REVIEW_SAMPLE_CSV
    
    # 1. 모델 로드
    with st.spinner("🤖 AI 모델 로드 중... (처음 실행 시 다운로드로 인해 시간이 걸릴 수 있습니다)"):
        try:
            sbert_model, sentiment_pipeline, sentiment_model_name = load_models()
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
        # cache_version을 변경하면 캐시가 무효화됩니다 (카페명 위치 정보 추가 로직 적용)
        df_reviews = load_data(file_path, cache_version="v2.1")
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return
    
    if df_reviews.empty:
        st.warning("로드된 리뷰 데이터가 없습니다.")
        return
    
    # 탭 구조 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "☕ 카페 추천",
        "📈 카페별 요인 분석",
        "🖼️ 멀티모달 데모",
        "📊 데이터 분석",
        "📋 리뷰 데이터"
    ])
    
    with tab1:
        # 카페 추천
        render_cafe_recommendation()
    
    with tab2:
        # 카페별 요인 점수 분석
        render_cafe_factor_analysis()
    
    with tab3:
        render_multimodal_space_demo(df_reviews)
    
    with tab4:
        # 3. 데이터 미리보기
        render_data_preview(file_path, sentiment_pipeline, sentiment_model_name, tab_suffix="_tab3")
        
        # 4. 장소성 요인 점수 계산
        render_placeness_calculation(df_reviews, sbert_model, sentiment_pipeline, sentiment_model_name)
        
        # 5. 개별 리뷰 감성 분석
        render_sentiment_analysis(df_reviews, sentiment_pipeline, sentiment_model_name)
    
    with tab5:
        # 데이터 미리보기만 표시
        render_data_preview(file_path, sentiment_pipeline, sentiment_model_name, tab_suffix="_tab4")


if __name__ == "__main__":
    main()
