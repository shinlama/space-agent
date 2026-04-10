"""
Streamlit UI 구성 모듈
"""
import streamlit as st
import pandas as pd
import re
import io
import traceback
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
from modules.config import (
    ALL_FACTORS,
    CAFE_INFO_CSV,
    CAFE_AVG_SENTIMENT_CSV_CANDIDATES,
    FACTOR_CATEGORIES,
    FACTOR_NAMES,
    FACTOR_PREFERENCE_LABELS,
    LEGACY_FACTOR_MERGES,
    LEGACY_FACTOR_RENAMES,
    PLACENESS_METRICS_CSV_CANDIDATES,
    REVIEW_PLACENESS_CSV_CANDIDATES,
    REVIEWS_WITH_SENTIMENT_CSV_CANDIDATES,
    SIMILARITY_THRESHOLD,
)

# Streamlit 버전 호환성 처리
def get_dataframe_width_param():
    """
    Streamlit 버전에 따라 적절한 width 파라미터를 반환합니다.
    실제로 width='stretch'를 시도하고, 실패하면 use_container_width를 사용합니다.
    """
    # 안전하게 use_container_width 사용 (모든 버전에서 지원)
    return {'use_container_width': True}
from modules.sentiment import run_sentiment_analysis
from modules.score import calculate_place_scores, calculate_final_research_metrics
from modules.preprocess import load_csv_raw, is_numeric_only, is_metadata_only, truncate_text_for_bert
from modules.sentiment import process_sentiment_result
from modules.vision_analysis import analyze_cafe_images_with_openai, build_vlm_fingerprint

# 한글 형태소 분석 (선택적, 지연 초기화)
HAS_KONLPY = False
okt = None


def _resolve_existing_csv(candidates):
    base_dir = Path(__file__).resolve().parent.parent
    for file_name in candidates:
        path = base_dir / file_name
        if path.exists():
            return path
    return None


def _rename_legacy_factor_columns(df, source_factor, target_factor):
    rename_map = {}
    for template in [
        "점수_{}",
        "리뷰수_{}",
        "점수_{}_calc",
        "리뷰수_{}_calc",
        "Wi_{}",
    ]:
        old_col = template.format(source_factor)
        new_col = template.format(target_factor)
        if old_col in df.columns and new_col not in df.columns:
            rename_map[old_col] = new_col
    return df.rename(columns=rename_map)


def _normalize_legacy_metrics_columns(df):
    normalized = df.copy()

    for old_factor, new_factor in LEGACY_FACTOR_RENAMES.items():
        normalized = _rename_legacy_factor_columns(normalized, old_factor, new_factor)

    for new_factor, old_factors in LEGACY_FACTOR_MERGES.items():
        score_cols = [f"점수_{factor}" for factor in old_factors if f"점수_{factor}" in normalized.columns]
        if score_cols and f"점수_{new_factor}" not in normalized.columns:
            normalized[f"점수_{new_factor}"] = normalized[score_cols].mean(axis=1, skipna=True)

        count_cols = [f"리뷰수_{factor}" for factor in old_factors if f"리뷰수_{factor}" in normalized.columns]
        if count_cols and f"리뷰수_{new_factor}" not in normalized.columns:
            normalized[f"리뷰수_{new_factor}"] = normalized[count_cols].fillna(0).sum(axis=1)

        calc_score_cols = [f"점수_{factor}_calc" for factor in old_factors if f"점수_{factor}_calc" in normalized.columns]
        if calc_score_cols and f"점수_{new_factor}_calc" not in normalized.columns:
            normalized[f"점수_{new_factor}_calc"] = normalized[calc_score_cols].mean(axis=1, skipna=True)

        calc_count_cols = [f"리뷰수_{factor}_calc" for factor in old_factors if f"리뷰수_{factor}_calc" in normalized.columns]
        if calc_count_cols and f"리뷰수_{new_factor}_calc" not in normalized.columns:
            normalized[f"리뷰수_{new_factor}_calc"] = normalized[calc_count_cols].fillna(0).sum(axis=1)

        weight_cols = [f"Wi_{factor}" for factor in old_factors if f"Wi_{factor}" in normalized.columns]
        if weight_cols and f"Wi_{new_factor}" not in normalized.columns:
            normalized[f"Wi_{new_factor}"] = normalized[weight_cols].fillna(0).sum(axis=1).clip(upper=1.0)

    return normalized


@st.cache_data
def _load_metrics_dataframe():
    csv_path = _resolve_existing_csv(PLACENESS_METRICS_CSV_CANDIDATES)
    if csv_path is None:
        return None, None

    try:
        df_metrics = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        try:
            df_metrics = pd.read_csv(csv_path)
        except Exception:
            return None, None

    return _normalize_legacy_metrics_columns(df_metrics), csv_path


@st.cache_data
def _load_csv_candidates(candidates):
    csv_path = _resolve_existing_csv(candidates)
    if csv_path is None:
        return None, None

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return None, None

    return df, csv_path


def preload_result_csvs_to_session_state():
    """Preload generated result CSVs so UI tabs use the latest saved outputs by default."""
    if st.session_state.get("precomputed_results_loaded"):
        return

    df_metrics, metrics_path = _load_metrics_dataframe()
    if df_metrics is not None:
        st.session_state.df_final_metrics = df_metrics
        st.session_state.precomputed_metrics_csv = metrics_path.name if metrics_path else None

    df_review_scores, review_scores_path = _load_csv_candidates(REVIEW_PLACENESS_CSV_CANDIDATES)
    if df_review_scores is not None:
        st.session_state.df_review_scores = df_review_scores
        st.session_state.precomputed_review_scores_csv = review_scores_path.name if review_scores_path else None

    df_reviews_with_sentiment, reviews_sentiment_path = _load_csv_candidates(REVIEWS_WITH_SENTIMENT_CSV_CANDIDATES)
    if df_reviews_with_sentiment is not None:
        st.session_state.df_reviews_with_sentiment = df_reviews_with_sentiment
        st.session_state.precomputed_reviews_with_sentiment_csv = (
            reviews_sentiment_path.name if reviews_sentiment_path else None
        )

    df_avg_sentiment, avg_sentiment_path = _load_csv_candidates(CAFE_AVG_SENTIMENT_CSV_CANDIDATES)
    if df_avg_sentiment is not None:
        st.session_state.df_avg_sentiment = df_avg_sentiment
        st.session_state.precomputed_avg_sentiment_csv = avg_sentiment_path.name if avg_sentiment_path else None

    st.session_state.precomputed_results_loaded = True

def _init_konlpy():
    """konlpy를 지연 초기화합니다. Java가 없으면 None을 반환합니다."""
    global HAS_KONLPY, okt
    if HAS_KONLPY and okt is not None:
        return okt
    
    try:
        from konlpy.tag import Okt
        okt = Okt()
        HAS_KONLPY = True
        return okt
    except (ImportError, Exception) as e:
        HAS_KONLPY = False
        okt = None
        return None


def render_data_preview(file_path, sentiment_pipeline, sentiment_model_name, tab_suffix=""):
    """데이터 미리보기 섹션 렌더링
    
    Args:
        file_path: CSV 파일 경로
        sentiment_pipeline: 감성 분석 파이프라인
        sentiment_model_name: 감성 분석 모델 이름
        tab_suffix: 탭별 구분을 위한 접미사 (버튼 key 중복 방지용)
    """
    st.header("📋 리뷰 데이터 미리보기")
    
    # 전체 데이터 로드 (미리보기용, 원본 컬럼명 유지)
    df_preview = load_csv_raw(file_path)
    
    # 필요한 컬럼 확인 및 선택
    required_cols = ['상호명', '시군구명', '행정동명', '평점', '리뷰']
    available_cols = [col for col in required_cols if col in df_preview.columns]
    
    if len(available_cols) == len(required_cols):
        # 행정구별로 정렬 (시군구명, 상호명, 행정동명 순)
        df_preview_sorted = df_preview[available_cols].copy()
        df_preview_sorted = df_preview_sorted.sort_values(by=['시군구명', '상호명', '행정동명'], ascending=[True, True, True])
        
        # 표를 화면 전체 너비로 표시하기 위한 CSS 스타일
        st.markdown("""
<style>
        .stDataFrame {
            width: 100% !important;
        }
        div[data-testid="stDataFrame"] {
            width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

        st.dataframe(
            df_preview_sorted,
            hide_index=True,
            **get_dataframe_width_param()
        )
        st.caption(f"전체 {len(df_preview_sorted):,}개 리뷰 (행정구별 정렬)")
        
        # 감성 분석 추가 버튼
        st.markdown("---")
        
        # 세션 상태에 저장된 결과가 있으면 표시
        if 'preview_sentiment_result' in st.session_state and st.session_state.preview_sentiment_result is not None:
            df_preview_with_sentiment = st.session_state.preview_sentiment_result
            st.success(f"✅ 감성 분석 결과 (총 {len(df_preview_with_sentiment):,}개 리뷰)")
            
            # 결과 표시
            st.dataframe(
                df_preview_with_sentiment,
                hide_index=True,
            )
            
            # 통계 정보
            sentiment_labels = df_preview_with_sentiment['감성분석'].tolist()
            col1, col2, col3 = st.columns(3)
            with col1:
                positive_count = sentiment_labels.count('긍정')
                st.metric("긍정 리뷰", f"{positive_count:,}개 ({positive_count/len(sentiment_labels)*100:.1f}%)")
            with col2:
                negative_count = sentiment_labels.count('부정')
                st.metric("부정 리뷰", f"{negative_count:,}개 ({negative_count/len(sentiment_labels)*100:.1f}%)")
            with col3:
                neutral_count = sentiment_labels.count('중립')
                if neutral_count > 0:
                    st.metric("중립 리뷰", f"{neutral_count:,}개 ({neutral_count/len(sentiment_labels)*100:.1f}%)")
            
            # 다운로드 버튼
            csv = df_preview_with_sentiment.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 감성 분석 결과 CSV 다운로드",
                data=csv,
                file_name="google_reviews_with_sentiment.csv",
                mime="text/csv",
                key=f"download_preview_sentiment{tab_suffix}"
            )
            
            # 재실행 버튼
            if False and st.button("🔄 감성 분석 다시 실행", type="secondary", key=f"preview_sentiment_rerun{tab_suffix}"):
                st.session_state.preview_sentiment_result = None
                st.rerun()
        else:
            # 감성 분석 실행 버튼
            if False and st.button("🔍 감성 분석 추가 (긍정/부정/중립)", type="secondary", key=f"preview_sentiment_analyze{tab_suffix}"):
                _run_preview_sentiment_analysis(df_preview_sorted, sentiment_pipeline, sentiment_model_name)
    else:
        st.warning(f"필요한 컬럼이 없습니다. 현재 컬럼: {list(df_preview.columns)}")


def _run_preview_sentiment_analysis(df_preview_sorted, sentiment_pipeline, sentiment_model_name):
    """미리보기 섹션의 감성 분석 실행"""
    with st.spinner(f"감성 분석 모델을 사용하여 리뷰별 감성 분석 중... (시간이 걸릴 수 있습니다)"):
        # 리뷰 텍스트 및 평점 추출
        review_texts = df_preview_sorted['리뷰'].astype(str).tolist()
        ratings = df_preview_sorted['평점'].astype(float).tolist() if '평점' in df_preview_sorted.columns else [None] * len(review_texts)
        
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        batch_size = 32
        total_batches = (len(review_texts) + batch_size - 1) // batch_size
        
        sentiment_labels = []
        sentiment_scores = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(review_texts))
            batch_texts = review_texts[start_idx:end_idx]
            batch_ratings = ratings[start_idx:end_idx] if ratings else [None] * len(batch_texts)
            
            progress = (batch_idx + 1) / total_batches
            progress_bar.progress(progress)
            status_text.text(f"처리 중: {batch_idx + 1}/{total_batches} 배치 ({len(batch_texts)}개 리뷰)")
            
            try:
                # 숫자-only 리뷰와 일반 텍스트 리뷰 분리
                text_batch = []
                batch_results_map = {}
                
                for idx, text in enumerate(batch_texts):
                    rating = batch_ratings[idx] if idx < len(batch_ratings) else None
                    
                    # 숫자-only 리뷰는 별점 기반으로 처리
                    if is_numeric_only(text):
                        try:
                            rating_value = float(text)
                            if rating_value >= 4.0:
                                batch_results_map[idx] = ("긍정", 0.9)
                            elif rating_value >= 3.0:
                                batch_results_map[idx] = ("중립", 0.5)
                            else:
                                batch_results_map[idx] = ("부정", 0.1)
                        except ValueError:
                            batch_results_map[idx] = ("중립", 0.5)
                    # 메타데이터-only 리뷰도 별점 기반으로 처리
                    elif is_metadata_only(text) and rating is not None:
                        try:
                            rating_value = float(rating)
                            if rating_value >= 4.0:
                                batch_results_map[idx] = ("긍정", 0.9)
                            elif rating_value >= 3.0:
                                batch_results_map[idx] = ("중립", 0.5)
                            else:
                                batch_results_map[idx] = ("부정", 0.1)
                        except (ValueError, TypeError):
                            batch_results_map[idx] = ("중립", 0.5)
                    else:
                        # 일반 텍스트 리뷰는 모델 사용을 위해 수집
                        text_batch.append((idx, text))
                
                # 일반 텍스트 리뷰는 모델 사용
                if text_batch:
                    text_only = [text for _, text in text_batch]
                    truncated_texts = [truncate_text_for_bert(text) for text in text_only]
                    model_results = sentiment_pipeline(truncated_texts, truncation=True, max_length=512)
                    
                    # 모델 결과를 인덱스에 매핑
                    for (idx, _), res in zip(text_batch, model_results):
                        label, score = process_sentiment_result(res, sentiment_model_name)
                        batch_results_map[idx] = (label, score)
                
                # 원래 순서대로 결과 추가
                for idx in range(len(batch_texts)):
                    label, score = batch_results_map[idx]
                    sentiment_labels.append(label)
                    sentiment_scores.append(score)
                    
            except Exception as e:
                st.warning(f"배치 {batch_idx+1} 처리 중 오류: {e}")
                sentiment_labels.extend(['중립'] * len(batch_texts))
                sentiment_scores.extend([0.5] * len(batch_texts))
        
        progress_bar.empty()
        status_text.empty()
        
        # 결과를 데이터프레임에 추가
        df_preview_with_sentiment = df_preview_sorted.copy()
        df_preview_with_sentiment['감성분석'] = sentiment_labels
        df_preview_with_sentiment['감성점수'] = [f"{s:.3f}" for s in sentiment_scores]
        
        # 컬럼 순서 재정렬
        column_order = ['상호명', '시군구명', '행정동명', '평점', '리뷰', '감성분석', '감성점수']
        df_preview_with_sentiment = df_preview_with_sentiment[column_order]
        
        # 세션 상태에 저장
        st.session_state.preview_sentiment_result = df_preview_with_sentiment
        
        st.success(f"✅ 감성 분석 완료! {len(sentiment_labels):,}개 리뷰 분석됨")
        st.rerun()


def render_placeness_calculation(df_reviews, sbert_model, sentiment_pipeline, sentiment_model_name):
    """장소성 요인 점수 계산 섹션 렌더링"""
    st.header("📊 1. 장소성 요인별 정량 점수 계산")
    st.caption(f"유사도 임계값: {SIMILARITY_THRESHOLD} (코드 내 고정값)")
    st.caption(f"⚠️ 언급 0인 요인은 fsi=0.5, Wi=0 처리되어 Mu에 영향 없음")
    
    total_reviews_count = len(df_reviews)

    precomputed_csv = st.session_state.get("precomputed_metrics_csv")
    if st.session_state.df_final_metrics is not None:
        if precomputed_csv:
            st.success(f"저장된 장소성 결과 CSV를 불러와 표시 중입니다: `{precomputed_csv}`")
        _render_placeness_results()
        st.markdown("---")
    
    if False and st.button("장소성 요인 점수 다시 계산", type="secondary", key="placeness_calculation_start"):
        with st.spinner("장소성 요인별 점수 계산 및 연구 지표 산출 중..."):
            try:
                df_place_scores, df_review_scores = calculate_place_scores(
                    df_reviews.copy(), 
                    sbert_model, 
                    sentiment_pipeline, 
                    ALL_FACTORS, 
                    similarity_threshold=SIMILARITY_THRESHOLD,
                    sentiment_model_name=sentiment_model_name
                )
                
                df_final_metrics = calculate_final_research_metrics(
                    df_place_scores, 
                    list(ALL_FACTORS.keys()), 
                    total_reviews_count
                )
                
                st.session_state.df_review_scores = df_review_scores
                st.session_state.df_final_metrics = df_final_metrics
                st.session_state.df_place_scores = df_place_scores
                st.session_state.precomputed_metrics_csv = None
                st.session_state.precomputed_review_scores_csv = None
                
            except Exception as e:
                st.error(f"점수 계산 중 오류 발생: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

            st.rerun()


def _render_placeness_results():
    """장소성 계산 결과 표시"""
    st.header("장소성 종합 점수")
    
    df_final_metrics = st.session_state.df_final_metrics
    
    # Final_PlaceScore_Summary와 강점/약점만 표시
    display_summary_cols = ['cafe_name', 'Final_PlaceScore_Summary', '강점_요인(+df+)', '약점_요인(-df-)']
    if all(col in df_final_metrics.columns for col in display_summary_cols):
        st.dataframe(
            df_final_metrics[display_summary_cols].set_index('cafe_name'), 
            **get_dataframe_width_param()
        )
    
    st.subheader("세부 지표 점수 (fsi)")
    fsi_cols = ['cafe_name', '종합_장소성_점수_Mu', '요인_점수_표준편차_Sigma'] + [f'점수_{factor}' for factor in ALL_FACTORS.keys()]
    if all(col in df_final_metrics.columns for col in fsi_cols):
        st.dataframe(
            df_final_metrics[fsi_cols].set_index('cafe_name'), 
            **get_dataframe_width_param()
        )
    
    # 가중치 정보 표시
    with st.expander("📊 가중치 (Wi) 상세 정보"):
        wi_cols = ['cafe_name'] + [f'Wi_{factor}' for factor in ALL_FACTORS.keys()]
        if all(col in df_final_metrics.columns for col in wi_cols):
            st.dataframe(
                df_final_metrics[wi_cols].set_index('cafe_name'), 
                **get_dataframe_width_param()
            )
    
    # 결과 다운로드
    csv = df_final_metrics.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "장소성 최종 연구 지표 CSV 다운로드 (Wi, Mu, Sigma 포함)",
        data=csv,
        file_name="placeness_final_research_metrics_real.csv",
        mime="text/csv"
    )


def render_sentiment_analysis(df_reviews, sentiment_pipeline, sentiment_model_name):
    """개별 리뷰 감성 분석 섹션 렌더링"""
    st.header("2. 개별 리뷰 감성 분석 및 카페별 평균")
    
    if st.button("KoBERT 개별 리뷰 감성 분석 시작", type="primary", key="sentiment_analysis_start"):
        with st.spinner("개별 리뷰 긍정/부정 감성 점수 계산 중 (KoBERT/KoELECTRA)..."):
            try:
                df_reviews_with_sentiment, df_avg_sentiment = run_sentiment_analysis(
                    df_reviews.copy(), 
                    sentiment_pipeline,
                    sentiment_model_name
                )
                
                st.session_state.df_reviews_with_sentiment = df_reviews_with_sentiment
                st.session_state.df_avg_sentiment = df_avg_sentiment
                
            except Exception as e:
                st.error(f"감성 분석 중 오류 발생: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # 결과 표시
    if st.session_state.df_reviews_with_sentiment is not None and st.session_state.df_avg_sentiment is not None:
        st.subheader("✅ 카페별 평균 감성 점수")
        st.dataframe(st.session_state.df_avg_sentiment.set_index('cafe_name'), **get_dataframe_width_param())
        
        st.subheader("✅ 개별 리뷰 감성 분석 결과 (샘플)")
        sample_df = st.session_state.df_reviews_with_sentiment[['cafe_name', 'review_text', 'sentiment_label', 'sentiment_score']].head(20)
        st.dataframe(sample_df, **get_dataframe_width_param())
        
        # 결과 다운로드
        csv = st.session_state.df_reviews_with_sentiment.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 개별 리뷰 감성 분석 결과 CSV 다운로드",
            data=csv,
            file_name="reviews_with_sentiment_real.csv",
            mime="text/csv"
        )


def render_detailed_results():
    """리뷰별 상세 분석 결과 섹션 렌더링"""
    st.header("📊 리뷰별 상세 분석 결과")
    
    has_sentiment = st.session_state.df_reviews_with_sentiment is not None
    has_placeness = st.session_state.df_review_scores is not None
    
    if not has_sentiment and not has_placeness:
        st.info("👆 위의 두 분석을 모두 실행하면 리뷰별 상세 결과를 확인할 수 있습니다.")
    else:
        if has_sentiment and has_placeness:
            _render_merged_results()
        elif has_sentiment:
            st.info("장소성 요인 점수 계산을 실행하면 리뷰별 상세 결과를 확인할 수 있습니다.")
            # 전체 리뷰 표시
            display_cols = ['cafe_name', 'review_text', 'sentiment_label', 'sentiment_score']
            available_cols = [col for col in display_cols if col in st.session_state.df_reviews_with_sentiment.columns]
            st.dataframe(
                st.session_state.df_reviews_with_sentiment[available_cols], 
                hide_index=True, 
            )
            st.caption(f"총 {len(st.session_state.df_reviews_with_sentiment):,}개 리뷰")
            
            # CSV 다운로드 버튼
            csv = st.session_state.df_reviews_with_sentiment[available_cols].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 리뷰별 상세 결과 CSV 다운로드",
                data=csv,
                file_name="reviews_with_sentiment_real.csv",
                mime="text/csv"
            )
        elif has_placeness:
            st.info("감성 분석을 실행하면 리뷰별 상세 결과를 확인할 수 있습니다.")
            # 전체 12개 요인 점수 표시
            factor_names = list(ALL_FACTORS.keys())
            factor_score_cols = [f'{factor}_점수' for factor in factor_names]
            display_cols = ['cafe_name', 'review_text'] + factor_score_cols
            available_cols = [col for col in display_cols if col in st.session_state.df_review_scores.columns]
            
            # 점수 포맷팅 (표시용)
            display_df = st.session_state.df_review_scores[available_cols].copy()
            for col in factor_score_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
            
            st.dataframe(
                display_df, 
                hide_index=True, 
            )
            st.caption(f"총 {len(st.session_state.df_review_scores):,}개 리뷰 ({len(FACTOR_NAMES)}개 요인 전체 표시)")
            
            # CSV 다운로드 버튼 (원본 데이터, 포맷팅 없이)
            csv = st.session_state.df_review_scores[available_cols].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 리뷰별 상세 결과 CSV 다운로드",
                data=csv,
                file_name="review_placeness_scores_real.csv",
                mime="text/csv"
            )
            
            # 요인별 키워드 분석 추가
            st.markdown("---")
            visualize_factor_keywords(st.session_state.df_review_scores, factor_names, top_n=15)


def _render_merged_results():
    """병합된 결과 표시"""
    df_sentiment = st.session_state.df_reviews_with_sentiment.copy()
    df_placeness = st.session_state.df_review_scores.copy()
    
    df_sentiment['review_index'] = df_sentiment.index
    df_placeness['review_index'] = df_placeness['review_index']
    
    df_merged = pd.merge(
        df_sentiment[['review_index', 'cafe_name', 'review_text', 'sentiment_label', 'sentiment_score']],
        df_placeness,
        on=['review_index', 'cafe_name', 'review_text'],
        how='outer'
    )
    
    factor_names = list(ALL_FACTORS.keys())
    factor_score_cols = [f'{factor}_점수' for factor in factor_names]
    display_cols = ['cafe_name', 'review_text', 'sentiment_label', 'sentiment_score'] + factor_score_cols
    # 실제 존재하는 컬럼만 필터링
    available_cols = [col for col in display_cols if col in df_merged.columns]
    
    # 누락된 요인 컬럼 확인
    missing_factors = [f for f in factor_names if f'{f}_점수' not in df_merged.columns]
    if missing_factors:
        st.warning(f"⚠️ 다음 요인 컬럼이 데이터에 없습니다: {', '.join(missing_factors)}")
    
    st.subheader(f"✅ 리뷰별 감성 분석 + 장소성 요인 점수 (전체 {len(FACTOR_NAMES)}개 요인)")
    st.caption(f"총 {len(df_merged):,}개 리뷰 중 필터링된 결과 표시")
    
    # 필터 옵션
    col1, col2 = st.columns(2)
    with col1:
        selected_cafe = st.selectbox(
            "카페 선택 (전체 보기)",
            options=['전체'] + sorted(df_merged['cafe_name'].unique().tolist()),
            key="review_detail_cafe_filter"
        )
    with col2:
        selected_sentiment = st.selectbox(
            "감성 필터",
            options=['전체', '긍정', '부정'],
            key="review_detail_sentiment_filter"
        )
    
    # 필터링
    filtered_df = df_merged.copy()
    if selected_cafe != '전체':
        filtered_df = filtered_df[filtered_df['cafe_name'] == selected_cafe]
    if selected_sentiment != '전체':
        filtered_df = filtered_df[filtered_df['sentiment_label'] == selected_sentiment]
    
    # 결과 표시
    if len(filtered_df) > 0:
        display_df = filtered_df[available_cols].copy()
        
        # 점수 포맷팅
        for col in factor_score_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        
        if 'sentiment_score' in display_df.columns:
            display_df['sentiment_score'] = display_df['sentiment_score'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        
        # 전체 리뷰 표시 (높이를 크게 설정하여 스크롤 가능)
        st.dataframe(
            display_df,
            hide_index=True,
        )

        st.caption(f"총 {len(filtered_df):,}개 리뷰 표시 ({len(FACTOR_NAMES)}개 요인 전체)")
        
        # 다운로드 버튼
        csv = filtered_df[available_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 리뷰별 상세 결과 CSV 다운로드",
            data=csv,
            file_name="review_detailed_analysis_real.csv",
            mime="text/csv"
        )
        
        # 요인별 키워드 분석 추가
        st.markdown("---")
        factor_names = list(ALL_FACTORS.keys())
        visualize_factor_keywords(filtered_df, factor_names, top_n=15)
    else:
        st.warning("선택한 조건에 해당하는 리뷰가 없습니다.")


def visualize_factor_keywords(df_review_scores, factor_names, top_n=15, top_reviews_per_factor=200):
    """
    각 요인별로 TF-IDF를 이용하여 '특색 있는' 주요 키워드를 추출하고 시각화합니다.
    유사도가 가장 높은 상위 리뷰에서만 키워드를 추출합니다.
    
    Args:
        df_review_scores: 리뷰별 요인 점수/유사도 DataFrame
        factor_names: 요인 이름 리스트
        top_n: 상위 N개 키워드 표시
        top_reviews_per_factor: 각 요인별로 유사도 상위 몇 개 리뷰를 사용할지 (기본값: 200)
    """
    st.subheader("🔎 요인별 핵심 키워드 분석 (TF-IDF 기반)")
    st.info(f"유사도가 가장 높은 상위 {top_reviews_per_factor}개 리뷰에서만 키워드를 추출합니다. TF-IDF를 사용하여 각 장소성 요인을 가장 잘 대표하는 차별화된 단어들을 추출합니다.")
    
    # 1. 요인별 텍스트 문서 생성
    # 각 요인에 매칭된 리뷰 중 유사도 상위 리뷰만 사용
    factor_documents = []
    valid_factors = []  # 리뷰가 하나라도 있는 요인만 추적
    
    for factor in factor_names:
        score_col = f'{factor}_점수'
        sim_col = f'{factor}_유사도'
        
        if score_col not in df_review_scores.columns:
            factor_documents.append("")
            continue
            
        # 해당 요인에 매칭된 리뷰만 필터링
        relevant_df = df_review_scores[
            pd.to_numeric(df_review_scores[score_col], errors='coerce').notnull()
        ].copy()
        
        if not relevant_df.empty:
            # 유사도 컬럼이 있으면 유사도 기준으로, 없으면 점수 기준으로 정렬
            if sim_col in relevant_df.columns:
                # 유사도 상위 리뷰만 선택
                top_relevant_df = relevant_df.sort_values(by=sim_col, ascending=False).head(top_reviews_per_factor)
            else:
                # 점수 기준으로 정렬
                top_relevant_df = relevant_df.sort_values(by=score_col, ascending=False).head(top_reviews_per_factor)
            
            # 텍스트 전처리 (한글만 남기기)
            text = " ".join(top_relevant_df['review_text'].astype(str).tolist())
            text = re.sub(r'[^가-힣\s]', '', text)  # 한글과 공백만 남김
            factor_documents.append(text)
            valid_factors.append(factor)
        else:
            # 매칭된 리뷰가 없으면 빈 문자열 추가 (인덱스 유지를 위해)
            factor_documents.append("")
    
    if not any(factor_documents):
        st.warning("분석할 텍스트 데이터가 없습니다.")
        return
    
    # 2. TF-IDF 벡터화
    # 불용어 설정 (모든 요인에서 공통적으로 너무 많이 나오는 단어들 제거)
    stop_words = [
        '카페', '너무', '진짜', '정말', '많이', '가서', '먹고', '있는', '하는', '그리고', '그래서', 
        '좋아요', '있어요', '같아요', '맛있어요', '분위기', '생각', '느낌', '방문', '곳', '것', '수',
        '있습니다', '있었', '있고', '있는데', '있어서', '있어', '있음',
        '좋습니다', '좋았', '좋고', '좋은', '좋아', '좋다', '좋음',
        '맛있습니다', '맛있었', '맛있고', '맛있는', '맛있어',
        '이거', '그거', '저거', '이것', '그것', '저것',
        '그런데', '하지만', '그러나',
        '이런', '그런', '저런', '이렇게', '그렇게', '저렇게',
        '때문', '위해', '통해', '대해', '관련', '따라',
        '자리', '자리가', '자리도', '자리를', '자리에',
        '매장', '매장이', '매장도', '매장을', '매장에',
        '사람', '사람이', '사람들', '사람도',
        # 음료/음식 관련 일반 단어
        '커피', '커피도', '커피를', '커피가', '커피는',
        '음료', '음료도', '음료를',
        '디저트', '디저트도',
        # 일반 형용사/부사
        '다양한', '다양', '다른', '매우', '아주', '정말로',
        '있다', '있어', '있고', '있는데', '있어서', '있음'
    ]
    
    try:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words=stop_words,
            token_pattern=r"(?u)\b\w\w+\b"  # 2글자 이상 단어만
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(factor_documents)
        feature_names = tfidf_vectorizer.get_feature_names_out()
    except ValueError as e:
        st.warning(f"TF-IDF 분석을 위한 충분한 텍스트가 없거나, 모든 단어가 불용어로 처리되었습니다: {e}")
        return
    
    # 3. 시각화
    tabs = st.tabs(factor_names)
    
    for i, factor in enumerate(factor_names):
        with tabs[i]:
            score_col = f'{factor}_점수'
            sim_col = f'{factor}_유사도'
            
            if score_col not in df_review_scores.columns:
                st.warning(f"데이터에 {score_col} 컬럼이 없습니다.")
                continue
            
            # 해당 요인 점수가 있는(매칭된) 리뷰들만 추출
            relevant_df = df_review_scores[
                pd.to_numeric(df_review_scores[score_col], errors='coerce').notnull()
            ].copy()
            
            if relevant_df.empty or not factor_documents[i].strip():
                st.write("매칭된 리뷰가 없습니다.")
                continue
            
            # -------------------------------------------------------
            # [검증 방법 1] 유사도 상위 리뷰 리스트
            # -------------------------------------------------------
            st.markdown(f"#### 1. '{factor}'와 유사도가 가장 높은 리뷰 Top 10")
            
            # 유사도 컬럼이 있다면 사용, 없다면 점수 기준
            if sim_col in relevant_df.columns:
                top_reviews = relevant_df.sort_values(by=sim_col, ascending=False).head(10)
                for idx, row in top_reviews.iterrows():
                    score_val = row[score_col] if pd.notna(row[score_col]) else "N/A"
                    sim_val = row[sim_col] if pd.notna(row[sim_col]) else "N/A"
                    st.success(f"**유사도 {sim_val:.3f} | 점수 {score_val:.3f}**: {row['review_text']}")
            else:
                st.warning("유사도 컬럼을 찾을 수 없어 점수 기준으로 정렬합니다.")
                top_reviews = relevant_df.sort_values(by=score_col, ascending=False).head(10)
                for idx, row in top_reviews.iterrows():
                    score_val = row[score_col] if pd.notna(row[score_col]) else "N/A"
                    st.success(f"**점수 {score_val:.3f}**: {row['review_text']}")
            
            # -------------------------------------------------------
            # [검증 방법 2] TF-IDF 기반 키워드 분석 (Bar Chart)
            # -------------------------------------------------------
            st.markdown(f"#### 2. '{factor}' 관련 리뷰 내 주요 키워드")
            
            # 해당 요인(문서)의 TF-IDF 점수 가져오기
            tfidf_scores = tfidf_matrix[i].toarray().flatten()
            
            # 해당 요인 내에서의 단어 빈도도 계산 (하이브리드 방식)
            # TF-IDF 점수와 요인 내 빈도를 결합하여 더 정확한 키워드 추출
            factor_text = factor_documents[i]
            factor_word_counts = Counter(factor_text.split())
            total_words_in_factor = sum(factor_word_counts.values())
            
            # TF-IDF 점수와 요인 내 상대 빈도를 결합한 점수 계산
            hybrid_scores = []
            for idx, word in enumerate(feature_names):
                tfidf_score = tfidf_scores[idx]
                # 해당 요인 내에서의 상대 빈도 (0~1)
                word_freq_in_factor = factor_word_counts.get(word, 0) / max(total_words_in_factor, 1)
                # 하이브리드 점수: TF-IDF * (1 + 요인 내 상대 빈도)
                # 이렇게 하면 TF-IDF가 높고 해당 요인에서도 자주 나오는 단어가 우선순위가 높아짐
                hybrid_score = tfidf_score * (1 + word_freq_in_factor * 2)
                if hybrid_score > 0:
                    hybrid_scores.append((word, hybrid_score, tfidf_score, word_freq_in_factor))
            
            # 하이브리드 점수 기준으로 정렬
            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            top_keywords = hybrid_scores[:top_n]
            
            if top_keywords:
                # 데이터프레임 변환 (하이브리드 점수 사용)
                df_keywords = pd.DataFrame(
                    [(word, hybrid_score) for word, hybrid_score, _, _ in top_keywords],
                    columns=['단어', '하이브리드 점수']
                )
                df_keywords = df_keywords.sort_values('하이브리드 점수', ascending=True)
                
                # Streamlit Bar Chart
                st.bar_chart(df_keywords.set_index('단어'), height=400)
                
                # 상세 테이블 (TF-IDF 점수와 빈도 정보 포함)
                with st.expander("상세 키워드 점수 보기"):
                    df_detail = pd.DataFrame(
                        [(word, f"{hybrid_score:.4f}", f"{tfidf_score:.4f}", f"{freq:.4f}") 
                         for word, hybrid_score, tfidf_score, freq in top_keywords],
                        columns=['단어', '하이브리드 점수', 'TF-IDF 점수', '요인 내 상대 빈도']
                    )
                    st.dataframe(df_detail, hide_index=True, **get_dataframe_width_param())
                    st.caption("하이브리드 점수 = TF-IDF × (1 + 요인 내 상대 빈도 × 2)")
            else:
                st.write("유의미한 키워드를 추출하지 못했습니다.")
            
            # -------------------------------------------------------
            # [검증 방법 3] 긍정 리뷰만 필터링하여 보기
            # -------------------------------------------------------
            st.markdown(f"#### 3. '{factor}' 점수가 높은(0.9 이상) 긍정 리뷰 패턴")
            high_score_df = relevant_df[
                pd.to_numeric(relevant_df[score_col], errors='coerce') >= 0.9
            ]
            
            if not high_score_df.empty:
                display_cols = ['review_text', score_col]
                if sim_col in high_score_df.columns:
                    display_cols.insert(1, sim_col)
                st.dataframe(
                    high_score_df[display_cols].sort_values(by=score_col, ascending=False),
                    hide_index=True,
                )
                st.caption(f"총 {len(high_score_df)}개 리뷰 (점수 0.9 이상)")
            else:
                st.info("0.9점 이상의 매우 긍정적인 리뷰가 없습니다.")
            
            # -------------------------------------------------------
            # [검증 방법 4] 통계 정보
            # -------------------------------------------------------
            st.markdown(f"#### 4. '{factor}' 관련 통계")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("관련 리뷰 수", f"{len(relevant_df):,}개")
            with col2:
                avg_score = pd.to_numeric(relevant_df[score_col], errors='coerce').mean()
                st.metric("평균 점수", f"{avg_score:.3f}" if pd.notna(avg_score) else "N/A")
            with col3:
                if sim_col in relevant_df.columns:
                    avg_sim = pd.to_numeric(relevant_df[sim_col], errors='coerce').mean()
                    st.metric("평균 유사도", f"{avg_sim:.3f}" if pd.notna(avg_sim) else "N/A")
                else:
                    st.metric("평균 유사도", "N/A")
            with col4:
                high_count = len(relevant_df[pd.to_numeric(relevant_df[score_col], errors='coerce') >= 0.9])
                st.metric("긍정 리뷰 (≥0.9)", f"{high_count}개")
            
            # -------------------------------------------------------
            # [검증 방법 5] 행정구별 상위 10% 카페 분포 시각화
            # -------------------------------------------------------
            st.markdown(f"#### 5. '{factor}' 점수 상위 10% 카페 행정구별 분포")
            
            # df_place_scores에서 해당 요인 점수 상위 10% 카페 추출
            if 'df_place_scores' in st.session_state and st.session_state.df_place_scores is not None:
                df_place_scores = st.session_state.df_place_scores
                factor_score_col = f'점수_{factor}'
                
                if factor_score_col in df_place_scores.columns:
                    # 점수가 있는 카페만 필터링
                    valid_scores = df_place_scores[
                        pd.to_numeric(df_place_scores[factor_score_col], errors='coerce').notna()
                    ].copy()
                    
                    if not valid_scores.empty:
                        # 상위 10% 임계값 계산
                        threshold = valid_scores[factor_score_col].quantile(0.9)
                        top_10_percent = valid_scores[valid_scores[factor_score_col] >= threshold].copy()
                        
                        # 원본 데이터에서 시군구명 가져오기
                        df_reviews_for_district = None
                        if 'df_reviews' in st.session_state and st.session_state.df_reviews is not None:
                            df_reviews_for_district = st.session_state.df_reviews.copy()
                        
                        # 시군구명 컬럼이 없으면 원본 CSV에서 로드
                        if df_reviews_for_district is None or '시군구명' not in df_reviews_for_district.columns:
                            try:
                                from pathlib import Path
                                from modules.config import GOOGLE_REVIEW_SAMPLE_CSV
                                df_reviews_for_district = load_csv_raw(Path(GOOGLE_REVIEW_SAMPLE_CSV))
                                
                                # 컬럼명 정규화 (상호명 -> cafe_name)
                                if '상호명' in df_reviews_for_district.columns and 'cafe_name' not in df_reviews_for_district.columns:
                                    df_reviews_for_district['cafe_name'] = df_reviews_for_district['상호명']
                            except Exception as e:
                                st.warning(f"시군구명 데이터 로드 실패: {e}")
                                df_reviews_for_district = None
                        
                        if df_reviews_for_district is not None and '시군구명' in df_reviews_for_district.columns:
                            # cafe_name 컬럼 확인 및 생성
                            if 'cafe_name' not in df_reviews_for_district.columns:
                                if '상호명' in df_reviews_for_district.columns:
                                    df_reviews_for_district['cafe_name'] = df_reviews_for_district['상호명']
                                else:
                                    st.warning("cafe_name 또는 상호명 컬럼을 찾을 수 없습니다.")
                                    df_reviews_for_district = None
                            
                            if df_reviews_for_district is not None:
                                # original_cafe_name이 있으면 사용, 없으면 cafe_name 사용
                                if 'original_cafe_name' in df_reviews_for_district.columns:
                                    cafe_to_district = df_reviews_for_district.groupby('original_cafe_name')['시군구명'].first().to_dict()
                                    # top_10_percent의 cafe_name에서 위치 정보 제거하여 original_cafe_name 추출
                                    top_10_percent['original_cafe_name'] = top_10_percent['cafe_name'].str.split().str[0]
                                    top_10_percent['시군구명'] = top_10_percent['original_cafe_name'].map(cafe_to_district)
                                else:
                                    # cafe_name에서 위치 정보 제거 시도 (공백으로 분리된 첫 번째 부분)
                                    # 먼저 원본 cafe_name으로 매핑 시도
                                    cafe_to_district = df_reviews_for_district.groupby('cafe_name')['시군구명'].first().to_dict()
                                    
                                    # 카페명에서 위치 정보 제거 (시군구명과 행정동명이 추가된 경우)
                                    # 예: "투썸플레이스 강남구 역삼동" -> "투썸플레이스"
                                    top_10_percent['base_cafe_name'] = top_10_percent['cafe_name'].str.split().str[0]
                                    
                                    # 먼저 전체 cafe_name으로 매핑 시도, 없으면 base_cafe_name으로 시도
                                    top_10_percent['시군구명'] = top_10_percent['cafe_name'].map(cafe_to_district)
                                    # 매핑되지 않은 경우 base_cafe_name으로 재시도
                                    missing_mask = top_10_percent['시군구명'].isna()
                                    if missing_mask.any():
                                        base_cafe_to_district = df_reviews_for_district.groupby('cafe_name')['시군구명'].first().to_dict()
                                        # base_cafe_name으로 매핑 시도
                                        for idx in top_10_percent[missing_mask].index:
                                            base_name = top_10_percent.loc[idx, 'base_cafe_name']
                                            # base_name과 일치하는 cafe_name 찾기 (부분 매칭)
                                            matched_district = None
                                            for cafe_name, district in base_cafe_to_district.items():
                                                if cafe_name.startswith(base_name) or base_name in cafe_name:
                                                    matched_district = district
                                                    break
                                            if matched_district:
                                                top_10_percent.loc[idx, '시군구명'] = matched_district
                            
                            # 시군구명이 있는 카페만 사용
                            top_10_percent_with_district = top_10_percent[
                                top_10_percent['시군구명'].notna()
                            ]
                            
                            if not top_10_percent_with_district.empty:
                                # 행정구별 카페 수 집계
                                district_counts = top_10_percent_with_district['시군구명'].value_counts().sort_values(ascending=True)
                                
                                # 막대 그래프
                                df_district = pd.DataFrame({
                                    '행정구': district_counts.index,
                                    '상위 10% 카페 수': district_counts.values
                                })
                                
                                st.bar_chart(df_district.set_index('행정구'), height=400)
                                
                                # 상세 테이블
                                with st.expander("행정구별 상세 정보"):
                                    st.dataframe(
                                        df_district.sort_values('상위 10% 카페 수', ascending=False),
                                        hide_index=True
                                    )
                                
                                st.caption(f"총 {len(top_10_percent_with_district)}개 카페 (점수 임계값: {threshold:.3f} 이상)")
                                
                                # 위도/경도 가져오기 및 지도 시각화
                                if CAFE_INFO_CSV.exists() and HAS_FOLIUM:
                                    try:
                                        # 카페 정보 CSV에서 위도/경도 가져오기
                                        df_cafe_info = pd.read_csv(CAFE_INFO_CSV, encoding='utf-8-sig')
                                        
                                        # 상호명, 시군구명, 행정동명으로 매칭
                                        # top_10_percent_with_district에 위도/경도 추가
                                        top_10_percent_with_district['위도'] = None
                                        top_10_percent_with_district['경도'] = None
                                        
                                        # 카페명에서 위치 정보 제거 (base_cafe_name 사용)
                                        if 'base_cafe_name' not in top_10_percent_with_district.columns:
                                            top_10_percent_with_district['base_cafe_name'] = top_10_percent_with_district['cafe_name'].str.split().str[0]
                                        
                                        # 행정동명도 가져오기
                                        if 'df_reviews' in st.session_state and st.session_state.df_reviews is not None:
                                            df_reviews = st.session_state.df_reviews
                                            if '행정동명' in df_reviews.columns:
                                                if 'original_cafe_name' in df_reviews.columns:
                                                    cafe_to_dong = df_reviews.groupby('original_cafe_name')['행정동명'].first().to_dict()
                                                    top_10_percent_with_district['행정동명'] = top_10_percent_with_district.get('original_cafe_name', top_10_percent_with_district['base_cafe_name']).map(cafe_to_dong)
                                                else:
                                                    cafe_to_dong = df_reviews.groupby('cafe_name')['행정동명'].first().to_dict()
                                                    top_10_percent_with_district['행정동명'] = top_10_percent_with_district['cafe_name'].map(cafe_to_dong)
                                        
                                        # 카페 정보와 매칭
                                        for idx, row in top_10_percent_with_district.iterrows():
                                            cafe_name = row.get('base_cafe_name', row['cafe_name'].split()[0])
                                            district = row['시군구명']
                                            dong = row.get('행정동명', None)
                                            
                                            matched = None
                                            
                                            # 매칭 우선순위:
                                            # 1. 정확한 상호명 일치 + 시군구명 + 행정동명
                                            # 2. 정확한 상호명 일치 + 시군구명
                                            # 3. 상호명이 cafe_name으로 시작 + 시군구명 + 행정동명
                                            # 4. 상호명이 cafe_name으로 시작 + 시군구명
                                            
                                            if dong and pd.notna(dong):
                                                # 1순위: 정확한 일치 + 시군구명 + 행정동명
                                                matched = df_cafe_info[
                                                    (df_cafe_info['상호명'] == cafe_name) &
                                                    (df_cafe_info['시군구명'] == district) &
                                                    (df_cafe_info['행정동명'] == dong)
                                                ]
                                                
                                                if matched.empty:
                                                    # 2순위: 상호명이 cafe_name으로 시작 + 시군구명 + 행정동명
                                                    matched = df_cafe_info[
                                                        (df_cafe_info['상호명'].str.startswith(cafe_name, na=False)) &
                                                        (df_cafe_info['시군구명'] == district) &
                                                        (df_cafe_info['행정동명'] == dong)
                                                    ]
                                            else:
                                                # 행정동명이 없으면 시군구명만으로 매칭
                                                # 1순위: 정확한 일치 + 시군구명
                                                matched = df_cafe_info[
                                                    (df_cafe_info['상호명'] == cafe_name) &
                                                    (df_cafe_info['시군구명'] == district)
                                                ]
                                                
                                                if matched.empty:
                                                    # 2순위: 상호명이 cafe_name으로 시작 + 시군구명
                                                    matched = df_cafe_info[
                                                        (df_cafe_info['상호명'].str.startswith(cafe_name, na=False)) &
                                                        (df_cafe_info['시군구명'] == district)
                                                    ]
                                            
                                            if not matched.empty:
                                                # 첫 번째 매칭 사용
                                                top_10_percent_with_district.loc[idx, '위도'] = matched.iloc[0]['위도']
                                                top_10_percent_with_district.loc[idx, '경도'] = matched.iloc[0]['경도']
                                        
                                        # 위도/경도가 있는 카페만 필터링
                                        cafes_with_location = top_10_percent_with_district[
                                            top_10_percent_with_district['위도'].notna() & 
                                            top_10_percent_with_district['경도'].notna()
                                        ]
                                        
                                        if not cafes_with_location.empty:
                                            st.markdown("##### 지도 시각화")
                                            
                                            # 서울 중심 좌표
                                            seoul_center = [37.5665, 126.9780]
                                            
                                            # Folium 지도 생성
                                            m = folium.Map(
                                                location=seoul_center,
                                                zoom_start=11,
                                                tiles='OpenStreetMap'
                                            )
                                            
                                            # 마커 추가
                                            for idx, row in cafes_with_location.iterrows():
                                                lat = float(row['위도'])
                                                lng = float(row['경도'])
                                                cafe_name = row['cafe_name']
                                                score = row[factor_score_col]
                                                
                                                # 팝업 정보
                                                popup_html = f"""
                                                <div style="font-family: Arial; min-width: 150px;">
                                                    <b>{cafe_name}</b><br>
                                                    {factor} 점수: {score:.3f}<br>
                                                    {row.get('시군구명', 'N/A')} {row.get('행정동명', '')}
                                                </div>
                                                """
                                                
                                                folium.Marker(
                                                    location=[lat, lng],
                                                    popup=folium.Popup(popup_html, max_width=300),
                                                    tooltip=f"{cafe_name} ({score:.3f})",
                                                    icon=folium.Icon(color='red', icon='info-sign')
                                                ).add_to(m)
                                            
                                            # 지도 표시
                                            st_folium(m, width=700, height=500)
                                            st.caption(f"지도에 표시된 카페: {len(cafes_with_location)}개 / 총 {len(top_10_percent_with_district)}개")
                                        else:
                                            st.info("위도/경도 정보를 찾을 수 있는 카페가 없습니다.")
                                    except Exception as e:
                                        st.warning(f"지도 시각화 중 오류 발생: {e}")
                                elif not HAS_FOLIUM:
                                    st.info("지도 시각화를 위해 folium과 streamlit-folium 패키지가 필요합니다.")
                            else:
                                st.info("시군구명 정보가 있는 카페가 없습니다.")
                        else:
                            st.info("리뷰 데이터를 찾을 수 없습니다.")
                    else:
                        st.info("유효한 점수 데이터가 없습니다.")
                else:
                    st.info(f"'{factor_score_col}' 컬럼을 찾을 수 없습니다.")
            else:
                st.info("장소성 점수 데이터를 찾을 수 없습니다. 먼저 장소성 점수를 계산해주세요.")


def _display_cafe_reviews(selected_cafe):
    """카페 리뷰 표시 헬퍼 함수"""
    st.subheader("📝 해당 카페 리뷰")
    
    # 리뷰 데이터 로드
    # config에서 경로 가져오기 (배포 환경 호환성)
    from modules.config import BASE_DIR
    review_file_path = BASE_DIR / "google_reviews_scraped_cleaned.csv"
    
    # 파일 존재 여부 확인
    if not review_file_path.exists():
        st.warning(f"⚠️ 리뷰 데이터 파일을 찾을 수 없습니다: {review_file_path}")
        st.info("💡 배포 환경에서는 파일이 프로젝트 루트에 있어야 합니다.")
        return
    
    try:
        # 리뷰 데이터 로드 (캐시 사용)
        @st.cache_data
        def load_reviews_for_cafe():
            df_reviews = pd.read_csv(review_file_path, encoding='utf-8-sig')
            return df_reviews
        
        df_all_reviews = load_reviews_for_cafe()
        
        # 카페명에서 상호명, 시군구명, 행정동명 추출
        # selected_cafe 형식: "카페명 구 동" (예: "스타벅스 강남구 역삼동")
        parts = selected_cafe.split() if selected_cafe else []
        
        # 상호명 추출 (구가 나오기 전까지)
        base_cafe_name = ""
        district = None
        dong = None
        
        SEOUL_DISTRICTS = [
            "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구",
            "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구",
            "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구",
            "서초구", "강남구", "송파구", "강동구"
        ]
        
        for i, part in enumerate(parts):
            if part in SEOUL_DISTRICTS:
                district = part
                base_cafe_name = " ".join(parts[:i])  # 구 이전까지가 상호명
                if i + 1 < len(parts):
                    dong = parts[i + 1]  # 구 다음이 행정동명
                break
        
        # 구를 찾지 못한 경우 첫 번째 단어를 상호명으로
        if not base_cafe_name:
            base_cafe_name = parts[0] if parts else ""
        
        # 필터링: 상호명, 시군구명, 행정동명이 모두 일치하는 리뷰만
        cafe_reviews = pd.DataFrame()
        
        if '상호명' in df_all_reviews.columns:
            # 상호명으로 필터링
            mask = df_all_reviews['상호명'] == base_cafe_name
            
            # 시군구명으로 추가 필터링
            if district and '시군구명' in df_all_reviews.columns:
                mask = mask & (df_all_reviews['시군구명'] == district)
            
            # 행정동명으로 추가 필터링
            if dong and '행정동명' in df_all_reviews.columns:
                mask = mask & (df_all_reviews['행정동명'] == dong)
            
            cafe_reviews = df_all_reviews[mask].copy()
        else:
            st.warning("상호명 컬럼을 찾을 수 없습니다.")
        
        if not cafe_reviews.empty:
            # 리뷰 수 표시
            st.info(f"총 {len(cafe_reviews)}개의 리뷰가 있습니다.")
            
            # 표시할 컬럼 선택
            display_cols = []
            if '리뷰' in cafe_reviews.columns:
                display_cols.append('리뷰')
            elif 'review_text' in cafe_reviews.columns:
                display_cols.append('review_text')
            
            if '평점' in cafe_reviews.columns:
                display_cols.insert(0, '평점')
            elif 'rating' in cafe_reviews.columns:
                display_cols.insert(0, 'rating')
            
            if '시군구명' in cafe_reviews.columns:
                display_cols.insert(0, '시군구명')
            if '행정동명' in cafe_reviews.columns:
                display_cols.insert(0, '행정동명')
            
            # 사용 가능한 컬럼만 필터링
            available_cols = [col for col in display_cols if col in cafe_reviews.columns]
            
            if available_cols:
                # 리뷰 표시 (최대 100개)
                max_reviews = min(100, len(cafe_reviews))
                st.dataframe(
                    cafe_reviews[available_cols].head(max_reviews),
                    hide_index=True,
                    **get_dataframe_width_param()
                )
                
                if len(cafe_reviews) > max_reviews:
                    st.caption(f"상위 {max_reviews}개 리뷰만 표시됩니다. (전체 {len(cafe_reviews)}개)")
            else:
                st.warning("표시할 수 있는 리뷰 컬럼이 없습니다.")
        else:
            st.warning(f"'{selected_cafe}'에 해당하는 리뷰를 찾을 수 없습니다.")
            st.info("💡 팁: 카페명이 정확히 일치하지 않을 수 있습니다. 원본 리뷰 데이터의 카페명 형식을 확인해주세요.")
                
    except Exception as e:
        st.error(f"리뷰 데이터 로드 중 오류 발생: {e}")
        import traceback
        st.code(traceback.format_exc())


def _display_cafe_reviews_for_recommendation(cafe_name, selected_factors=None):
    """추천 결과용 카페 리뷰 표시 및 요약 헬퍼 함수"""
    st.subheader("📝 리뷰 요약 및 분석")
    
    # 리뷰 데이터 로드
    from modules.config import BASE_DIR
    review_file_path = BASE_DIR / "google_reviews_scraped_cleaned.csv"
    
    # 파일 존재 여부 확인
    if not review_file_path.exists():
        st.warning(f"⚠️ 리뷰 데이터 파일을 찾을 수 없습니다: {review_file_path}")
        return
    
    try:
        # 리뷰 데이터 로드 (캐시 사용)
        @st.cache_data
        def load_reviews_for_cafe():
            df_reviews = pd.read_csv(review_file_path, encoding='utf-8-sig')
            return df_reviews
        
        df_all_reviews = load_reviews_for_cafe()
        
        # 카페명에서 상호명, 시군구명, 행정동명 추출
        # cafe_name 형식: "카페명 구 동" (예: "스타벅스 강남구 역삼동")
        parts = cafe_name.split() if cafe_name else []
        
        # 상호명 추출 (구가 나오기 전까지)
        base_cafe_name = ""
        district = None
        dong = None
        
        SEOUL_DISTRICTS = [
            "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구",
            "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구",
            "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구",
            "서초구", "강남구", "송파구", "강동구"
        ]
        
        for i, part in enumerate(parts):
            if part in SEOUL_DISTRICTS:
                district = part
                base_cafe_name = " ".join(parts[:i])  # 구 이전까지가 상호명
                if i + 1 < len(parts):
                    dong = parts[i + 1]  # 구 다음이 행정동명
                break
        
        # 구를 찾지 못한 경우 첫 번째 단어를 상호명으로
        if not base_cafe_name:
            base_cafe_name = parts[0] if parts else ""
        
        # 필터링: 상호명, 시군구명, 행정동명이 모두 일치하는 리뷰만
        cafe_reviews = pd.DataFrame()
        
        if '상호명' in df_all_reviews.columns:
            # 상호명으로 필터링
            mask = df_all_reviews['상호명'] == base_cafe_name
            
            # 시군구명으로 추가 필터링
            if district and '시군구명' in df_all_reviews.columns:
                mask = mask & (df_all_reviews['시군구명'] == district)
            
            # 행정동명으로 추가 필터링
            if dong and '행정동명' in df_all_reviews.columns:
                mask = mask & (df_all_reviews['행정동명'] == dong)
            
            cafe_reviews = df_all_reviews[mask].copy()
        
        if not cafe_reviews.empty:
            # 리뷰 수 표시
            st.caption(f"총 {len(cafe_reviews)}개의 리뷰")
            
            # 표시할 컬럼 선택
            display_cols = []
            if '평점' in cafe_reviews.columns:
                display_cols.append('평점')
            elif 'rating' in cafe_reviews.columns:
                display_cols.append('rating')
            
            if '리뷰' in cafe_reviews.columns:
                display_cols.append('리뷰')
            elif 'review_text' in cafe_reviews.columns:
                display_cols.append('review_text')
            
            # 사용 가능한 컬럼만 필터링
            available_cols = [col for col in display_cols if col in cafe_reviews.columns]
            
            if available_cols:
                # 리뷰 텍스트 추출
                review_text_col = None
                if '리뷰' in cafe_reviews.columns:
                    review_text_col = '리뷰'
                elif 'review_text' in cafe_reviews.columns:
                    review_text_col = 'review_text'
                
                if review_text_col:
                    # 리뷰 텍스트 수집 (최대 30개)
                    max_reviews_for_summary = min(30, len(cafe_reviews))
                    review_texts = cafe_reviews[review_text_col].head(max_reviews_for_summary).tolist()
                    review_texts = [str(text).strip() for text in review_texts if pd.notna(text) and str(text).strip()]
                    
                    if review_texts:
                        # OpenAI를 사용한 리뷰 요약 및 분석
                        with st.spinner("🤖 리뷰 분석 중..."):
                            summary = _generate_review_summary_with_openai(
                                cafe_name, 
                                review_texts, 
                                selected_factors or []
                            )
                        
                        if summary:
                            st.markdown("### 💡 공간 특성 요약")
                            st.markdown(summary.get('summary', ''))
                            
                            if summary.get('recommendation'):
                                st.markdown("### 🎯 추천 이유")
                                st.markdown(summary.get('recommendation', ''))
                        else:
                            # 요약 실패 시 기본 리뷰 표시
                            st.dataframe(
                                cafe_reviews[available_cols].head(10),
                                hide_index=True,
                                use_container_width=True
                            )
                    else:
                        st.info("리뷰 텍스트를 찾을 수 없습니다.")
                else:
                    # 리뷰 텍스트 컬럼이 없으면 기본 표시
                    st.dataframe(
                        cafe_reviews[available_cols].head(10),
                        hide_index=True,
                        use_container_width=True
                    )
            else:
                st.info("표시할 수 있는 리뷰 컬럼이 없습니다.")
        else:
            st.info(f"'{cafe_name}'에 해당하는 리뷰를 찾을 수 없습니다.")
                
    except Exception as e:
        st.warning(f"리뷰 데이터 로드 중 오류 발생: {e}")


def _get_openai_client():
    """OpenAI 클라이언트 초기화"""
    try:
        import os
        from openai import OpenAI
        
        api_key = None
        
        # 1. .env 파일에서 우선 로드 (로컬 및 배포 환경 모두)
        try:
            from dotenv import load_dotenv
            from modules.config import BASE_DIR
            env_path = BASE_DIR / ".env"
            if env_path.exists():
                # override=True: .env 파일의 값으로 환경 변수를 덮어씀
                load_dotenv(env_path, override=True)
                api_key = os.getenv("OPENAI_API_KEY")
        except ImportError:
            pass
        except Exception:
            pass
        
        # 2. .env에서 못 찾으면 환경 변수에서 확인 (배포 환경용)
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        # 3. Streamlit secrets에서 확인 (Streamlit Cloud용)
        if not api_key:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", None)
            except (AttributeError, FileNotFoundError, KeyError):
                pass
        
        if not api_key:
            return None
        
        return OpenAI(api_key=api_key)
    except ImportError:
        return None
    except Exception:
        return None


def _generate_review_summary_with_openai(cafe_name, review_texts, selected_factors):
    """OpenAI GPT-4o를 사용하여 리뷰 요약 및 추천 이유 생성"""
    client = _get_openai_client()
    if not client:
        return None
    
    # 리뷰 텍스트 결합 (최대 5000자)
    reviews_text = "\n".join(review_texts[:30])
    if len(reviews_text) > 5000:
        reviews_text = reviews_text[:5000] + "..."
    
    # 선택한 세부 항목 설명
    factor_descriptions = FACTOR_PREFERENCE_LABELS
    
    selected_factors_desc = [factor_descriptions.get(f, f) for f in selected_factors]
    
    prompt = f"""다음은 '{cafe_name}' 카페에 대한 고객 리뷰입니다. 장소성 관점에서 이 공간을 분석하고 요약해주세요.

## 리뷰 내용
{reviews_text}

## 사용자가 선택한 선호 특성
{', '.join(selected_factors_desc) if selected_factors_desc else '없음'}

## 요청 사항
1. **공간 특성 요약**: 리뷰를 바탕으로 이 카페의 공간적 특성을 장소성 관점에서 요약해주세요. 인테리어, 분위기, 공간 구성, 쾌적성, 접근성, 활동적 특성, 지역 정체성이나 개인의 경험에 의미가 있는지 의미적 특성 등을 포함해주세요.

2. **추천 이유**: 사용자가 선택한 선호 특성({', '.join(selected_factors) if selected_factors else '없음'})과 연결하여, 어떤 사용자에게 이 공간을 추천할 수 있는지, 그리고 추천 이유를 설명해주세요.

## 출력 형식 (JSON)
{{
  "summary": "공간 특성 요약 (3-5문단)",
  "recommendation": "추천 이유 및 대상 사용자 설명 (2-3문단)"
}}

JSON 형식으로만 응답해주세요."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1000,
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        st.warning(f"리뷰 요약 생성 중 오류: {str(e)}")
        return None


def render_cafe_factor_analysis():
    """카페별 요인 점수 분석 탭 렌더링"""
    st.header("카페별 요인 점수 분석")
    
    df_metrics, csv_path = _load_metrics_dataframe()

    if csv_path is None or df_metrics is None:
        st.error("⚠️ 결과 CSV 파일을 찾을 수 없습니다.")
        st.info(f"다음 파일 중 하나가 프로젝트 루트에 있는지 확인해주세요: {', '.join(PLACENESS_METRICS_CSV_CANDIDATES)}")
        return
    
    if df_metrics.empty:
        st.warning("로드된 데이터가 없습니다.")
        return

    st.caption(f"현재 분석 결과 파일: {csv_path.name}")
    
    # 카페 목록 가져오기
    cafe_list = sorted(df_metrics['cafe_name'].unique().tolist())
    
    if not cafe_list:
        st.warning("카페 데이터가 없습니다.")
        return
    
    # 카페 선택
    selected_cafe = st.selectbox(
        "카페 선택",
        options=cafe_list,
        key="cafe_factor_analysis_select",
        help="분석할 카페를 선택하세요"
    )
    
    if not selected_cafe:
        return
    
    # 선택한 카페의 데이터 추출
    cafe_data = df_metrics[df_metrics['cafe_name'] == selected_cafe].iloc[0]
    
    st.markdown("---")
    
    # 종합 점수 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        mu_score = cafe_data.get('종합_장소성_점수_Mu', 0)
        st.metric("종합 장소성 점수 (μ)", f"{mu_score:.3f}" if pd.notna(mu_score) else "N/A")
    with col2:
        sigma_score = cafe_data.get('요인_점수_표준편차_Sigma', 0)
        st.metric("표준편차 (σ)", f"{sigma_score:.3f}" if pd.notna(sigma_score) else "N/A")
    with col3:
        summary = cafe_data.get('Final_PlaceScore_Summary', 'N/A')
        st.metric("요약", summary if pd.notna(summary) else "N/A")
    
    st.markdown("---")
    
    # 요인별 점수 추출 (calc 컬럼 사용)
    factor_names = list(ALL_FACTORS.keys())
    factor_scores = {}
    
    # 사용 가능한 컬럼 목록 확인
    available_cols = cafe_data.index.tolist()
    
    for factor in factor_names:
        score = None
        
        # calc 컬럼 우선 사용
        calc_col = f'점수_{factor}_calc'
        if calc_col in available_cols:
            score = cafe_data[calc_col]
        else:
            # 일반 컬럼 사용
            normal_col = f'점수_{factor}'
            if normal_col in available_cols:
                score = cafe_data[normal_col]
        
        # 점수 처리 (0.5는 기본값이지만 유효한 데이터로 간주)
        if pd.notna(score):
            try:
                score_val = float(score)
                factor_scores[factor] = score_val
            except (ValueError, TypeError):
                factor_scores[factor] = None
        else:
            factor_scores[factor] = None
    
    # 요인별 점수 그래프 (방사형 차트)
    st.subheader("📈 요인별 점수 그래프")
    
    # 데이터 준비 (None이 아닌 모든 점수 포함, 0.5도 포함)
    valid_factors = {k: v for k, v in factor_scores.items() if v is not None}
    
    if valid_factors:
        # 요인을 카테고리별로 그룹화
        factor_categories = FACTOR_CATEGORIES
        
        if HAS_PLOTLY:
            # 카테고리별로 탭 생성
            tabs = st.tabs(["전체 요인", "물리적 특성", "활동적 특성", "의미적 특성"])
            
            def create_radar_chart(factors_dict, title, max_value=1.0):
                """방사형 차트 생성 함수"""
                theta = list(factors_dict.keys())
                r = list(factors_dict.values())
                
                # 차트를 닫기 위해 첫 번째 값을 마지막에 추가
                theta_closed = theta + [theta[0]]
                r_closed = r + [r[0]]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=r_closed,
                    theta=theta_closed,
                    fill='toself',
                    name='요인 점수',
                    line=dict(color='rgb(32, 201, 151)', width=2),
                    fillcolor='rgba(32, 201, 151, 0.25)',
                    hovertemplate='<b>%{theta}</b><br>점수: %{r:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max_value],
                            tickmode='linear',
                            tick0=0,
                            dtick=0.2,
                            tickfont=dict(size=10),
                            gridcolor='rgba(200, 200, 200, 0.3)'
                        ),
                        angularaxis=dict(
                            rotation=90,
                            direction='counterclockwise',
                            tickfont=dict(size=11)
                        )
                    ),
                    title=dict(
                        text=title,
                        x=0.5,
                        font=dict(size=16, color='#1f77b4')
                    ),
                    height=500,
                    showlegend=False,
                    paper_bgcolor='white',
                    plot_bgcolor='white'
                )
                
                return fig
            
            with tabs[0]:
                # 방사형 차트와 상세 점수를 나란히 배치
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    # 전체 요인 방사형 차트
                    fig_all = create_radar_chart(valid_factors, f"{selected_cafe}")
                    st.plotly_chart(fig_all, use_container_width=True, key=f"cafe_factor_radar_all_{selected_cafe}")
                
                with col_right:
                    # 상세 점수 보기
                    st.subheader("상세 점수 보기")
                    df_detail = pd.DataFrame({
                        '요인': list(valid_factors.keys()),
                        '점수': [f"{v:.3f}" for v in valid_factors.values()]
                    })
                    st.dataframe(df_detail, hide_index=True, **get_dataframe_width_param())
            
            # 카테고리별 탭
            for tab_idx, (category, factors) in enumerate(factor_categories.items(), 1):
                with tabs[tab_idx]:
                    category_scores = {k: v for k, v in valid_factors.items() if k in factors}
                    
                    if category_scores:
                        # 카테고리별 방사형 차트
                        fig_category = create_radar_chart(
                            category_scores, 
                            f"{selected_cafe} - {category}",
                            max_value=1.0
                        )
                        st.plotly_chart(fig_category, use_container_width=True, key=f"cafe_factor_radar_{category}_{selected_cafe}")
                        
                        # 평균 점수
                        avg_score = sum(category_scores.values()) / len(category_scores)
                        st.metric(f"{category} 평균 점수", f"{avg_score:.3f}")
                    else:
                        st.info(f"{category} 관련 데이터가 없습니다.")
        else:
            # plotly가 없으면 막대 그래프로 대체
            st.warning("⚠️ plotly가 설치되지 않아 막대 그래프로 표시됩니다. 방사형 차트를 보려면 `pip install plotly`를 실행하세요.")
            
            # 카테고리별로 탭 생성
            tabs = st.tabs(["전체 요인", "물리적 특성", "활동적 특성", "의미적 특성"])
            
            with tabs[0]:
                # 막대 그래프와 상세 점수를 나란히 배치
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    # 전체 요인 막대 그래프
                    df_chart = pd.DataFrame({
                        '요인': list(valid_factors.keys()),
                        '점수': list(valid_factors.values())
                    })
                    df_chart = df_chart.sort_values('점수', ascending=True)
                    
                    st.bar_chart(df_chart.set_index('요인'), height=400)
                
                with col_right:
                    # 상세 점수 보기
                    st.subheader("상세 점수 보기")
                    df_detail = pd.DataFrame({
                        '요인': list(valid_factors.keys()),
                        '점수': [f"{v:.3f}" for v in valid_factors.values()]
                    })
                    st.dataframe(df_detail, hide_index=True, **get_dataframe_width_param())
            
            # 카테고리별 탭
            for tab_idx, (category, factors) in enumerate(factor_categories.items(), 1):
                with tabs[tab_idx]:
                    category_scores = {k: v for k, v in valid_factors.items() if k in factors}
                    
                    if category_scores:
                        df_category = pd.DataFrame({
                            '요인': list(category_scores.keys()),
                            '점수': list(category_scores.values())
                        })
                        df_category = df_category.sort_values('점수', ascending=True)
                        
                        st.bar_chart(df_category.set_index('요인'), height=300)
                        
                        # 평균 점수
                        avg_score = sum(category_scores.values()) / len(category_scores)
                        st.metric(f"{category} 평균 점수", f"{avg_score:.3f}")
                    else:
                        st.info(f"{category} 관련 데이터가 없습니다.")
    else:
        st.warning("표시할 요인 점수 데이터가 없습니다.")
    
    st.markdown("---")
    
    # 강점/약점 요인 표시
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("(+)강점 요인")
        strength_factors = cafe_data.get('강점_요인(+df+)', 'N/A')
        if pd.notna(strength_factors) and strength_factors != 'N/A':
            # 쉼표로 구분된 요인들을 리스트로 변환
            if isinstance(strength_factors, str):
                strength_list = [f.strip() for f in strength_factors.split(',') if f.strip()]
            else:
                strength_list = []
            
            if strength_list:
                for factor in strength_list:
                    score = valid_factors.get(factor, None)
                    if score is not None:
                        st.success(f"**{factor}**: {score:.3f}")
                    else:
                        st.success(f"**{factor}**")
            else:
                st.info("강점 요인이 없습니다.")
        else:
            st.info("강점 요인이 없습니다.")
    
    with col2:
        st.subheader("(-)약점 요인")
        weakness_factors = cafe_data.get('약점_요인(-df-)', 'N/A')
        if pd.notna(weakness_factors) and weakness_factors != 'N/A':
            # 쉼표로 구분된 요인들을 리스트로 변환
            if isinstance(weakness_factors, str):
                weakness_list = [f.strip() for f in weakness_factors.split(',') if f.strip()]
            else:
                weakness_list = []
            
            if weakness_list:
                for factor in weakness_list:
                    score = valid_factors.get(factor, None)
                    if score is not None:
                        st.error(f"**{factor}**: {score:.3f}")
                    else:
                        st.error(f"**{factor}**")
            else:
                st.info("약점 요인이 없습니다.")
        else:
            st.info("약점 요인이 없습니다.")
    
    # 강점/약점 요인 아래에 리뷰 표시
    st.markdown("---")
    _display_cafe_reviews(selected_cafe)
    
    st.markdown("---")
    
    # 전체 데이터 표시 (확장 가능)
    with st.expander("📋 전체 데이터 보기"):
        # _calc로 끝나는 컬럼 제외
        filtered_data = cafe_data[~cafe_data.index.str.endswith('_calc', na=False)]
        st.dataframe(filtered_data.to_frame().T, **get_dataframe_width_param())


def render_cafe_recommendation():
    """카페 추천 탭 렌더링"""
    # 파스텔톤 primary 버튼 스타일 적용
    st.markdown("""
    <style>
    /* Primary 버튼 - 파스텔톤 하늘색 */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #BEE3F8 !important; /* 부드러운 파우더 블루 */
        color: #2C5282 !important; /* 텍스트는 짙은 네이비로 가독성 확보 */
        border: none !important;
        border-radius: 12px !important; /* 둥글게 처리하면 더 귀여움 */
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #90CDF4 !important; /* 호버 시 조금 더 진한 하늘색 */
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(144, 205, 244, 0.4) !important;
    }
    
    /* Multiselect 선택된 태그 - 파스텔톤 노란색 */
    div[data-baseweb="select"] p[data-baseweb="tag"],
    div[data-baseweb="select"] span[data-baseweb="tag"],
    div[data-baseweb="select"] div[data-baseweb="tag"],
    div[data-testid="stMultiSelect"] p[data-baseweb="tag"],
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"],
    div[data-testid="stMultiSelect"] div[data-baseweb="tag"] {
        background-color: #FEF3C7 !important; /* 부드러운 파스텔톤 노란색 */
        color: #92400E !important; /* 텍스트는 짙은 갈색으로 가독성 확보 */
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    /* Multiselect 태그의 X 버튼 */
    div[data-baseweb="select"] button[aria-label],
    div[data-testid="stMultiSelect"] button[aria-label] {
        color: #92400E !important;
    }
    
    /* Multiselect 태그 호버 효과 */
    div[data-baseweb="select"] p[data-baseweb="tag"]:hover,
    div[data-baseweb="select"] span[data-baseweb="tag"]:hover,
    div[data-baseweb="select"] div[data-baseweb="tag"]:hover,
    div[data-testid="stMultiSelect"] p[data-baseweb="tag"]:hover,
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"]:hover,
    div[data-testid="stMultiSelect"] div[data-baseweb="tag"]:hover {
        background-color: #FDE68A !important; /* 호버 시 조금 더 진한 노란색 */
    }
    </style>
    """, unsafe_allow_html=True)

    st.caption("선호하는 특성에 맞는 카페를 추천해드립니다.")
    
    df_metrics, csv_path = _load_metrics_dataframe()

    if csv_path is None or df_metrics is None:
        st.error("⚠️ 결과 CSV 파일을 찾을 수 없습니다.")
        st.info(f"다음 파일 중 하나가 프로젝트 루트에 있는지 확인해주세요: {', '.join(PLACENESS_METRICS_CSV_CANDIDATES)}")
        return
    
    if df_metrics.empty:
        st.warning("로드된 데이터가 없습니다.")
        return

    st.caption(f"현재 추천 기준 파일: {csv_path.name}")
    
    # 행정구 파싱 (cafe_name에서 추출)
    # 서울시 실제 행정구 목록
    SEOUL_DISTRICTS = [
        "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구",
        "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구",
        "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구",
        "서초구", "강남구", "송파구", "강동구"
    ]
    
    def parse_district(cafe_name):
        """카페명에서 실제 행정구만 추출"""
        if pd.isna(cafe_name):
            return None
        parts = str(cafe_name).split()
        for part in parts:
            # 실제 서울시 행정구 목록에 있는 것만 반환
            if part in SEOUL_DISTRICTS:
                return part
        return None
    
    df_metrics['행정구'] = df_metrics['cafe_name'].apply(parse_district)
    available_districts = sorted([d for d in df_metrics['행정구'].dropna().unique() if d])
    
    # 1. 행정구 선택
    st.subheader("📍 지역 선택")
    selected_districts = st.multiselect(
        "원하는 행정구를 선택하세요 (복수 선택 가능, 선택하지 않으면 전체 지역)",
        options=available_districts,
        default=[],
        key="recommendation_districts"
    )
    
    # 필터링
    if selected_districts:
        df_filtered = df_metrics[df_metrics['행정구'].isin(selected_districts)].copy()
    else:
        df_filtered = df_metrics.copy()
    
    if df_filtered.empty:
        st.warning("선택한 지역에 해당하는 카페가 없습니다.")
        return
    
    st.markdown("---")
    
    # 2. 선호 특성 선택 (가로 버튼 배치)
    st.subheader("선호하는 특성 선택")
    
    # 세션 상태 초기화 (안전하게)
    if 'recommendation_preference_type' not in st.session_state:
        st.session_state.recommendation_preference_type = None
    if 'recommendation_selected_details' not in st.session_state:
        st.session_state.recommendation_selected_details = []
    
    # preference_type을 세션 상태에서 읽기 (항상 최신 상태 보장)
    preference_type = st.session_state.get('recommendation_preference_type', None)
    
    # 가로로 3개 버튼 배치
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏛️ 물리적 특성", 
                    use_container_width=True,
                    type="primary" if preference_type == "물리적 특성" else "secondary",
                    key="btn_physical"):
            st.session_state.recommendation_preference_type = "물리적 특성"
            st.session_state.recommendation_selected_details = []
            st.rerun()
    
    with col2:
        if st.button("🎭 활동적 특성",
                    use_container_width=True,
                    type="primary" if preference_type == "활동적 특성" else "secondary",
                    key="btn_activity"):
            st.session_state.recommendation_preference_type = "활동적 특성"
            st.session_state.recommendation_selected_details = []
            st.rerun()
    
    with col3:
        if st.button("💭 의미적 특성",
                    use_container_width=True,
                    type="primary" if preference_type == "의미적 특성" else "secondary",
                    key="btn_semantic"):
            st.session_state.recommendation_preference_type = "의미적 특성"
            st.session_state.recommendation_selected_details = []
            st.rerun()
    
    # 3. 세부 항목 선택 (선택된 특성 아래에 표시)
    # preference_type을 다시 한 번 확인 (버튼 클릭 후 최신 상태)
    preference_type = st.session_state.get('recommendation_preference_type', None)
    
    if preference_type:
        st.markdown("---")
        st.subheader(f"{preference_type} 세부 항목 선택")
        
        detail_options = [
            (factor_name, FACTOR_PREFERENCE_LABELS.get(factor_name, factor_name))
            for factor_name in FACTOR_CATEGORIES.get(preference_type, [])
        ]
        
        # 세부 항목을 버튼으로 표시 (수형도처럼)
        st.markdown("<div style='margin-left: 20px;'>", unsafe_allow_html=True)
        
        # 버튼을 그리드로 배치 (2열)
        num_cols = 2
        cols = st.columns(num_cols)
        
        for idx, (factor_key, factor_desc) in enumerate(detail_options):
            col_idx = idx % num_cols
            with cols[col_idx]:
                # 세션 상태에서 최신 선택 상태 확인
                current_selected = st.session_state.get('recommendation_selected_details', [])
                is_selected = factor_key in current_selected
                button_type = "primary" if is_selected else "secondary"
                
                if st.button(
                    f"✓ {factor_desc}" if is_selected else factor_desc,
                    use_container_width=True,
                    type=button_type,
                    key=f"detail_btn_{preference_type}_{factor_key}"
                ):
                    # preference_type이 유지되도록 명시적으로 보장
                    st.session_state.recommendation_preference_type = preference_type
                    
                    # 세션 상태에서 최신 리스트 가져오기
                    if 'recommendation_selected_details' not in st.session_state:
                        st.session_state.recommendation_selected_details = []
                    
                    # 리스트 복사본으로 작업 (참조 문제 방지)
                    current_list = list(st.session_state.recommendation_selected_details)
                    
                    if is_selected:
                        # 선택 해제
                        if factor_key in current_list:
                            current_list.remove(factor_key)
                    else:
                        # 선택 추가
                        if factor_key not in current_list:
                            current_list.append(factor_key)
                    
                    # 세션 상태 업데이트
                    st.session_state.recommendation_selected_details = current_list
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        selected_details = st.session_state.get('recommendation_selected_details', [])
    else:
        selected_details = []
    
    # 4. 추천 실행
    if preference_type:
        st.markdown("---")
        if st.button("🔍 추천 받기", type="primary", key="recommendation_search", use_container_width=True):
            if not selected_details:
                st.warning("⚠️ 최소 하나의 세부 항목을 선택해주세요.")
            else:
                # 추천 로직
                recommendations = _calculate_recommendations(df_filtered, selected_details)
                st.caption("추천 점수는 선택한 세부 요인의 점수를 동일 가중 평균한 값입니다. 값이 비어 있으면 중립값 0.5로 반영합니다.")
                
                if recommendations.empty:
                    st.warning("선택한 조건에 맞는 카페를 찾을 수 없습니다.")
                else:
                    st.success(f"✅ {len(recommendations)}개의 카페를 찾았습니다!")
                    st.markdown("---")
                    
                    # 상위 3개 추천
                    top_3 = recommendations.head(3)
                    
                    for idx, (_, row) in enumerate(top_3.iterrows(), 1):
                        with st.container():
                            # 카페 이름과 기본 정보
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"### {idx}. {row['cafe_name']}")
                                if pd.notna(row.get('행정구')):
                                    st.caption(f"📍 {row['행정구']}")
                            
                            with col2:
                                mu_score = row.get('종합_장소성_점수_Mu', 0)
                                if pd.notna(mu_score) and mu_score > 0:
                                    st.metric("종합 점수 (μ)", f"{mu_score:.3f}")
                                else:
                                    st.metric("종합 점수 (μ)", "N/A")
                            
                            with col3:
                                sigma_score = row.get('요인_점수_표준편차_Sigma', 0)
                                if pd.notna(sigma_score) and sigma_score > 0:
                                    st.metric("표준편차 (σ)", f"{sigma_score:.3f}")
                                else:
                                    st.metric("표준편차 (σ)", "N/A")
                        
                            
                            # 요인별 점수 추출 (calc 컬럼 우선 사용)
                            factor_names = list(ALL_FACTORS.keys())
                            factor_scores = {}
                            
                            for factor in factor_names:
                                score = None
                                
                                # calc 컬럼 우선 사용
                                calc_col = f'점수_{factor}_calc'
                                if calc_col in row.index:
                                    score = row[calc_col]
                                else:
                                    # 일반 컬럼 사용
                                    normal_col = f'점수_{factor}'
                                    if normal_col in row.index:
                                        score = row[normal_col]
                                
                                # 점수 처리
                                if pd.notna(score):
                                    try:
                                        score_val = float(score)
                                        factor_scores[factor] = score_val
                                    except (ValueError, TypeError):
                                        factor_scores[factor] = None
                                else:
                                    factor_scores[factor] = None
                            
                            # 유효한 요인만 필터링
                            valid_factors = {k: v for k, v in factor_scores.items() if v is not None}
                            
                            if valid_factors and HAS_PLOTLY:
                                # Radial Chart 표시
                                st.subheader("📈 요인별 점수 그래프")
                                
                                def create_radar_chart(factors_dict, title, max_value=1.0):
                                    """방사형 차트 생성 함수"""
                                    theta = list(factors_dict.keys())
                                    r = list(factors_dict.values())
                                    
                                    # 차트를 닫기 위해 첫 번째 값을 마지막에 추가
                                    theta_closed = theta + [theta[0]]
                                    r_closed = r + [r[0]]
                                    
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Scatterpolar(
                                        r=r_closed,
                                        theta=theta_closed,
                                        fill='toself',
                                        name='요인 점수',
                                        line=dict(color='rgb(32, 201, 151)', width=2),
                                        fillcolor='rgba(32, 201, 151, 0.25)',
                                        hovertemplate='<b>%{theta}</b><br>점수: %{r:.3f}<extra></extra>'
                                    ))
                                    
                                    fig.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                                range=[0, max_value],
                                                tickmode='linear',
                                                tick0=0,
                                                dtick=0.2,
                                                tickfont=dict(size=10),
                                                gridcolor='rgba(200, 200, 200, 0.3)'
                                            ),
                                            angularaxis=dict(
                                                rotation=90,
                                                direction='counterclockwise',
                                                tickfont=dict(size=11)
                                            )
                                        ),
                                        title=dict(
                                            text=title,
                                            x=0.5,
                                            font=dict(size=16, color='#1f77b4')
                                        ),
                                        height=500,
                                        showlegend=False,
                                        paper_bgcolor='white',
                                        plot_bgcolor='white'
                                    )
                                    
                                    return fig
                                
                                # 전체 요인 방사형 차트
                                cafe_name = row['cafe_name']
                                fig_all = create_radar_chart(valid_factors, f"{cafe_name}")
                                st.plotly_chart(fig_all, use_container_width=True, key=f"recommendation_radar_{idx}_{cafe_name}")
                            elif valid_factors:
                                # plotly가 없으면 막대 그래프로 대체
                                st.subheader("📈 요인별 점수 그래프")
                                st.warning("⚠️ plotly가 설치되지 않아 막대 그래프로 표시됩니다.")
                                df_chart = pd.DataFrame({
                                    '요인': list(valid_factors.keys()),
                                    '점수': list(valid_factors.values())
                                })
                                df_chart = df_chart.sort_values('점수', ascending=True)
                                st.bar_chart(df_chart.set_index('요인'), height=400)
                            
                            # 카페 리뷰 표시 및 요약
                            _display_cafe_reviews_for_recommendation(row['cafe_name'], selected_details)
                            
                            st.markdown("---")


def _calculate_recommendations_legacy(df: pd.DataFrame, selected_factors: list) -> pd.DataFrame:
    """선택한 요인에 따라 카페를 추천합니다."""
    # 각 요인별 점수 컬럼명
    factor_score_cols = [f"점수_{factor}" for factor in selected_factors]
    
    # 유효한 점수 컬럼만 사용
    valid_cols = [col for col in factor_score_cols if col in df.columns]
    
    if not valid_cols:
        return pd.DataFrame()
    
    # 각 요인별 점수 계산 (0.5는 기본값이므로 제외)
    df_scored = df.copy()
    
    # 추천 점수 계산: 선택한 요인들의 평균 점수
    scores = []
    for _, row in df_scored.iterrows():
        factor_scores = []
        for col in valid_cols:
            score = row[col]
            if pd.notna(score) and score != 0.5:  # 기본값 제외
                factor_scores.append(score)
        
        if factor_scores:
            avg_score = sum(factor_scores) / len(factor_scores)
            scores.append(avg_score)
        else:
            scores.append(0)
    
    df_scored['추천_점수'] = scores
    
    # 추천 점수가 0보다 큰 카페만 필터링
    df_scored = df_scored[df_scored['추천_점수'] > 0].copy()
    
    # 추천 점수 기준으로 정렬 (내림차순)
    df_scored = df_scored.sort_values('추천_점수', ascending=False)
    
    return df_scored


def _calculate_recommendations(df: pd.DataFrame, selected_factors: list) -> pd.DataFrame:
    """선택한 요인에 따라 카페를 추천합니다."""
    if not selected_factors:
        return pd.DataFrame()

    def _resolve_factor_score(row: pd.Series, factor: str) -> float:
        """선택된 요인은 모두 동일 가중으로 반영하고, 빈 값은 중립값 0.5로 처리합니다."""
        candidate_cols = [f"점수_{factor}_calc", f"점수_{factor}"]

        for col in candidate_cols:
            if col not in row.index:
                continue

            score = row[col]
            if pd.isna(score):
                continue

            try:
                return float(score)
            except (TypeError, ValueError):
                continue

        return 0.5

    df_scored = df.copy()
    scores = []

    for _, row in df_scored.iterrows():
        factor_scores = [_resolve_factor_score(row, factor) for factor in selected_factors]
        scores.append(sum(factor_scores) / len(factor_scores))

    df_scored["추천_점수"] = scores
    df_scored = df_scored.sort_values("추천_점수", ascending=False)

    return df_scored


@st.cache_data
def _load_multimodal_demo_metrics():
    """멀티모달 데모에서 사용할 기존 장소성 점수 CSV를 로드합니다."""
    df_metrics, _csv_path = _load_metrics_dataframe()
    return df_metrics


def _open_uploaded_image(uploaded_file):
    """업로드된 이미지를 RGB PIL 이미지로 엽니다."""
    if not HAS_PIL:
        return None

    try:
        return Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    except Exception:
        return None


def _calculate_image_stats(image):
    """간단한 시각 통계량을 계산합니다."""
    if image is None:
        return None

    image_np = np.asarray(image).astype(np.float32)
    if image_np.ndim != 3 or image_np.shape[2] < 3:
        return None

    red = image_np[:, :, 0]
    green = image_np[:, :, 1]
    blue = image_np[:, :, 2]

    brightness = float(image_np.mean())
    contrast = float(image_np.std())

    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    colorfulness = float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) +
        0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )

    width, height = image.size
    return {
        "width": int(width),
        "height": int(height),
        "brightness": brightness,
        "contrast": contrast,
        "colorfulness": colorfulness,
        "megapixels": float((width * height) / 1_000_000),
    }
def _build_3dgs_concept_result(image_stats_list, shot_types, capture_sequence):
    """실제 3DGS 재구성 대신 촬영 준비도를 계산합니다."""
    valid_stats = [stats for stats in image_stats_list if stats is not None]
    image_count = len(valid_stats)
    unique_shot_types = {shot_type for shot_type in shot_types if shot_type and shot_type != "미분류"}

    count_score = min(40, image_count * 4)
    diversity_score = min(25, len(unique_shot_types) * 6)
    sequence_score = 20 if capture_sequence else 0

    avg_megapixels = float(np.mean([stats["megapixels"] for stats in valid_stats])) if valid_stats else 0.0
    resolution_score = 10 if avg_megapixels >= 1.2 else 5 if avg_megapixels >= 0.6 else 0

    coverage_bonus = 5 if {"실내 전경", "좌석 영역", "카운터/바", "창가/뷰"}.intersection(unique_shot_types) else 0
    readiness_score = int(min(100, round(count_score + diversity_score + sequence_score + resolution_score + coverage_bonus)))

    if readiness_score >= 75:
        readiness_label = "높음"
    elif readiness_score >= 45:
        readiness_label = "보통"
    else:
        readiness_label = "낮음"

    extractable_metrics = []
    if {"실내 전경", "좌석 영역"}.issubset(unique_shot_types):
        extractable_metrics.append("좌석 배치와 동선 연속성")
    if {"창가/뷰", "외관"}.intersection(unique_shot_types):
        extractable_metrics.append("개방감과 시야 축")
    if {"카운터/바", "실내 전경"}.issubset(unique_shot_types):
        extractable_metrics.append("출입구-카운터-좌석 간 구획 구조")
    if "디테일/소품" in unique_shot_types:
        extractable_metrics.append("재료감과 콘셉트 디테일 앵커링")
    if not extractable_metrics:
        extractable_metrics.append("현재 업로드 구성만으로는 공간 구조 지표 추출이 제한적")

    recommendations = []
    if image_count < 8:
        recommendations.append("3DGS용으로는 최소 8장 이상, 가능하면 12장 이상을 권장합니다.")
    if len(unique_shot_types) < 3:
        recommendations.append("실내 전경, 좌석 영역, 카운터, 창가/뷰 등 서로 다른 시점이 필요합니다.")
    if not capture_sequence:
        recommendations.append("연속 동선으로 촬영한 이미지나 walkthrough 영상이 있으면 재구성 안정성이 높아집니다.")
    if avg_megapixels < 1.2:
        recommendations.append("해상도가 더 높은 이미지를 사용하면 세부 구조 복원에 유리합니다.")
    if not recommendations:
        recommendations.append("현재 업로드 구성은 데모 수준의 3DGS 실험 설계에 비교적 적합합니다.")

    return {
        "readiness_score": readiness_score,
        "readiness_label": readiness_label,
        "image_count": image_count,
        "shot_type_count": len(unique_shot_types),
        "avg_megapixels": avg_megapixels,
        "extractable_metrics": extractable_metrics,
        "recommendations": recommendations,
    }


def render_multimodal_space_demo(df_reviews: pd.DataFrame):
    """수동 업로드 이미지 기반의 멀티모달 공간 분석 데모 탭을 렌더링합니다."""
    st.header("🖼️ 멀티모달 공간 분석 데모")
    st.caption("카페를 선택하고 수동으로 찾은 이미지를 업로드하면, 실제 VLM으로 장소성 요인을 분석하고 3DGS 촬영 준비도는 별도 개념 결과로 보여줍니다.")
    st.info("VLM 결과는 실제 OpenAI vision 모델을 호출합니다. 3DGS 영역은 아직 재구성 실행이 아니라 촬영 준비도/설계 가이드입니다.")

    cafe_options = sorted(df_reviews["cafe_name"].dropna().unique().tolist())
    if not cafe_options:
        st.info("선택할 수 있는 카페 데이터가 없습니다.")
        return

    metrics_df = _load_multimodal_demo_metrics()

    selected_cafe = st.selectbox(
        "카페 선택",
        options=cafe_options,
        key="multimodal_demo_cafe_select",
        help="기존 텍스트 기반 장소성 점수와 비교할 카페를 선택하세요."
    )

    if metrics_df is not None and "cafe_name" in metrics_df.columns:
        matched_metrics = metrics_df[metrics_df["cafe_name"] == selected_cafe]
        if not matched_metrics.empty:
            cafe_metrics = matched_metrics.iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                mu_score = cafe_metrics.get("종합_장소성_점수_Mu", 0)
                st.metric("기존 텍스트 점수 μ", f"{mu_score:.3f}" if pd.notna(mu_score) else "N/A")
            with col2:
                sigma_score = cafe_metrics.get("요인_점수_표준편차_Sigma", 0)
                st.metric("기존 표준편차 σ", f"{sigma_score:.3f}" if pd.notna(sigma_score) else "N/A")
            with col3:
                st.metric("기존 요약", cafe_metrics.get("Final_PlaceScore_Summary", "N/A"))

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "카페 공간 이미지를 업로드하세요",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="multimodal_demo_uploader",
        help="실내 전경, 좌석, 카운터, 외관, 창가 뷰 등을 여러 장 올리면 모델이 더 안정적으로 분석할 수 있습니다."
    )

    capture_sequence = st.checkbox(
        "연속된 동선으로 촬영한 이미지다 (3DGS 실험 가정)",
        value=False,
        key="multimodal_demo_sequence"
    )

    analyst_note = st.text_area(
        "분석 메모 (선택)",
        placeholder="예: 한옥 리노베이션 느낌, 외부 골목과의 연결감이 좋음, 포토존이 확실함 등",
        key="multimodal_demo_note"
    )

    model_name = st.selectbox(
        "VLM 모델",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
        key="multimodal_demo_model_name",
        help="`gpt-4o-mini`는 비용이 낮고 빠르며, `gpt-4o`는 더 강한 해석 품질을 기대할 수 있습니다."
    )

    if not uploaded_files:
        st.info("이미지를 업로드하면 여기서 실제 VLM 결과와 3DGS 준비도 결과가 표시됩니다.")
        return

    st.markdown("---")
    st.subheader("업로드 이미지")

    image_stats_list = []
    shot_types = []
    preview_rows = []

    for idx, uploaded_file in enumerate(uploaded_files):
        image = _open_uploaded_image(uploaded_file)
        stats = _calculate_image_stats(image)
        image_stats_list.append(stats)

        with st.expander(f"이미지 {idx + 1}: {uploaded_file.name}", expanded=(idx == 0)):
            col1, col2 = st.columns([1, 1])

            with col1:
                if image is not None:
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                else:
                    st.warning("이미지 미리보기를 불러오지 못했습니다.")

            with col2:
                shot_type = st.selectbox(
                    "촬영 유형",
                    options=["미분류", "실내 전경", "좌석 영역", "카운터/바", "창가/뷰", "외관", "디테일/소품"],
                    index=0,
                    key=f"multimodal_demo_shot_type_{idx}"
                )
                shot_types.append(shot_type)

                if stats is not None:
                    st.metric("해상도", f"{stats['width']} × {stats['height']}")
                    st.metric("밝기", f"{stats['brightness']:.1f}")
                    st.metric("색채 다양성", f"{stats['colorfulness']:.1f}")
                    preview_rows.append({
                        "파일명": uploaded_file.name,
                        "촬영 유형": shot_type,
                        "해상도": f"{stats['width']} × {stats['height']}",
                        "밝기": f"{stats['brightness']:.1f}",
                        "색채 다양성": f"{stats['colorfulness']:.1f}",
                    })
                else:
                    st.info("이미지 통계를 계산할 수 없어 업로드 파일 정보만 사용합니다.")
                    preview_rows.append({
                        "파일명": uploaded_file.name,
                        "촬영 유형": shot_type,
                        "해상도": "N/A",
                        "밝기": "N/A",
                        "색채 다양성": "N/A",
                    })

    st.dataframe(pd.DataFrame(preview_rows), hide_index=True, **get_dataframe_width_param())

    if len(uploaded_files) > 10:
        st.warning("현재는 비용과 응답 시간을 고려해 최대 10장까지만 실제 VLM 분석에 사용합니다.")

    vlm_fingerprint = build_vlm_fingerprint(
        cafe_name=selected_cafe,
        uploaded_files=uploaded_files[:10],
        shot_types=shot_types[:10],
        analyst_note=analyst_note,
        model_name=model_name,
    )

    previous_fingerprint = st.session_state.get("multimodal_vlm_fingerprint")
    if previous_fingerprint != vlm_fingerprint:
        st.session_state.multimodal_vlm_result = None
        st.session_state.multimodal_vlm_error = None
        st.session_state.multimodal_vlm_fingerprint = vlm_fingerprint

    analyze_clicked = st.button(
        "실제 VLM 분석 실행",
        type="primary",
        key="multimodal_demo_run_vlm",
        use_container_width=True
    )

    if analyze_clicked:
        with st.spinner(f"{model_name} 모델로 업로드 이미지를 분석 중입니다..."):
            try:
                vlm_result = analyze_cafe_images_with_openai(
                    cafe_name=selected_cafe,
                    uploaded_files=uploaded_files[:10],
                    shot_types=shot_types[:10],
                    analyst_note=analyst_note,
                    model_name=model_name,
                    streamlit_secrets=st.secrets,
                )
                st.session_state.multimodal_vlm_result = vlm_result
                st.session_state.multimodal_vlm_error = None
            except Exception as e:
                st.session_state.multimodal_vlm_result = None
                st.session_state.multimodal_vlm_error = f"{e}\n\n{traceback.format_exc()}"

    vlm_result = st.session_state.get("multimodal_vlm_result")
    vlm_error = st.session_state.get("multimodal_vlm_error")
    gs_result = _build_3dgs_concept_result(image_stats_list, shot_types, capture_sequence)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("실제 VLM 분석 결과")

        if vlm_error:
            st.error("실제 VLM 분석 중 오류가 발생했습니다.")
            st.code(vlm_error)
        elif vlm_result is None:
            st.info("`실제 VLM 분석 실행` 버튼을 누르면 업로드 이미지들을 실제 비전 모델로 분석합니다.")
        else:
            st.caption(f"모델: `{vlm_result.get('model_name', model_name)}`")
            if vlm_result.get("overall_summary"):
                st.markdown("#### 요약")
                st.write(vlm_result["overall_summary"])

            result_rows = []
            factor_score_map = {}
            for item in vlm_result.get("factors", []):
                factor_name = item.get("name")
                score = float(item.get("score", 0.5))
                confidence = float(item.get("confidence", 0.3))
                factor_score_map[factor_name] = score
                evidence = " / ".join(item.get("evidence", [])[:2]) if item.get("evidence") else "근거 없음"
                result_rows.append({
                    "요인": factor_name,
                    "VLM 점수": round(score, 3),
                    "신뢰도": round(confidence, 3),
                    "시각 근거": evidence,
                })

            result_df = pd.DataFrame(result_rows).sort_values("VLM 점수", ascending=False)
            st.dataframe(result_df, hide_index=True, **get_dataframe_width_param())

            if HAS_PLOTLY and factor_score_map:
                factor_names = list(factor_score_map.keys())
                factor_values = [factor_score_map[factor] for factor in factor_names]
                theta_closed = factor_names + [factor_names[0]]
                r_closed = factor_values + [factor_values[0]]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=r_closed,
                    theta=theta_closed,
                    fill="toself",
                    name="VLM 점수",
                    line=dict(color="rgb(8, 145, 178)", width=2),
                    fillcolor="rgba(8, 145, 178, 0.18)",
                    hovertemplate="<b>%{theta}</b><br>점수: %{r:.3f}<extra></extra>"
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1]),
                        angularaxis=dict(rotation=90, direction="counterclockwise")
                    ),
                    height=520,
                    showlegend=False,
                    title=dict(text=f"{selected_cafe} 실제 VLM 기반 요인 점수", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True, key=f"multimodal_demo_radar_{selected_cafe}")
            elif factor_score_map:
                chart_df = pd.DataFrame({
                    "요인": list(factor_score_map.keys()),
                    "점수": list(factor_score_map.values())
                }).sort_values("점수", ascending=True)
                st.bar_chart(chart_df.set_index("요인"), height=450)

            with st.expander("요인별 상세 근거 보기"):
                for item in vlm_result.get("factors", []):
                    st.markdown(f"**{item.get('name', '요인')}**")
                    st.write(f"- 점수: {float(item.get('score', 0.5)):.3f}")
                    st.write(f"- 신뢰도: {float(item.get('confidence', 0.3)):.3f}")
                    st.write(f"- 시각 근거 부족 여부: {'예' if item.get('insufficient_visual_evidence') else '아니오'}")
                    visible_cues = item.get("visible_cues", [])
                    evidence = item.get("evidence", [])
                    if visible_cues:
                        st.write(f"- visible cues: {', '.join(visible_cues)}")
                    if evidence:
                        st.write(f"- evidence: {' / '.join(evidence)}")

            if vlm_result.get("cross_image_observations"):
                st.markdown("#### 다중 이미지 종합 관찰")
                for observation in vlm_result["cross_image_observations"]:
                    st.write(f"- {observation}")

            if vlm_result.get("limitations"):
                st.markdown("#### 시각 분석 한계")
                for limitation in vlm_result["limitations"]:
                    st.write(f"- {limitation}")

    with col_right:
        st.subheader("3DGS Concept 결과")
        st.metric("재구성 준비도", f"{gs_result['readiness_score']}/100")
        st.metric("준비도 레이블", gs_result["readiness_label"])
        st.metric("업로드 이미지 수", f"{gs_result['image_count']}장")
        st.metric("촬영 유형 다양성", f"{gs_result['shot_type_count']}종")
        st.metric("평균 해상도", f"{gs_result['avg_megapixels']:.2f} MP")

        st.markdown("#### 추후 추출 가능성이 있는 공간 지표")
        for metric_name in gs_result["extractable_metrics"]:
            st.write(f"- {metric_name}")

        st.markdown("#### 다음 단계 제안")
        for recommendation in gs_result["recommendations"]:
            st.write(f"- {recommendation}")

        if vlm_result is not None and vlm_result.get("recommended_3dgs_capture"):
            st.markdown("#### VLM이 제안한 추가 촬영 컷")
            for recommendation in vlm_result["recommended_3dgs_capture"]:
                st.write(f"- {recommendation}")

    if metrics_df is not None and "cafe_name" in metrics_df.columns and vlm_result is not None:
        matched_metrics = metrics_df[metrics_df["cafe_name"] == selected_cafe]
        if not matched_metrics.empty:
            cafe_metrics = matched_metrics.iloc[0]
            factor_score_map = {
                item.get("name"): float(item.get("score", 0.5))
                for item in vlm_result.get("factors", [])
                if item.get("name")
            }
            compare_rows = []
            for factor in ALL_FACTORS.keys():
                text_score = cafe_metrics.get(f"점수_{factor}_calc", cafe_metrics.get(f"점수_{factor}", np.nan))
                image_score = factor_score_map.get(factor, np.nan)
                compare_rows.append({
                    "요인": factor,
                    "텍스트 점수": round(float(text_score), 3) if pd.notna(text_score) else np.nan,
                    "이미지 VLM 점수": round(float(image_score), 3) if pd.notna(image_score) else np.nan,
                    "차이(이미지-텍스트)": round(float(image_score) - float(text_score), 3) if pd.notna(text_score) and pd.notna(image_score) else np.nan,
                })

            st.markdown("---")
            st.subheader("텍스트 기반 점수와 비교")
            st.dataframe(pd.DataFrame(compare_rows), hide_index=True, **get_dataframe_width_param())


def render_sentiment_analysis_saved(df_reviews, sentiment_pipeline, sentiment_model_name):
    """Render sentiment results from saved CSVs by default, with optional recomputation."""
    st.header("2. 개별 리뷰 감성 분석 및 카페별 평균")

    precomputed_reviews_csv = st.session_state.get("precomputed_reviews_with_sentiment_csv")
    precomputed_avg_csv = st.session_state.get("precomputed_avg_sentiment_csv")

    if st.session_state.df_reviews_with_sentiment is not None and st.session_state.df_avg_sentiment is not None:
        source_csvs = [name for name in [precomputed_reviews_csv, precomputed_avg_csv] if name]
        if source_csvs:
            st.success(f"저장된 감성 분석 결과 CSV를 불러와 표시 중입니다: `{', '.join(source_csvs)}`")

        st.subheader("카페별 평균 감성 점수")
        st.dataframe(st.session_state.df_avg_sentiment.set_index("cafe_name"), **get_dataframe_width_param())

        st.subheader("개별 리뷰 감성 분석 결과 샘플")
        sample_df = st.session_state.df_reviews_with_sentiment[
            ["cafe_name", "review_text", "sentiment_label", "sentiment_score"]
        ].head(20)
        st.dataframe(sample_df, **get_dataframe_width_param())

        csv = st.session_state.df_reviews_with_sentiment.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "감성 분석 결과 CSV 다운로드",
            data=csv,
            file_name="reviews_with_sentiment_real.csv",
            mime="text/csv",
            key="download_reviews_with_sentiment_real",
        )

        st.markdown("---")

    if False and st.button("개별 리뷰 감성 분석 다시 계산", type="secondary", key="sentiment_analysis_saved_start"):
        with st.spinner("개별 리뷰 긍정/부정 감성 점수 계산 중...(KoBERT/KoELECTRA)..."):
            try:
                df_reviews_with_sentiment, df_avg_sentiment = run_sentiment_analysis(
                    df_reviews.copy(),
                    sentiment_pipeline,
                    sentiment_model_name,
                )

                st.session_state.df_reviews_with_sentiment = df_reviews_with_sentiment
                st.session_state.df_avg_sentiment = df_avg_sentiment
                st.session_state.precomputed_reviews_with_sentiment_csv = None
                st.session_state.precomputed_avg_sentiment_csv = None
            except Exception as e:
                st.error(f"감성 분석 중 오류 발생: {e}")
                st.code(traceback.format_exc())
                return

            st.rerun()
