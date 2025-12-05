"""
Streamlit UI 구성 모듈
"""
import streamlit as st
import pandas as pd
from modules.config import ALL_FACTORS, SIMILARITY_THRESHOLD
from modules.sentiment import run_sentiment_analysis
from modules.placeness_score import calculate_place_scores, calculate_final_research_metrics
from modules.preprocess import load_csv_raw, is_numeric_only, is_metadata_only, truncate_text_for_bert
from modules.sentiment import process_sentiment_result


def render_data_preview(file_path, sentiment_pipeline, sentiment_model_name):
    """데이터 미리보기 섹션 렌더링"""
    st.markdown("---")
    st.header("📋 데이터 미리보기")
    
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
            use_container_width=True,
            hide_index=True,
            height=600
        )
        st.caption(f"전체 {len(df_preview_sorted):,}개 리뷰 (행정구별 정렬)")
        
        # 감성 분석 추가 버튼
        st.markdown("---")
        
        # 세션 상태에 저장된 결과가 있으면 표시
        if st.session_state.preview_sentiment_result is not None:
            df_preview_with_sentiment = st.session_state.preview_sentiment_result
            st.success(f"✅ 감성 분석 결과 (총 {len(df_preview_with_sentiment):,}개 리뷰)")
            
            # 결과 표시
            st.dataframe(
                df_preview_with_sentiment,
                use_container_width=True,
                hide_index=True,
                height=600
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
                mime="text/csv"
            )
            
            # 재실행 버튼
            if st.button("🔄 감성 분석 다시 실행", type="secondary"):
                st.session_state.preview_sentiment_result = None
                st.rerun()
        else:
            # 감성 분석 실행 버튼
            if st.button("🔍 감성 분석 추가 (긍정/부정/중립)", type="secondary"):
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
    
    if st.button("장소성 요인 점수 계산 시작", type="primary"):
        with st.spinner("12개 장소성 요인별 점수 계산 및 연구 지표 산출 중..."):
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
                
            except Exception as e:
                st.error(f"점수 계산 중 오류 발생: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # 결과 표시
    if st.session_state.df_final_metrics is not None:
        _render_placeness_results()


def _render_placeness_results():
    """장소성 계산 결과 표시"""
    st.header("⭐ 최종 장소성 정량 평가 (연구 결과)")
    
    df_final_metrics = st.session_state.df_final_metrics
    
    # Final_PlaceScore_Summary와 강점/약점만 표시
    display_summary_cols = ['cafe_name', 'Final_PlaceScore_Summary', '강점_요인(+df+)', '약점_요인(-df-)']
    if all(col in df_final_metrics.columns for col in display_summary_cols):
        st.dataframe(
            df_final_metrics[display_summary_cols].set_index('cafe_name'), 
            use_container_width=True
        )
    
    st.subheader("세부 지표 점수 (fsi)")
    fsi_cols = ['cafe_name', '종합_장소성_점수_Mu', '요인_점수_표준편차_Sigma'] + [f'점수_{factor}' for factor in ALL_FACTORS.keys()]
    if all(col in df_final_metrics.columns for col in fsi_cols):
        st.dataframe(
            df_final_metrics[fsi_cols].set_index('cafe_name'), 
            use_container_width=True
        )
    
    # 가중치 정보 표시
    with st.expander("📊 가중치 (Wi) 및 언급 비율 (Ri) 상세 정보"):
        wi_cols = ['cafe_name'] + [f'Wi_{factor}' for factor in ALL_FACTORS.keys()] + [f'Ri_{factor}' for factor in ALL_FACTORS.keys()]
        if all(col in df_final_metrics.columns for col in wi_cols):
            st.dataframe(
                df_final_metrics[wi_cols].set_index('cafe_name'), 
                use_container_width=True
            )
    
    # 결과 다운로드
    csv = df_final_metrics.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "장소성 최종 연구 지표 CSV 다운로드 (Wi, Mu, Sigma 포함)",
        data=csv,
        file_name="placeness_final_research_metrics.csv",
        mime="text/csv"
    )


def render_sentiment_analysis(df_reviews, sentiment_pipeline, sentiment_model_name):
    """개별 리뷰 감성 분석 섹션 렌더링"""
    st.header("2. 개별 리뷰 감성 분석 및 카페별 평균")
    
    if st.button("KoBERT 개별 리뷰 감성 분석 시작", type="primary"):
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
        st.dataframe(st.session_state.df_avg_sentiment.set_index('cafe_name'), use_container_width=True)
        
        st.subheader("✅ 개별 리뷰 감성 분석 결과 (샘플)")
        sample_df = st.session_state.df_reviews_with_sentiment[['cafe_name', 'review_text', 'sentiment_label', 'sentiment_score']].head(20)
        st.dataframe(sample_df, use_container_width=True)
        
        # 결과 다운로드
        csv = st.session_state.df_reviews_with_sentiment.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 개별 리뷰 감성 분석 결과 CSV 다운로드",
            data=csv,
            file_name="review_sentiment_analysis.csv",
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
            sample_df = st.session_state.df_reviews_with_sentiment[['cafe_name', 'review_text', 'sentiment_label', 'sentiment_score']].head(50)
            st.dataframe(sample_df, use_container_width=True, hide_index=True, height=400)
        elif has_placeness:
            st.info("감성 분석을 실행하면 리뷰별 상세 결과를 확인할 수 있습니다.")
            sample_df = st.session_state.df_review_scores[['cafe_name', 'review_text'] + [f'{factor}_점수' for factor in list(ALL_FACTORS.keys())[:5]]].head(50)
            st.dataframe(sample_df, use_container_width=True, hide_index=True, height=400)


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
    available_cols = [col for col in display_cols if col in df_merged.columns]
    
    st.subheader("✅ 리뷰별 감성 분석 + 장소성 요인 점수")
    
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
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=600
        )

        st.caption(f"총 {len(filtered_df):,}개 리뷰 표시")
        
        # 다운로드 버튼
        csv = filtered_df[available_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 리뷰별 상세 결과 CSV 다운로드",
            data=csv,
            file_name="review_detailed_analysis.csv",
            mime="text/csv"
        )
    else:
        st.warning("선택한 조건에 해당하는 리뷰가 없습니다.")

