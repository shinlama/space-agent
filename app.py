import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --- 1. 장소성 요인 정의 (Sentence-BERT Input) ---
FACTOR_DEFINITIONS = {
    "물리적 특성": {
        "심미성": "인테리어, 조명, 가구, 색채 등 시각적 아름다움과 미적 즐거움을 통해 심리적 욕구를 충족시키는 정도입니다. 키워드: 아름다움, 예쁨, 디자인, 인테리어, 조명, 세련됨, 감각적, 분위기, 시각적, 조화로움, 미적 즐거움",
        "형태성": "공간의 중심, 축, 방향성, 경계, 에워쌈을 통해 구조적 질서를 구축하고, 공간 지각 및 정위(Orientation)에 도움을 주는 물리적 구성입니다. 키워드: 중심, 축, 방향성, 경계, 에워쌈, 개방감, 구조, 배치, 동선, 공간 구성, 효율성, 질서정연, 체계적",
        "감각적 경험": "배경 음악, 향기, 가구 질감, 색채 등 오감을 자극하는 실내디자인 요소가 쾌적하고 돋보이며, 장소에 대한 특별한 기억과 의미를 만들어 줍니다. 키워드: 음악, 향기, 냄새, 질감, 촉감, 오감, 감각적, 청각, 후각, 분위기, 색채, 기억",
        "접근성": "대중교통 접근, 도보 가능성 등 장소를 쉽게 찾아오고 이용할 수 있는 정도를 나타내며, 공간지각의 중요한 요소입니다. 키워드: 접근성, 위치, 거리, 교통, 버스, 지하철, 도보, 이동, 편리함, 주차, 진입, Traffic accessibility, Walkability",
        "쾌적성": "채광, 온습도, 청결, 안전 등 공간 이용자가 느끼는 물리적 안락감과 쾌적함을 의미하며, 공간지각의 핵심 요소입니다. 키워드: 청결, 온도, 채광, 통풍, 위생, 밝음, 냉난방, 공기, 정돈, 안전, 안락함, Safe and clean"
    },
    "활동적 특성": {
        "활동성": "대화, 업무, 휴식, 식사 등 다양한 활동이 자연스럽게 이루어지는 정도를 의미하며, 기능의 복합성과 자유로운 행동 선택을 가능하게 합니다. 키워드: 대화, 업무, 회의, 식사, 휴식, 활동, 모임, 일, 작업, 이용, 기능적 복합성, 다양성",
        "사회성": "다른 사람들과 자연스럽게 어울리거나 교류할 수 있는 개방적이고 친근한 분위기를 의미하며, 공동 유대감 형성 및 사회적 욕구를 충족시킵니다. 키워드: 교류, 소통, 친근, 친절, 서비스, 어울림, 대인 관계, 커뮤니티, 개방적, 교감, 사회적, 함께, 파티, Social participation",
        "참여성": "이용자가 이벤트, 체험, 클래스 등 공간 내에서 능동적으로 참여하고 경험할 수 있는 정도를 의미하며, 환경에 주체적으로 영향을 주려는 통제 욕구를 충족시킵니다. 키워드: 참여, 체험, 클래스, 원데이, 워크숍, 행사, 이벤트, 활동, 직접, 경험, 주체적, Self participation"
    },
    "의미적 특성": {
        "고유성": "다른 장소와 차별화되는 독특한 콘셉트나 상징적 디자인으로 장소만의 정체성을 형성하며, 이용자가 자아를 표출하는 수단으로 활용됩니다. 키워드: 독특, 개성, 차별화, 컨셉, 상징, 유니크, 독창적, 아이덴티티, 고유, 정체성, Preference, Meaning, Personal identity",
        "기억/경험": "특별한 추억이나 의미 있는 경험을 제공하여 오래 기억에 남으며, 심리적 요소가 개입된 건축적 체험을 통해 장소성을 지속시킵니다. 키워드: 추억, 기억, 경험, 감동, 인상적, 회상, 스토리, 의미, 특별함, 회고, Functional attachment",
        "지역 정체성": "장소가 위치한 지역의 문화, 상징을 반영하여 고유한 지역 이미지를 형성하며, 지역성이나 전통성 부각을 통해 강한 장소성을 갖습니다. 키워드: 지역성, 지역 이미지, 문화, 전통, 상징, 로컬, 동네, 핫플레이스, 지역 기반, Cultural image, Regional landmark",
        "문화적 맥락": "공간이 위치한 지역의 역사, 문화적 배경, 스토리 등을 반영하며, 문명권의 문화적 체계에 따른 의미적 질서를 통해 장소의 맥락을 강화합니다. 키워드: 역사, 문화, 배경, 스토리, 전통, 서사, 지역성, 의미, 맥락, 오래된"
    }
}

ALL_FACTORS = {k: v for outer_dict in FACTOR_DEFINITIONS.values() for k, v in outer_dict.items()}

# --- 2. 데이터 파일 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parent
GOOGLE_REVIEW_SAMPLE_CSV = BASE_DIR / "google_reviews_scraped_cleaned.csv"

# --- 2-1. 알고리즘 하이퍼파라미터 설정 ---
SIMILARITY_THRESHOLD = 0.4  # 리뷰와 요인 정의 간 최소 코사인 유사도 (0.0~1.0)
# 권장값: 0.5 (중간 필터링) - 너무 낮으면(0.3~0.4) 관련 없는 리뷰 포함, 너무 높으면(0.7~0.8) 관련 리뷰 누락

# --- 3. 모델 로드 및 캐싱 (Streamlit 성능 최적화) ---
@st.cache_resource
def load_models():
    """Sentence-BERT와 감성 분석 모델을 로드합니다."""
    # 1. Sentence-BERT 모델 로드 (임베딩 및 유사도 계산용)
    with st.spinner("모델 로드 중: Sentence-BERT (유사도용)..."):
        try:
            sbert_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        except Exception as e:
            st.warning(f"기본 모델 로드 실패, 대체 모델 사용: {e}")
            sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    
    # 2. 감성 분석 모델 로드 (KoBERT/KoELECTRA 기반)
    with st.spinner("모델 로드 중: KoELECTRA/KoBERT 기반 감성분석..."):
        try:
            # KoELECTRA 우선 시도
            sentiment_model_name = "beomi/KcELECTRA-base"
            tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, num_labels=2)
            device = 0 if torch.cuda.is_available() else -1
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=device
            )
        except Exception as e:
            st.warning(f"KoELECTRA 로드 실패, KoBERT 사용: {e}")
            try:
                # KoBERT 대체 시도
                sentiment_model_name = "monologg/kobert-base-v1"
                tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
                model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, num_labels=2)
                device = 0 if torch.cuda.is_available() else -1
                sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )
            except Exception as e2:
                st.error(f"감성 분석 모델 로드 실패: {e2}")
                st.stop()
    
    return sbert_model, sentiment_pipeline

# --- 4. 데이터 로드 및 전처리 ---
@st.cache_data
def load_data(file_path: Path):
    """리뷰 데이터를 로드하고 전처리합니다."""
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="cp949")
    
    # 컬럼명 정규화 (한국어 컬럼명 처리)
    column_mapping = {
        "상호명": "cafe_name",
        "리뷰": "review_text",
        "cafe_name": "cafe_name",
        "review_text": "review_text"
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # 필요한 컬럼 확인
    if "cafe_name" not in df.columns or "review_text" not in df.columns:
        st.error(f"필수 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")
        st.stop()
    
    # 결측치 제거
    initial_count = len(df)
    initial_cafe_count = df['cafe_name'].nunique() if 'cafe_name' in df.columns else 0
    
    df = df[['cafe_name', 'review_text']].dropna()
    after_dropna_count = len(df)
    after_dropna_cafe_count = df['cafe_name'].nunique() if 'cafe_name' in df.columns else 0
    
    df = df[df['review_text'].astype(str).str.strip() != '']
    final_count = len(df)
    final_cafe_count = df['cafe_name'].nunique() if 'cafe_name' in df.columns else 0
    
    st.success(f"리뷰 데이터 로드 완료: 총 {final_count}건")
    st.info(f"📊 고유 카페 수: {final_cafe_count}개 (초기: {initial_cafe_count}개, 결측치 제거 후: {after_dropna_cafe_count}개)")
    
    if initial_cafe_count > final_cafe_count:
        excluded = initial_cafe_count - final_cafe_count
        st.warning(f"⚠️ {excluded}개 카페가 빈 리뷰로 인해 제외되었습니다.")
    
    return df

# --- 5. 알고리즘 핵심: 감성 분석 및 유사도 기반 요인 점수 계산 ---
def calculate_place_scores(df_reviews, sbert_model, sentiment_pipeline, factor_defs, similarity_threshold=0.5):
    """
    Sentence-BERT와 감성 분석을 사용하여 장소성 요인별 점수를 계산합니다.
    리뷰별 점수도 함께 반환합니다.
    """
    st.subheader("1. Sentence-BERT 임베딩 생성")
    
    # 1. 장소성 정의 문장 임베딩 (고정 벡터)
    factor_sentences = list(factor_defs.values())
    factor_names = list(factor_defs.keys())
    
    with st.spinner("장소성 요인 정의 임베딩 생성 중..."):
        factor_embeddings = sbert_model.encode(factor_sentences, convert_to_tensor=True, show_progress_bar=False)
    
    # 결과를 저장할 빈 리스트
    results_list = []
    review_scores_list = []  # 리뷰별 점수 저장
    
    # 카페별로 그룹화하여 처리
    cafe_groups = df_reviews.groupby('cafe_name')
    total_cafes = len(cafe_groups)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (cafe_name, group) in enumerate(cafe_groups):
        status_text.text(f"처리 중: {cafe_name} ({idx+1}/{total_cafes})")
        progress_bar.progress((idx + 1) / total_cafes)
        
        # 2. 개별 리뷰 임베딩
        review_texts = group['review_text'].astype(str).tolist()
        review_indices = group.index.tolist()
        
        with st.spinner(f"{cafe_name} 리뷰 임베딩 생성 중..."):
            review_embeddings = sbert_model.encode(review_texts, convert_to_tensor=True, show_progress_bar=False)
        
        # 3. 코사인 유사도 계산 (리뷰 문장 vs. 12개 요인 정의 문장)
        similarity_matrix = cosine_similarity(
            review_embeddings.cpu().numpy(), 
            factor_embeddings.cpu().numpy()
        )
        
        # 4. 요인별 점수 집계
        cafe_scores = {'cafe_name': cafe_name}
        
        # 각 리뷰별로 요인 점수 계산
        for review_idx, (review_text, review_original_idx) in enumerate(zip(review_texts, review_indices)):
            review_factor_scores = {
                'review_index': review_original_idx,
                'cafe_name': cafe_name,
                'review_text': review_text
            }
            
            # 각 요인별로 반복
            for i, factor_name in enumerate(factor_names):
                similarity_score = similarity_matrix[review_idx, i]
                
                # 유사도 임계값 이상인 경우에만 점수 계산
                if similarity_score >= similarity_threshold:
                    # 해당 리뷰에 대한 감성 분석
                    try:
                        sentiment_result = sentiment_pipeline([review_text])[0]
                        label = sentiment_result['label']
                        score = sentiment_result['score']
                        
                        # 긍정 확률 계산
                        if any(pos in str(label).upper() for pos in ['POSITIVE', '긍정', 'LABEL_1', '1']):
                            positive_prob = score
                        else:
                            positive_prob = 1 - score
                        
                        # 유사도와 감성 점수를 결합 (가중 평균)
                        combined_score = 0.6 * similarity_score + 0.4 * positive_prob
                        review_factor_scores[f'{factor_name}_점수'] = combined_score
                        review_factor_scores[f'{factor_name}_유사도'] = similarity_score
                    except Exception as e:
                        review_factor_scores[f'{factor_name}_점수'] = np.nan
                        review_factor_scores[f'{factor_name}_유사도'] = similarity_score
                else:
                    review_factor_scores[f'{factor_name}_점수'] = np.nan
                    review_factor_scores[f'{factor_name}_유사도'] = similarity_score
            
            review_scores_list.append(review_factor_scores)
        
        # 각 요인별로 반복 (카페별 평균 점수 계산)
        for i, factor_name in enumerate(factor_names):
            # 4-1. 유사도 임계값 이상인 문장 선별
            relevant_review_indices = np.where(similarity_matrix[:, i] >= similarity_threshold)[0]
            
            if len(relevant_review_indices) > 0:
                relevant_texts = [review_texts[idx] for idx in relevant_review_indices]
                
                # 4-2. BERT 감성 분석 적용 (0~1 긍정 점수)
                try:
                    sentiment_results = sentiment_pipeline(relevant_texts)
                    
                    # 감성 점수 추출 (레이블에 따라 긍정 확률 계산)
                    sentiment_scores = []
                    for res in sentiment_results:
                        label = res['label']
                        score = res['score']
                        
                        # 레이블이 'POSITIVE', '긍정', 'LABEL_1' 등인 경우
                        if any(pos in str(label).upper() for pos in ['POSITIVE', '긍정', 'LABEL_1', '1']):
                            sentiment_scores.append(score)
                        else:
                            sentiment_scores.append(1 - score)
                    
                    # 4-3. 세부 항목 최종 점수 산출 (산술 평균)
                    avg_score = np.mean(sentiment_scores) if sentiment_scores else 0.5
                    cafe_scores[f'점수_{factor_name}'] = avg_score
                    cafe_scores[f'리뷰수_{factor_name}'] = len(relevant_texts)
                    
                except Exception as e:
                    st.warning(f"{cafe_name} - {factor_name} 감성 분석 오류: {e}")
                    cafe_scores[f'점수_{factor_name}'] = np.nan
                    cafe_scores[f'리뷰수_{factor_name}'] = 0
            else:
                # 관련 리뷰가 없으면 NaN 처리
                cafe_scores[f'점수_{factor_name}'] = np.nan
                cafe_scores[f'리뷰수_{factor_name}'] = 0
        
        results_list.append(cafe_scores)
    
    progress_bar.empty()
    status_text.empty()
    
    df_cafe_scores = pd.DataFrame(results_list)
    df_review_scores = pd.DataFrame(review_scores_list)
    
    return df_cafe_scores, df_review_scores

# --- 6. 알고리즘 핵심: 개별 리뷰 감성 분석 (KoBERT 활용) ---
def run_sentiment_analysis(df_reviews, sentiment_pipeline):
    """
    개별 리뷰 텍스트에 대해 KoBERT/KoELECTRA 기반 감성 분석을 수행합니다.
    """
    st.subheader("2. 개별 리뷰 감성 분석 (KoBERT/KoELECTRA 기반)")
    
    review_texts = df_reviews['review_text'].astype(str).tolist()
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    batch_size = 32
    total_batches = (len(review_texts) + batch_size - 1) // batch_size
    
    sentiment_scores = []
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(review_texts))
        batch_texts = review_texts[start_idx:end_idx]
        
        progress_bar.progress((batch_idx + 1) / total_batches)
        
        try:
            # KoBERT/KoELECTRA 모델을 사용하여 긍정 확률을 산출
            batch_results = sentiment_pipeline(batch_texts)
            
            # 긍정 점수 추출
            for res in batch_results:
                label = res['label']
                score = res['score']
                
                if any(pos in str(label).upper() for pos in ['POSITIVE', '긍정', 'LABEL_1', '1']):
                    sentiment_scores.append(score)
                else:
                    sentiment_scores.append(1 - score)
        except Exception as e:
            st.warning(f"배치 {batch_idx+1} 처리 중 오류: {e}")
            # 오류 발생 시 중립 점수 할당
            sentiment_scores.extend([0.5] * len(batch_texts))
    
    progress_bar.empty()
    
    # 리뷰 데이터프레임에 추가
    df_reviews = df_reviews.copy()
    df_reviews['sentiment_score'] = sentiment_scores
    
    # 감성 레이블 추가 (0.5 기준으로 긍정/부정 분류)
    df_reviews['sentiment_label'] = df_reviews['sentiment_score'].apply(
        lambda x: '긍정' if x >= 0.5 else '부정'
    )
    
    # 카페별 평균 감성 점수 산출
    avg_sentiment = df_reviews.groupby('cafe_name')['sentiment_score'].mean().reset_index()
    avg_sentiment.rename(columns={'sentiment_score': '평균_리뷰_감성점수'}, inplace=True)
    
    st.success("개별 리뷰 감성 분석 및 카페별 평균 산출 완료.")
    return df_reviews, avg_sentiment

# --- 7. Streamlit UI 구성 ---
def main():
    st.title("장소성 기반 공간 정량 평가 시스템 (LLM & BERT)")
    st.markdown("---")
    
    # 파일 경로 설정
    file_path = GOOGLE_REVIEW_SAMPLE_CSV
    
    # 1. 모델 로드
    sbert_model, sentiment_pipeline = load_models()
    
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
    
    # 데이터 미리보기
    st.markdown("---")
    st.header("📋 데이터 미리보기")
    preview_cols = ['cafe_name', 'review_text']
    if len(df_reviews) > 0:
        preview_df = df_reviews[preview_cols].head(10000)
        st.dataframe(
            preview_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        st.caption(f"상위 10000개 리뷰 미리보기 (전체 {len(df_reviews):,}개)")
    
    # 세션 상태 초기화
    if 'df_review_scores' not in st.session_state:
        st.session_state.df_review_scores = None
    if 'df_reviews_with_sentiment' not in st.session_state:
        st.session_state.df_reviews_with_sentiment = None
    
    # --- 3. 실행 파트: 장소성 요인 점수 계산 ---
    st.header("📊 1. 장소성 요인별 정량 점수 계산")
    st.caption(f"유사도 임계값: {SIMILARITY_THRESHOLD} (코드 내 고정값)")
    
    # Sentence-BERT를 사용한 요인 점수 계산 실행
    if st.button("장소성 요인 점수 계산 시작", type="primary"):
        with st.spinner("12개 장소성 요인별 점수 계산 중 (Sentence-BERT & Sentiment Analysis)..."):
            try:
                df_place_scores, df_review_scores = calculate_place_scores(
                    df_reviews.copy(), 
                    sbert_model, 
                    sentiment_pipeline, 
                    ALL_FACTORS, 
                    similarity_threshold=SIMILARITY_THRESHOLD
                )
                
                # 세션 상태에 저장
                st.session_state.df_review_scores = df_review_scores
                
                st.subheader("✅ 카페별 장소성 요인 점수 (0~1)")
                st.dataframe(df_place_scores.set_index('cafe_name'), use_container_width=True)
                
                # 결과 다운로드
                csv = df_place_scores.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "장소성 요인 점수 CSV 다운로드",
                    data=csv,
                    file_name="placeness_factor_scores.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"점수 계산 중 오류 발생: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # --- 4. 실행 파트: 개별 리뷰 감성 분석 (KoBERT) ---
    st.header("2. 개별 리뷰 감성 분석 및 카페별 평균")
    
    # KoBERT를 사용한 개별 리뷰 감성 분석 실행
    if st.button("KoBERT 개별 리뷰 감성 분석 시작", type="primary"):
        with st.spinner("개별 리뷰 긍정/부정 감성 점수 계산 중 (KoBERT/KoELECTRA)..."):
            try:
                df_reviews_with_sentiment, df_avg_sentiment = run_sentiment_analysis(
                    df_reviews.copy(), 
                    sentiment_pipeline
                )
                
                # 세션 상태에 저장
                st.session_state.df_reviews_with_sentiment = df_reviews_with_sentiment
                
                st.subheader("✅ 카페별 평균 감성 점수")
                st.dataframe(df_avg_sentiment.set_index('cafe_name'), use_container_width=True)
                
                st.subheader("✅ 개별 리뷰 감성 분석 결과 (샘플)")
                sample_df = df_reviews_with_sentiment[['cafe_name', 'review_text', 'sentiment_label', 'sentiment_score']].head(20)
                st.dataframe(sample_df, use_container_width=True)
                
                # 결과 다운로드
                csv = df_reviews_with_sentiment.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 개별 리뷰 감성 분석 결과 CSV 다운로드",
                    data=csv,
                    file_name="review_sentiment_analysis.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"감성 분석 중 오류 발생: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # --- 5. 리뷰별 상세 결과 표시 ---
    st.header("📊 리뷰별 상세 분석 결과")
    
    # 두 분석이 모두 완료되었는지 확인
    has_sentiment = st.session_state.df_reviews_with_sentiment is not None
    has_placeness = st.session_state.df_review_scores is not None
    
    if not has_sentiment and not has_placeness:
        st.info("👆 위의 두 분석을 모두 실행하면 리뷰별 상세 결과를 확인할 수 있습니다.")
    else:
        # 결과 병합
        if has_sentiment and has_placeness:
            # 두 결과를 병합
            df_sentiment = st.session_state.df_reviews_with_sentiment.copy()
            df_placeness = st.session_state.df_review_scores.copy()
            
            # 인덱스를 기준으로 병합
            df_sentiment['review_index'] = df_sentiment.index
            df_placeness['review_index'] = df_placeness['review_index']
            
            # 병합
            df_merged = pd.merge(
                df_sentiment[['review_index', 'cafe_name', 'review_text', 'sentiment_label', 'sentiment_score']],
                df_placeness,
                on=['review_index', 'cafe_name', 'review_text'],
                how='outer'
            )
            
            # 12개 요인 점수 컬럼 추출
            factor_names = list(ALL_FACTORS.keys())
            factor_score_cols = [f'{factor}_점수' for factor in factor_names]
            
            # 표시할 컬럼 선택
            display_cols = ['cafe_name', 'review_text', 'sentiment_label', 'sentiment_score'] + factor_score_cols
            
            # 존재하는 컬럼만 선택
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
                
                # 점수 포맷팅 (소수점 3자리)
                for col in factor_score_cols:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                
                # 감성 점수 포맷팅
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
        
        elif has_sentiment:
            st.info("장소성 요인 점수 계산을 실행하면 리뷰별 상세 결과를 확인할 수 있습니다.")
            sample_df = st.session_state.df_reviews_with_sentiment[['cafe_name', 'review_text', 'sentiment_label', 'sentiment_score']].head(50)
            st.dataframe(sample_df, use_container_width=True, hide_index=True, height=400)
        
        elif has_placeness:
            st.info("감성 분석을 실행하면 리뷰별 상세 결과를 확인할 수 있습니다.")
            sample_df = st.session_state.df_review_scores[['cafe_name', 'review_text'] + [f'{factor}_점수' for factor in list(ALL_FACTORS.keys())[:5]]].head(50)
            st.dataframe(sample_df, use_container_width=True, hide_index=True, height=400)

if __name__ == "__main__":
    main()
