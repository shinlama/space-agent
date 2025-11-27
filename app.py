import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import warnings
import re

# Streamlit 페이지 설정 (wide 모드로 전체 너비 사용)
st.set_page_config(layout="wide")

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
# 전역 변수로 모델 이름 저장
_sentiment_model_name = None

@st.cache_resource
def load_models():
    """Sentence-BERT와 감성 분석 모델을 로드합니다."""
    global _sentiment_model_name
    # 1. Sentence-BERT 모델 로드 (임베딩 및 유사도 계산용)
    with st.spinner("모델 로드 중: Sentence-BERT (유사도용)..."):
        try:
            sbert_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        except Exception as e:
            st.warning(f"기본 모델 로드 실패, 대체 모델 사용: {e}")
            sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    
    # 2. 감성 분석 모델 로드 (한국어 리뷰 감성 분석 특화 모델)
    with st.spinner("모델 로드 중: 한국어 감성 분석 모델..."):
        sentiment_pipeline = None
        model_loaded = False
        
        # 우선순위 1: 한국어 감성 분석 전용 fine-tuned 모델
        model_candidates = [
            {
                "name": "matthewburke/korean_sentiment",
                "description": "한국어 감성 분석 전용 모델"
            },
            {
                "name": "nlptown/bert-base-multilingual-uncased-sentiment",
                "description": "다국어 감성 분석 모델 (한국어 포함, 5단계 감성)"
            },
            {
                "name": "beomi/KoELECTRA-v3-discriminator",
                "description": "KoELECTRA v3 (최신 버전)"
            },
            {
                "name": "beomi/KcELECTRA-base",
                "description": "KoELECTRA base (기존)"
            },
            {
                "name": "monologg/kobert-base-v1",
                "description": "KoBERT (fallback)"
            }
        ]
        
        for model_info in model_candidates:
            try:
                sentiment_model_name = model_info["name"]
                st.info(f"시도 중: {model_info['description']} ({sentiment_model_name})")
                
                # 특별 처리: nlptown 모델은 이미 fine-tuned되어 있음
                if "nlptown" in sentiment_model_name:
                    sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model=sentiment_model_name,
                        device=0 if torch.cuda.is_available() else -1
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
                    # num_labels 확인 (nlptown은 5, 나머지는 2 또는 3)
                    if "nlptown" in sentiment_model_name or "multilingual" in sentiment_model_name:
                        num_labels = 5
                    else:
                        num_labels = 2
                    
                    model = AutoModelForSequenceClassification.from_pretrained(
                        sentiment_model_name, 
                        num_labels=num_labels
                    )
                    device = 0 if torch.cuda.is_available() else -1
                    sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model=model,
                        tokenizer=tokenizer,
                        device=device
                    )
                
                st.success(f"✅ 모델 로드 성공: {model_info['description']}")
                _sentiment_model_name = sentiment_model_name
                model_loaded = True
                break
                
            except Exception as e:
                st.warning(f"모델 로드 실패 ({model_info['name']}): {e}")
                continue
        
        if not model_loaded or sentiment_pipeline is None:
            st.error("모든 감성 분석 모델 로드 실패. 인터넷 연결을 확인하거나 다른 모델을 시도해주세요.")
            st.stop()

    return sbert_model, sentiment_pipeline, _sentiment_model_name

# --- 3-1. 숫자-only 텍스트 확인 함수 ---
def is_numeric_only(text: str) -> bool:
    """
    텍스트가 숫자만 포함되어 있는지 확인합니다.
    
    Args:
        text: 확인할 텍스트
    
    Returns:
        bool: 숫자만 포함되어 있으면 True
    """
    if text is None:
        return False
    text = str(text).strip()
    return bool(re.fullmatch(r"[0-9]+(\.[0-9]+)?", text))

# --- 3-1-1. 메타데이터-only 텍스트 확인 함수 ---
def is_metadata_only(text: str) -> bool:
    """
    텍스트가 메타데이터만 포함되어 있는지 확인합니다.
    (예: "서비스매장 내 식사식사 유형아침 식사", "식사 유형브런치" 등)
    
    Args:
        text: 확인할 텍스트
    
    Returns:
        bool: 메타데이터만 포함되어 있으면 True
    """
    if text is None:
        return False
    text = str(text).strip()
    
    # 메타데이터 패턴들
    metadata_patterns = [
        r'^서비스.*식사.*유형',
        r'^식사.*유형',
        r'^서비스.*매장.*내.*식사',
        r'^음식:\s*\d+.*서비스:\s*\d+.*분위기:\s*\d+$',  # "음식: 5서비스: 5분위기: 5" 같은 패턴
        r'^음식:\s*\d+$',  # "음식: 5" 같은 패턴
        r'^서비스:\s*\d+$',
        r'^분위기:\s*\d+$',
    ]
    
    for pattern in metadata_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    
    # 매우 짧은 텍스트 (10자 이하)도 메타데이터로 간주할 수 있음
    # 하지만 이건 너무 광범위할 수 있으므로 주석 처리
    # if len(text) <= 10:
    #     return True
    
    return False

# --- 3-2. 감성 분석 결과 처리 헬퍼 함수 ---
def process_sentiment_result(result, model_name=""):
    """
    다양한 감성 분석 모델의 결과를 통일된 형식(긍정/부정, 점수)으로 변환합니다.
    
    Args:
        result: sentiment_pipeline의 결과 (dict 또는 list)
        model_name: 사용된 모델 이름 (선택적)
    
    Returns:
        tuple: (label: str, score: float) - '긍정'/'부정'/'중립', 0.0~1.0 점수
    """
    if isinstance(result, list):
        # 배치 결과인 경우 첫 번째 결과 사용
        result = result[0] if len(result) > 0 else {}
    
    label = str(result.get('label', '')).upper()
    score = float(result.get('score', 0.5))
    
    # nlptown 모델 처리 (5단계: 1-5점)
    if 'nlptown' in model_name.lower() or 'multilingual' in model_name.lower():
        # label 형식: "1 star", "2 stars", "3 stars", "4 stars", "5 stars"
        if '5' in label or 'FIVE' in label:
            return ('긍정', 0.9)
        elif '4' in label or 'FOUR' in label:
            return ('긍정', 0.7)
        elif '3' in label or 'THREE' in label:
            return ('중립', 0.5)
        elif '2' in label or 'TWO' in label:
            return ('부정', 0.3)
        elif '1' in label or 'ONE' in label:
            return ('부정', 0.1)
        else:
            # 점수 기반으로 판단
            if score >= 0.6:
                return ('긍정', score)
            elif score <= 0.4:
                return ('부정', 1 - score)
            else:
                return ('중립', 0.5)
    
    # 일반적인 2단계 모델 처리 (긍정/부정)
    if any(pos in label for pos in ['POSITIVE', '긍정', 'LABEL_1', '1', 'POS']):
        return ('긍정', score)
    elif any(neg in label for neg in ['NEGATIVE', '부정', 'LABEL_0', '0', 'NEG']):
        return ('부정', 1 - score)
    else:
        # 레이블을 알 수 없는 경우 점수로 판단
        if score >= 0.6:
            return ('긍정', score)
        elif score <= 0.4:
            return ('부정', 1 - score)
        else:
            return ('중립', 0.5)

# --- 4. 데이터 로드 및 전처리 ---
@st.cache_data
def load_data(file_path: Path):
    """리뷰 데이터를 로드하고 전처리합니다."""
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    try:
        df = pd.read_csv(
            file_path, 
            encoding="utf-8-sig",
            on_bad_lines='skip',  # 잘못된 라인은 건너뛰기
            quoting=1,  # QUOTE_ALL
            escapechar='\\'
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            file_path, 
            encoding="cp949",
            on_bad_lines='skip',
            quoting=1,
            escapechar='\\'
        )
    except Exception as e:
        st.warning(f"CSV 읽기 중 일부 오류 발생: {e}")
        # 오류가 있어도 계속 진행
        try:
            df = pd.read_csv(
                file_path, 
                encoding="utf-8-sig",
                on_bad_lines='skip',
                engine='python'
            )
        except:
            df = pd.read_csv(
                file_path, 
                encoding="utf-8-sig",
                on_bad_lines='skip',
                sep=',',
                quotechar='"',
                escapechar='\\',
                engine='python'
            )
    
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
                        label, positive_prob = process_sentiment_result(sentiment_result, _sentiment_model_name)
                        
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
                
                # 4-2. 감성 분석 적용 (0~1 긍정 점수)
                try:
                    sentiment_results = sentiment_pipeline(relevant_texts)
                    
                    # 헬퍼 함수를 사용하여 감성 점수 추출
                    sentiment_scores = []
                    for res in sentiment_results:
                        label, score = process_sentiment_result(res, _sentiment_model_name)
                        sentiment_scores.append(score)
                    
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

# --- 6. 알고리즘 핵심: 개별 리뷰 감성 분석 (한국어 감성 분석 모델 활용) ---
def run_sentiment_analysis(df_reviews, sentiment_pipeline, model_name="", ratings=None):
    """
    개별 리뷰 텍스트에 대해 한국어 감성 분석 모델을 사용하여 감성 분석을 수행합니다.
    
    Args:
        df_reviews: 리뷰 데이터프레임
        sentiment_pipeline: 감성 분석 파이프라인
        model_name: 모델 이름
        ratings: 평점 리스트 (선택적, 메타데이터-only 리뷰 처리용)
    """
    st.subheader("2. 개별 리뷰 감성 분석 (한국어 감성 분석 모델)")
    
    review_texts = df_reviews['review_text'].astype(str).tolist()
    
    # 평점 정보 추출 (있으면 사용)
    if ratings is None:
        if '평점' in df_reviews.columns:
            ratings = df_reviews['평점'].astype(float).tolist()
        elif 'rating' in df_reviews.columns:
            ratings = df_reviews['rating'].astype(float).tolist()
        else:
            ratings = [None] * len(review_texts)
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    batch_size = 32
    total_batches = (len(review_texts) + batch_size - 1) // batch_size
    
    sentiment_scores = []
    sentiment_labels = []
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(review_texts))
        batch_texts = review_texts[start_idx:end_idx]
        
        progress_bar.progress((batch_idx + 1) / total_batches)
        
        try:
            # 숫자-only 리뷰와 일반 텍스트 리뷰 분리
            text_batch = []
            batch_results_map = {}  # 인덱스 -> 결과 매핑
            
            for idx, text in enumerate(batch_texts):
                global_idx = start_idx + idx
                rating = ratings[global_idx] if global_idx < len(ratings) and ratings[global_idx] is not None else None
                
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
                        # 숫자 변환 실패 시 중립 처리
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
                        # 평점 변환 실패 시 중립 처리
                        batch_results_map[idx] = ("중립", 0.5)
                else:
                    # 일반 텍스트 리뷰는 모델 사용을 위해 수집
                    text_batch.append((idx, text))
            
            # 일반 텍스트 리뷰는 모델 사용
            if text_batch:
                text_only = [text for _, text in text_batch]
                model_results = sentiment_pipeline(text_only)
                
                # 모델 결과를 인덱스에 매핑
                for (idx, _), res in zip(text_batch, model_results):
                    label, score = process_sentiment_result(res, model_name)
                    batch_results_map[idx] = (label, score)
            
            # 원래 순서대로 결과 추가
            for idx in range(len(batch_texts)):
                label, score = batch_results_map[idx]
                sentiment_labels.append(label)
                sentiment_scores.append(score)
                
        except Exception as e:
            st.warning(f"배치 {batch_idx+1} 처리 중 오류: {e}")
            # 오류 발생 시 중립 점수 할당
            sentiment_labels.extend(['중립'] * len(batch_texts))
            sentiment_scores.extend([0.5] * len(batch_texts))
    
    progress_bar.empty()
    
    # 리뷰 데이터프레임에 추가
    df_reviews = df_reviews.copy()
    df_reviews['sentiment_score'] = sentiment_scores
    df_reviews['sentiment_label'] = sentiment_labels
    
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
    sbert_model, sentiment_pipeline, sentiment_model_name = load_models()
    
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
    
    # 전체 데이터 로드 (미리보기용)
    try:
        df_preview = pd.read_csv(
            file_path, 
            encoding="utf-8-sig",
            on_bad_lines='skip',  # 잘못된 라인은 건너뛰기
            quoting=1,  # QUOTE_ALL (모든 필드를 따옴표로 감싸기)
            escapechar='\\'  # 이스케이프 문자
        )
    except UnicodeDecodeError:
        df_preview = pd.read_csv(
            file_path, 
            encoding="cp949",
            on_bad_lines='skip',
            quoting=1,
            escapechar='\\'
        )
    except Exception as e:
        st.warning(f"CSV 읽기 중 일부 오류 발생: {e}")
        # 오류가 있어도 계속 진행 (최선의 노력으로 읽기)
        try:
            df_preview = pd.read_csv(
                file_path, 
                encoding="utf-8-sig",
                on_bad_lines='skip',
                engine='python'  # Python 엔진 사용 (더 관대함)
            )
        except:
            df_preview = pd.read_csv(
                file_path, 
                encoding="utf-8-sig",
                on_bad_lines='skip',
                sep=',',
                quotechar='"',
                escapechar='\\',
                engine='python'
            )
    
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
        if st.button("🔍 감성 분석 추가 (긍정/부정/중립)", type="secondary"):
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
                        batch_results_map = {}  # 인덱스 -> 결과 매핑
                        
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
                                    # 숫자 변환 실패 시 중립 처리
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
                                    # 평점 변환 실패 시 중립 처리
                                    batch_results_map[idx] = ("중립", 0.5)
                            else:
                                # 일반 텍스트 리뷰는 모델 사용을 위해 수집
                                text_batch.append((idx, text))
                        
                        # 일반 텍스트 리뷰는 모델 사용
                        if text_batch:
                            text_only = [text for _, text in text_batch]
                            model_results = sentiment_pipeline(text_only)
                            
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
                        # 오류 발생 시 중립 처리
                        sentiment_labels.extend(['중립'] * len(batch_texts))
                        sentiment_scores.extend([0.5] * len(batch_texts))
                
                progress_bar.empty()
                status_text.empty()
                
                # 결과를 데이터프레임에 추가
                df_preview_with_sentiment = df_preview_sorted.copy()
                df_preview_with_sentiment['감성분석'] = sentiment_labels
                df_preview_with_sentiment['감성점수'] = [f"{s:.3f}" for s in sentiment_scores]
                
                # 컬럼 순서 재정렬 (감성분석 컬럼을 리뷰 옆에 배치)
                column_order = ['상호명', '시군구명', '행정동명', '평점', '리뷰', '감성분석', '감성점수']
                df_preview_with_sentiment = df_preview_with_sentiment[column_order]
                
                st.success(f"✅ 감성 분석 완료! {len(sentiment_labels):,}개 리뷰 분석됨")
                
                # 결과 표시
                st.dataframe(
                    df_preview_with_sentiment,
                    use_container_width=True,
                            hide_index=True,
                    height=600
                )
                
                # 통계 정보
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
    else:
        st.warning(f"필요한 컬럼이 없습니다. 현재 컬럼: {list(df_preview.columns)}")
        # 기본 컬럼으로 표시
        if '상호명' in df_preview.columns or 'cafe_name' in df_preview.columns:
            cafe_col = '상호명' if '상호명' in df_preview.columns else 'cafe_name'
            review_col = '리뷰' if '리뷰' in df_preview.columns else 'review_text'
            preview_cols = [cafe_col, review_col]
            if all(col in df_preview.columns for col in preview_cols):
                df_preview_sorted = df_preview[preview_cols].copy()
                if '시군구명' in df_preview.columns:
                    df_preview_sorted = df_preview_sorted.sort_values(by='시군구명', ascending=True)
                st.dataframe(
                    df_preview_sorted,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
                st.caption(f"전체 {len(df_preview_sorted):,}개 리뷰")
    
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
                    sentiment_pipeline,
                    sentiment_model_name
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
