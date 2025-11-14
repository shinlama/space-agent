import streamlit as st
import googlemaps
from openai import OpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
import json
import re
from wordcloud import WordCloud
import numpy as np
import warnings # 경고 메시지 처리를 위해 추가
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import time
import random
import tab4_analysis
from scipy import stats

# NLP 모델 관련 임포트 (Hugging Face Transformers)
try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    from transformers import logging as transformers_logging # 트랜스포머 로깅 임포트
    # 명시적 로드를 위한 추가 import
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    st.error("필요한 라이브러리(sentence-transformers, transformers)가 설치되지 않았습니다. `pip install sentence-transformers transformers` 명령어로 설치해주세요.")
    st.stop()


# ----------------------------------------------------
# 1. 환경 설정, 캐싱 및 유틸리티
# ----------------------------------------------------

# 폰트 경로 캐싱 (WordCloud용)
@st.cache_resource(show_spinner=False)
def get_font_path():
    # 실제 환경에 맞춰 폰트 파일 경로를 지정해야 합니다.
    # 예시: NotoSansKR-Regular.ttf 파일이 'fonts' 폴더에 있다고 가정
    # 로컬 환경에서는 경로를 수정하거나, 시스템 폰트를 사용하도록 설정해야 합니다.
    # 현재는 더미 경로로 설정하며, 실행 환경에 따라 수정 필요
    try:
        # Streamlit 폰트 경로를 사용하거나 실제 폰트 경로를 지정
        return os.path.join(os.path.dirname(__file__), 'fonts', 'NotoSansKR-Regular.ttf')
    except:
        return None # 폰트 경로를 찾을 수 없는 경우

font_path = get_font_path()

# 표본 CSV 경로
BASE_DIR = Path(__file__).resolve().parent
SAMPLED_CAFE_CSV = BASE_DIR / "서울시_상권_카페빵_표본.csv"
GOOGLE_REVIEW_SAMPLE_CSV = BASE_DIR / "google_reviews_sample.csv"  # 사전 수집(2500개) 파일
GOOGLE_REVIEW_LIVE_CSV = BASE_DIR / "google_reviews_live.csv"      # 탭3에서 새로 수집한 결과
FULL_CAFE_CSV = Path(__file__).resolve().parent / "서울시_상권_카페빵.csv"
SAMPLED_CAFE_WITH_TRANSIT_CSV = BASE_DIR / "서울시_상권_카페빵_표본_with_transit.csv"


@st.cache_data(show_spinner="표본 데이터 불러오는 중...")
def load_sampled_cafes(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")
    return df

@st.cache_data(show_spinner="전체 카페 데이터 불러오는 중...")
def load_full_cafes(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")
    return df


@st.cache_data(show_spinner="Google 리뷰 데이터 불러오는 중...")
def load_google_reviews_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")
    return df


@st.cache_data(show_spinner=False)
def load_csv_generic(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")
    try:
        return pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="cp949")

# WordCloud 이미지 캐싱 (PIL 이미지 반환)
@st.cache_data(show_spinner=False)
def generate_wordcloud(text: str, font_path: str, colormap: str = "Greens", **kwargs):
    if not text or not text.strip():
        return None
    
    # 폰트 경로가 유효하지 않으면 기본 설정을 사용하거나 경고
    if font_path and os.path.exists(font_path):
        font_kwarg = {'font_path': font_path}
    else:
        # st.warning("WordCloud 폰트 파일 경로가 유효하지 않아 기본 폰트를 사용합니다.")
        font_kwarg = {}
        
    wc = WordCloud(
        **font_kwarg,
        background_color="white",
        width=600,
        height=300,
        scale=2,
        max_words=180,
        prefer_horizontal=0.9,
        colormap=colormap,
        collocations=True,
        normalize_plurals=False,
        relative_scaling=0.35,
        min_font_size=8,
        max_font_size=90,
        random_state=42,
        regexp=r"[가-힣a-zA-Z]+" # 한글, 영문만 포함
    ).generate(text)
    return wc.to_image()

# 환경변수 로드
load_dotenv()

st.set_page_config(page_title="Seoul Place Recommendation", page_icon="🗺️", layout="wide")

# 의미 단위로 문장을 분리하는 함수 (OpenAI 기반)
def semantic_split(text: str) -> list[str]:
    """
    OpenAI API를 사용해 의미 단위로 문장을 분리합니다.
    """
    try:
        prompt = f"""
        아래 리뷰 텍스트를 의미 단위(하나의 감정이나 평가를 담은 단락)로 분리하세요.
        각 문장은 독립적인 판단이 가능한 단위여야 하며, 불필요한 접속사는 제거하세요.
        출력은 JSON 리스트로만 작성하세요.

        예시:
        입력: "카페 분위기가 좋고 커피는 맛있지만 좌석이 좁아요."
        출력: ["카페 분위기가 좋다.", "커피가 맛있다.", "좌석이 좁다."]

        리뷰 텍스트:
        {text}
        """

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "semantic_split_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentences": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["sentences"],
                        "additionalProperties": False
                    }
                }
            },
            temperature=0
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content) if content else {"sentences": []}
        if isinstance(parsed, list):
            candidates = parsed
        else:
            candidates = parsed.get("sentences", [])
        cleaned = [c.strip() for c in candidates if isinstance(c, str) and len(c.strip()) > 2]
        if cleaned:
            return cleaned
    except Exception as e:
        print(f"Semantic split error: {e}")
        # 폴백은 아래 일반 분기에서 수행
        pass
    # 폴백: 문장부호 → 접속 표현 2단계 분할
    fallback_units = []
    for s in re.split(r'[.!?]\s*', text):
        if not s or not s.strip():
            continue
        parts = re.split(r'(?:하지만|그러나|그런데|인데|지만|는데)', s)
        for p in parts:
            p = p.strip()
            if len(p) > 2:
                fallback_units.append(p)
    return fallback_units


# 캐시된 의미 분할 래퍼
@st.cache_data(show_spinner=False)
def cached_semantic_split(text: str) -> List[str]:
    return semantic_split(text)


# 캐시된 LLM 요약/키워드 추출 래퍼
@st.cache_data(show_spinner=False)
def cached_unified_summary(review_text: str):
    try:
        sample = review_text[:1200]
        unified_prompt = f"""
        다음 리뷰들을 바탕으로 장소를 분석하여 한 번에 결과만 JSON으로 응답하세요. 한국어로 작성합니다.
        1) positive_keywords: 장소/장소성 관련 긍정 핵심 단어 최대 10개 (문자열 배열)
        2) negative_keywords: 장소/장소성 관련 부정 핵심 단어 최대 10개 (문자열 배열)
        3) summary: 전반적 분위기/공간 특성/주요 경험 중심의 5~8문장 요약 (문자열)

        리뷰 텍스트는 다음과 같습니다. 정의 구간 안의 텍스트만 참고하세요:
        ```
        {sample}
        ```
        """
        unified_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": unified_prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "unified_summary_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "positive_keywords": {"type": "array", "items": {"type": "string"}},
                            "negative_keywords": {"type": "array", "items": {"type": "string"}},
                            "summary": {"type": "string"}
                        },
                        "required": ["positive_keywords", "negative_keywords", "summary"],
                        "additionalProperties": False
                    }
                }
            },
            temperature=0
        )
        content = unified_response.choices[0].message.content
        parsed = json.loads(content) if content else {"positive_keywords": [], "negative_keywords": [], "summary": ""}
        return {
            "positive_keywords": parsed.get("positive_keywords", []) or [],
            "negative_keywords": parsed.get("negative_keywords", []) or [],
            "summary": (parsed.get("summary") or "").strip() or "리뷰 내용이 충분하지 않아 LLM 요약이 어렵습니다."
        }
    except Exception as e:
        print(f"LLM 요약/키워드 추출 중 오류 발생: {e}")
        return {
            "positive_keywords": [],
            "negative_keywords": [],
            "summary": "LLM 요약 실패. NLP 분석만 진행됨."
        }

# ----------------------------------------------------
# 2. API 키 및 세션 상태 초기화
# ----------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# API 키 로드
gmaps_key = os.getenv("Maps_API_KEY") or st.secrets.get("Maps_API_KEY", "")
openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

if "gmaps_key" not in st.session_state:
    st.session_state.gmaps_key = gmaps_key or ""
if "openai_key" not in st.session_state:
    st.session_state.openai_key = openai_key or ""

# API 키 입력 UI (키가 없는 경우)
if not st.session_state.gmaps_key or not st.session_state.openai_key:
    st.title("🗺️ Seoul Place Recommendation and Spatial Evaluation System")

    st.info("API 키를 `.env` 파일에 설정하거나 수동으로 입력하세요.")
    st.markdown("---")
    gmaps_input = st.text_input("Google Maps API Key", type="password")
    openai_input = st.text_input("OpenAI API Key", type="password")

    if st.button("Start"):
        if gmaps_input and openai_input:
            st.session_state.gmaps_key = gmaps_input
            st.session_state.openai_key = openai_input
            st.rerun()
        else:
            st.warning("Please enter both API keys.")
    st.stop()

# 클라이언트 초기화 (OpenAI client는 폴백 로직에서 사용되므로 전역적으로 유지)
try:
    gmaps = googlemaps.Client(key=st.session_state.gmaps_key)
    # LLM 클라이언트는 그대로 유지
    client = OpenAI(api_key=st.session_state.openai_key)
except Exception as e:
    st.error(f"API 클라이언트 초기화 중 오류 발생: {e}")
    st.stop()


# ----------------------------------------------------
# 3. LangGraph State 정의 및 모델 로딩 (캐시)
# ----------------------------------------------------

class AgentState(BaseModel):
    query: str
    places: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    answer: Optional[str] = ""

# 장소성 요인 임베딩 및 모델 로드 (Sentence-BERT)
def _compute_factors_hash(path: str = "factors.json") -> str:
    try:
        with open(path, "rb") as f:
            import hashlib
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""

@st.cache_resource(show_spinner="임베딩 모델 로드 중...")
def load_category_embeddings(factors_hash: str):
    try:
        # 한국어 문장 임베딩에 최적화된 모델 사용
        model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        # factors.json 파일 로드
        with open("factors.json", "r", encoding="utf-8") as f:
            factors = json.load(f)
    except FileNotFoundError:
        st.error("오류: 'factors.json' 파일을 찾을 수 없습니다. 프로젝트 루트에 파일을 생성했는지 확인하세요.")
        st.stop()
    except Exception as e:
        st.error(f"임베딩 모델 로드 중 오류 발생: {e}")
        st.stop()

    embeddings = {}
    score_structure = {}
    
    # 11개 세부 요인의 정의 문장을 임베딩
    for main_cat, subcats in factors.items():
        score_structure[main_cat] = {}
        for subcat, definition in subcats.items():
            emb = model.encode(definition, normalize_embeddings=True)
            embeddings[subcat] = emb
            score_structure[main_cat][subcat] = None
            
    return embeddings, model, score_structure

factors_hash = _compute_factors_hash()
category_embeddings, embed_model, new_score_structure_template = load_category_embeddings(factors_hash)

@st.cache_resource(show_spinner="감성 분석 모델 로드 중...")
def load_sentiment_model_tabularis():
    """
    공개 감정 분포형 모델 (tabularisai/multilingual-sentiment-analysis) 기반
    - 출력: 0.0 ~ 1.0 연속 점수 (부정→긍정)
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import numpy as np

    model_name = "tabularisai/multilingual-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        truncation=True,
        max_length=512,
    )
    weights = np.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]

    def predict_score(sentences: List[str]):
        if not sentences:
            return []
        results = pipe(sentences, truncation=True, max_length=512)
        scores = []
        for res in results:
            probs = np.array([r['score'] for r in res])
            score = float(np.dot(probs, weights))
            scores.append(score)
        return scores

    return predict_score


sentiment_model = load_sentiment_model_tabularis()


# ----------------------------------------------------
# 4. LangGraph Node 정의
# ----------------------------------------------------

def search_places(state: AgentState):
    """Google Maps API를 사용하여 장소를 검색하는 함수"""
    if state.places is None:
        state.places = []
    try:
        # 서울시청 기준(37.5665,126.9780) 반경 10km 검색
        res = gmaps.places(query=state.query, language="ko", location="37.5665,126.9780", radius=10000)
        state.places = res.get('results', [])[:5]
    except Exception as e:
        st.error(f"Google Maps 장소 검색 중 오류 발생: {e}")
        st.error("Google Maps API 키가 유효한지, 또는 Places API가 활성화되어 있는지 확인하세요.")
        state.places = []
    return state.dict()

def analyze_reviews(state: AgentState):
    """SBERT + Sentiment 기반 정량 평가 + LLM 해석"""
    if state.places is None:
        state.places = []

    place_infos = []
    
    SIMILARITY_THRESHOLD = 0.35
    ALPHA, BETA = 0.75, 0.25  # 유사도 비중 추가 상향
    
    # factors.json은 한 번만 로드 (속도 개선)
    with open("factors.json", "r", encoding="utf-8") as f:
        factor_definitions = json.load(f)
    
    for place in state.places:
        place_id = place.get("place_id")
        if not place_id:
            continue

        details = gmaps.place(place_id=place_id, language="ko").get('result', {})
        reviews = details.get('reviews', [])[:10]
        review_texts = [r['text'] for r in reviews if r.get('text')]
        if not review_texts:
            continue

        # 1) 의미 단위 분리 (LLM 사용)
        review_text = "\n".join(review_texts)
        review_units = cached_semantic_split(review_text)
        if not review_units:
            review_units = cached_semantic_split(" ".join(review_texts))

        # 2) SBERT + 감성모델 기반 점수 계산
        factor_sentiments = {f: [] for f in category_embeddings.keys()}
        # 리뷰 요약/키워드 (summary)와 점수 해설(explanation)은 분리 생성
        summary = "리뷰 요약 생성 중 오류가 발생했습니다."
        positive_keywords: List[str] = []
        negative_keywords: List[str] = []

        # 2-1) 리뷰 요약 및 키워드 추출 (summary 전용)
        try:
            cached = cached_unified_summary(review_text)
            positive_keywords = cached.get("positive_keywords", []) or []
            negative_keywords = cached.get("negative_keywords", []) or []
            summary = cached.get("summary", "") or "리뷰 내용이 충분하지 않아 LLM 요약이 어렵습니다."
        except Exception as e:
            print(f"LLM 요약/키워드 추출 중 오류 발생: {e}")
            summary = "LLM 요약 실패. NLP 분석만 진행됨."
            positive_keywords, negative_keywords = [], []

        # 감성(0~1) 스코어 및 배치 임베딩/유사도
        sentiment_scores = sentiment_model(review_units)
        unit_embs = embed_model.encode(review_units, normalize_embeddings=True)
        subcat_list = list(category_embeddings.keys())
        factor_mat = np.stack([category_embeddings[s] for s in subcat_list], axis=0)
        sim_mat = np.matmul(unit_embs, factor_mat.T)

        for i, unit in enumerate(review_units):
            raw_sent = float(sentiment_scores[i]) if i < len(sentiment_scores) else 0.5
            # 감성 보정: 하한을 0.2까지 낮추고 범위를 0.6으로 확장
            sent_adj = np.clip((raw_sent - 0.2) / 0.6, 0, 1)
            sims = sim_mat[i]
            for j, sim in enumerate(sims):
                # 유사도 보정: 0.2 기준으로 완화하고 상한 폭 확장
                sim_adj = np.clip((float(sim) - 0.2) / 0.4, 0, 1)
                if sim_adj > 0:
                    f_name = subcat_list[j]
                    combined = ALPHA * sim_adj + BETA * sent_adj
                    # 시그모이드: 중심 0.3, 기울기 1.6로 더 부드럽게
                    score_scaled = 1 / (1 + np.exp(-1.6 * (combined - 0.3)))
                    factor_sentiments[f_name].append(float(score_scaled))

        # 3) 키워드 기반 부스팅 (임베딩 한계 보완)
        keyword_boosts = {
            "고유성": ["독특", "유니크", "차별", "컨셉", "테마", "특색", "개성", "특별한", "아이덴티티", "유일", "독창"],
            "문화적 맥락": ["전통", "역사", "년", "오래", "옛", "고풍", "문화", "배경", "스토리", "세월", "내력", "유서", "레트로", "빈티지", "클래식", "앤티크", "과거", "옛날"],
            "지역 정체성": ["지역", "동네", "마을", "근처", "주변", "명소", "랜드마크", "상징", "대표", "신촌", "홍대", "강남", "이태원", "연남", "성수", "을지로", "익선동", "북촌", "삼청동", "종로", "명동"],
            "기억/경험": ["추억", "감동", "인상", "특별", "잊을 수", "기억", "회상", "경험", "느낌"],
            "심미성": ["예쁘", "아름", "멋지", "세련", "야경", "뷰", "인테리어", "디자인", "조명", "아늑", "분위기", "감성"],
            "감각적 경험": ["음악", "향", "냄새", "질감", "맛", "오감", "감각", "소리", "촉감"],
            "쾌적성": ["청결", "깨끗", "밝", "통풍", "화장실", "위생", "정돈", "쾌적"],
            "접근성": ["가깝", "접근", "역", "정류장", "도보", "분 거리", "편리", "지하철역", "버스정류장", "역에서", "역까지", "정류장에서", "정류장까지", "대중교통", "교통편", "오기 쉬", "찾기 쉬", "위치 좋", "교통 좋"],
            "활동성": ["대화", "업무", "작업", "회의", "공부", "활동", "모임", "스터디"],
            "사회성": ["친절", "서비스", "교류", "소통", "친근", "인사", "배려"],
            "형태성": ["넓", "공간", "구조", "배치", "개방", "동선", "층", "룸"],
        }
        
        # 키워드 매칭으로 직접 고점수 할당
        for factor, keywords in keyword_boosts.items():
            matched_kws = [kw for kw in keywords if kw in review_text]
            match_count = len(matched_kws)
            if match_count > 0:
                # 매칭된 키워드 수에 비례해 0.75~0.95 할당
                boosted_score = min(0.75 + (match_count * 0.05), 0.95)
                # 여러 번 추가해 평균에서도 높은 가중치 유지
                for _ in range(3):
                    factor_sentiments[factor].append(boosted_score)
                print(f"[BOOST] {factor}: {match_count}개 키워드 매칭 ({', '.join(matched_kws[:3])}) → {boosted_score:.2f}")
        
        # 3-0) 접근성 전용 패턴 매칭 (시간+거리 표현 강화)
        accessibility_patterns = [
            # 도보 시간 표현: "도보로 5분", "도보 10분", "5분 도보" 등
            r'도보\s*(?:로\s*)?(\d+)\s*분',
            r'(\d+)\s*분\s*도보',
            r'도보\s*로\s*(\d+)\s*분\s*(?:이면|만에|걸림|걸려|가능)',
            # 거리 표현: "5분 거리", "10분 거리"
            r'(\d+)\s*분\s*거리',
            # 역/정류장 + 시간: "지하철역에서 5분", "역까지 10분"
            r'(?:지하철역|역|버스정류장|정류장)(?:에서|까지)\s*(\d+)\s*분',
            r'(\d+)\s*분\s*(?:이면|만에|걸림|걸려)\s*(?:갈\s*수|도착|가능)',
            # "5분이면 갈 수 있다" 같은 표현
            r'(\d+)\s*분\s*이면\s*(?:갈\s*수|도착|가능)',
        ]
        
        accessibility_score_boost = 0.0
        matched_patterns = []
        for pattern in accessibility_patterns:
            matches = re.finditer(pattern, review_text, re.IGNORECASE)
            for match in matches:
                time_str = match.group(1) if match.groups() else None
                if time_str:
                    try:
                        time_minutes = int(time_str)
                        # 5분 이하: 매우 높은 점수 (0.90~0.95)
                        # 10분 이하: 높은 점수 (0.85~0.90)
                        # 15분 이하: 보통 점수 (0.75~0.85)
                        if time_minutes <= 5:
                            boost = 0.95
                        elif time_minutes <= 10:
                            boost = 0.90
                        elif time_minutes <= 15:
                            boost = 0.85
                        else:
                            boost = 0.80
                        
                        if boost > accessibility_score_boost:
                            accessibility_score_boost = boost
                        matched_patterns.append(f"{time_minutes}분 ({match.group(0)})")
                    except ValueError:
                        pass
        
        if matched_patterns:
            # 패턴 매칭으로 매우 높은 점수 부여 (5번 추가로 평균 가중치 강화)
            for _ in range(5):
                factor_sentiments["접근성"].append(accessibility_score_boost)
            print(f"[ACCESSIBILITY PATTERN] 접근성: {len(matched_patterns)}개 패턴 매칭 ({', '.join(matched_patterns[:3])}) → {accessibility_score_boost:.2f}")
        
        # 3-1) 세부요인별 평균 점수 (정규화 포함)
        scores = json.loads(json.dumps(new_score_structure_template))
        all_vals = []
        for vals in factor_sentiments.values():
            all_vals.extend(vals)
        if all_vals:
            vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
        else:
            vmin, vmax = 0.5, 0.5

        for main_cat, subcats in scores.items():
            for subcat in subcats.keys():
                vals = factor_sentiments.get(subcat, [])
                if vals and vmax > vmin:
                    raw = float(np.mean(vals))
                    # 0.20~1.0 범위로 min-max 정규화
                    normed = 0.20 + 0.80 * ((raw - vmin) / (vmax - vmin + 1e-8))
                    scores[main_cat][subcat] = float(np.clip(normed, 0.20, 1.0))
                elif vals:
                    scores[main_cat][subcat] = float(np.clip(vals[0], 0.20, 1.0))
                else:
                    scores[main_cat][subcat] = 0.5

        # 4) LLM 기반 점수 검증 및 보정 (GPT-4o 추론)
        corrected_scores = json.loads(json.dumps(scores))  # 보정 전 복사
        correction_log = []
        try:
            sample_reviews = "\n".join(review_texts[:3])  # 리뷰 5개→3개로 축소
            
            validation_prompt = f"""
당신은 장소성 평가 감사자입니다.
입력된 점수는 SBERT + 감성 회귀모델로 산출된 값입니다.
각 요인별 점수의 타당성을 **요인의 정의에 따라** 정확히 검토하세요.

## 요인 정의 (반드시 참고)
{json.dumps(factor_definitions, ensure_ascii=False, indent=2)}

## 현재 점수
{json.dumps(scores, ensure_ascii=False, indent=2)}

## 리뷰 내용
{sample_reviews}

## 검토 규칙
1. 각 요인의 정의와 키워드를 **정확히** 확인하세요.
   예: "감각적 경험"은 음악, 향기, 질감 등 오감 자극 / "문화적 맥락"은 역사, 전통, 지역 배경
2. 리뷰에서 해당 요인 정의에 맞는 언급이 있는데 점수가 낮거나, 언급이 없는데 점수가 높으면 delta 제안
3. delta는 -0.5 ~ +0.5 범위
4. 근거는 한 문장으로만 작성

## 접근성 특별 검토 가이드
**접근성**은 대중교통 접근, 도보 가능성 등 장소를 쉽게 찾아오고 이용할 수 있는 정도입니다.
다음과 같은 표현이 리뷰에 있으면 접근성 점수가 높아야 합니다:
- "지하철역에서 도보로 5분이면 갈 수 있다" → 접근성 매우 높음 (0.9 이상)
- "버스정류장에서 10분 거리" → 접근성 높음 (0.85 이상)
- "역까지 5분", "도보 5분", "5분 거리" → 접근성 높음
- "대중교통 접근이 편리하다", "교통편이 좋다" → 접근성 높음
- "가까운 곳", "접근하기 쉬운 위치" → 접근성 높음

만약 리뷰에 위와 같은 표현이 있는데 접근성 점수가 낮다면(0.7 이하), delta를 +0.2~+0.3으로 제안하세요.

## 출력 형식 (JSON만)
{{
  "corrections": [
    {{"factor": "쾌적성", "delta": 0.15, "reason": "청결, 화장실, 충전시설 긍정 언급 많음"}},
    {{"factor": "감각적 경험", "delta": 0.12, "reason": "디저트 맛과 다양성 강조"}}
  ]
}}

보정 불필요 시: {{"corrections": []}}
"""
            resp = client.chat.completions.create(
                model="gpt-4o",  # 보정은 정확한 추론이 필요하므로 gpt-4o 사용
                messages=[{"role": "user", "content": validation_prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=500,  # 충분한 토큰으로 정확한 보정
            )
            correction_result = json.loads(resp.choices[0].message.content)
            corrections = correction_result.get("corrections", [])
            
            print(f"[DEBUG] GPT-4o 응답: {correction_result}")  # 디버깅용
            
            # 보정 적용
            for correction in corrections:
                if isinstance(correction, dict):
                    factor_name = correction.get("factor", "")
                    delta = float(correction.get("delta", 0))
                    reason = correction.get("reason", "")
                    
                    # 요인명 매칭 후 점수 보정
                    for main_cat, subcats in corrected_scores.items():
                        if factor_name in subcats:
                            old_val = subcats[factor_name]
                            new_val = np.clip(old_val + delta, 0.20, 1.0)
                            corrected_scores[main_cat][factor_name] = float(new_val)
                            correction_log.append({
                                "factor": factor_name,
                                "original": round(old_val, 2),
                                "adjusted": round(new_val, 2),
                                "delta": round(delta, 2),
                                "reason": reason
                            })
                            break
            
            # 보정된 점수를 최종 점수로 사용
            scores = corrected_scores
            
            if correction_log:
                print(f"[INFO] {len(correction_log)}개 요인 보정됨")
            else:
                print(f"[INFO] 보정 필요 없음")
            
        except Exception as e:
            print(f"[ERROR] LLM 점수 보정 중 오류: {e}")
            import traceback
            traceback.print_exc()
            correction_log = []

        # 5) LLM 기반 해석(explanation) - 보정된 점수 기준 (간략화)
        try:
            # 대표 점수만 추출 (상위 3개 + 하위 2개)
            flat_scores = [(f"{mc}/{sc}", v) for mc, subs in scores.items() for sc, v in subs.items()]
            flat_scores.sort(key=lambda x: x[1], reverse=True)
            top_factors = flat_scores[:3]
            low_factors = flat_scores[-2:]
            
            explanation_prompt = f"""
아래 점수에서 상위/하위 요인의 이유를 3문장으로 설명하세요.
상위: {", ".join([f"{f}({v:.2f})" for f, v in top_factors])}
하위: {", ".join([f"{f}({v:.2f})" for f, v in low_factors])}
리뷰: {sample_reviews[:500]}
"""
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": explanation_prompt}],
                temperature=0.2,
                max_tokens=300,  # 토큰 제한
            )
            explanation = resp.choices[0].message.content.strip()
        except Exception as e:
            explanation = f"LLM 해석 실패: {e}"

        # 6) 결과 저장
        place_infos.append({
            'name': place.get('name', '이름 없음'), 
            'summary': summary,
            'address': place.get('formatted_address', place.get('vicinity', '주소 정보 없음')),
            'scores': scores, 
            'geometry': place.get('geometry', {}), 
            'place_id': place.get('place_id', ''),
            'positive_keywords': positive_keywords, 
            'negative_keywords': negative_keywords, 
            'explanation': explanation,
            'corrections': correction_log,  # 보정 내역 추가
        })

    state.places = place_infos
    return state.dict()

# ----------------------------------------------------
# 5. LangGraph 구성
# ----------------------------------------------------

graph = StateGraph(AgentState)
graph.add_node("search_places", search_places)
graph.add_node("analyze_reviews", analyze_reviews)
graph.set_entry_point("search_places")
graph.add_edge("search_places", "analyze_reviews")
graph.add_edge("analyze_reviews", END)
agent = graph.compile()


# ----------------------------------------------------
# 6. 실험용 데이터 수집 및 분석 함수
# ----------------------------------------------------

# 서울시 25개 구 목록
SEOUL_DISTRICTS = [
    "종로구", "중구", "용산구", "성동구", "광진구", 
    "동대문구", "중랑구", "성북구", "강북구", "도봉구",
    "노원구", "은평구", "서대문구", "마포구", "양천구",
    "강서구", "구로구", "금천구", "영등포구", "동작구",
    "관악구", "서초구", "강남구", "송파구", "강동구"
]

# 서울시 중심 좌표
SEOUL_CENTER = (37.5665, 126.9780)

def collect_cafes_in_district(district: str, max_results: int = 50) -> List[Dict]:
    """
    특정 구의 카페 데이터를 수집합니다.
    """
    try:
        query = f"서울특별시 {district} 카페"
        # Places API로 검색
        results = gmaps.places(
            query=query,
            language="ko",
            type="cafe"
        ).get('results', [])
        
        # place_id, 이름, 위치, 주소 등 기본 정보 추출
        cafes = []
        for place in results[:max_results]:
            if place.get('geometry') and place.get('place_id'):
                cafes.append({
                    'place_id': place['place_id'],
                    'name': place.get('name', ''),
                    'lat': place['geometry']['location']['lat'],
                    'lng': place['geometry']['location']['lng'],
                    'address': place.get('formatted_address', place.get('vicinity', '')),
                    'district': district,
                    'rating': place.get('rating', None),
                    'user_ratings_total': place.get('user_ratings_total', 0)
                })
        
        print(f"[INFO] {district}: {len(cafes)}개 카페 수집")
        return cafes
    
    except Exception as e:
        print(f"[ERROR] {district} 카페 수집 실패: {e}")
        return []

@st.cache_data(ttl=3600*24, show_spinner="서울 전역 카페 데이터 수집 중...")
def collect_all_cafes_seoul(_gmaps_client, max_per_district: int = 30) -> pd.DataFrame:
    """
    서울 전체 25개 구의 카페 데이터를 병렬 수집합니다.
    """
    all_cafes = []
    
    # 순차 수집 (API quota 고려)
    for district in SEOUL_DISTRICTS:
        cafes = collect_cafes_in_district(district, max_per_district)
        all_cafes.extend(cafes)
        time.sleep(0.5)  # API rate limit 방지
    
    df = pd.DataFrame(all_cafes)
    
    # 중복 제거 (place_id 기준)
    if not df.empty:
        df = df.drop_duplicates(subset=['place_id']).reset_index(drop=True)
        print(f"[INFO] 총 {len(df)}개 카페 수집 완료")
    
    return df

def calculate_transit_accessibility(lat: float, lng: float, max_distance: int = 600) -> Tuple[float, str, str, float, float]:
    """
    특정 위치에서 가장 가까운 지하철역/버스정류장까지의 도보 시간을 계산합니다.
    
    Returns:
        (도보_분, 최근접_역명, 타입, 도보_거리(m), 직선_거리(m))
    """
    try:
        print(f"[DEBUG] 접근성 계산 시작: lat={lat}, lng={lng}")
        
        # 반경 600m 내 지하철역 검색
        try:
            subway_results = gmaps.places_nearby(
                location=(lat, lng),
                radius=max_distance,
                type='subway_station',
                language='ko'
            ).get('results', [])
            print(f"[DEBUG] 지하철역 검색 결과: {len(subway_results)}개")
        except Exception as e:
            print(f"[ERROR] 지하철역 검색 실패: {e}")
            subway_results = []
        
        # 반경 600m 내 버스 정류장 검색
        try:
            bus_results = gmaps.places_nearby(
                location=(lat, lng),
                radius=max_distance,
                type='bus_station',
                language='ko'
            ).get('results', [])
            print(f"[DEBUG] 버스정류장 검색 결과: {len(bus_results)}개")
        except Exception as e:
            print(f"[ERROR] 버스정류장 검색 실패: {e}")
            bus_results = []
        
        # 검색 결과가 없으면 조기 반환
        if not subway_results and not bus_results:
            print(f"[WARN] 600m 내 역/정류장 없음")
            return None, "정보 없음", "없음", float("nan"), float("nan")
        
        from math import radians, cos, sin, asin, sqrt

        def haversine(lat1, lon1, lat2, lon2):
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            return 6371 * c * 1000  # meters

        def compute_walking_time(target_lat, target_lng, target_name, target_type):
            attempts = [
                ((lat, lng), (target_lat, target_lng), "카페→대중교통"),
                ((target_lat, target_lng), (lat, lng), "대중교통→카페"),
            ]
            for origin, destination, attempt_label in attempts:
                try:
                    result = gmaps.distance_matrix(
                        origins=[origin],
                        destinations=[destination],
                        mode="walking",
                        language="ko",
                        region="kr",
                    )
                    status = result["rows"][0]["elements"][0]["status"]
                    print(f"[DEBUG] Distance Matrix 응답 status ({attempt_label}): {status}")
                    if status == "OK":
                        duration_sec = result["rows"][0]["elements"][0]["duration"]["value"]
                        distance_m = result["rows"][0]["elements"][0]["distance"]["value"]
                        print(
                            f"[DEBUG] ✓ {target_name} ({target_type}) - Distance Matrix({attempt_label}): "
                            f"{duration_sec/60:.1f}분 ({distance_m}m)"
                        )
                        return duration_sec / 60.0, distance_m, "distance_matrix"
                except Exception as e:
                    print(f"[WARN] Distance Matrix 호출 실패 ({attempt_label}): {e}")

            try:
                directions = gmaps.directions(
                    origin=(lat, lng),
                    destination=(target_lat, target_lng),
                    mode="walking",
                    language="ko",
                    region="kr",
                )
                if directions:
                    leg = directions[0]["legs"][0]
                    duration_sec = leg["duration"]["value"]
                    distance_m = leg["distance"]["value"]
                    print(
                        f"[DEBUG] ✓ {target_name} ({target_type}) - Directions API: "
                        f"{duration_sec/60:.1f}분 ({distance_m}m)"
                    )
                    return duration_sec / 60.0, distance_m, "directions"
                print(f"[WARN] Directions API 결과 없음: {target_name} ({target_type})")
            except Exception as e:
                print(f"[WARN] Directions API 호출 실패: {target_name} ({target_type}) → {e}")
            distance_m = haversine(lat, lng, target_lat, target_lng)
            actual_distance_m = distance_m * 1.4
            duration = actual_distance_m / 67.0
            print(
                f"[DEBUG] ⚠ {target_name} ({target_type}): {duration:.1f}분 "
                f"(직선 {distance_m:.0f}m → 실제경로 추정 {actual_distance_m:.0f}m)"
            )
            return duration, actual_distance_m, "fallback"
        
        # 가장 가까운 역/정류장 찾기
        min_walk_time = 999
        min_walk_distance = float("inf")
        min_straight_distance = float("inf")
        nearest_name = "정보 없음"
        nearest_type = "없음"
        distance_matrix_success = False
        
        # 지하철역 처리
        for idx, station in enumerate(subway_results[:3]):  # 상위 3개만 검사
            try:
                station_loc = station["geometry"]["location"]
                station_name = station.get("name", "지하철역")
                print(f"[DEBUG] [{idx+1}/3] 지하철역 도보 시간 계산: {station_name}")

                straight_distance = haversine(lat, lng, station_loc["lat"], station_loc["lng"])
                duration, distance_m, _ = compute_walking_time(
                    station_loc["lat"], station_loc["lng"], station_name, "지하철역"
                )
                distance_matrix_success = True
                if duration < min_walk_time:
                    min_walk_time = duration
                    min_walk_distance = distance_m
                    min_straight_distance = straight_distance
                    nearest_name = station_name
                    nearest_type = "지하철역"

            except Exception as e:
                print(f"[ERROR] 지하철역 '{station_name}' 처리 오류: {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()

        # 버스정류장 처리
        for idx, bus in enumerate(bus_results[:3]):
            try:
                bus_loc = bus["geometry"]["location"]
                bus_name = bus.get("name", "버스정류장")
                print(f"[DEBUG] [{idx+1}/3] 버스정류장 도보 시간 계산: {bus_name}")

                straight_distance = haversine(lat, lng, bus_loc["lat"], bus_loc["lng"])
                duration, distance_m, _ = compute_walking_time(
                    bus_loc["lat"], bus_loc["lng"], bus_name, "버스정류장"
                )
                distance_matrix_success = True
                if duration < min_walk_time:
                    min_walk_time = duration
                    min_walk_distance = distance_m
                    min_straight_distance = straight_distance
                    nearest_name = bus_name
                    nearest_type = "버스정류장"

            except Exception as e:
                print(f"[ERROR] 버스정류장 '{bus_name}' 처리 오류: {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()
        
        # Distance Matrix API가 한 번도 성공하지 못했다면
        if not distance_matrix_success:
            print(f"[ERROR] Distance Matrix API 호출이 모두 실패했습니다")
            print(f"[INFO] 가능한 원인:")
            print(f"  1. Distance Matrix API가 활성화되지 않음")
            print(f"  2. API 키에 Distance Matrix API 권한이 없음")
            print(f"  3. Billing이 활성화되지 않음")
            print(f"  4. API quota 초과")
            return None, "Distance Matrix 실패", "오류", float("nan"), float("nan")
        
        # 결과 반환
        if min_walk_time < 999 and min_walk_distance != float("inf"):
            print(f"[SUCCESS] 최근접: {nearest_name} ({nearest_type}), 도보 {min_walk_time:.1f}분")
            return (
                round(min_walk_time, 1),
                nearest_name,
                nearest_type,
                float(min_walk_distance),
                float(min_straight_distance),
            )
        else:
            print(f"[WARN] Distance Matrix 호출은 성공했지만 유효한 경로를 찾지 못함")
            return None, "경로 없음", "없음", float("nan"), float("nan")
    
    except Exception as e:
        print(f"[CRITICAL ERROR] 접근성 계산 치명적 오류: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, "치명적 오류", "오류", float("nan"), float("nan")

def calculate_placeness_batch(df: pd.DataFrame, sample_size: int = None, progress_callback=None) -> pd.DataFrame:
    """
    카페 데이터프레임에 대해 장소성 점수를 일괄 계산합니다.
    """
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    results = []
    total = len(df)
    
    for idx, row in df.iterrows():
        # 진행률 업데이트
        if progress_callback:
            progress_callback(idx + 1, total, row.get('name', '?'))
        try:
            # Place Details API로 리뷰 가져오기
            details = gmaps.place(place_id=row['place_id'], language='ko').get('result', {})
            reviews = details.get('reviews', [])[:10]
            review_texts = [r['text'] for r in reviews if r.get('text')]
            
            if not review_texts:
                # 리뷰가 없으면 스킵
                continue
            
            # 리뷰 텍스트 병합
            review_text = "\n".join(review_texts)
            
            # 의미 단위 분리
            review_units = cached_semantic_split(review_text)
            if not review_units:
                continue
            
            # SBERT + 감성 모델 기반 점수 계산 (analyze_reviews 로직 재사용)
            factor_sentiments = {f: [] for f in category_embeddings.keys()}
            
            sentiment_scores = sentiment_model(review_units)
            unit_embs = embed_model.encode(review_units, normalize_embeddings=True)
            subcat_list = list(category_embeddings.keys())
            factor_mat = np.stack([category_embeddings[s] for s in subcat_list], axis=0)
            sim_mat = np.matmul(unit_embs, factor_mat.T)
            
            ALPHA, BETA = 0.75, 0.25
            
            for i, unit in enumerate(review_units):
                raw_sent = float(sentiment_scores[i]) if i < len(sentiment_scores) else 0.5
                sent_adj = np.clip((raw_sent - 0.3) / 0.7, 0, 1)
                sims = sim_mat[i]
                for j, sim in enumerate(sims):
                    sim_adj = np.clip((float(sim) - 0.3) / 0.5, 0, 1)
                    if sim_adj > 0:
                        f_name = subcat_list[j]
                        combined = ALPHA * sim_adj + BETA * sent_adj
                        score_scaled = 1 / (1 + np.exp(-2.2 * (combined - 0.4)))
                        factor_sentiments[f_name].append(float(score_scaled))
            
            # 키워드 부스팅 (접근성 포함)
            keyword_boosts = {
                "고유성": ["독특", "유니크", "차별", "컨셉", "테마", "특색", "개성", "유일", "독창"],
                "문화적 맥락": ["전통", "역사", "년", "오래", "옛", "고풍", "문화", "배경", "스토리", "세월", "내력", "유서", "레트로", "빈티지", "클래식", "앤티크", "과거", "옛날"],
                "지역 정체성": ["지역", "동네", "마을", "근처", "주변", "명소", "랜드마크", "상징", "대표", "신촌", "홍대", "강남", "이태원", "연남", "성수", "을지로", "익선동", "북촌", "삼청동", "종로", "명동"],
                "심미성": ["예쁘", "아름", "멋지", "세련", "야경", "뷰", "인테리어", "디자인", "조명", "아늑", "분위기", "감성"],
                "접근성": ["가깝", "접근", "역", "정류장", "도보", "편리"],
            }
            
            for factor, keywords in keyword_boosts.items():
                matched_kws = [kw for kw in keywords if kw in review_text]
                if matched_kws:
                    boosted_score = min(0.75 + (len(matched_kws) * 0.05), 0.95)
                    for _ in range(2):
                        factor_sentiments[factor].append(boosted_score)
            
            # 접근성 패턴 매칭
            accessibility_patterns = [
                r'도보\s*(?:로\s*)?(\d+)\s*분',
                r'(\d+)\s*분\s*도보',
                r'(\d+)\s*분\s*거리',
            ]
            
            for pattern in accessibility_patterns:
                matches = re.finditer(pattern, review_text)
                for match in matches:
                    time_str = match.group(1)
                    if time_str:
                        try:
                            time_minutes = int(time_str)
                            if time_minutes <= 5:
                                boost = 0.95
                            elif time_minutes <= 10:
                                boost = 0.90
                            else:
                                boost = 0.85
                            factor_sentiments["접근성"].append(boost)
                        except:
                            pass
            
            # 점수 정규화
            scores = json.loads(json.dumps(new_score_structure_template))
            all_vals = []
            for vals in factor_sentiments.values():
                all_vals.extend(vals)
            
            if all_vals:
                vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
            else:
                vmin, vmax = 0.5, 0.5
            
            for main_cat, subcats in scores.items():
                for subcat in subcats.keys():
                    vals = factor_sentiments.get(subcat, [])
                    if vals and vmax > vmin:
                        raw = float(np.mean(vals))
                        normed = 0.30 + 0.70 * ((raw - vmin) / (vmax - vmin + 1e-8))
                        scores[main_cat][subcat] = float(np.clip(normed, 0.30, 1.0))
                    elif vals:
                        scores[main_cat][subcat] = float(np.clip(vals[0], 0.30, 1.0))
                    else:
                        scores[main_cat][subcat] = 0.5
            
            # Overall 점수 계산 (전체 평균)
            all_scores = [s for main_cat, sub_scores in scores.items() for s in sub_scores.values() if s is not None]
            overall_score = np.mean(all_scores) if all_scores else 0.5
            
            # 결과 저장
            result = row.to_dict()
            result['overall_score'] = round(overall_score, 3)
            result['accessibility_score'] = round(scores.get('물리적 특성', {}).get('접근성', 0.5), 3)
            result['scores'] = scores
            results.append(result)
            
            print(f"[{idx+1}/{len(df)}] {row['name']}: overall={overall_score:.2f}, 접근성={result['accessibility_score']:.2f}")
            
        except Exception as e:
            print(f"[ERROR] {row.get('name', '?')} 점수 계산 실패: {e}")
            continue
    
    return pd.DataFrame(results)


# ----------------------------------------------------
# 7. Streamlit UI
# ----------------------------------------------------

st.title("장소성 요인 기반 공간 정량 평가 도구")

# CSS로 텍스트 색상 강제 (가독성 개선)
st.markdown("""
<style>
    .stMarkdown, .stCaption, p, div {
        color: #000000 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    /* 예시 텍스트는 회색 유지 */
    .example-text {
        color: #888888 !important;
    }
</style>
""", unsafe_allow_html=True)

# 탭 구성
tab1, tab2, tab4 = st.tabs(["🔍 개별 장소 분석", "🗺️ 서울 전역 실험", "📁 Google 리뷰 분석"])

# ========================================
# 탭 1: 개별 장소 분석 (기존 기능)
# ========================================
with tab1:
    st.markdown("분석할 공간의 위치와 감성/기능적 특성을 입력하십시오. "
                 "<span class='example-text'>(예: 신촌 조용한 카페, 종로구 전통적인 음식점, 마포구 산책로 공원)</span>", 
                 unsafe_allow_html=True)
    query = st.text_input("", placeholder="예: 신촌 조용한 카페", key="query_tab1")

    if st.button("장소성 정량 분석 시작", key="btn_analyze_tab1"):
        if not query.strip():
            st.warning("장소를 입력해주세요.")
        else:
            with st.spinner("NLP 및 LLM을 사용하여 사용자 리뷰를 기반으로 장소성 요인을 정량 평가하는 중..."):
                result = agent.invoke({"query": query, "places": [], "answer": ""})
                places = result.get('places', [])
                st.session_state.history.append((query, places))
                st.rerun()

    # 결과 출력
    if st.session_state.history:
        latest_query, latest_places = st.session_state.history[-1]
        st.markdown(f"---")
        st.markdown(f"### '{latest_query}'에 대한 장소성 평가 결과")

        for i, place in enumerate(latest_places):
            with st.container(border=True):
                st.subheader(place.get('name', '이름 정보 없음'))
                st.markdown(f"**📍 주소:** {place.get('address', '주소 정보 없음')}")
                
                # 2열 레이아웃: 왼쪽(시각화), 오른쪽(보정/해설)
                col_left, col_right = st.columns([1.2, 1])
                
                scores = place.get('scores')
                
                # ========== 왼쪽 열: 리뷰 요약 + 시각화 ==========
                with col_left:
                    st.markdown(f"**📝 리뷰 요약**")
                    st.markdown(place.get('summary', '요약 정보 없음'))
                
                with col_left:
                    if scores:
                        st.markdown(f"**📊 장소성 종합 평가**")

                        # Sunburst 차트 데이터 생성
                        labels = []
                        parents = []
                        values = []
                        colors = []

                        # 부드러운 파스텔톤 색상 맵 (factors.json 구조와 동일한 대분류 라벨)
                        color_map = {
                            "물리적 특성": "rgb(173, 216, 230)",     # 연한 파란색 (Light Blue)
                            "활동적 특성": "rgb(152, 251, 152)",   # 연한 연두색 (Light Lime Green)
                            "의미적 특성": "rgb(255, 182, 193)" # 연한 분홍색 (Light Pink)
                        }

                        # 루트 노드 추가 (전체 점수의 평균으로 설정)
                        all_scores = [s for main_cat, sub_scores in scores.items() for s in sub_scores.values() if s is not None]
                        total_score = sum(all_scores)
                        score_count = len(all_scores)
                        root_value = total_score / score_count if score_count > 0 else 0.5

                        labels.append(place['name'])
                        parents.append("")
                        values.append(root_value)
                        colors.append("#FFFFFF")

                        # 대분류와 세부 분류 추가
                        for main_cat, sub_scores in scores.items():
                            main_scores = [s for s in sub_scores.values() if s is not None]
                            main_avg = sum(main_scores) / len(main_scores) if main_scores else 0
                
                            labels.append(main_cat)
                            parents.append(place['name'])
                            values.append(main_avg)
                            colors.append(color_map.get(main_cat, "#CCCCCC"))
                
                            for sub_cat, score in sub_scores.items():
                                if score is not None:
                                    labels.append(f"{sub_cat}: {score:.2f}") # 점수를 라벨에 포함
                                    parents.append(main_cat)
                                    values.append(float(score))
                                    colors.append(color_map.get(main_cat, "#CCCCCC"))
                
                        # Sunburst 차트 생성
                        try:
                            fig_sunburst = go.Figure(go.Sunburst(
                                labels=labels,
                                parents=parents,
                                values=values,
                                branchvalues="remainder",
                                marker=dict(colors=colors),
                                hovertemplate='<b>%{customdata[0]}</b><br>점수: %{value:.2f}', # customdata는 사용하지 않으므로 value만 표시
                                maxdepth=2,
                                insidetextorientation='radial'
                            ))
                
                            fig_sunburst.update_layout(
                                margin=dict(t=20, l=10, r=10, b=10),
                                height=400,
                                title_text=f"{place['name']} 장소성 종합 평가",
                                font=dict(size=12, family="NotoSansKR, sans-serif")
                            )
                
                            st.plotly_chart(fig_sunburst, use_container_width=True, key=f"sunburst_{i}_{place.get('place_id','')}")
                
                        except Exception as e:
                            # Sunburst 실패 시 Treemap 시도
                            st.error(f"Sunburst 차트 생성 중 오류: {e}")
                            pass

                        # Radar Chart 생성 함수 정의
                        def make_radar_chart(scores_dict, title="장소성 요인 특성 분포"):
                            # 색상 매핑 (대분류 기준)
                            fill_color_map = {
                                "물리적 특성": "rgba(173, 216, 230, 0.5)",  # 연한 파란색
                                "활동적 특성": "rgba(152, 251, 152, 0.5)",  # 연한 초록색
                                "의미적 특성": "rgba(255, 182, 193, 0.5)"   # 연한 분홍색
                            }
                
                            # 전체 요인을 순서대로 배치 + 색상 매핑
                            categories = []
                            values = []
                            colors = []
                
                            for main_cat, subcats in scores_dict.items():
                                for subcat, val in subcats.items():
                                    categories.append(subcat)
                                    values.append(val if val is not None else 0.5)
                                    colors.append(fill_color_map.get(main_cat, "rgba(200,200,200,0.5)"))
                
                            fig = go.Figure()
                
                            # Barpolar로 각 축별 색상 구분
                            fig.add_trace(go.Barpolar(
                                r=values,
                                theta=categories,
                                marker=dict(
                                    color=colors,
                                    line=dict(color="rgba(80,80,80,0.3)", width=1)
                                ),
                                hovertemplate='<b>%{theta}</b><br>점수: %{r:.2f}<extra></extra>',
                                name="요인별 점수"
                            ))
                
                            # 윤곽선을 위한 Scatterpolar 추가
                            categories_closed = categories + categories[:1]
                            values_closed = values + values[:1]
                
                            fig.add_trace(go.Scatterpolar(
                                r=values_closed,
                                theta=categories_closed,
                                mode='lines',
                                line=dict(color="rgba(60, 60, 60, 0.8)", width=2.5),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True, 
                                        range=[0, 1], 
                                        tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                                        showline=True,
                                        gridcolor="rgba(200,200,200,0.5)"
                                    ),
                                    angularaxis=dict(rotation=90, direction="clockwise")
                                ),
                                showlegend=False,
                                height=580,
                                margin=dict(l=140, r=140, t=110, b=110),  # 여백 최대 확대
                                title=dict(text=title, x=0.5, font=dict(size=14, family="NotoSansKR"))
                            )
                
                            return fig
                
                        st.markdown(f"**📊 장소성 요인 특성 분포도**")
                        # Radar Chart 출력
                        fig_radar = make_radar_chart(scores, title=f"{place['name']} 장소성 특성 분포")
                    st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{i}_{place.get('place_id','')}")

            
                # ========== 오른쪽 열: LLM 보정 + 해설 ==========
                with col_right:
                    # LLM 보정 내역 표시
                    corrections = place.get('corrections', [])
                    if corrections:
                        st.markdown("**⚙️ LLM 점수 보정**")
                        st.caption("GPT-4o 검증 결과")
                
                        correction_df = pd.DataFrame(corrections)
                        correction_df = correction_df.rename(columns={
                            "factor": "요인",
                            "original": "원점수",
                            "adjusted": "보정",
                            "delta": "Δ",
                            "reason": "근거"
                        })
                        st.dataframe(
                            correction_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "요인": st.column_config.TextColumn(width="small"),
                                "원점수": st.column_config.NumberColumn(format="%.2f", width="small"),
                                "보정": st.column_config.NumberColumn(format="%.2f", width="small"),
                                "Δ": st.column_config.NumberColumn(format="%+.2f", width="small"),
                                "근거": st.column_config.TextColumn(width="medium"),
                            }
                        )
                    else:
                        st.markdown("**⚙️ LLM 점수 보정**")
                        st.caption("보정 필요 없음")
                
                    # 점수 해설 (보정 후 최종 점수 기준)
                    if place.get('explanation'):
                        st.markdown("**🔎 최종 점수 해설**")
                        st.markdown(place.get('explanation'))
                
                    # 워드 클라우드 시각화 (오른쪽 열 하단, 좌우 배치)
                    if place.get('positive_keywords') or place.get('negative_keywords'):
                        st.markdown("**📝 키워드 분석**")
                
                        wc_col1, wc_col2 = st.columns(2)
                
                        # 긍정 워드 클라우드
                        if place.get('positive_keywords'):
                            with wc_col1:
                                st.caption("✅ 긍정")
                                text = " ".join(place['positive_keywords'])
                                if text:
                                    img = generate_wordcloud(text, font_path, colormap="Greens")
                                    if img is not None:
                                        st.image(img, use_container_width=True)
                
                        # 부정 워드 클라우드
                        if place.get('negative_keywords'):
                            with wc_col2:
                                st.caption("❌ 부정")
                                text = " ".join(place['negative_keywords'])
                                if text:
                                    img = generate_wordcloud(text, font_path, colormap="Reds")
                                    if img is not None:
                                        st.image(img, use_container_width=True)
                
                # 지도 및 로드뷰 (기존 로직 유지)
                if place.get('geometry') and place['geometry'].get('location'):
                    lat, lng = place['geometry']['location']['lat'], place['geometry']['location']['lng']
                
                    map_key = f"map_{i}_{place['place_id']}"
                    streetview_key = f"street_{i}_{place['place_id']}"
                
                    if map_key not in st.session_state:
                        st.session_state[map_key] = False
                    if streetview_key not in st.session_state:
                        st.session_state[streetview_key] = False
                
                    col1, col2 = st.columns(2)
                
                    # 버튼 클릭 시 상태 토글 후 재실행하여 지도 표시
                    if col1.button("🗺️ 지도 보기", key=f"btn_{map_key}"):
                        st.session_state[map_key] = not st.session_state[map_key]
                        st.rerun()
                
                    if col2.button("🚗 로드뷰 보기", key=f"btn_{streetview_key}"):
                        st.session_state[streetview_key] = not st.session_state[streetview_key]
                        st.rerun()
                
                    if st.session_state[map_key] or st.session_state[streetview_key]:
                        st.markdown("**📍 위치 정보**")
                
                        map_col1, map_col2 = st.columns(2)
                
                        if st.session_state[map_key]:
                            with map_col1:
                                st.markdown("**🗺️ 지도**")
                                # Google Maps Embed API (전체 폭 사용)
                                map_url = f"https://www.google.com/maps/embed/v1/place?key={st.session_state.gmaps_key}&q={lat},{lng}"
                                st.markdown(
                                    f'<iframe src="{map_url}" width="100%" height="450" style="border:0;" allowfullscreen="" loading="lazy"></iframe>',
                                    unsafe_allow_html=True
                                )
                
                        if st.session_state[streetview_key]:
                            with map_col2:
                                st.markdown("**🚗 로드뷰**")
                                # Google Maps Street View Embed API (전체 폭 사용)
                                streetview_url = f"https://www.google.com/maps/embed/v1/streetview?key={st.session_state.gmaps_key}&location={lat},{lng}"
                                st.markdown(
                                    f'<iframe src="{streetview_url}" width="100%" height="450" style="border:0;" allowfullscreen="" loading="lazy"></iframe>',
                                    unsafe_allow_html=True
                                )
                else:
                    st.info("📍 위치 정보가 없어 지도를 표시할 수 없습니다.")

# ========================================
# 탭 2: 서울 전역 실험 (논문용 데이터 수집 및 검증)
# ========================================
with tab2:
    st.markdown("### 🗺️ 서울 카페 장소성 실험 및 검증")
    st.markdown("""
    이 탭에서는 서울 전역의 카페를 대상으로 대규모 장소성 평가를 수행하고,
    접근성 점수와 실제 대중교통 도보 시간 간의 상관관계를 검증합니다.
    """)
    
    # 데이터 수집 옵션
    st.subheader("1️⃣ 데이터 수집")
    
    col_config1, col_config2 = st.columns(2)
    with col_config1:
        sample_per_district = st.slider(
            "구당 최대 카페 수",
            min_value=10,
            max_value=50,
            value=20,
            help="각 구에서 수집할 최대 카페 개수"
        )
    
    with col_config2:
        score_sample_size = st.slider(
            "점수 계산 샘플 수",
            min_value=10,
            max_value=300,
            value=30,
            help="장소성 점수를 계산할 카페 샘플 수 (API quota 고려)"
        )
        st.caption(f"⏱️ 예상 시간: 약 {score_sample_size * 30 / 60:.0f}분 (카페당 ~30초)")
        if score_sample_size > 50:
            st.warning("⚠️ 50개 이상은 시간이 오래 걸립니다 (25분+)")
    
    if st.button("🔄 서울 전역 카페 데이터 수집 시작", key="btn_collect_cafes"):
        with st.spinner(f"서울 25개 구에서 카페 데이터 수집 중... (구당 최대 {sample_per_district}개)"):
            # 카페 기본 정보 수집
            cafes_df = collect_all_cafes_seoul(gmaps, max_per_district=sample_per_district)
            
            if not cafes_df.empty:
                st.session_state['cafes_df'] = cafes_df
                st.success(f"✅ 총 {len(cafes_df)}개 카페 수집 완료!")
                
                # 간단한 통계
                district_counts = cafes_df['district'].value_counts()
                st.dataframe(district_counts.head(10), use_container_width=True)
            else:
                st.error("카페 데이터 수집 실패")
    
    # 장소성 점수 계산
    if 'cafes_df' in st.session_state and not st.session_state['cafes_df'].empty:
        st.markdown("---")
        st.subheader("2️⃣ 장소성 점수 계산")
        
        if st.button("📊 장소성 점수 일괄 계산", key="btn_calc_scores"):
            cafes_df = st.session_state['cafes_df']
            
            # 예상 시간 표시
            estimated_time = score_sample_size * 30 / 60  # 카페당 30초 가정
            st.info(f"⏱️ 예상 소요 시간: 약 {estimated_time:.1f}분 (카페당 ~30초)")
            
            # 프로그레스 바 및 상태 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()
            
            import time
            start_time = time.time()
            
            def update_progress(current, total, cafe_name):
                progress = current / total
                progress_bar.progress(progress)
                
                elapsed = time.time() - start_time
                avg_time = elapsed / current if current > 0 else 30
                remaining = (total - current) * avg_time
                
                status_text.text(f"진행 중: {current}/{total} - {cafe_name}")
                time_text.text(f"⏱️ 경과: {elapsed/60:.1f}분 | 예상 남은 시간: {remaining/60:.1f}분 | 평균: {avg_time:.1f}초/카페")
            
            scored_df = calculate_placeness_batch(
                cafes_df, 
                sample_size=score_sample_size,
                progress_callback=update_progress
            )
            
            # 정리
            progress_bar.empty()
            status_text.empty()
            time_text.empty()
            
            if not scored_df.empty:
                st.session_state['scored_df'] = scored_df
                total_time = time.time() - start_time
                st.success(f"✅ {len(scored_df)}개 카페 점수 계산 완료! (소요 시간: {total_time/60:.1f}분)")
                
                # 기본 통계
                st.markdown("**점수 분포 통계**")
                stats_df = scored_df[['overall_score', 'accessibility_score']].describe()
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.error("점수 계산 실패")
    
    # 대중교통 접근성 계산
    if 'scored_df' in st.session_state and not st.session_state['scored_df'].empty:
        st.markdown("---")
        st.subheader("3️⃣ 대중교통 접근성 데이터 수집")
        
        st.info("""
        💡 **이 단계는 선택사항입니다**
        - 지도 마커에 **최근접 역/정류장, 도보 시간** 정보 추가
        - H1 가설 검증 (접근성 점수 vs 실제 도보 시간 상관관계)에 필요
        """)
        
        transit_sample = st.number_input(
            "접근성 계산 샘플 수 (Distance Matrix API quota 고려)",
            min_value=10,
            max_value=100,
            value=30,
            help="최근접 역/정류장까지의 도보 시간을 계산할 카페 수"
        )
        st.caption(f"⏱️ 예상 시간: 약 {transit_sample * 0.5:.0f}분 (카페당 ~30초)")
        
        if st.button("🚇 대중교통 접근성 계산", key="btn_calc_transit"):
            scored_df = st.session_state['scored_df']
            sample_df = scored_df.sample(n=min(transit_sample, len(scored_df)), random_state=42)
            
            results = []
            error_log = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            error_text = st.empty()
            
            for counter, (idx, row) in enumerate(sample_df.iterrows(), start=1):
                status_text.text(f"계산 중: {row['name']} ({counter}/{len(sample_df)})")
                
                try:
                    walk_time, nearest_name, transit_type, walk_distance, straight_distance = calculate_transit_accessibility(
                        row['lat'], row['lng']
                    )
                    results.append({
                        'place_id': row['place_id'],
                        'name': row['name'],
                        'lat': row['lat'],
                        'lng': row['lng'],
                        'district': row['district'],
                        'overall_score': row['overall_score'],
                        'accessibility_score': row['accessibility_score'],
                        'walk_time_minutes': walk_time,
                        'nearest_station': nearest_name,
                        'transit_type': transit_type,
                        'walk_distance_m': walk_distance,
                        'straight_distance_m': straight_distance,
                    })
                    
                    if walk_time is None:
                        error_log.append(f"{row['name']}: 주변에 역/정류장 없음")
                    
                except Exception as e:
                    error_log.append(f"{row['name']}: {str(e)}")
                    results.append({
                        'place_id': row['place_id'],
                        'name': row['name'],
                        'lat': row['lat'],
                        'lng': row['lng'],
                        'district': row['district'],
                        'overall_score': row['overall_score'],
                        'accessibility_score': row['accessibility_score'],
                        'walk_time_minutes': None,
                        'nearest_station': '오류',
                        'transit_type': '오류',
                        'walk_distance_m': None,
                        'straight_distance_m': None,
                    })
                
                progress_bar.progress(counter / len(sample_df))
                
                # 에러 로그 표시
                if error_log:
                    error_text.text(f"⚠️ 오류: {len(error_log)}개 | 마지막: {error_log[-1]}")
                
                time.sleep(0.3)  # API rate limit
            
            transit_df = pd.DataFrame(results)
            # None 값 필터링 (접근성 계산 실패한 경우)
            valid_transit_df = transit_df[transit_df['walk_time_minutes'].notna()].reset_index(drop=True)
            
            if not valid_transit_df.empty:
                st.session_state['transit_df'] = valid_transit_df
                st.success(f"✅ {len(valid_transit_df)}개 카페 접근성 데이터 수집 완료!")
                
                if error_log:
                    st.warning(f"⚠️ {len(error_log)}개 카페는 접근성 계산 실패 (600m 내 역/정류장 없음)")
                    with st.expander("오류 상세 보기"):
                        for err in error_log:
                            st.text(err)
                
                st.dataframe(valid_transit_df.head(10), use_container_width=True)
            else:
                st.error("❌ 접근성 데이터 수집 실패: 모든 카페에서 600m 내 역/정류장을 찾지 못했습니다")
                st.info("""
                **가능한 원인:**
                1. Google Maps API의 Places Nearby API가 비활성화됨
                2. API 키의 quota 초과
                3. 선택된 카페들이 대중교통에서 너무 멀리 위치
                
                **해결 방법:**
                - Google Cloud Console에서 Places API, Distance Matrix API 활성화 확인
                - 다른 구역의 카페로 재시도
                """)
                
                if error_log:
                    with st.expander("오류 상세 보기"):
                        for err in error_log:
                            st.text(err)
            
            status_text.empty()
            progress_bar.empty()
            error_text.empty()
    
    # 시각화 및 분석
    if 'scored_df' in st.session_state and not st.session_state['scored_df'].empty:
        st.markdown("---")
        st.subheader("4️⃣ 시각화 및 분석")
        
        scored_df = st.session_state['scored_df']
        
        # E1: 구 단위 Choropleth
        st.markdown("**E1: 구 단위 장소성 강도**")
        district_scores = scored_df.groupby('district')['overall_score'].mean().sort_values(ascending=False)
        
        # Choropleth 지도
        st.markdown("**행정구별 장소성 히트맵**")
        
        # GeoJSON URL (서울시 구 경계)
        seoul_geo_url = "https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json"
        
        try:
            import requests
            geo_response = requests.get(seoul_geo_url)
            seoul_geo = geo_response.json()
            
            # Folium Choropleth 지도 생성
            m_choropleth = folium.Map(location=SEOUL_CENTER, zoom_start=11, tiles='OpenStreetMap')
            
            # district_scores를 DataFrame으로 변환
            choropleth_data = district_scores.reset_index()
            choropleth_data.columns = ['district', 'score']
            
            # Choropleth 레이어 추가
            folium.Choropleth(
                geo_data=seoul_geo,
                name='choropleth',
                data=choropleth_data,
                columns=['district', 'score'],
                key_on='feature.properties.name',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='평균 장소성 점수',
                highlight=True
            ).add_to(m_choropleth)
            
            # 구별 평균 점수 표시 (툴팁)
            style_function = lambda x: {'fillColor': '#ffffff', 'color':'#000000', 'fillOpacity': 0.1, 'weight': 0.1}
            highlight_function = lambda x: {'fillColor': '#000000', 'color':'#000000', 'fillOpacity': 0.50, 'weight': 0.1}
            
            folium.GeoJson(
                seoul_geo,
                style_function=style_function,
                control=False,
                highlight_function=highlight_function,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['name'],
                    aliases=['구:'],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                )
            ).add_to(m_choropleth)
            
            folium.LayerControl().add_to(m_choropleth)
            st_folium(m_choropleth, width=None, height=500)
            
        except Exception as e:
            st.error(f"Choropleth 지도 생성 실패: {e}")
            # 막대 차트로 대체
            fig_district = go.Figure(go.Bar(
                x=district_scores.values,
                y=district_scores.index,
                orientation='h',
                marker=dict(color=district_scores.values, colorscale='Viridis', showscale=True),
                text=[f"{v:.2f}" for v in district_scores.values],
                textposition='auto'
            ))
            fig_district.update_layout(
                title="서울시 자치구별 평균 장소성 점수",
                xaxis_title="평균 장소성 점수",
                yaxis_title="자치구",
                height=600,
                showlegend=False
            )
            st.plotly_chart(fig_district, use_container_width=True)
        
        # E2: 포인트 히트맵 (Folium)
        st.markdown("**E2: 카페 위치 및 장소성 히트맵**")
        
        # transit_df가 있으면 대중교통 정보 포함, 없으면 기본 정보만
        use_transit_data = 'transit_df' in st.session_state and not st.session_state['transit_df'].empty
        
        if use_transit_data:
            st.info("✅ 대중교통 접근성 정보 포함")
            display_df = st.session_state['transit_df']
        else:
            st.info("ℹ️ 기본 정보만 표시 (대중교통 정보는 3단계에서 계산 가능)")
            display_df = scored_df
        
        # Folium 지도 생성
        m = folium.Map(location=SEOUL_CENTER, zoom_start=11, tiles='OpenStreetMap')
        
        # 히트맵 데이터 준비
        heat_data = [[row['lat'], row['lng'], row['overall_score']] 
                     for _, row in display_df.iterrows() if row['overall_score'] > 0]
        
        HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)
        
        # 마커 클러스터
        marker_cluster = MarkerCluster().add_to(m)
        
        for _, row in display_df.head(100).iterrows():
            has_transit_info = (use_transit_data and 'walk_time_minutes' in row and 
                               row.get('walk_time_minutes') is not None and pd.notna(row.get('walk_time_minutes')))
            
            if has_transit_info:
                popup_html = f"""
                <div style="font-family: NotoSansKR, sans-serif; min-width: 200px;">
                    <h4 style="margin: 0 0 10px 0; color: #1f77b4;">{row['name']}</h4>
                    <hr style="margin: 5px 0;">
                    <b>📊 장소성 점수</b>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        <li>전체: <b>{row['overall_score']:.2f}</b></li>
                        <li>접근성: <b>{row['accessibility_score']:.2f}</b></li>
                    </ul>
                    <b>🚇 대중교통 접근성</b>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        <li>최근접: <b>{row['nearest_station']}</b></li>
                        <li>도보: <b style="color: {'green' if row['walk_time_minutes'] <= 5 else 'orange' if row['walk_time_minutes'] <= 10 else 'red'};">{row['walk_time_minutes']:.1f}분</b></li>
                        <li>유형: {row['transit_type']}</li>
                    </ul>
                    <b>📍 위치</b><br>
                    <span style="color: #666;">{row['district']}</span>
                </div>
                """
                marker_color = 'green' if row['walk_time_minutes'] <= 5 else 'blue' if row['walk_time_minutes'] <= 10 else 'orange'
            else:
                popup_html = f"""
                <div style="font-family: NotoSansKR, sans-serif; min-width: 180px;">
                    <h4 style="margin: 0 0 10px 0; color: #1f77b4;">{row['name']}</h4>
                    <hr style="margin: 5px 0;">
                    <b>📊 점수</b>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        <li>장소성: <b>{row['overall_score']:.2f}</b></li>
                        <li>접근성: <b>{row['accessibility_score']:.2f}</b></li>
                    </ul>
                    <b>📍 구</b><br>
                    <span style="color: #666;">{row['district']}</span>
                    <hr style="margin: 5px 0;">
                    <small style="color: #999;">💡 대중교통 정보는<br>3단계에서 추가 가능</small>
                </div>
                """
                marker_color = 'blue' if row['overall_score'] > 0.7 else 'gray'
            
            folium.Marker(
                location=[row['lat'], row['lng']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=row['name'],
                icon=folium.Icon(color=marker_color, icon='coffee', prefix='fa')
            ).add_to(marker_cluster)
        
        st_folium(m, width=None, height=600)
    
    # H1 검증: 접근성 점수 vs 도보 시간 상관관계
    if 'transit_df' in st.session_state and not st.session_state['transit_df'].empty:
        st.markdown("---")
        st.subheader("5️⃣ H1 검증: 접근성 점수 vs 대중교통 도보 시간")
        
        # 결측치 제거
        transit_df = st.session_state['transit_df'].dropna(subset=['walk_time_minutes', 'accessibility_score'])
        
        st.markdown("""
        **가설 H1**: "최근접 대중교통 도보 시간이 짧을수록 접근성 점수가 높다."
        
        상관계수가 음수(-) 값을 가지면 가설이 지지됩니다.
        (도보 시간 ↓ → 접근성 점수 ↑)
        """)
        
        # ① 정규성 검정
        shapiro_walk = stats.shapiro(transit_df['walk_time_minutes'])
        shapiro_acc = stats.shapiro(transit_df['accessibility_score'])
        is_normal = shapiro_walk.pvalue > 0.05 and shapiro_acc.pvalue > 0.05
    
        # ② 상관계수 계산 (정규분포면 Pearson, 아니면 Spearman)
        if is_normal:
            corr_type = "Pearson"
            correlation, p_value = stats.pearsonr(
                transit_df['walk_time_minutes'], transit_df['accessibility_score']
            )
        else:
            corr_type = "Spearman"
            correlation, p_value = stats.spearmanr(
                transit_df['walk_time_minutes'], transit_df['accessibility_score']
            )
    
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric(f"{corr_type} 상관계수 (r)", f"{correlation:.3f}")
        with col_stat2:
            st.metric("p-value", f"{p_value:.4f}")
        
        if p_value < 0.05:
            if correlation < 0:
                st.success(f"✅ 가설 지지: 접근성 점수와 도보 시간 간 유의미한 음의 상관관계 (r={correlation:.3f}, p<0.05)")
            else:
                st.warning(f"⚠️ 가설 기각: 양의 상관관계 발견 (r={correlation:.3f}, p<0.05)")
        else:
            st.info(f"📊 유의미한 상관관계 없음 (p={p_value:.4f} > 0.05)")
        
        # 산점도
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=transit_df['walk_time_minutes'],
            y=transit_df['accessibility_score'],
            mode='markers',
            marker=dict(
                size=10,
                color=transit_df['accessibility_score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="접근성<br>점수"),
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=[f"{row['name']}<br>{row['nearest_station']}" for _, row in transit_df.iterrows()],
            hovertemplate='<b>%{text}</b><br>도보: %{x:.1f}분<br>접근성: %{y:.2f}<extra></extra>'
        ))
        
        # 회귀선 추가
        from scipy.stats import linregress
        slope, intercept, r_value, p_value_reg, std_err = linregress(
            transit_df['walk_time_minutes'],
            transit_df['accessibility_score']
        )
        
        x_range = np.linspace(
            transit_df['walk_time_minutes'].min(),
            transit_df['walk_time_minutes'].max(),
            100
        )
        y_pred = slope * x_range + intercept
        
        fig_scatter.add_trace(go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            name=f'회귀선 (y={slope:.3f}x+{intercept:.3f})',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_scatter.update_layout(
            title=f"접근성 점수 vs 대중교통 도보 시간 (r={correlation:.3f}, p={p_value:.4f})",
            xaxis_title="최근접 역/정류장까지 도보 시간 (분)",
            yaxis_title="접근성 점수 (모델 예측)",
            height=500,
            hovermode='closest',
            showlegend=True
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # 데이터 테이블
        st.markdown("**상세 데이터**")
        display_df_table = transit_df[['name', 'district', 'overall_score', 'accessibility_score', 
                                 'walk_time_minutes', 'nearest_station', 'transit_type']].copy()
        display_df_table.columns = ['카페명', '구', '전체 장소성', '접근성 점수', 
                              '도보(분)', '최근접 역/정류장', '유형']
        st.dataframe(display_df_table, use_container_width=True, height=300)


# ========================================
# 탭 4: Google 리뷰 CSV 기반 분석
# ========================================
with tab4:
    st.markdown("### 📚 데이터셋 확인")
    dataset_targets = [
        ("서울시 카페 표본 (접근성 포함)", SAMPLED_CAFE_WITH_TRANSIT_CSV),
        ("Google 리뷰 표본", GOOGLE_REVIEW_SAMPLE_CSV),
    ]
    for dataset_title, dataset_path in dataset_targets:
        st.markdown(f"#### {dataset_title}")
        if dataset_path.exists():
            try:
                dataset_df = load_csv_generic(dataset_path)
                preview_height = min(520, max(240, 35 * min(len(dataset_df), 50)))
                st.dataframe(
                    dataset_df,
                    use_container_width=True,
                    hide_index=True,
                    height=preview_height,
                )
                st.download_button(
                    f"📥 {dataset_path.name} 다운로드",
                    data=dataset_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=dataset_path.name,
                    mime="text/csv",
                    key=f"download_{dataset_path.stem}",
                )
            except Exception as dataset_err:
                st.error(f"데이터 로딩에 실패했습니다: {dataset_err}")
        else:
            st.info(f"`{dataset_path.name}` 파일을 찾을 수 없습니다.")

    st.divider()
    st.markdown("### 📁 Google 리뷰 데이터 분석")
    st.caption("리뷰 CSV(표본/최근 수집)를 불러와 감성 분석과 평점 상관 관계를 탐색합니다.")

    if not GOOGLE_REVIEW_SAMPLE_CSV.exists() and not GOOGLE_REVIEW_LIVE_CSV.exists():
        st.warning("Google 리뷰 CSV 파일이 없습니다. 탭 3에서 리뷰를 수집하거나 CSV를 준비해주세요.")
    else:
        available_sources = {}
        if GOOGLE_REVIEW_SAMPLE_CSV.exists():
            available_sources[f"표본 (전체 2500개) - {GOOGLE_REVIEW_SAMPLE_CSV.name}"] = GOOGLE_REVIEW_SAMPLE_CSV
        if GOOGLE_REVIEW_LIVE_CSV.exists():
            available_sources[f"최근 수집 (탭3) - {GOOGLE_REVIEW_LIVE_CSV.name}"] = GOOGLE_REVIEW_LIVE_CSV

        selected_label = st.selectbox(
            "분석할 리뷰 CSV 선택",
            options=list(available_sources.keys()),
            index=0,
        )
        selected_path = available_sources[selected_label]

        try:
            base_review_df = load_google_reviews_csv(selected_path)
        except Exception as e:
            st.error(f"CSV 로딩 실패: {e}")
            base_review_df = None

        if base_review_df is not None:
            if "리뷰" not in base_review_df.columns:
                st.error("CSV에 '리뷰' 컬럼이 없어 감성 분석을 진행할 수 없습니다.")
            else:
                st.success(f"✅ {len(base_review_df):,}개 리뷰 로드 완료 ({selected_path.name})")

                with st.expander("🔍 필터", expanded=False):
                    district_options = (
                        sorted(base_review_df["시군구명"].dropna().unique().tolist())
                        if "시군구명" in base_review_df.columns
                        else []
                    )
                    selected_districts = st.multiselect(
                        "시군구 선택",
                        district_options,
                        placeholder="전체 시군구",
                        key="tab4_filter_district",
                    )

                    keyword = st.text_input(
                        "상호명/리뷰 검색어",
                        value="",
                        placeholder="예: 홍대, 분위기, 친절",
                        key="tab4_filter_keyword",
                    ).strip()
                    keyword_lower = keyword.lower()

                    min_review_per_place = st.slider(
                        "최소 리뷰 수 (집계 대상)",
                        min_value=1,
                        max_value=50,
                        value=3,
                        step=1,
                        key="tab4_min_reviews",
                    )

                filtered_reviews = base_review_df.copy()
                if selected_districts and "시군구명" in filtered_reviews.columns:
                    filtered_reviews = filtered_reviews[filtered_reviews["시군구명"].isin(selected_districts)]
                if keyword_lower:
                    filtered_reviews = filtered_reviews[
                        filtered_reviews.apply(
                            lambda row: any(
                                keyword_lower in str(row.get(col, "")).lower()
                                for col in ["상호명", "리뷰", "행정동명", "도로명주소"]
                            ),
                            axis=1,
                        )
                    ]

                st.info(f"필터링 결과 리뷰 수: {len(filtered_reviews):,}건")

                current_config = {
                    "source": selected_path.resolve().as_posix(),
                    "districts": tuple(selected_districts),
                    "keyword": keyword_lower,
                    "min_reviews": min_review_per_place,
                }
                prev_config = st.session_state.get("tab4_analysis_config")
                if prev_config != current_config:
                    st.session_state["tab4_analysis_config"] = current_config
                    st.session_state.pop("tab4_grouped_df", None)
                    st.session_state.pop("tab4_analysis_results_raw", None)
                    st.session_state.pop("tab4_analysis_results_final", None)
                    st.session_state["tab4_analysis_active"] = False

                display_cols = [
                    col
                    for col in ["상호명", "시군구명", "행정동명", "place_id", "평점", "리뷰", "작성일"]
                    if col in filtered_reviews.columns
                ]
                table_height = min(600, max(250, 35 * min(len(filtered_reviews), 50)))
                st.dataframe(
                    filtered_reviews[display_cols].head(1000),
                    use_container_width=True,
                    hide_index=True,
                    height=table_height,
                )

                group_cols = [col for col in ["place_id", "상호명", "시군구명", "행정동명"] if col in filtered_reviews.columns]
                if not group_cols:
                    st.error("집계를 위한 식별 컬럼(place_id 또는 상호명/시군구명 등)이 없습니다.")
                else:
                    st.markdown("#### ⚙️ 감성 분석 실행")
                    analysis_active = st.session_state.get("tab4_analysis_active", False)
                    run_analysis = st.button("🚀 감성 분석 실행", key="tab4_run_analysis")
                    if run_analysis:
                        st.session_state["tab4_analysis_active"] = True
                        analysis_active = True

                    if not analysis_active:
                        st.info("필터를 설정한 뒤 ‘감성 분석 실행’ 버튼을 눌러 분석을 시작하세요.")
                        st.stop()

                    grouped_state = st.session_state.get("tab4_grouped_df")
                    if run_analysis or grouped_state is None:
                        aggregated = filtered_reviews.copy()
                        aggregated["리뷰"] = aggregated["리뷰"].fillna("").astype(str)
                        aggregated["리뷰정제"] = aggregated["리뷰"].str.strip()
                        aggregated["리뷰감성점수"] = np.nan

                        valid_mask = aggregated["리뷰정제"] != ""
                        if not valid_mask.any():
                            st.info("유효한 리뷰 텍스트가 없어 감성 분석을 수행할 수 없습니다.")
                            st.session_state["tab4_analysis_active"] = False
                            st.stop()

                        texts = aggregated.loc[valid_mask, "리뷰정제"].tolist()
                        sentiment_scores = sentiment_model(texts)
                        aggregated.loc[valid_mask, "리뷰감성점수"] = sentiment_scores

                        grouped = (
                            aggregated.groupby(group_cols, dropna=False)
                            .agg(
                                평균평점=("평점", "mean"),
                                리뷰수=("리뷰정제", lambda s: int((s != "").sum())),
                                평균감성점수=("리뷰감성점수", "mean"),
                            )
                            .reset_index()
                        )

                        grouped = grouped[grouped["리뷰수"] >= min_review_per_place]

                        if grouped.empty:
                            st.warning("조건을 만족하는 집계 데이터가 없습니다.")
                            st.session_state["tab4_analysis_active"] = False
                            st.stop()

                        st.session_state["tab4_grouped_df"] = grouped.copy()
                    else:
                        grouped = grouped_state.copy()

                    st.markdown("#### 📊 감성 분석 결과")
                    base_analysis_cols = [c for c in group_cols if c != "place_id"] + ["평균평점", "평균감성점수", "리뷰수"]
                    base_height = min(400, max(240, 38 * len(grouped)))
                    st.dataframe(
                        grouped[base_analysis_cols],
                        use_container_width=True,
                        hide_index=True,
                        height=base_height,
                    )

                    base_corr_df = grouped.dropna(subset=["평균평점", "평균감성점수"])
                    if len(base_corr_df) >= 2:
                        base_corr, base_p = stats.pearsonr(
                            base_corr_df["평균평점"].astype(float),
                            base_corr_df["평균감성점수"].astype(float),
                        )
                        base_p_text = f"{base_p:.4f}" if base_p >= 1e-4 else f"{base_p:.2e}"
                        if base_p < 0.05:
                            if base_corr > 0:
                                base_msg = " → 통계적으로 유의미한 양의 상관관계가 확인됩니다 (α=0.05)."
                            else:
                                base_msg = " → 통계적으로 유의미한 음의 상관관계가 확인됩니다 (α=0.05).\n   ※ 기대와 다른 방향의 상관관계이므로 데이터 분포나 모델 결과를 추가 검토하세요."
                        else:
                            base_msg = " → 통계적으로 유의미한 상관관계로 보기 어렵습니다 (α=0.05)."
                        st.markdown("#### 📈 리뷰 평점 vs 감성 분석 점수 (장소성 요인 분석 전)")
                        st.write(f"상관계수(피어슨 r): **{base_corr:.3f}** (p-value={base_p_text}){base_msg}")
                        base_slope, base_intercept, _, _, _ = stats.linregress(
                            base_corr_df["평균평점"].astype(float),
                            base_corr_df["평균감성점수"].astype(float),
                        )
                        st.caption(f"회귀선: y = {base_slope:.3f}x + {base_intercept:.3f}")
                        base_fig = px.scatter(
                            base_corr_df,
                            x="평균평점",
                            y="평균감성점수",
                            hover_data=[col for col in group_cols if col != "place_id" and col in base_corr_df.columns],
                            trendline="ols",
                            labels={"평균평점": "Google 평점 평균", "평균감성점수": "감성 점수 평균"},
                            title="평균 평점 vs 감성 점수",
                        )
                        st.plotly_chart(base_fig, use_container_width=True, key="tab4_pre_factor_corr")
                    else:
                        st.info("기본 감성 분석 상관관계를 계산하기 위한 데이터가 충분하지 않습니다.")

                    st.markdown("---")
                    st.markdown("#### 🧪 장소성 요인별 상세 분석")
                    st.caption("장소성 요인별 분석에는 다소 시간이 소요될 수 있습니다.")

                    with open("factors.json", "r", encoding="utf-8") as f:
                        factor_definitions_tab4 = json.load(f)

                    raw_results_state = st.session_state.get("tab4_analysis_results_raw")
                    final_results_state = st.session_state.get("tab4_analysis_results_final")
                    if run_analysis or raw_results_state is None:
                        progress_placeholder = st.empty()
                        status_placeholder = st.empty()
                        with progress_placeholder:
                            with st.spinner("장소성 요인별 분석 수행 중..."):
                                raw_results = tab4_analysis.analyze_review_groups(
                                    review_df=filtered_reviews,
                                    group_cols=group_cols,
                                    sentiment_model=sentiment_model,
                                    embed_model=embed_model,
                                    category_embeddings=category_embeddings,
                                    score_template=new_score_structure_template,
                                    semantic_split_fn=cached_semantic_split,
                                    llm_client=client,
                                    factor_definitions=factor_definitions_tab4,
                                    llm_delta_limit=0.5,
                                    progress_callback=lambda current, total, context: (
                                        status_placeholder.text(f"분석 진행 중: {current}/{total} - {context}")
                                    ),
                                )
                        progress_placeholder.empty()
                        status_placeholder.empty()
                        st.session_state["tab4_analysis_results_raw"] = raw_results.copy()
                    else:
                        raw_results = raw_results_state.copy()

                    if run_analysis or final_results_state is None:
                        analysis_results = raw_results.copy()

                        # 문장 단위 감성 평균을 별도 보관하고, 테이블용 평균감성점수는 상단 집계값으로 덮어쓰기
                        if "평균감성점수" in analysis_results.columns:
                            analysis_results = analysis_results.rename(
                                columns={"평균감성점수": "평균감성점수(문장단위)"}
                            )

                        merge_keys = [col for col in group_cols if col in grouped.columns and col in analysis_results.columns]
                        if merge_keys:
                            base_sentiments = grouped[merge_keys + ["평균감성점수"]].copy()
                            analysis_results = analysis_results.merge(base_sentiments, on=merge_keys, how="left")
                        else:
                            analysis_results["평균감성점수"] = np.nan

                        analysis_results = analysis_results[analysis_results["리뷰수"] >= min_review_per_place]

                        if analysis_results.empty:
                            st.warning("장소성 요인 분석을 위한 데이터가 충분하지 않습니다.")
                            st.stop()
                    else:
                        analysis_results = final_results_state.copy()

                    score_columns = []
                    accessibility_col = None
                    if not analysis_results["scores"].dropna().empty:
                        example_scores = analysis_results["scores"].dropna().iloc[0]
                        score_columns = [
                            f"{main}/{sub}"
                            for main, subdict in example_scores.items()
                            for sub in subdict.keys()
                        ]
                        expanded_scores = analysis_results["scores"].apply(
                            lambda score_dict: {
                                f"{main}/{sub}": subdict.get(sub)
                                for main, subdict in score_dict.items()
                                for sub in subdict.keys()
                            }
                            if isinstance(score_dict, dict)
                            else {}
                        )
                        expanded_df = pd.DataFrame(list(expanded_scores), index=analysis_results.index)
                        new_score_cols = [col for col in expanded_df.columns if col not in analysis_results.columns]
                        if new_score_cols:
                            analysis_results = pd.concat([analysis_results, expanded_df[new_score_cols]], axis=1)
                        if "물리적 특성/접근성" in analysis_results.columns:
                            accessibility_col = "물리적 특성/접근성"

                    st.markdown("#### 📊 장소성 분석 결과 (요인별 평균)")
                    analysis_cols = [
                        col
                        for col in (
                            [c for c in group_cols if c != "place_id"]
                            + ["평균평점", "평균감성점수", "평균장소성점수", "리뷰수", "리뷰문장수"]
                        )
                        if col in analysis_results.columns
                    ]
                    score_display_cols = [col for col in score_columns if col in analysis_results.columns]
                    display_cols = analysis_cols + score_display_cols
                    analysis_height = min(600, max(240, 38 * len(analysis_results)))
                    st.dataframe(
                        analysis_results[display_cols] if display_cols else analysis_results,
                    use_container_width=True,
                    hide_index=True,
                        height=analysis_height,
                    )

                    if "corrections" in analysis_results.columns:
                        total_corrections = analysis_results["corrections"].apply(lambda logs: len(logs) if isinstance(logs, list) else 0).sum()
                        st.caption(f"LLM 보정 건수: {total_corrections}건")
                        with st.expander("LLM 보정 상세", expanded=False):
                            correction_rows = []
                            for _, row in analysis_results.iterrows():
                                logs = row.get("corrections") or []
                                for log in logs:
                                    correction_rows.append(
                                        {
                                            "상호명": row.get("상호명"),
                                            "시군구명": row.get("시군구명"),
                                            "행정동명": row.get("행정동명"),
                                            **log,
                                        }
                                    )
                            if correction_rows:
                                correction_df = pd.DataFrame(correction_rows)
                                st.dataframe(correction_df, use_container_width=True, hide_index=True)
                            else:
                                st.write("보정된 항목이 없습니다.")

                    corr_df = analysis_results.dropna(subset=["평균평점", "평균장소성점수"])
                    if len(corr_df) >= 2:
                        corr_value, corr_p = stats.pearsonr(
                            corr_df["평균평점"].astype(float),
                            corr_df["평균장소성점수"].astype(float),
                        )
                        st.markdown("#### 📈 리뷰 평점 vs 장소성 점수 (요인 평균)")
                        p_text = f"{corr_p:.4f}" if corr_p >= 1e-4 else f"{corr_p:.2e}"
                        if corr_p < 0.05:
                            if corr_value > 0:
                                significance_msg = " → 통계적으로 유의미한 양의 상관관계가 확인됩니다 (α=0.05)."
                            else:
                                significance_msg = " → 통계적으로 유의미한 음의 상관관계가 확인됩니다 (α=0.05).\n   ※ 기대와 다른 방향의 상관관계이므로 데이터 분포나 모델 결과를 추가 검토하세요."
                        else:
                            significance_msg = " → 통계적으로 유의미한 상관관계로 보기 어렵습니다 (α=0.05)."
                        st.write(f"상관계수(피어슨 r): **{corr_value:.3f}** (p-value={p_text}){significance_msg}")
                        slope, intercept, _, _, _ = stats.linregress(
                            corr_df["평균평점"].astype(float),
                            corr_df["평균장소성점수"].astype(float),
                        )
                        st.caption(f"회귀선: y = {slope:.3f}x + {intercept:.3f}")
                        fig_corr = px.scatter(
                            corr_df,
                            x="평균평점",
                            y="평균장소성점수",
                            hover_data=[col for col in group_cols if col != "place_id"],
                            trendline="ols",
                            labels={"평균평점": "Google 평점 평균", "평균장소성점수": "장소성 점수 평균"},
                            title="평균 평점 vs 장소성 점수 (요인 평균)",
                        )
                        st.plotly_chart(fig_corr, use_container_width=True, key="tab4_sentiment_corr")
                    else:
                        st.info("상관 분석을 수행하기 위한 데이터가 충분하지 않습니다.")

                    # 좌표/접근성 데이터 보강
                    if (
                        ("lat" not in analysis_results.columns or analysis_results["lat"].isna().all())
                        and "place_id" in analysis_results.columns
                    ):
                        with st.spinner("지도 좌표 정보를 불러오는 중입니다..."):
                            coord_cache = {}
                            for pid in analysis_results["place_id"].dropna().unique():
                                try:
                                    detail = gmaps.place(place_id=pid, language="ko")
                                    loc = (
                                        detail.get("result", {})
                                        .get("geometry", {})
                                        .get("location", {})
                                    )
                                    if loc:
                                        coord_cache[pid] = (loc.get("lat"), loc.get("lng"))
                                except Exception:
                                    coord_cache[pid] = (np.nan, np.nan)
                            analysis_results["lat"] = analysis_results["place_id"].map(
                                lambda pid: coord_cache.get(pid, (np.nan, np.nan))[0]
                            )
                            analysis_results["lng"] = analysis_results["place_id"].map(
                                lambda pid: coord_cache.get(pid, (np.nan, np.nan))[1]
                            )

                    accessibility_required_cols = {"walk_time_minutes", "walk_distance_m", "straight_distance_m", "nearest_station", "transit_type"}
                    has_accessibility_cols = accessibility_required_cols.issubset(analysis_results.columns)
                    needs_accessibility = not has_accessibility_cols
                    if has_accessibility_cols:
                        needs_accessibility = analysis_results[
                            list({"walk_time_minutes", "walk_distance_m", "straight_distance_m"} & set(analysis_results.columns))
                        ].isna().any().any()

                    if needs_accessibility and {"lat", "lng"}.issubset(analysis_results.columns):
                        with st.spinner("접근성(도보 시간) 데이터를 계산 중입니다..."):
                            # 사전 계산된 접근성 CSV 로드 시도
                            if "precomputed_transit" not in st.session_state:
                                transit_csv = Path("서울시_상권_카페빵_표본_with_transit.csv")
                                if transit_csv.exists():
                                    try:
                                        precomputed_df = pd.read_csv(transit_csv)
                                        required_cols = {"상호명", "행정동명", "nearest_station", "transit_type", "walk_time_minutes", "walk_distance_m", "straight_distance_m"}
                                        if required_cols.issubset(precomputed_df.columns):
                                            precomputed_df = precomputed_df.copy()
                                            st.session_state["precomputed_transit"] = precomputed_df
                                        else:
                                            st.warning("사전 계산된 접근성 CSV에 필요한 컬럼이 모두 존재하지 않습니다. API 호출을 진행합니다.")
                                    except Exception as transit_load_err:
                                        st.warning(f"사전 계산된 접근성 CSV 로딩 실패: {transit_load_err}")
                                else:
                                    st.info("사전 계산된 접근성 CSV가 없어 API 기반으로 접근성을 계산합니다.")

                            precomputed_df = st.session_state.get("precomputed_transit")

                            if precomputed_df is not None and not precomputed_df.empty:
                                analysis_results = analysis_results.copy()

                                # 행정동명 + 상호명으로 매칭
                                merge_keys = ["상호명", "행정동명"]
                                if all(col in analysis_results.columns for col in merge_keys) and all(col in precomputed_df.columns for col in merge_keys):
                                    merged = analysis_results.merge(
                                        precomputed_df[
                                            [
                                                "상호명",
                                                "행정동명",
                                                "nearest_station",
                                                "transit_type",
                                                "walk_time_minutes",
                                                "walk_distance_m",
                                                "straight_distance_m",
                                            ]
                                        ],
                                        how="left",
                                        on=merge_keys,
                                        suffixes=("", "_precomputed"),
                                    )

                                    for col in ["walk_time_minutes", "walk_distance_m", "straight_distance_m", "nearest_station", "transit_type"]:
                                        pre_col = f"{col}_precomputed"
                                        if pre_col in merged.columns:
                                            merged[col] = merged[col].combine_first(merged[pre_col])
                                            merged = merged.drop(columns=[pre_col])

                                    analysis_results = merged
                                else:
                                    st.warning("매칭을 위한 '상호명' 또는 '행정동명' 컬럼이 없어 사전 계산된 접근성 데이터를 사용할 수 없습니다.")

                            missing_after_merge = analysis_results[
                                ["walk_time_minutes", "walk_distance_m", "straight_distance_m"]
                            ].isna().any(axis=1)

                            if missing_after_merge.any():
                                st.info("사전 계산된 데이터가 없는 카페에 대해 API 호출로 접근성을 계산합니다.")
                                walk_times, walk_path_distances, walk_straight_distances, stations, transit_types = [], [], [], [], []
                                for _, row in analysis_results.iterrows():
                                    lat_val, lng_val = row.get("lat"), row.get("lng")
                                    if pd.isna(lat_val) or pd.isna(lng_val):
                                        walk_times.append(np.nan)
                                        walk_path_distances.append(np.nan)
                                        walk_straight_distances.append(np.nan)
                                        stations.append(None)
                                        transit_types.append(None)
                                        continue

                                    if (
                                        row.get("walk_time_minutes") is not None
                                        and not pd.isna(row.get("walk_time_minutes"))
                                        and row.get("walk_distance_m") is not None
                                        and not pd.isna(row.get("walk_distance_m"))
                                    ):
                                        walk_times.append(row.get("walk_time_minutes"))
                                        walk_path_distances.append(row.get("walk_distance_m"))
                                        walk_straight_distances.append(row.get("straight_distance_m"))
                                        stations.append(row.get("nearest_station"))
                                        transit_types.append(row.get("transit_type"))
                                        continue

                                    try:
                                        (
                                            walk_time,
                                            station_name,
                                            transit_type,
                                            walk_distance,
                                            straight_distance,
                                        ) = calculate_transit_accessibility(lat_val, lng_val)
                                    except Exception:
                                        walk_time, station_name, transit_type, walk_distance, straight_distance = (
                                            np.nan,
                                            None,
                                            None,
                                            np.nan,
                                            np.nan,
                                        )
                                    walk_times.append(walk_time)
                                    walk_path_distances.append(walk_distance)
                                    walk_straight_distances.append(straight_distance)
                                    stations.append(station_name)
                                    transit_types.append(transit_type)
                                analysis_results["walk_time_minutes"] = walk_times
                                analysis_results["walk_distance_m"] = walk_path_distances
                                analysis_results["straight_distance_m"] = walk_straight_distances
                                analysis_results["nearest_station"] = stations
                                analysis_results["transit_type"] = transit_types
                                analysis_results = analysis_results.drop(columns=[col for col in ["lat_key", "lng_key"] if col in analysis_results.columns])

                    if run_analysis or final_results_state is None:
                        st.session_state["tab4_analysis_results_final"] = analysis_results.copy()

                    for col in ["walk_time_minutes", "walk_distance_m", "straight_distance_m"]:
                        if col in analysis_results.columns:
                            analysis_results[col] = pd.to_numeric(analysis_results[col], errors="coerce")
                    access_info_cols = [
                        col
                        for col in ["walk_time_minutes", "walk_distance_m", "straight_distance_m", "nearest_station", "transit_type", accessibility_col]
                        if col and col in analysis_results.columns
                    ]
                    if access_info_cols:
                        st.markdown("#### 🚇 주변 대중교통 접근성 요약")
                        st.caption("Google Places Nearby + Distance Matrix API 기반으로 최근접 역/정류장과 직선거리를 가져와서 예상 도보 시간을 추정합니다.")
                        access_display_cols = [c for c in group_cols if c != "place_id"] + access_info_cols
                        access_height = min(400, max(240, 38 * len(analysis_results)))
                        st.dataframe(
                            analysis_results[access_display_cols],
                            use_container_width=True,
                            hide_index=True,
                            height=access_height,
                        )

                    download_df = analysis_results.copy()
                    download_df["scores"] = download_df["scores"].apply(lambda s: json.dumps(s, ensure_ascii=False))
                    if "corrections" not in download_df.columns:
                        download_df["corrections"] = [[] for _ in range(len(download_df))]
                    download_df["corrections"] = download_df["corrections"].apply(
                        lambda logs: json.dumps(logs, ensure_ascii=False) if isinstance(logs, list) else "[]"
                    )
                    download_df["원본파일"] = selected_path.name
                    st.download_button(
                        "📥 감성 분석 결과 CSV 다운로드",
                        data=download_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="google_review_sentiment_summary.csv",
                        mime="text/csv",
                        key="tab4_download_summary",
                    )

                    # 지도 시각화
                    valid_map_df = (
                        analysis_results.dropna(subset=["lat", "lng"])
                        if {"lat", "lng"}.issubset(analysis_results.columns)
                        else pd.DataFrame()
                    )
                    if not valid_map_df.empty:
                        st.markdown("#### 🗺️ 서울 지도 시각화")

                        metric_options = ["평균장소성점수", "평균감성점수"] + score_columns
                        selected_metric = st.selectbox(
                            "지도에 표시할 지표",
                            options=metric_options,
                            index=0,
                            key="tab4_map_metric",
                        )
                        st.caption("HeatMap은 선택한 지표 값을 이용해 가중치를 계산합니다.")

                        try:
                            placeness_map = tab4_analysis.build_placeness_map(
                                valid_map_df,
                                value_col=selected_metric,
                            )
                            st_folium(placeness_map, width=None, height=500, key="tab4_map")
                        except Exception as map_err:
                            st.warning(f"지도 시각화에 실패했습니다: {map_err}")
                    else:
                        st.info("지도 표시를 위한 좌표 정보가 부족합니다.")

                    # 접근성 비교 (도보 시간 vs 접근성 점수)
                    if accessibility_col and accessibility_col in analysis_results.columns:
                        analysis_results[accessibility_col] = pd.to_numeric(
                            analysis_results[accessibility_col], errors="coerce"
                        )

                    if accessibility_col and {"walk_time_minutes", accessibility_col}.issubset(analysis_results.columns):
                        st.markdown("#### 🚇 접근성 실험")
                        valid_access_df = analysis_results.dropna(subset=["walk_time_minutes", accessibility_col])
                        if len(valid_access_df) >= 2:
                            try:
                                corr_access, p_access = stats.pearsonr(
                                    valid_access_df["walk_time_minutes"].astype(float),
                                    valid_access_df[accessibility_col].astype(float),
                                )
                                p_access_text = f"{p_access:.4f}" if p_access >= 1e-4 else f"{p_access:.2e}"
                                relation_access = "양의" if corr_access > 0 else "음의"
                                st.write(
                                    f"접근성 상관계수: **{corr_access:.3f}** (p-value={p_access_text}) "
                                    f"→ 통계적으로 유의미한 {relation_access} 상관관계가 확인됩니다 (α=0.05)."
                                    if p_access < 0.05
                                    else (
                                        f"접근성 상관계수: **{corr_access:.3f}** (p-value={p_access_text}) "
                                        "→ 통계적으로 유의미한 상관관계로 보기 어렵습니다 (α=0.05)."
                                    )
                                )

                                access_fig = px.scatter(
                                    valid_access_df,
                                    x="walk_time_minutes",
                                    y=accessibility_col,
                                    trendline="ols",
                                    labels={"walk_time_minutes": "도보 시간(분)", accessibility_col: "접근성 요인 점수"},
                                    title="도보 시간 vs 접근성 요인 점수",
                                )
                                st.plotly_chart(access_fig, use_container_width=True, key="tab4_access_corr")
                                slope_access, intercept_access, _, _, _ = stats.linregress(
                                    valid_access_df["walk_time_minutes"].astype(float),
                                    valid_access_df[accessibility_col].astype(float),
                                )
                                st.caption(f"회귀선: y = {slope_access:.3f}x + {intercept_access:.3f}")
                            except Exception as access_err:
                                st.warning(f"도보 시간 상관 분석 중 오류가 발생했습니다: {access_err}")
                        else:
                            st.info("도보 시간 상관 분석을 위해서는 최소 2개 이상의 유효한 데이터가 필요합니다.")

                        if "straight_distance_m" in valid_access_df.columns and valid_access_df["straight_distance_m"].notna().any():
                            st.markdown("#### 🚇 직선거리 기반 접근성 실험")
                            valid_distance_df = valid_access_df.dropna(subset=["straight_distance_m"])
                            if len(valid_distance_df) >= 2:
                                try:
                                    corr_dist, p_dist = stats.pearsonr(
                                        valid_distance_df["straight_distance_m"].astype(float),
                                        valid_distance_df[accessibility_col].astype(float),
                                    )
                                    p_dist_text = f"{p_dist:.4f}" if p_dist >= 1e-4 else f"{p_dist:.2e}"
                                    relation_dist = "양의" if corr_dist > 0 else "음의"
                                    st.write(
                                        f"직선거리 상관계수: **{corr_dist:.3f}** (p-value={p_dist_text}) "
                                        f"→ 통계적으로 유의미한 {relation_dist} 상관관계가 확인됩니다 (α=0.05)."
                                        if p_dist < 0.05
                                        else (
                                            f"직선거리 상관계수: **{corr_dist:.3f}** (p-value={p_dist_text}) "
                                            "→ 통계적으로 유의미한 상관관계로 보기 어렵습니다 (α=0.05)."
                                        )
                                    )
                                    distance_fig = px.scatter(
                                        valid_distance_df,
                                        x="straight_distance_m",
                                        y=accessibility_col,
                                        trendline="ols",
                                        labels={"straight_distance_m": "직선 거리(m)", accessibility_col: "접근성 요인 점수"},
                                        title="직선 거리 vs 접근성 요인 점수",
                                    )
                                    st.plotly_chart(distance_fig, use_container_width=True, key="tab4_access_dist_corr")
                                    slope_dist, intercept_dist, _, _, _ = stats.linregress(
                                        valid_distance_df["straight_distance_m"].astype(float),
                                        valid_distance_df[accessibility_col].astype(float),
                                    )
                                    st.caption(f"회귀선: y = {slope_dist:.3f}x + {intercept_dist:.3f}")
                                except Exception as dist_err:
                                    st.warning(f"직선거리 상관 분석 중 오류가 발생했습니다: {dist_err}")
                            else:
                                st.info("직선거리 상관 분석을 위해서는 최소 2개 이상의 유효한 데이터가 필요합니다.")
                        else:
                            st.info("접근성 비교를 위한 데이터가 충분하지 않습니다.")
                    else:
                        st.info("상관 분석을 수행하기 위한 데이터가 충분하지 않습니다.")
