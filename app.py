import streamlit as st
import googlemaps
from openai import OpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from dotenv import load_dotenv
import json
import re
from wordcloud import WordCloud
import numpy as np
import warnings # 경고 메시지 처리를 위해 추가

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
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    weights = np.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]

    def predict_score(sentences: List[str]):
        if not sentences:
            return []
        results = pipe(sentences)
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
            # 감성 보정: 하한 0.3 기준으로 추가 완화
            sent_adj = np.clip((raw_sent - 0.3) / 0.7, 0, 1)
            sims = sim_mat[i]
            for j, sim in enumerate(sims):
                # 유사도 보정: 0.3 기준으로 추가 완화 (더 많은 문장 포함)
                sim_adj = np.clip((float(sim) - 0.3) / 0.5, 0, 1)
                if sim_adj > 0:
                    f_name = subcat_list[j]
                    combined = ALPHA * sim_adj + BETA * sent_adj
                    # 시그모이드: 중심 0.4, 기울기 2.2로 상한 확장
                    score_scaled = 1 / (1 + np.exp(-2.2 * (combined - 0.4)))
                    factor_sentiments[f_name].append(float(score_scaled))

        # 3) 키워드 기반 부스팅 (임베딩 한계 보완)
        keyword_boosts = {
            "고유성": ["독특", "유니크", "차별", "컨셉", "테마", "특색", "개성", "특별한", "아이덴티티"],
            "문화적 맥락": ["전통", "역사", "년", "오래", "옛", "고풍", "문화", "배경", "스토리"],
            "기억/경험": ["추억", "감동", "인상", "특별", "잊을 수", "기억", "회상"],
            "심미성": ["예쁘", "아름", "멋지", "세련", "야경", "뷰", "인테리어", "디자인", "조명", "아늑"],
            "감각적 경험": ["음악", "향", "냄새", "질감", "맛", "오감", "감각"],
            "쾌적성": ["청결", "깨끗", "밝", "통풍", "화장실", "위생", "정돈"],
            "접근성": ["가깝", "접근", "역", "정류장", "도보", "분 거리", "편리"],
            "활동성": ["대화", "업무", "작업", "회의", "공부", "활동"],
            "사회성": ["친절", "서비스", "교류", "소통", "친근"],
            "형태성": ["넓", "공간", "구조", "배치", "개방", "동선"],
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
                    # 0.30~1.0 범위로 min-max 정규화
                    normed = 0.30 + 0.70 * ((raw - vmin) / (vmax - vmin + 1e-8))
                    scores[main_cat][subcat] = float(np.clip(normed, 0.30, 1.0))
                elif vals:
                    scores[main_cat][subcat] = float(np.clip(vals[0], 0.30, 1.0))
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
3. delta는 -0.3 ~ +0.3 범위
4. 근거는 한 문장으로만 작성

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
                            new_val = np.clip(old_val + delta, 0.30, 1.0)
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
# 6. Streamlit UI
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

st.markdown("분석할 공간의 위치와 감성/기능적 특성을 입력하십시오. "
             "<span class='example-text'>(예: 신촌 조용한 카페, 종로구 전통적인 음식점, 마포구 산책로 공원)</span>", 
             unsafe_allow_html=True)
query = st.text_input("", placeholder="예: 신촌 조용한 카페")

if st.button("장소성 정량 분석 시작"):
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
