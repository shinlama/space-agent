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

st.set_page_config(page_title="Seoul Place Recommendation", page_icon="🗺️", layout="centered")

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
    """Sentence-BERT + Tabularis 기반 감성 분석으로 장소성 정량 평가"""
    if state.places is None:
        state.places = []

    place_infos = []
    
    # 유사도 임계값 설정 (완화)
    SIMILARITY_THRESHOLD = 0.35
    
    for place in state.places:
        place_id = place.get("place_id")
        if not place_id:
            continue

        details = gmaps.place(place_id=place_id, language="ko").get('result', {})
        reviews = details.get('reviews', [])[:10] # 최대 10개 리뷰
        review_text = "\n".join([review['text'] for review in reviews if review.get('text')])

        scores = json.loads(json.dumps(new_score_structure_template))
        
        # LLM 호출을 위한 키워드/요약 변수
        summary = "분석 중..."
        positive_keywords = [] # LLM이 추출한 단어
        negative_keywords = [] # LLM이 추출한 단어
        
        # 장소성 정량 평가 (NLP 기반)
        if review_text.strip():
            # OpenAI 기반 의미 단위 분리 (문서 전체를 입력) - 캐시 사용
            review_units = cached_semantic_split(review_text)
            if not review_units:
                # 추가 방어: 폴백 분할 재실행 (캐시 사용)
                review_units = cached_semantic_split(" ".join([r.get('text','') for r in reviews]))
            
            # 1. LLM을 사용하여 요약 및 워드클라우드 키워드 추출 (OpenAI API 사용)
            try:
                cached = cached_unified_summary(review_text)
                positive_keywords = cached["positive_keywords"]
                negative_keywords = cached["negative_keywords"]
                summary = cached["summary"]
            except Exception as e:
                print(f"LLM 요약/키워드 추출 중 오류 발생: {e}")
                summary = "LLM 요약 실패. NLP 분석만 진행됨."
                positive_keywords, negative_keywords = [], []


            # 2. 장소성 세부 항목별 점수 산정 (SBERT + SA 기반)
            
            # 장소성 세부 항목 이름: [관련 의미단위의 감성 점수 리스트]
            factor_sentiment_map = {f_name: [] for f_name in category_embeddings.keys()}
            
            # 의미 단위별 감성 점수 계산 (0~1 연속값)
            sentiment_scores = sentiment_model(review_units)

            # 감성 분석 결과와 의미 단위를 매핑 (배치 임베딩 + 벡터화 유사도)
            # 과도한 길이 방지: 의미 단위 상한
            MAX_UNITS = 60
            if len(review_units) > MAX_UNITS:
                review_units = review_units[:MAX_UNITS]
                sentiment_scores = sentiment_scores[:MAX_UNITS]

            try:
                unit_embs = embed_model.encode(review_units, normalize_embeddings=True)
                # 요인 임베딩 행렬 구성 (순서 고정)
                subcat_list = list(category_embeddings.keys())
                factor_mat = np.stack([category_embeddings[s] for s in subcat_list], axis=0)
                # 유사도 행렬: [num_units, num_subcats]
                sim_mat = np.matmul(unit_embs, factor_mat.T)

                for i, unit in enumerate(review_units):
                    s_score = float(sentiment_scores[i])
                    sims = sim_mat[i]
                    for j, sim in enumerate(sims):
                        if sim > SIMILARITY_THRESHOLD:
                            f_name = subcat_list[j]
                            factor_sentiment_map[f_name].append(s_score)
            except Exception as e:
                print(f"배치 임베딩/유사도 계산 오류: {e}")

            # 3. 항목별 최종 점수 계산 (0~1 스케일 그대로 평균)
            for main_cat, subcats in scores.items():
                for subcat in subcats.keys():
                    vals = factor_sentiment_map.get(subcat, [])
                    if vals:
                        scores[main_cat][subcat] = float(np.mean(vals))
                    else:
                        # 관련 문장이 없는 경우 중립 점수 (0.5) 부여
                        scores[main_cat][subcat] = 0.5
        
        # 최종 정보 리스트에 추가
        place_infos.append({
            'name': place.get('name', '이름 없음'), 
            'summary': summary,
            'address': place.get('formatted_address', place.get('vicinity', '주소 정보 없음')),
            'scores': scores, 
            'geometry': place.get('geometry', {}), 
            'place_id': place.get('place_id', ''),
            'positive_keywords': positive_keywords, 
            'negative_keywords': negative_keywords, 
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

st.markdown("분석할 공간의 위치와 감성/기능적 특성을 입력하십시오.  \n"
             "<span style='color:gray'>(예: 신촌 조용한 카페, 종로구 전통적인 음식점, 마포구 산책로 공원)</span>", 
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
            st.markdown(f"**📝 리뷰 요약 (LLM 생성):** {place.get('summary', '요약 정보 없음')}")

            scores = place.get('scores')
            if scores:
                st.markdown(f"**📊 장소성 종합 평가 (NLP 기반)**")

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

                st.markdown(f"**📊 장소성 대분류 평가**")
                main_scores = {main: round(sum(filter(None, sub.values())) / len(sub), 2) for main, sub in scores.items() if any(s is not None for s in sub.values())}
                if main_scores:
                    df = pd.DataFrame(list(main_scores.items()), columns=['분류', '점수'])
                    fig_bar = px.bar(df, x='분류', y='점수', color='분류', color_discrete_map=color_map, range_y=[0, 1], text_auto='.2f')
                    fig_bar.update_layout(showlegend=False, title_text="")
                    st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{i}_{place.get('place_id','')}")
                else:
                    st.warning("정량 평가 결과가 없습니다.")

                # 워드 클라우드 시각화 (LLM 추출 키워드 사용)
                if place.get('positive_keywords') or place.get('negative_keywords'):
                    st.markdown("---")
                    st.markdown("**📝 리뷰 키워드 분석 (LLM 추출)**")
                    
                    col_pos, col_neg = st.columns(2)
                    
                    # 긍정 워드 클라우드
                    if place.get('positive_keywords'):
                        with col_pos:
                            st.markdown("#### ✅ 긍정 키워드")
                            text = " ".join(place['positive_keywords'])
                            if text:
                                img = generate_wordcloud(text, font_path, colormap="Greens")
                                if img is not None:
                                    st.image(img, use_container_width=True)
                            else:
                                st.info("긍정 키워드 없음")
                    
                    # 부정 워드 클라우드
                    if place.get('negative_keywords'):
                        with col_neg:
                            st.markdown("#### ❌ 부정 키워드")
                            text = " ".join(place['negative_keywords'])
                            if text:
                                img = generate_wordcloud(text, font_path, colormap="Reds")
                                if img is not None:
                                    st.image(img, use_container_width=True)
                            else:
                                st.info("부정 키워드 없음")
                
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
                        
                        if st.session_state[map_key]:
                            st.markdown("**🗺️ 지도**")
                            # Google Maps Embed API
                            map_url = f"https://www.google.com/maps/embed/v1/place?key={st.session_state.gmaps_key}&q={lat},{lng}"
                            st.components.v1.iframe(map_url, height=400, width=700)
                        
                        if st.session_state[streetview_key]:
                            st.markdown("**🚗 로드뷰**")
                            # Google Maps Street View Embed API
                            streetview_url = f"https://www.google.com/maps/embed/v1/streetview?key={st.session_state.gmaps_key}&location={lat},{lng}"
                            st.components.v1.iframe(streetview_url, height=400, width=700)
                else:
                    st.info("📍 위치 정보가 없어 지도를 표시할 수 없습니다.")
