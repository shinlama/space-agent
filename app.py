import streamlit as st
import googlemaps
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from dotenv import load_dotenv
import json
import re

# 환경변수 로드
load_dotenv()

st.set_page_config(page_title="Seoul Place Recommendation", page_icon="🗺️", layout="centered")

# 세션 상태 변수 초기화
if "history" not in st.session_state:
    st.session_state.history = []

# --- [수정] API 키 환경변수 이름 표준화 ---
gmaps_key = os.getenv("Maps_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

# 세션 상태 초기화
if "gmaps_key" not in st.session_state:
    st.session_state.gmaps_key = gmaps_key or ""
if "openai_key" not in st.session_state:
    st.session_state.openai_key = openai_key or ""

# API 키가 없으면 입력 요청
if not st.session_state.gmaps_key or not st.session_state.openai_key:
    st.title("🗺️ Seoul Place Recommendation and Spatial Evaluation System")
    
    st.info("""
    🔑 **API 키 설정 방법**
    
    1. 프로젝트 루트에 `.env` 파일을 생성하세요
    2. 다음 내용을 추가하세요:
    ```
    # [수정] 환경변수 이름을 표준화했습니다.
    Maps_API_KEY=your_Maps_api_key_here
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    3. 앱을 다시 시작하세요
    """)
    
    if not gmaps_key or not openai_key:
        st.markdown("---")
        st.markdown("**또는 수동으로 입력:**")
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

# 클라이언트 초기화
try:
    gmaps = googlemaps.Client(key=st.session_state.gmaps_key)
    client = OpenAI(api_key=st.session_state.openai_key)
except Exception as e:
    st.error(f"API 클라이언트 초기화 중 오류 발생: {e}")
    st.stop()

class AgentState(BaseModel):
    query: str
    places: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    answer: Optional[str] = ""

def search_places(state: AgentState):
    """Google Maps API를 사용하여 장소를 검색하는 함수"""
    # None 가드
    if state.places is None:
        state.places = []
    try:
        res = gmaps.places(query=state.query, language="ko", location="37.5665,126.9780", radius=5000)
        state.places = res.get('results', [])[:5]
    except Exception as e:
        st.error(f"Google Maps 장소 검색 중 오류 발생: {e}")
        st.error("Google Maps API 키가 유효한지, 또는 API가 활성화되어 있는지 확인하세요.")
        state.places = []
    return state.dict()

def analyze_reviews(state: AgentState):
    """장소 리뷰를 분석하고 새로운 장소성 지표로 정량 평가하는 함수"""
    if state.places is None:
        state.places = []
    place_infos = []
    new_score_structure = {
        "물리적 환경": {"심미성": None, "형태성": None, "감각적 경험": None, "고유성": None},
        "사회적 상호작용": {"활동성": None, "사회성": None, "참여성": None},
        "개인적/문화적 의미": {"기억/경험": None, "정체성/애착": None, "문화적 맥락": None}
    }

    for place in state.places:
        place_id = place.get("place_id")
        if not place_id:
            continue
            
        details = gmaps.place(place_id=place_id, language="ko").get('result', {})
        reviews = details.get('reviews', [])[:5]
        review_text = "\n".join([review['text'] for review in reviews if review.get('text')])

        summary = "리뷰 정보가 부족합니다."
        scores = json.loads(json.dumps(new_score_structure))

        if review_text.strip():
            summary_prompt = f"다음 리뷰들을 종합하여 장소의 전반적인 분위기, 실내 공간 디자인 특성, 방문객들의 주요 경험, 긍정적 및 부정적 피드백을 중심으로 요약해줘:\n\n{review_text}\n\n요약:"
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o", messages=[{"role": "user", "content": summary_prompt}], max_tokens=400
                )
                summary = completion.choices[0].message.content.strip() or "리뷰 내용이 충분하지 않아 요약이 어렵습니다."
            except Exception as e:
                summary = f"요약 생성 중 오류 발생: {e}"

            scoring_prompt = f"""다음 리뷰를 '장소성' 관점에서 분석하여 각 세부 지표를 0.0부터 1.0 사이의 숫자로 평가하세요. 판단 근거가 부족하면 0.5로 평가하고, 평가는 반드시 아래에 제시된 JSON 구조와 키를 그대로 사용해야 합니다. 다른 텍스트는 절대 포함하지 마세요.
            ### 평가 지표 정의:
            **1. 물리적 환경 (Physical Setting): 공간의 물리적 디자인과 특성**
            - **심미성**: 인테리어, 조명, 가구 등 시각적인 아름다움과 분위기.
            - **형태성**: 공간의 구조, 개방감, 좌석 배치 등 공간의 물리적 구성.
            - **감각적 경험**: 배경 음악, 향기, 식기의 질감 등 오감을 자극하는 요소.
            - **고유성**: 다른 곳과 차별화되는 독특한 디자인, 컨셉, 상징적 요소.
            **2. 사회적 상호작용 (Social Interaction): 공간 내에서의 활동과 관계**
            - **활동성**: 대화, 작업, 휴식 등 다양한 활동이 이루어지는 정도.
            - **사회성**: 다른 사람들과 자연스럽게 어울리거나 교류할 수 있는 분위기.
            - **참여성**: 이벤트, 오픈 키친, 클래스 등 고객이 참여할 수 있는 요소.
            **3. 개인적/문화적 의미 (Personal/Cultural Meaning): 공간과 맺는 정서적, 문화적 관계**
            - **기억/경험**: 특별한 추억이나 의미 있는 경험을 제공하는 정도.
            - **정체성/애착**: 방문객이 자신의 취향이나 정체성과 연결하며 애착을 느끼게 하는 정도.
            - **문화적 맥락**: 지역의 역사, 문화적 스토리를 반영하고 있는 정도.
            ### 리뷰:
            {review_text}
            ### 응답 형식 (오직 JSON 형식으로만 응답):
            {{"물리적 환경": {{"심미성": 0.8, "형태성": 0.7, "감각적 경험": 0.6, "고유성": 0.9}},"사회적 상호작용": {{"활동성": 0.7, "사회성": 0.6, "참여성": 0.4}},"개인적/문화적 의미": {{"기억/경험": 0.8, "정체성/애착": 0.9, "문화적 맥락": 0.5}}}}"""

            try:
                score_response = client.chat.completions.create(
                    model="gpt-4o", messages=[{"role": "user", "content": scoring_prompt}], response_format={"type": "json_object"}
                )
                parsed_scores = json.loads(score_response.choices[0].message.content)
                for main_key, sub_dict in new_score_structure.items():
                    if main_key in parsed_scores and isinstance(parsed_scores[main_key], dict):
                        for sub_key in sub_dict:
                            value = parsed_scores[main_key].get(sub_key)
                            if isinstance(value, (int, float)):
                                scores[main_key][sub_key] = float(value)
            except Exception as e:
                print(f"JSON 파싱 또는 API 오류: {e}")

        place_infos.append({
            'name': place.get('name', '이름 없음'), 'summary': summary,
            'address': place.get('formatted_address', place.get('vicinity', '주소 정보 없음')),
            'scores': scores, 'geometry': place.get('geometry', {}), 'place_id': place.get('place_id', '')
        })

    state.places = place_infos
    return state.dict()

# LangGraph 구성
graph = StateGraph(AgentState)
graph.add_node("search_places", search_places)
graph.add_node("analyze_reviews", analyze_reviews)
graph.set_entry_point("search_places")
graph.add_edge("search_places", "analyze_reviews")
graph.add_edge("analyze_reviews", END)
agent = graph.compile()

# Streamlit UI
st.title("🗺️ 장소성 기반 서울시 공간 정량 평가 시스템 ")
query = st.text_input("🔍 평가하고 싶은 장소나 테마를 입력하세요", placeholder="예: 신촌 카페, 종로구 맛집")

if st.button("장소성 분석하기"):
    if not query.strip():
        st.warning("장소나 테마를 입력해주세요.")
    else:
        with st.spinner("리뷰를 분석하여 장소성을 평가하는 중..."):
            result = agent.invoke({"query": query, "places": [], "answer": ""})
            places = result.get('places', [])
            st.session_state.history.append((query, places))
            st.rerun()

# 결과 출력
if st.session_state.history:
    latest_query, latest_places = st.session_state.history[-1]
    st.markdown(f"---")
    st.markdown(f"### 🔍 '{latest_query}'에 대한 분석 결과")

    for i, place in enumerate(latest_places):
        with st.container(border=True):
            st.subheader(place.get('name', '이름 정보 없음'))
            st.markdown(f"**📍 주소:** {place.get('address', '주소 정보 없음')}")
            st.markdown(f"**📝 리뷰 요약:** {place.get('summary', '요약 정보 없음')}")

            scores = place.get('scores')
            if scores:
                st.markdown(f"**📊 장소성 종합 평가**")
                
                # Sunburst 차트 데이터 생성
                labels = []
                parents = []
                values = []
                colors = []
                
                # 부드러운 파스텔톤 색상 맵
                color_map = {
                    "물리적 환경": "rgb(173, 216, 230)",      # 연한 파란색 (Light Blue)
                    "사회적 상호작용": "rgb(152, 251, 152)",   # 연한 연두색 (Light Lime Green)
                    "개인적/문화적 의미": "rgb(255, 182, 193)" # 연한 분홍색 (Light Pink)
                }
                
                # 루트 노드 추가 (전체 점수의 평균으로 설정)
                total_score = 0
                score_count = 0
                for main_cat, sub_scores in scores.items():
                    for score in sub_scores.values():
                        if score is not None:
                            total_score += score
                            score_count += 1
                
                root_value = total_score / score_count if score_count > 0 else 0.5
                
                labels.append(place['name'])
                parents.append("")
                values.append(root_value)
                colors.append("#FFFFFF")
                
                # 대분류와 세부 분류 추가
                for main_cat, sub_scores in scores.items():
                    # 대분류 평균 점수 계산
                    main_scores = [s for s in sub_scores.values() if s is not None]
                    main_avg = sum(main_scores) / len(main_scores) if main_scores else 0
                    
                    # 대분류 추가
                    labels.append(main_cat)
                    parents.append(place['name'])
                    values.append(main_avg)
                    colors.append(color_map.get(main_cat, "#CCCCCC"))
                    
                    # 세부 분류 추가
                    for sub_cat, score in sub_scores.items():
                        if score is not None:
                            labels.append(sub_cat)
                            parents.append(main_cat)
                            values.append(float(score))
                            colors.append(color_map.get(main_cat, "#CCCCCC"))
                

                
                # Sunburst 차트 생성
                try:
                    fig_sunburst = go.Figure(go.Sunburst(
                        labels=labels,
                        parents=parents,
                        values=values,
                        branchvalues="remainder",  # total 대신 remainder 사용
                        marker=dict(colors=colors),
                        hovertemplate='<b>%{label}</b><br>점수: %{value:.2f}',
                        maxdepth=2,
                        insidetextorientation='radial'
                    ))
                    
                    fig_sunburst.update_layout(
                        margin=dict(t=20, l=10, r=10, b=10),
                        height=400,
                        title_text=f"{place['name']} 장소성 종합 평가",
                        font=dict(size=12)
                    )
                    
                    st.plotly_chart(fig_sunburst, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Sunburst 차트 생성 중 오류: {e}")
                    
                    # 대안: Treemap 차트 시도
                    try:
                        st.info("Sunburst 차트 대신 Treemap 차트를 표시합니다.")
                        fig_treemap = go.Figure(go.Treemap(
                            labels=labels,
                            parents=parents,
                            values=values,
                            marker=dict(colors=colors),
                            hovertemplate='<b>%{label}</b><br>점수: %{value:.2f}'
                        ))
                        fig_treemap.update_layout(
                            margin=dict(t=20, l=10, r=10, b=10),
                            height=400,
                            title_text=f"{place['name']} 장소성 종합 평가"
                        )
                        st.plotly_chart(fig_treemap, use_container_width=True)
                    except Exception as e2:
                        st.error(f"Treemap 차트 생성 중 오류: {e2}")

                st.markdown(f"**📊 장소성 대분류 평가**")
                main_scores = {main: round(sum(filter(None, sub.values())) / len(sub), 2) for main, sub in scores.items() if any(s is not None for s in sub.values())}
                if main_scores:
                    df = pd.DataFrame(list(main_scores.items()), columns=['분류', '점수'])
                    fig_bar = px.bar(df, x='분류', y='점수', color='분류', color_discrete_map=color_map, range_y=[0, 1], text_auto='.2f')
                    fig_bar.update_layout(showlegend=False, title_text="")
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("정량 평가 결과가 없습니다.")

            if place.get('geometry') and place['geometry'].get('location'):
                lat, lng = place['geometry']['location']['lat'], place['geometry']['location']['lng']
                
                # 세션 상태 초기화
                map_key = f"map_{i}_{place['place_id']}"
                streetview_key = f"street_{i}_{place['place_id']}"
                
                if map_key not in st.session_state:
                    st.session_state[map_key] = False
                if streetview_key not in st.session_state:
                    st.session_state[streetview_key] = False
                
                col1, col2 = st.columns(2)
                
                # 버튼 클릭 처리
                if col1.button("🗺️ 지도 보기", key=f"btn_{map_key}"):
                    st.session_state[map_key] = not st.session_state[map_key]
                    st.rerun()
                
                if col2.button("🚗 로드뷰 보기", key=f"btn_{streetview_key}"):
                    st.session_state[streetview_key] = not st.session_state[streetview_key]
                    st.rerun()
                
                # 지도와 로드뷰를 세로로 쌓아서 표시
                if st.session_state[map_key] or st.session_state[streetview_key]:
                    st.markdown("**📍 위치 정보**")
                    
                    # 지도 표시
                    if st.session_state[map_key]:
                        st.markdown("**🗺️ 지도**")
                        map_url = f"https://www.google.com/maps/embed/v1/place?key={st.session_state.gmaps_key}&q={lat},{lng}&zoom=16"
                        st.components.v1.iframe(map_url, height=400, width=700)
                    
                    # 로드뷰 표시
                    if st.session_state[streetview_key]:
                        st.markdown("**🚗 로드뷰**")
                        streetview_url = f"https://www.google.com/maps/embed/v1/streetview?key={st.session_state.gmaps_key}&location={lat},{lng}&heading=210&pitch=10"
                        st.components.v1.iframe(streetview_url, height=400, width=700)
            else:
                st.info("📍 위치 정보가 없어 지도를 표시할 수 없습니다.")