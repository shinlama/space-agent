import streamlit as st
import googlemaps
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any
import plotly.express as px
import pandas as pd
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

st.set_page_config(page_title="Seoul Place Recommendation", page_icon="🗺️", layout="centered")

# 세션 상태 변수 초기화
if "history" not in st.session_state:
    st.session_state.history = []

# 환경변수에서 API 키 가져오기
gmaps_key = os.getenv("GOOGLE_MAPS_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

# 세션 상태 초기화
if "gmaps_key" not in st.session_state:
    st.session_state.gmaps_key = gmaps_key or ""
if "openai_key" not in st.session_state:
    st.session_state.openai_key = openai_key or ""

# API 키가 없으면 입력 요청
if not st.session_state.gmaps_key or not st.session_state.openai_key:
    st.title("🗺️ Seoul Place Recommendation and Spatial Evaluation System")
    
    # 환경변수 설정 안내
    st.info("""
    🔑 **API 키 설정 방법**
    
    1. 프로젝트 루트에 `.env` 파일을 생성하세요
    2. 다음 내용을 추가하세요:
    ```
    GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    3. 앱을 다시 시작하세요
    """)
    
    # 수동 입력 옵션 (환경변수가 없을 때만)
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
gmaps = googlemaps.Client(key=st.session_state.gmaps_key)
client = OpenAI(api_key=st.session_state.openai_key)

class AgentState(BaseModel):
    query: str
    places: List[Dict[str, Any]] = Field(default_factory=list)
    answer: str = ""

# 장소 검색

def search_places(state: AgentState):
    res = gmaps.places(query=state.query, language="ko", location="37.5665,126.9780", radius=5000)
    state.places = res.get('results', [])[:5]
    return state.dict()

# 리뷰 분석 및 정량 평가 노드

def analyze_reviews(state: AgentState):
    import json
    place_infos = []

    for place in state.places:
        place_id = place["place_id"]
        details = gmaps.place(place_id=place_id, language="ko")

        reviews = details.get('result', {}).get('reviews', [])[:5]
        review_text = "\n".join([review['text'] for review in reviews])

        # 기본 값
        summary = "리뷰 정보가 부족합니다."
        scores = {k: None for k in ["심미성", "형태성", "활동성", "접근성", "청결도"]}

        # 리뷰가 있을 경우 GPT로 요약 & 정량 평가 요청
        if review_text.strip():
            # 1. 요약
            summary_prompt = f"""
            다음 리뷰들을 읽고 장소의 분위기, 실내 공간 디자인 특성, 가구나 채광, 접근성, 청결도, 전체적 추천 여부를 요약해줘:\n\n{review_text}\n\n요약:
            """
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=400,
                )
                summary = completion.choices[0].message.content.strip()
                if not summary:
                    summary = "리뷰 내용이 충분하지 않아 요약이 어렵습니다."
            except:
                summary = "요약 생성 중 오류가 발생했습니다."

            # 2. 정량 평가
            scoring_prompt = f"""
            다음 리뷰를 분석하여 각 지표를 0~1 사이의 숫자로 평가하세요. 
            평가 지표 중 판단이 어려운 경우 0.5로 평가하세요.
            반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.

            평가 지표:
            - 심미성: 시각적 인상, 분위기와 감성, 가구의 색채 및 재질의 이미지, 채광
            - 형태성: 공간 구조, 공간 배치
            - 활동성: 다양한 활동, 참여 가능성
            - 접근성: 위치, 진입 편리성
            - 청결도: 위생, 정리 상태

            리뷰:
            {review_text}

            응답 형식 (다른 텍스트 없이 JSON만):
            {{
              "심미성": 0.8,
              "형태성": 0.6,
              "활동성": 0.5,
              "접근성": 0.9,
              "청결도": 0.7
            }}
            """

            try:
                score_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": scoring_prompt}],
                    max_tokens=300,
                )
                raw_json = score_response.choices[0].message.content.strip()
                
                # JSON 추출을 위한 정규식 패턴들
                import re
                
                # 여러 JSON 패턴 시도
                json_patterns = [
                    r'\{[^{}]*"[^"]*"[^{}]*\}',  # 기본 패턴
                    r'\{[^{}]*"[^"]*"[^{}]*"[^"]*"[^{}]*\}',  # 더 복잡한 패턴
                    r'\{[^{}]*\}',  # 가장 단순한 패턴
                ]
                
                parsed = None
                for pattern in json_patterns:
                    json_match = re.search(pattern, raw_json)
                    if json_match:
                        try:
                            json_str = json_match.group()
                            parsed = json.loads(json_str)
                            break
                        except json.JSONDecodeError:
                            continue
                
                # 직접 JSON 파싱 시도
                if parsed is None:
                    try:
                        parsed = json.loads(raw_json)
                    except json.JSONDecodeError:
                        pass
                
                if parsed:
                    # 유효성 검사: 누락된 항목 보정
                    for key in scores.keys():
                        value = parsed.get(key)
                        if value is not None:
                            try:
                                scores[key] = float(value)
                            except (ValueError, TypeError):
                                scores[key] = None
                        else:
                            scores[key] = None
                else:
                    print("JSON 패턴을 찾을 수 없습니다:", raw_json)
                    scores = {k: None for k in scores.keys()}

            except Exception as e:
                print("JSON 파싱 오류:", e)
                print("원본 응답:", raw_json)
                scores = {k: None for k in scores.keys()}

        # 장소 정보 저장
        place_infos.append({
            'name': place.get('name', '이름 없음'),
            'summary': summary,
            'address': place.get('formatted_address', place.get('vicinity', '주소 정보 없음')),
            'scores': scores,
            'geometry': place.get('geometry', {}),
            'place_id': place.get('place_id', '')
        })

    # 텍스트 출력용 응답 생성
    state.answer = "\n\n".join(
        [f"🔸 **{info['name']}**\n주소: {info['address']}\n요약: {info['summary']}" for info in place_infos]
    )
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
st.title("🗺️ 서울시 공간 정량 평가 및 장소 추천 시스템")
query = st.text_input("🔍 장소 또는 테마 입력")

if st.button("장소 추천받기"):
    with st.spinner("찾는 중..."):
        result = agent.invoke({
            "query": query,
            "places": [],
            "answer": ""
        })

        places = result.get('places', [])
        answer = result.get('answer', '')
        
        st.session_state.history.append((query, answer, places))

# 결과 출력
for i, (q, a, places) in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**질문:** {q}")
    st.markdown(f"**추천 결과:**\n")

    for place in places:
        st.subheader(place.get('name', '이름 정보 없음'))

        # 주소 정보 안전하게 접근
        address = place.get('address') or place.get('formatted_address') or place.get('vicinity', '주소 정보 없음')
        st.markdown(f"**주소:** {address}")

        st.markdown(f"**리뷰 요약:** {place.get('summary', '요약 정보 없음')}")

        # 정량 평가 결과 출력
        scores = place.get('scores')
        if scores:
            st.json(scores)

            if None not in scores.values():
                import pandas as pd
                import plotly.express as px
                import plotly.graph_objects as go

                # 데이터를 명시적으로 순서대로 정렬
                ordered_metrics = ["심미성", "형태성", "활동성", "접근성", "청결도"]
                ordered_scores = [scores[metric] for metric in ordered_metrics]
                
                # 첫 번째 점수를 마지막에 추가하여 완전한 루프 생성
                theta_values = ordered_metrics + [ordered_metrics[0]]
                r_values = ordered_scores + [ordered_scores[0]]

                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=theta_values,
                    fill='toself',
                    name='장소성 평가',
                    line_color='rgb(32, 201, 151)',
                    fillcolor='rgba(32, 201, 151, 0.3)'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        )),
                    showlegend=False,
                    title=f"{place['name']} 장소성 정량 평가",
                    title_x=0.5
                )

                st.plotly_chart(fig)
            else:
                st.warning("정량 평가에 충분한 데이터가 없습니다.")
        else:
            st.warning("정량 평가 결과가 없습니다.")

        # Google Maps 지도 표시 (레이더 차트 다음에 배치)
        if place.get('geometry') and place['geometry'].get('location'):
            lat = place['geometry']['location']['lat']
            lng = place['geometry']['location']['lng']
            
            # 지도와 로드뷰 버튼을 나란히 배치
            col1, col2 = st.columns(2)
            
            # 세션 상태로 지도/로드뷰 표시 여부 관리
            map_key = f"show_map_{place.get('name', '')}_{i}"
            streetview_key = f"show_streetview_{place.get('name', '')}_{i}"
            
            if map_key not in st.session_state:
                st.session_state[map_key] = False
            if streetview_key not in st.session_state:
                st.session_state[streetview_key] = False
            
            with col1:
                if st.button(f"🗺️ 지도 보기", key=f"map_{place.get('name', '')}_{i}"):
                    st.session_state[map_key] = not st.session_state[map_key]
                    st.rerun()
            
            with col2:
                if st.button(f"🚗 로드뷰 보기", key=f"streetview_{place.get('name', '')}_{i}"):
                    st.session_state[streetview_key] = not st.session_state[streetview_key]
                    st.rerun()
            
            # 지도 표시
            if st.session_state[map_key]:
                st.markdown("### 📍 위치 지도")
                
                # Google Maps 링크
                maps_url = f"https://www.google.com/maps?q={lat},{lng}"
                st.markdown(f"🔗 [Google Maps에서 보기]({maps_url})")
                
                # 지도 임베드 (Google Maps Embed API 사용)
                try:
                    map_embed_url = f"https://www.google.com/maps/embed/v1/place?key={st.session_state.gmaps_key}&q={lat},{lng}&zoom=16"
                    st.components.v1.iframe(map_embed_url, height=300)
                except Exception as e:
                    st.error(f"지도를 불러올 수 없습니다: {e}")
                    st.info("Google Maps 링크를 클릭하여 지도를 확인하세요.")
            
            # 로드뷰 표시
            if st.session_state[streetview_key]:
                st.markdown("### 🚗 로드뷰")
                
                # Street View 링크
                streetview_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lng}"
                st.markdown(f"🔗 [Google Street View에서 보기]({streetview_url})")
                
                # 로드뷰 임베드 (zoom 파라미터 제거)
                try:
                    streetview_embed_url = f"https://www.google.com/maps/embed/v1/streetview?key={st.session_state.gmaps_key}&location={lat},{lng}&heading=210&pitch=10"
                    st.components.v1.iframe(streetview_embed_url, height=300)
                except Exception as e:
                    st.error(f"로드뷰를 불러올 수 없습니다: {e}")
                    st.info("Street View 링크를 클릭하여 로드뷰를 확인하세요.")
        else:
            st.info("📍 위치 정보가 없어 지도를 표시할 수 없습니다.")

        st.divider()
