import streamlit as st
import googlemaps
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any

# --------------------------
# Streamlit 설정
# --------------------------
st.set_page_config(page_title="Seoul Place Recommendation", page_icon="🗺️", layout="centered")

# --------------------------
# 세션 상태 초기화
# --------------------------
if "gmaps_key" not in st.session_state:
    st.session_state.gmaps_key = ""
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""

# --------------------------
# ① API 키 입력 화면
# --------------------------
if not st.session_state.gmaps_key or not st.session_state.openai_key:
    st.title("🗺️ Seoul Place Recommendation Chatbot")
    st.markdown("""
    To use this chatbot, please enter your API keys below.  
    If you don't have them yet, follow these links to generate them:
    
    - 🌐 [Get your **Google Maps API Key**](https://developers.google.com/maps/documentation/javascript/get-api-key)
    - 🤖 [Get your **OpenAI API Key**](https://platform.openai.com/api-keys)
    """)

    gmaps_input = st.text_input("Enter your **Google Maps API Key**", type="password")
    openai_input = st.text_input("Enter your **OpenAI API Key**", type="password")

    if st.button("Start"):
        if gmaps_input and openai_input:
            st.session_state.gmaps_key = gmaps_input
            st.session_state.openai_key = openai_input
            st.rerun()
        else:
            st.warning("Please enter both keys to proceed.")
    st.stop()


# --------------------------
# ② 키 입력 후 서비스 실행
# --------------------------
# 클라이언트 초기화
gmaps = googlemaps.Client(key=st.session_state.gmaps_key)
client = OpenAI(api_key=st.session_state.openai_key)

# 상태 정의
class AgentState(BaseModel):
    query: str
    places: List[Dict[str, Any]] = Field(default_factory=list)
    answer: str = ""

# 장소 검색 노드
def search_places(state: AgentState):
    res = gmaps.places(query=state.query, language="ko", location="37.5665,126.9780", radius=5000)
    state.places = res.get('results', [])[:5]
    return state.dict()

# 리뷰 분석 노드
def analyze_reviews(state: AgentState):
    place_infos = []
    for place in state.places:
        place_id = place["place_id"]
        details = gmaps.place(place_id=place_id, language="ko")
        reviews = details.get('result', {}).get('reviews', [])[:3]
        review_text = "\n".join([review['text'] for review in reviews])

        prompt = f"""
        다음 리뷰를 읽고 장소의 분위기, 접근성, 청결도, 전체적 추천 여부를 요약해줘:\n\n{review_text}\n\n요약:
        """

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        summary = completion.choices[0].message.content

        place_infos.append({
            'name': place['name'],
            'summary': summary,
            'address': place.get('formatted_address', place.get('vicinity'))
        })

    state.answer = "\n\n".join(
        [f"🔸 **{info['name']}**\n주소: {info['address']}\n요약: {info['summary']}" for info in place_infos]
    )
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
st.title("🗺️ 서울 공간 추천 챗봇")
st.caption("서울에서 방문하고 싶은 장소나 테마를 입력하세요 (예: '예쁜 카페', '산책하기 좋은 공원', '쇼핑하기 좋은 곳')")

if "history" not in st.session_state:
    st.session_state["history"] = []

query = st.text_input("🔍 장소 또는 테마 입력")

if st.button("장소 추천받기"):
    with st.spinner("찾는 중..."):
        result = agent.invoke({
            "query": query,
            "places": [],
            "answer": ""
        })
        st.session_state.history.append((query, result["answer"]))

# --------------------------
# 결과 출력
# --------------------------
for i, (q, a) in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**질문:** {q}")
    st.markdown(f"**추천 결과:**\n")

    for place_block in a.split("🔸"):
        if not place_block.strip():
            continue
        lines = place_block.strip().split("\n")
        title_line = lines[0]
        info_lines = lines[1:]
        address_line = [line for line in info_lines if line.startswith("주소:")]

        with st.container():
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #ddd;
                    border-radius: 12px;
                    padding: 20px;
                    margin: 10px 0;
                    background-color: #f9f9f9;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                ">
                    <h4 style="margin-bottom: 0.5rem;">🧭 {title_line.strip('* ')}</h4>
                    <p style="margin: 0.2rem 0;"><b>📍 주소:</b> {address_line[0].replace("주소:", "").strip() if address_line else '정보 없음'}</p>
                    <p style="margin-top: 0.5rem;"><b>📝 요약:</b></p>
                    <p style="white-space: pre-line; margin-left: 0.5rem;">
                        {"".join(line.replace("요약:", "").strip() for line in info_lines if line.startswith("요약:"))}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 지도, 로드뷰 버튼
        if address_line:
            address = address_line[0].replace("주소:", "").strip()
            geocode_result = gmaps.geocode(address)
            if geocode_result:
                latlng = geocode_result[0]['geometry']['location']
                lat, lng = latlng['lat'], latlng['lng']
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🗺️ 지도 보기 ({title_line.strip('* ')})", key=f"map_{i}_{lat}"):
                        map_url = f"https://www.google.com/maps/embed/v1/place?key={st.session_state.gmaps_key}&q={lat},{lng}"
                        st.components.v1.iframe(map_url, width=600, height=400)
                with col2:
                    if st.button(f"🚶 로드뷰 보기 ({title_line.strip('* ')})", key=f"street_{i}_{lng}"):
                        street_url = f"https://www.google.com/maps/embed/v1/streetview?location={lat},{lng}&key={st.session_state.gmaps_key}"
                        st.components.v1.iframe(street_url, width=600, height=400)

    st.divider()
