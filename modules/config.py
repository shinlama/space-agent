"""
상수(config), 장소성 요인 정의, 기본 설정 관리
"""
from pathlib import Path


def _compose_factor_text(definition: str, examples: list[str]) -> str:
    """Sentence-BERT 입력용으로 설명 문장과 리뷰 예시를 하나의 텍스트로 합칩니다."""
    example_text = " ".join(example.strip() for example in examples if example.strip())
    if not example_text:
        return definition.strip()
    return f"{definition.strip()} 리뷰 예시: {example_text}"


FACTOR_METADATA = {
    "물리적 특성": {
        "심미성": {
            "english": "Aesthetics",
            "definition": (
                "독창적인 인테리어 디자인과 감각적인 마감재가 조화를 이루어 시각적으로 매우 아름답다. "
                "세련된 분위기 덕분에 공간 전체에서 감도 높은 미적 품질이 느껴진다."
            ),
            "examples": [
                "인테리어가 감성적이고 분위기가 예쁘다.",
                "전체적으로 공간디자인이 세련되고 감각적이다.",
                "사진 찍기 좋을 정도로 공간이 아름답다.",
            ],
            "preference_label": "인테리어가 예쁜 곳",
            "keywords": [
                "예쁘", "아름", "감성", "세련", "멋지", "분위기", "인테리어",
                "디자인", "조명", "사진", "뷰", "아늑",
            ],
        },
        "개방성": {
            "english": "Openness",
            "definition": (
                "유기적인 공간 배치를 통해 답답함 없는 시원한 개방감을 제공한다. "
                "통유리창을 통해 들어오는 풍부한 채광과 외부 조망이 내외부의 시각적 연속성을 강화한다."
            ),
            "examples": [
                "공간이 넓고 답답하지 않아서 좋다.",
                "창이 커서 개방감이 좋다.",
                "테라스가 있고 외부 뷰가 넓게 보인다.",
                "좌석 간격이 넓어서 답답하지 않다.",
            ],
            "preference_label": "개방감이 좋은 곳",
            "keywords": [
                "넓", "개방", "탁 트", "답답하지", "통유리", "창", "채광",
                "뷰", "테라스", "좌석 간격", "시원", "오픈",
            ],
        },
        "감각적 경험": {
            "english": "Sensory",
            "definition": (
                "매력적인 배경 음악과 고소한 커피 향이 어우러져 오감을 자극하는 입체적인 경험을 선사한다. "
                "가구의 다양한 질감과 섬세한 소품들이 다채로운 다감각적 환경을 조성한다."
            ),
            "examples": [
                "음악이랑 분위기가 잘 어울린다.",
                "커피 향이 퍼져서 향긋하다.",
                "조명이나 가구 색감, 소품이 감각적이다.",
            ],
            "preference_label": "감각적인 경험을 할 수 있는 곳",
            "keywords": [
                "음악", "향", "향긋", "냄새", "질감", "오감", "감각",
                "소리", "촉감", "색감", "소품", "커피 향",
            ],
        },
        "접근성": {
            "english": "Accessibility",
            "definition": (
                "대중교통 거점과 인접해 있어 찾아오기 쉽고, 편리한 주차 및 보행 환경을 갖추고 있다. "
                "약속 장소로 정하기에 물리적·심리적 제약이 없는 우수한 입지 조건을 갖추었다."
            ),
            "examples": [
                "위치가 좋아서 찾아가기 쉽다.",
                "역에서 가까워서 접근성이 좋다.",
                "주차나 이동이 편리한 편이다.",
                "약속 장소로 잡기 좋은 위치다.",
            ],
            "preference_label": "접근이 편리한 곳",
            "keywords": [
                "가깝", "접근", "역", "정류장", "도보", "주차",
                "찾아가기", "위치", "편리", "이동", "약속 장소",
            ],
        },
        "쾌적성": {
            "english": "Amenity",
            "definition": (
                "실내 조경과 쾌적한 온습도 관리 덕분에 머무는 내내 신체적·심리적 편안함을 느낄 수 있다. "
                "깨끗하고 청결한 환경과 자연 채광이 조화를 이루어 이용자에게 최적의 휴식을 제공한다."
            ),
            "examples": [
                "오래 있어도 편안하고 쾌적하다.",
                "실내가 위생적으로 깨끗하고 관리가 잘 되어 있다.",
                "온도나 공기가 쾌적해서 좋다.",
                "전체적으로 머물기 편한 환경이다.",
            ],
            "preference_label": "오래 머물기 편한 곳",
            "keywords": [
                "쾌적", "편안", "편하다", "청결", "깨끗", "위생",
                "온도", "공기", "통풍", "채광", "정돈", "관리",
            ],
        },
    },
    "활동적 특성": {
        "활동성": {
            "english": "Activity",
            "definition": (
                "휴식, 업무, 소모임 등 이용자의 방문 목적에 따른 다양한 행위를 수용할 수 있는 높은 공간 활용 수준을 보인다. "
                "개인적인 집중과 자유로운 대화가 모두 가능할 만큼 공간 활용도가 뛰어나다."
            ),
            "examples": [
                "단체 테이블이 있거나 좌석이 충분해 모임하기에 좋다.",
                "친구들과 대화하기 좋다.",
                "개인 작업이나 휴식하기에도 좋다.",
                "좌석에 오래 머물면서 여러 활동을 할 수 있다.",
            ],
            "preference_label": "업무·휴식·모임이 모두 가능한 곳",
            "keywords": [
                "대화", "업무", "작업", "회의", "공부", "휴식",
                "모임", "스터디", "오래 머물", "집중", "좌석",
            ],
        },
        "상호작용성": {
            "english": "Sociability",
            "definition": (
                "타인과의 우연한 만남이나 사회적 교류를 촉진하여 이용자 간의 소속감과 유대감을 자연스럽게 형성한다. "
                "능동적으로 참여할 수 있는 체험 프로그램과 소통의 장이 마련되어 활기찬 사회적 관계를 유도한다."
            ),
            "examples": [
                "사람들 간 소통이 자연스럽게 이루어진다.",
                "사람들끼리 대화하기 편한 분위기다.",
                "직원과 자연스러운 대화가 가능하다.",
                "참여할 수 있는 프로그램이 있다.",
            ],
            "preference_label": "사람들과 소통하기 좋은 곳",
            "keywords": [
                "교류", "소통", "대화", "친절", "서비스", "친근",
                "직원", "커뮤니티", "어울리", "유대", "프로그램", "참여",
            ],
        },
    },
    "의미적 특성": {
        "상징성": {
            "english": "Symbolism",
            "definition": (
                "다른 곳과 구별되는 독창적인 디자인 언어를 통해 해당 장소만의 개성 있는 정체성을 명확히 드러낸다. "
                "콘셉트와 브랜딩, 공간의 상징적 스타일이 일관되게 드러나 장소를 기억하게 만든다."
            ),
            "examples": [
                "이곳만의 개성이 뚜렷한 공간이다.",
                "다른 곳과 구별되는 콘셉트나 브랜딩이 잘 드러난다.",
                "공간 자체가 하나의 디자인 스타일을 보여준다.",
            ],
            "preference_label": "컨셉과 브랜딩이 뚜렷한 곳",
            "keywords": [
                "독특", "유니크", "개성", "컨셉", "브랜딩", "스타일",
                "상징", "시그니처", "아이덴티티", "특색", "차별",
            ],
        },
        "기억 및 선호": {
            "english": "Preference",
            "definition": (
                "비일상적이고 감성적인 공간 경험이 긍정적인 기억으로 남아 다시 방문하고 싶은 깊은 선호도를 형성한다. "
                "개인의 취향을 충족시키는 환경 덕분에 장소에 대한 정서적 만족감이 매우 높다."
            ),
            "examples": [
                "다시 방문하고 싶은 곳이다.",
                "개인적으로 취향이고 마음에 드는 공간이다.",
                "오래 기억에 남을 것 같은 곳이다.",
                "자주 오고 싶다는 생각이 든다.",
            ],
            "preference_label": "다시 방문하고 싶은 곳",
            "keywords": [
                "다시 방문", "재방문", "또 가고 싶", "기억", "추억",
                "인상적", "취향", "선호", "마음에 든", "오래 남",
            ],
        },
        "지역 정체성": {
            "english": "Local Identity",
            "definition": (
                "지역의 역사적 서사나 문화적 맥락을 디자인 요소로 승화시켜 장소가 가진 고유한 가치를 상징적으로 담아냈다. "
                "주변 경관 및 지역 자원과 조화를 이루며 사용자가 지역 사회의 일원임을 인지하게 하는 문화적 깊이가 있다."
            ),
            "examples": [
                "이 동네 분위기가 잘 느껴지는 곳이다.",
                "지역 특색이 잘 반영된 공간이다.",
                "주변 환경이랑 잘 어울린다.",
                "이 지역을 대표하는 장소이다.",
            ],
            "preference_label": "지역 분위기를 잘 담은 곳",
            "keywords": [
                "지역", "동네", "마을", "근처", "주변", "랜드마크",
                "대표", "로컬", "문화", "역사", "특색", "어울린다",
            ],
        },
    },
}


FACTOR_DEFINITIONS = {
    category: {
        factor_name: _compose_factor_text(metadata["definition"], metadata["examples"])
        for factor_name, metadata in factors.items()
    }
    for category, factors in FACTOR_METADATA.items()
}

FACTOR_CATEGORIES = {
    category: list(factors.keys())
    for category, factors in FACTOR_METADATA.items()
}

ALL_FACTORS = {
    factor_name: FACTOR_DEFINITIONS[category][factor_name]
    for category, factor_names in FACTOR_CATEGORIES.items()
    for factor_name in factor_names
}

FACTOR_NAMES = list(ALL_FACTORS.keys())

FACTOR_PREFERENCE_LABELS = {
    factor_name: metadata["preference_label"]
    for factors in FACTOR_METADATA.values()
    for factor_name, metadata in factors.items()
}

FACTOR_KEYWORDS = {
    factor_name: metadata["keywords"]
    for factors in FACTOR_METADATA.values()
    for factor_name, metadata in factors.items()
}

LEGACY_FACTOR_RENAMES = {
    "형태성": "개방성",
    "고유성": "상징성",
    "기억/경험": "기억 및 선호",
}

LEGACY_FACTOR_MERGES = {
    "상호작용성": ("사회성", "참여성"),
}

PLACENESS_METRICS_CSV_CANDIDATES = [
    "placeness_final_research_metrics_real.csv",
    "placeness_final_research_metrics (3).csv",
]

REVIEW_PLACENESS_CSV_CANDIDATES = [
    "review_placeness_scores_real.csv",
]

REVIEWS_WITH_SENTIMENT_CSV_CANDIDATES = [
    "reviews_with_sentiment_real.csv",
]

CAFE_AVG_SENTIMENT_CSV_CANDIDATES = [
    "cafe_avg_sentiment_real.csv",
]


# 데이터 파일 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
GOOGLE_REVIEW_SAMPLE_CSV = BASE_DIR / "google_reviews_scraped_cleaned.csv"
CAFE_INFO_CSV = BASE_DIR / "서울시_상권_카페빵_표본.csv"


# 알고리즘 하이퍼파라미터 설정
SIMILARITY_THRESHOLD = 0.4
DEVIATION_THRESHOLD = 0.05
