from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / "google_reviews_scraped_cleaned.csv"


FACTOR_GUIDE: Dict[str, Dict[str, Any]] = {
    "심미성": {
        "category": "물리적 특성",
        "english": "Aesthetics",
        "definition": "테마가 있는 인테리어 디자인과 감각적 마감재 혼합을 통한 시각적 아름다움",
        "keywords": ["인테리어 디자인", "시각적 아름다움", "분위기 연출", "예쁘다", "감성", "세련됨", "사진", "뷰", "디자인"],
        "positive_examples": ["인테리어가 감성적이고 분위기가 예쁘다.", "사진 찍기 좋을 정도로 공간이 아름답다."],
        "negative_examples": ["인테리어가 별로고 촌스럽다.", "공간 디자인이 어수선하고 예쁘지 않다."],
    },
    "개방성": {
        "category": "물리적 특성",
        "english": "Openness",
        "definition": "공간의 배치와 시야 구성이 외부 조망과 시각적 연결을 확보하여 확장감과 낮은 폐쇄감을 느끼게 하는 특성",
        "keywords": ["공간 배치", "시야 연속성", "채광", "외부 조망", "통창", "넓다", "답답하다", "좌석 간격", "테라스"],
        "positive_examples": ["공간이 넓고 답답하지 않아서 좋다.", "창이 커서 개방감이 좋고 외부 뷰가 잘 보인다."],
        "negative_examples": ["좌석 간격이 좁아서 답답하다.", "창이 작고 내부가 폐쇄적으로 느껴진다."],
    },
    "접근성": {
        "category": "물리적 특성",
        "english": "Accessibility",
        "definition": "대중교통 접근, 주차 연계, 도보 가능성 등 쉽게 찾아오고 편리하게 이용할 수 있는 정도",
        "keywords": ["대중교통 접근성", "보행 접근성", "주차 편의성", "역", "정류장", "도보", "위치", "찾아가기", "이동", "가깝다", "멀다"],
        "positive_examples": ["역에서 가까워서 접근성이 좋다.", "주차나 이동이 편리한 편이다."],
        "negative_examples": ["주차 지원이 안 돼서 불편하다.", "역에서 멀고 찾아가기 어렵다."],
    },
    "감각적 경험": {
        "category": "물리적 특성",
        "english": "Sensory",
        "definition": "조명, 배경 음악, 커피 및 베이커리 향기, 가구와 소품의 질감 등 오감을 자극하는 감각적 요소 제공",
        "keywords": ["조명 환경", "음향 환경", "향 요소", "가구 질감", "소품 질감", "음악", "소리", "냄새", "향", "질감", "색감"],
        "positive_examples": ["음악이랑 분위기가 잘 어울린다.", "커피 향이 퍼져서 향긋하고 조명이 감각적이다."],
        "negative_examples": ["음악이 너무 시끄러워서 불편하다.", "조명이 너무 어둡고 냄새가 별로다."],
    },
    "쾌적성": {
        "category": "물리적 특성",
        "english": "Comfort",
        "definition": "채광, 실내 조경, 온습도, 공기질, 청결 등 공간 이용자가 머무르며 느끼는 물리적 쾌적함",
        "keywords": ["실내 환경", "자연 친화적 요소", "오염 환경", "청결도", "위생", "온도", "습도", "환기", "공기", "통풍", "깨끗하다"],
        "positive_examples": ["실내가 위생적으로 깨끗하고 공기가 쾌적하다.", "온도와 통풍이 좋아 오래 있어도 편하다."],
        "negative_examples": ["매장이 덥고 환기가 잘 안 된다.", "테이블이 지저분하고 청결하지 않다."],
    },
    "활동성": {
        "category": "활동적 특성",
        "english": "Activity",
        "definition": "모임, 대화, 업무, 휴식, 식사 등 다양한 활동이 이루어지는 정도",
        "keywords": ["행위 수용성", "공간 활용도", "체류 형태 다양성", "작업", "공부", "휴식", "모임", "대화", "식사", "오래 머무름", "좌석"],
        "positive_examples": ["개인 작업이나 휴식하기에도 좋고 모임하기에도 좋다.", "좌석에 오래 머물면서 여러 활동을 할 수 있다."],
        "negative_examples": ["작업하기 불편하고 오래 머물기 어렵다.", "좌석이 부족해서 모임하기 애매하다."],
    },
    "상호작용성": {
        "category": "활동적 특성",
        "english": "Sociability",
        "definition": "능동적으로 참여할 수 있는 체험 프로그램과 이용자 간 소통 및 유대감을 강화할 수 있는 정도",
        "keywords": ["사회적 교류", "체험 프로그램", "커뮤니티 형성", "소통", "대화", "친절", "직원", "참여", "프로그램", "만남", "교류"],
        "positive_examples": ["사람들 간 소통이 자연스럽게 이루어진다.", "직원과 자연스러운 대화가 가능하고 참여할 수 있는 프로그램이 있다."],
        "negative_examples": ["직원 응대가 딱딱해서 소통이 어렵다.", "사람들과 교류하거나 참여할 만한 요소가 없다."],
    },
    "상징성": {
        "category": "의미적 특성",
        "english": "Symbolism",
        "definition": "다른 장소와 차별화되는 독창적인 공간디자인으로 해당 공간만의 개성 있는 정체성 형성",
        "keywords": ["브랜드 이미지", "디자인 개성", "콘셉트", "차별화", "시그니처", "독특함", "정체성", "개성", "브랜딩"],
        "positive_examples": ["이곳만의 개성이 뚜렷한 공간이다.", "다른 곳과 구별되는 콘셉트나 브랜딩이 잘 드러난다."],
        "negative_examples": ["컨셉이 애매하고 다른 카페와 차별점이 없다.", "브랜드 이미지가 잘 느껴지지 않는다."],
    },
    "기억 및 선호": {
        "category": "의미적 특성",
        "english": "Preference",
        "definition": "비일상적이고 감성적인 공간 경험과 개인 취향 충족을 통해 긍정적 감정과 재방문 의도를 형성하는 정도",
        "keywords": ["정서적 만족감", "취향 적합성", "기억 형성", "재방문", "다시 방문", "추천", "마음에 듦", "인상적", "오고 싶다", "취향"],
        "positive_examples": ["다시 방문하고 싶은 곳이다.", "개인적으로 취향이고 마음에 드는 공간이다."],
        "negative_examples": ["다시는 방문하고 싶지 않다.", "내 취향과 맞지 않아 기억에 남지 않는다."],
    },
    "지역 정체성": {
        "category": "의미적 특성",
        "english": "Local Identity",
        "definition": "지역의 역사적 서사, 유휴 시설의 재생, 지역 특유의 경관 및 자원을 디자인 요소로 반영하여 장소가 가진 문화적 맥락을 상징적으로 담아낸 정도",
        "keywords": ["지역 맥락 반영", "역사성", "지역 자원 활용", "동네", "마을", "주변 환경", "골목", "로컬", "지역 특색", "재생", "유휴 시설"],
        "positive_examples": ["이 동네 분위기가 잘 느껴지는 곳이다.", "지역 특색이 공간에 잘 반영되어 있다."],
        "negative_examples": ["주변 지역 분위기와 어울리지 않는다.", "지역 맥락 없이 뜬금없는 공간처럼 느껴진다."],
    },
}


FACTOR_NAMES = list(FACTOR_GUIDE.keys())


SYSTEM_PROMPT = """
당신은 실내건축/상업공간 연구를 위한 한국어 리뷰 주석자입니다.
목표는 카페 리뷰 문장을 10가지 장소성 요인 중 어떤 요인을 '언급하는지' 다중 라벨로 매핑하는 것입니다.

중요한 원칙:
1. 이 단계는 감성 점수화가 아니라 요인 매핑입니다. 좋다/나쁘다를 최종 점수로 계산하지 마세요.
2. 긍정 표현과 부정 표현은 모두 같은 요인으로 매핑합니다.
   예: "주차가 어렵다"는 접근성, "답답하다"는 개방성, "시끄럽다"는 감각적 경험, "더럽다"는 쾌적성입니다.
3. 요인 이름 자체나 긍정 예시와 문장 표현이 정확히 같지 않아도, 키워드와 공간적 의미가 같으면 매핑합니다.
4. 한 문장이 여러 요인을 언급하면 모두 매핑합니다.
5. 공간/장소성과 무관한 음식 맛, 가격, 메뉴 취향만 있는 문장은 매핑하지 않습니다.
   단, 향, 조명, 분위기, 체류, 재방문 의도처럼 공간 경험과 연결되면 매핑합니다.
6. "분위기"만으로 지역 정체성을 매핑하지 마세요. 지역 정체성은 동네, 지역, 역사, 주변 환경, 로컬 맥락이 명시될 때만 매핑합니다.
7. "좋다", "별로다" 같은 일반 평가만 있고 어떤 공간 요인인지 근거가 없으면 매핑하지 않습니다.
8. confidence는 0~1 사이로, 해당 요인 언급이라고 판단하는 확신입니다.
9. evidence는 원문에서 요인 판단 근거가 되는 짧은 구절입니다.
10. sentiment_hint는 근거 구절의 방향을 positive, negative, neutral, mixed 중 하나로만 표시합니다. 이것은 최종 점수가 아닙니다.

반드시 JSON만 출력하세요. 설명 문장을 JSON 밖에 쓰지 마세요.
"""


def _factor_guide_text() -> str:
    lines: List[str] = []
    for factor, guide in FACTOR_GUIDE.items():
        lines.append(f"- {factor} ({guide['english']}, {guide['category']})")
        lines.append(f"  정의: {guide['definition']}")
        lines.append(f"  키워드: {', '.join(guide['keywords'])}")
        lines.append(f"  긍정 예시: {' / '.join(guide['positive_examples'])}")
        lines.append(f"  부정 예시: {' / '.join(guide['negative_examples'])}")
    return "\n".join(lines)


USER_PROMPT_TEMPLATE = """
아래 장소성 요인 가이드를 기준으로 입력 리뷰 문장들을 요인 매핑하세요.

[장소성 요인 가이드]
{factor_guide}

[출력 JSON 형식]
{{
  "reviews": [
    {{
      "review_index": 0,
      "items": [
        {{
          "sentence_id": 0,
          "mappings": [
            {{
              "factor": "접근성",
              "confidence": 0.92,
              "evidence": "주차는 어려울 듯",
              "reason": "주차 편의성에 대한 부정 언급이므로 접근성 요인",
              "sentiment_hint": "negative"
            }}
          ]
        }}
      ]
    }}
  ]
}}

[입력 리뷰]
{review_payload}
"""


@dataclass
class ReviewItem:
    review_index: int
    cafe_name: str
    review_text: str
    sentences: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map Korean cafe review sentences to placeness factors with OpenAI API."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV, help="Input review CSV path.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="llm_factor_mapping",
        help="Output filename prefix. CSV/JSONL files are written to the project root.",
    )
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix appended to output filenames.")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8, help="Number of reviews per API request.")
    parser.add_argument("--start-row", type=int, default=0, help="0-based inclusive row index after cleaning.")
    parser.add_argument("--end-row", type=int, default=None, help="0-based exclusive row index after cleaning.")
    parser.add_argument("--start-cafe", type=int, default=None, help="0-based inclusive cafe index after cleaning.")
    parser.add_argument("--end-cafe", type=int, default=None, help="0-based exclusive cafe index after cleaning.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of reviews to process.")
    parser.add_argument("--max-sentence-chars", type=int, default=500)
    parser.add_argument("--resume", action="store_true", help="Skip review_index values already in raw JSONL.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing raw JSONL before running.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first prompt and exit without API calls.")
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def split_into_sentences(text: str) -> List[str]:
    if text is None:
        return []
    cleaned = re.sub(r"\s+", " ", str(text).strip())
    if not cleaned:
        return []

    parts = re.split(r"(?<=[.!?。！？])\s+|[\n\r]+", cleaned)
    sentences = [part.strip() for part in parts if len(part.strip()) > 1]
    return sentences or [cleaned]


def normalize_review_df(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    try:
        df = pd.read_csv(input_path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="cp949", on_bad_lines="skip")

    column_mapping = {
        "상호명": "cafe_name",
        "리뷰": "review_text",
        "cafe_name": "cafe_name",
        "review_text": "review_text",
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    if "cafe_name" not in df.columns or "review_text" not in df.columns:
        raise ValueError(f"Required columns are missing. Current columns: {list(df.columns)}")

    if "original_cafe_name" not in df.columns:
        df["original_cafe_name"] = df["cafe_name"]

    if "시군구명" in df.columns and "행정동명" in df.columns:
        has_location = df["시군구명"].notna() & df["행정동명"].notna()
        df.loc[has_location, "cafe_name"] = (
            df.loc[has_location, "cafe_name"].astype(str)
            + " "
            + df.loc[has_location, "시군구명"].astype(str)
            + " "
            + df.loc[has_location, "행정동명"].astype(str)
        )

    df = df.dropna(subset=["cafe_name", "review_text"]).copy()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"] != ""].copy()
    df = df.reset_index(drop=False).rename(columns={"index": "source_row"})
    df["review_index"] = df.index
    return df


def select_review_slice(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = df

    if args.start_cafe is not None or args.end_cafe is not None:
        cafe_names = list(dict.fromkeys(selected["cafe_name"].astype(str).tolist()))
        start = max(0, args.start_cafe or 0)
        end = args.end_cafe if args.end_cafe is not None else len(cafe_names)
        end = min(end, len(cafe_names))
        if start >= end:
            raise ValueError(f"Invalid cafe slice: start={start}, end={end}")
        selected_cafes = set(cafe_names[start:end])
        selected = selected[selected["cafe_name"].isin(selected_cafes)]

    start_row = max(0, args.start_row or 0)
    end_row = args.end_row if args.end_row is not None else len(selected)
    selected = selected.iloc[start_row:end_row]

    if args.limit is not None:
        selected = selected.head(max(0, args.limit))

    return selected.reset_index(drop=True)


def build_review_items(df: pd.DataFrame, max_sentence_chars: int) -> List[ReviewItem]:
    items: List[ReviewItem] = []
    for _, row in df.iterrows():
        sentences = []
        for sentence in split_into_sentences(row["review_text"]):
            sentence = sentence.strip()
            if len(sentence) > max_sentence_chars:
                sentence = sentence[:max_sentence_chars].rstrip() + "..."
            sentences.append(sentence)

        items.append(
            ReviewItem(
                review_index=int(row["review_index"]),
                cafe_name=str(row["cafe_name"]),
                review_text=str(row["review_text"]),
                sentences=sentences,
            )
        )
    return items


def chunked(items: List[ReviewItem], batch_size: int) -> Iterable[List[ReviewItem]]:
    batch_size = max(1, batch_size)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_prompt(batch: List[ReviewItem]) -> str:
    payload = [
        {
            "review_index": item.review_index,
            "cafe_name": item.cafe_name,
            "sentences": [
                {"sentence_id": idx, "text": sentence}
                for idx, sentence in enumerate(item.sentences)
            ],
        }
        for item in batch
    ]
    return USER_PROMPT_TEMPLATE.format(
        factor_guide=_factor_guide_text(),
        review_payload=json.dumps(payload, ensure_ascii=False, indent=2),
    )


def call_openai_json(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_retries: int,
) -> Dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            sleep_seconds = min(60, 2**attempt)
            print(f"OpenAI call failed ({attempt + 1}/{max_retries}). Retrying in {sleep_seconds}s: {exc}")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_error}")


def output_paths(prefix: str, suffix: str) -> Dict[str, Path]:
    clean_suffix = suffix.strip()
    if clean_suffix and not clean_suffix.startswith("_"):
        clean_suffix = "_" + clean_suffix
    return {
        "raw": PROJECT_ROOT / f"{prefix}_raw{clean_suffix}.jsonl",
        "sentence": PROJECT_ROOT / f"{prefix}_sentences{clean_suffix}.csv",
        "review": PROJECT_ROOT / f"{prefix}_reviews{clean_suffix}.csv",
    }


def load_processed_review_indices(raw_path: Path) -> set[int]:
    processed: set[int] = set()
    if not raw_path.exists():
        return processed

    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for review in record.get("reviews", []):
                try:
                    processed.add(int(review["review_index"]))
                except (KeyError, TypeError, ValueError):
                    continue
    return processed


def sanitize_factor_name(value: str) -> str | None:
    if value in FACTOR_NAMES:
        return value
    normalized = str(value).strip()
    return normalized if normalized in FACTOR_NAMES else None


def normalize_mapping(mapping: Dict[str, Any]) -> Dict[str, Any] | None:
    factor = sanitize_factor_name(str(mapping.get("factor", "")).strip())
    if factor is None:
        return None

    try:
        confidence = float(mapping.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    sentiment_hint = str(mapping.get("sentiment_hint", "neutral")).strip().lower()
    if sentiment_hint not in {"positive", "negative", "neutral", "mixed"}:
        sentiment_hint = "neutral"

    return {
        "factor": factor,
        "confidence": confidence,
        "evidence": str(mapping.get("evidence", "")).strip(),
        "reason": str(mapping.get("reason", "")).strip(),
        "sentiment_hint": sentiment_hint,
    }


def raw_jsonl_to_frames(raw_path: Path, review_items: List[ReviewItem], model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    review_lookup = {item.review_index: item for item in review_items}
    sentence_rows: List[Dict[str, Any]] = []

    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for review in record.get("reviews", []):
                    try:
                        review_index = int(review["review_index"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    item = review_lookup.get(review_index)
                    if item is None:
                        continue

                    for sentence_item in review.get("items", []):
                        try:
                            sentence_id = int(sentence_item.get("sentence_id", -1))
                        except (TypeError, ValueError):
                            continue
                        if sentence_id < 0 or sentence_id >= len(item.sentences):
                            continue
                        sentence = item.sentences[sentence_id]

                        mappings = sentence_item.get("mappings", [])
                        if not isinstance(mappings, list):
                            continue
                        for raw_mapping in mappings:
                            if not isinstance(raw_mapping, dict):
                                continue
                            mapping = normalize_mapping(raw_mapping)
                            if mapping is None:
                                continue
                            sentence_rows.append(
                                {
                                    "review_index": review_index,
                                    "cafe_name": item.cafe_name,
                                    "review_text": item.review_text,
                                    "sentence_id": sentence_id,
                                    "sentence": sentence,
                                    "factor": mapping["factor"],
                                    "confidence": mapping["confidence"],
                                    "evidence": mapping["evidence"],
                                    "reason": mapping["reason"],
                                    "sentiment_hint": mapping["sentiment_hint"],
                                    "model": model,
                                }
                            )

    sentence_df = pd.DataFrame(sentence_rows)

    review_rows: List[Dict[str, Any]] = []
    for item in review_items:
        row: Dict[str, Any] = {
            "review_index": item.review_index,
            "cafe_name": item.cafe_name,
            "review_text": item.review_text,
        }
        review_factor_rows = (
            sentence_df[sentence_df["review_index"] == item.review_index]
            if not sentence_df.empty
            else pd.DataFrame()
        )
        for factor in FACTOR_NAMES:
            if review_factor_rows.empty:
                factor_rows = pd.DataFrame()
            else:
                factor_rows = review_factor_rows[review_factor_rows["factor"] == factor]
            row[f"{factor}_mapped"] = not factor_rows.empty
            row[f"{factor}_sentence_count"] = int(len(factor_rows))
            row[f"{factor}_max_confidence"] = (
                float(factor_rows["confidence"].max()) if not factor_rows.empty else 0.0
            )
            row[f"{factor}_evidence"] = (
                " | ".join(dict.fromkeys(factor_rows["evidence"].dropna().astype(str).tolist()))
                if not factor_rows.empty
                else ""
            )
        review_rows.append(row)

    review_df = pd.DataFrame(review_rows)
    return sentence_df, review_df


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    df = normalize_review_df(args.input)
    selected_df = select_review_slice(df, args)
    review_items = build_review_items(selected_df, args.max_sentence_chars)
    paths = output_paths(args.output_prefix, args.suffix)

    print(f"Loaded {len(df):,} cleaned reviews.")
    print(f"Selected {len(review_items):,} reviews for LLM factor mapping.")
    print(f"Raw output: {paths['raw']}")

    if not review_items:
        raise ValueError("No reviews selected.")

    first_prompt = build_prompt(review_items[: min(args.batch_size, len(review_items))])
    if args.dry_run:
        print(first_prompt)
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or the environment.")

    if args.overwrite and paths["raw"].exists():
        paths["raw"].unlink()
    elif paths["raw"].exists() and not args.resume:
        raise FileExistsError(
            f"Raw output already exists: {paths['raw']}. "
            "Use --resume to continue or --overwrite to start over."
        )

    processed = load_processed_review_indices(paths["raw"]) if args.resume else set()
    if processed:
        print(f"Resume enabled: skipping {len(processed):,} already processed reviews.")

    pending_items = [item for item in review_items if item.review_index not in processed]
    client = OpenAI()

    with paths["raw"].open("a", encoding="utf-8") as raw_handle:
        for batch_no, batch in enumerate(chunked(pending_items, args.batch_size), start=1):
            prompt = build_prompt(batch)
            result = call_openai_json(
                client=client,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
            raw_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            raw_handle.flush()
            done = min(batch_no * args.batch_size, len(pending_items))
            print(f"Mapped {done:,}/{len(pending_items):,} pending reviews")

    sentence_df, review_df = raw_jsonl_to_frames(paths["raw"], review_items, args.model)
    sentence_df.to_csv(paths["sentence"], index=False, encoding="utf-8-sig")
    review_df.to_csv(paths["review"], index=False, encoding="utf-8-sig")
    print(f"Wrote sentence mappings: {paths['sentence']} ({len(sentence_df):,} rows)")
    print(f"Wrote review mappings: {paths['review']} ({len(review_df):,} rows)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise
