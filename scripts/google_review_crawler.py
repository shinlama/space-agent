from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from googlemaps import Client
from googlemaps.exceptions import ApiError, TransportError, Timeout
from tqdm import tqdm


DEFAULT_INPUT = Path("서울시_상권_카페빵_표본.csv")
DEFAULT_OUTPUT = Path("google_reviews_sample.csv")
DEFAULT_MAX_REVIEWS = 10
DEFAULT_SLEEP = (0.8, 1.6)
NAME_COL = "상호명"
DISTRICT_COL = "시군구명"
EUPMYEON_COL = "행정동명"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="서울시 카페 표본 데이터를 기반으로 Google Maps 리뷰를 수집합니다."
    )
    parser.add_argument("--api-key", type=str, default=None, help="Google Maps API 키 (미지정 시 환경변수 Maps_API_KEY 사용)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="카페 표본 CSV 경로")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="리뷰 저장 CSV 경로")
    parser.add_argument("--max-reviews", type=int, default=DEFAULT_MAX_REVIEWS, help="카페당 최대 리뷰 수")
    parser.add_argument(
        "--sleep-min",
        type=float,
        default=DEFAULT_SLEEP[0],
        help="API 호출 간 최소 대기 시간 (초)",
    )
    parser.add_argument(
        "--sleep-max",
        type=float,
        default=DEFAULT_SLEEP[1],
        help="API 호출 간 최대 대기 시간 (초)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="기존 수집 CSV 경로 (있다면 이어서 수집, 이미 처리한 카페는 건너뜀)",
    )
    return parser.parse_args()


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")
    return df


def filter_targets(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in (NAME_COL, DISTRICT_COL) if col not in df.columns]
    if cols:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(cols)}")
    return df.dropna(subset=[NAME_COL, DISTRICT_COL]).reset_index(drop=True)


def build_query(row: pd.Series) -> str:
    name = str(row[NAME_COL]).strip()
    district = str(row[DISTRICT_COL]).strip()
    eupmyeon = str(row.get(EUPMYEON_COL, "") or "").strip()
    if eupmyeon:
        return f"{district} {eupmyeon} {name}"
    return f"{district} {name}"


def get_google_reviews(
    gmaps: Client,
    name: str,
    district: str,
    eupmyeon: str,
    max_reviews: int,
) -> List[Dict[str, str]]:
    query_parts = [district]
    if eupmyeon:
        query_parts.append(eupmyeon)
    query_parts.append(name)
    query = " ".join(query_parts)

    reviews: List[Dict[str, str]] = []

    try:
        search_resp = gmaps.places(query=query, language="ko", region="kr")
    except (ApiError, TransportError, Timeout) as exc:
        print(f"[API 오류] {name} ({district}) → {exc}")
        return reviews
    except Exception as exc:
        print(f"[예외] {name} ({district}) 검색 실패: {exc}")
        return reviews

    results = search_resp.get("results") or []
    if not results:
        return reviews

    place_id = results[0].get("place_id")
    if not place_id:
        return reviews

    try:
        details = gmaps.place(place_id=place_id, language="ko")
    except (ApiError, TransportError, Timeout) as exc:
        print(f"[API 오류] {name} ({district}) 상세조회 실패 → {exc}")
        return reviews
    except Exception as exc:
        print(f"[예외] {name} ({district}) 상세조회 실패: {exc}")
        return reviews

    detail_result = details.get("result") or {}
    for review in (detail_result.get("reviews") or [])[:max_reviews]:
        reviews.append(
            {
                NAME_COL: name,
                DISTRICT_COL: district,
                "행정동명": eupmyeon,
                "작성자": review.get("author_name"),
                "평점": review.get("rating"),
                "리뷰": review.get("text"),
                "작성일": review.get("relative_time_description"),
                "언어": review.get("language"),
            }
        )
    return reviews


def main() -> int:
    args = parse_args()
    load_dotenv()
    api_key = args.api_key or os.getenv("Maps_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("Google Maps API 키를 찾을 수 없습니다. --api-key 옵션 또는 환경변수 Maps_API_KEY 설정이 필요합니다.")
        return 1

    cafes_df = filter_targets(load_dataframe(args.input))
    print(f"총 {len(cafes_df)}개 카페 표본 로드 완료")

    resume_df = None
    processed_keys = set()
    if args.resume and args.resume.exists():
        resume_df = filter_targets(load_dataframe(args.resume))
        processed_keys = set(zip(resume_df[NAME_COL], resume_df[DISTRICT_COL]))
        print(f"기존 리뷰 {len(resume_df)}건을 불러왔습니다. 이미 처리된 {len(processed_keys)}개 카페는 건너뜁니다.")

    gmaps = Client(key=api_key)
    collected: List[Dict[str, str]] = []
    if resume_df is not None:
        collected.extend(resume_df.to_dict(orient="records"))

    sleep_min, sleep_max = min(args.sleep_min, args.sleep_max), max(args.sleep_min, args.sleep_max)

    for _, row in tqdm(cafes_df.iterrows(), total=len(cafes_df), desc="구글 리뷰 수집"):
        name = str(row[NAME_COL]).strip()
        district = str(row[DISTRICT_COL]).strip()
        eupmyeon = str(row.get(EUPMYEON_COL, "") or "").strip()
        key = (name, district)

        if key in processed_keys:
            continue

        reviews = get_google_reviews(
            gmaps,
            name=name,
            district=district,
            eupmyeon=eupmyeon,
            max_reviews=args.max_reviews,
        )

        if reviews:
            collected.extend(reviews)

        delay = random.uniform(sleep_min, sleep_max)
        time.sleep(delay)

        processed_keys.add(key)

    if collected:
        output_df = pd.DataFrame(collected)
        output_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"✅ 총 {len(output_df)}개 리뷰 저장 완료 → {args.output}")
    else:
        print("❌ 리뷰가 수집되지 않았습니다.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

