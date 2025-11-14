import argparse
import json
import os
import sys
import time
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import googlemaps
from dotenv import load_dotenv


def load_api_key(env_key: str = "Maps_API_KEY") -> str:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    api_key = os.getenv(env_key)
    if not api_key:
        raise RuntimeError(f"{env_key} 환경 변수를 찾을 수 없습니다. .env 파일을 확인하세요.")
    return api_key


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c * 1000  # meters


def compute_walking_estimate(
    gmaps_client: googlemaps.Client,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    label: str,
) -> Tuple[Optional[float], Optional[float], str]:
    attempts = [
        (origin, destination, "카페→대중교통"),
        (destination, origin, "대중교통→카페"),
    ]
    for orig, dest, attempt_label in attempts:
        try:
            matrix = gmaps_client.distance_matrix(
                origins=[orig],
                destinations=[dest],
                mode="walking",
                language="ko",
                region="kr",
            )
            status = matrix["rows"][0]["elements"][0]["status"]
            if status == "OK":
                duration = matrix["rows"][0]["elements"][0]["duration"]["value"] / 60.0
                distance = matrix["rows"][0]["elements"][0]["distance"]["value"]
                return duration, distance, "distance_matrix"
        except Exception as exc:
            print(f"[WARN] Distance Matrix 실패 ({label}/{attempt_label}): {exc}", file=sys.stderr)

    try:
        directions = gmaps_client.directions(
            origin=origin,
            destination=destination,
            mode="walking",
            language="ko",
            region="kr",
        )
        if directions:
            leg = directions[0]["legs"][0]
            duration = leg["duration"]["value"] / 60.0
            distance = leg["distance"]["value"]
            return duration, distance, "directions"
    except Exception as exc:
        print(f"[WARN] Directions API 실패 ({label}): {exc}", file=sys.stderr)

    straight = haversine_distance(origin[0], origin[1], destination[0], destination[1])
    path = straight * 1.4
    duration = path / 67.0
    return duration, path, "fallback"


def calculate_transit_accessibility(
    gmaps_client: googlemaps.Client,
    lat: float,
    lng: float,
    radius: int = 600,
) -> Tuple[Optional[float], str, str, Optional[float], Optional[float]]:
    if np.isnan(lat) or np.isnan(lng):
        return None, "좌표 없음", "없음", np.nan, np.nan

    try:
        subway_results = gmaps_client.places_nearby(
            location=(lat, lng),
            radius=radius,
            type="subway_station",
            language="ko",
        ).get("results", [])
    except Exception as exc:
        print(f"[WARN] 지하철역 검색 실패: {exc}", file=sys.stderr)
        subway_results = []

    try:
        bus_results = gmaps_client.places_nearby(
            location=(lat, lng),
            radius=radius,
            type="bus_station",
            language="ko",
        ).get("results", [])
    except Exception as exc:
        print(f"[WARN] 버스정류장 검색 실패: {exc}", file=sys.stderr)
        bus_results = []

    if not subway_results and not bus_results:
        return None, "정보 없음", "없음", np.nan, np.nan

    candidates = []
    for station in subway_results[:3]:
        loc = station.get("geometry", {}).get("location", {})
        if not loc:
            continue
        candidates.append((station.get("name", "지하철역"), "지하철역", loc.get("lat"), loc.get("lng")))

    for bus in bus_results[:3]:
        loc = bus.get("geometry", {}).get("location", {})
        if not loc:
            continue
        candidates.append((bus.get("name", "버스정류장"), "버스정류장", loc.get("lat"), loc.get("lng")))

    nearest = None
    min_duration = float("inf")
    min_distance = float("inf")
    min_straight = float("inf")

    for name, label, t_lat, t_lng in candidates:
        if t_lat is None or t_lng is None:
            continue
        duration, distance, source = compute_walking_estimate(
            gmaps_client,
            (lat, lng),
            (t_lat, t_lng),
            f"{name}({label})",
        )
        straight = haversine_distance(lat, lng, t_lat, t_lng)
        if duration is not None and duration < min_duration:
            min_duration = duration
            min_distance = distance
            min_straight = straight
            nearest = (name, label, source)

    if nearest is None:
        return None, "경로 없음", "없음", np.nan, np.nan

    walk_minutes = round(float(min_duration), 1)
    return (
        walk_minutes,
        nearest[0],
        nearest[1],
        float(min_distance),
        float(min_straight),
    )


def enrich_dataset(
    csv_path: Path,
    output_path: Path,
    sleep: float = 0.2,
    radius: int = 600,
    limit: Optional[int] = None,
) -> None:
    df = pd.read_csv(csv_path)

    base_columns = [
        "상호명",
        "상권업종소분류명",
        "시군구명",
        "행정동코드",
        "행정동명",
        "법정동코드",
        "법정동명",
        "지번주소",
        "도로명주소",
        "층정보",
        "경도",
        "위도",
    ]
    missing_cols = [col for col in base_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {', '.join(missing_cols)}")
    df = df[base_columns].copy()

    if limit is not None:
        df = df.head(limit).reset_index(drop=True)

    api_key = load_api_key()
    gmaps_client = googlemaps.Client(key=api_key)

    results = {
        "nearest_station": [],
        "transit_type": [],
        "walk_time_minutes": [],
        "walk_distance_m": [],
        "straight_distance_m": [],
    }

    total = len(df)
    for idx, row in df.iterrows():
        lat = float(row["위도"])
        lng = float(row["경도"])
        walk_time, station_name, transit_type, walk_distance, straight_distance = calculate_transit_accessibility(
            gmaps_client,
            lat,
            lng,
            radius=radius,
        )
        results["nearest_station"].append(station_name)
        results["transit_type"].append(transit_type)
        results["walk_time_minutes"].append(walk_time)
        results["walk_distance_m"].append(walk_distance)
        results["straight_distance_m"].append(straight_distance)

        if (idx + 1) % 25 == 0 or idx == total - 1:
            print(f"[INFO] {idx + 1}/{total} 처리 완료")
        time.sleep(sleep)

    for key, values in results.items():
        df[key] = values

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] 저장 완료 → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="카페 표본 CSV에 접근성 정보를 추가합니다.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("서울시_상권_카페빵_표본.csv"),
        help="입력 CSV 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("서울시_상권_카페빵_표본_with_transit.csv"),
        help="결과 CSV 경로",
    )
    parser.add_argument("--radius", type=int, default=600, help="Places 검색 반경 (미터)")
    parser.add_argument("--sleep", type=float, default=0.2, help="API 호출 간 대기 시간(초)")
    parser.add_argument("--limit", type=int, default=None, help="상위 N개 행만 처리 (테스트용)")
    args = parser.parse_args()

    try:
        enrich_dataset(
            args.input,
            args.output,
            sleep=args.sleep,
            radius=args.radius,
            limit=args.limit,
        )
    except Exception as exc:
        print(f"[ERROR] 처리 중 오류 발생: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

