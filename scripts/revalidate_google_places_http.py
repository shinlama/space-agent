from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import pandas as pd
import requests

from revalidate_google_places import (
    DISTRICT,
    NAME,
    OUTPUT_COLUMNS,
    ROAD_ADDRESS,
    SOURCE_LAT,
    SOURCE_LNG,
    STORE_ID,
    address_similarity,
    append_result,
    haversine_m,
    name_similarity,
    normalize_name,
    processed_store_ids,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_TARGETS = Path(r"F:\JW\space-agent\outputs\google_place_revalidation_targets.csv")
DEFAULT_OUTPUT = Path(r"F:\JW\space-agent\outputs\google_place_revalidation_http.csv")
CID_PATTERN = re.compile(r"0x[0-9a-f]+:0x[0-9a-f]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Google 지도 데이터 응답으로 의심 장소를 빠르게 재검증합니다."
    )
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--store-id", action="append", default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay-min", type=float, default=0.05)
    parser.add_argument("--delay-max", type=float, default=0.15)
    return parser.parse_args()


def is_coordinate_list(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < 4:
        return False
    try:
        lat = float(value[2])
        lng = float(value[3])
    except (TypeError, ValueError):
        return False
    return 30 <= lat <= 40 and 120 <= lng <= 132


def candidate_from_record(record: list[Any]) -> dict[str, Any] | None:
    if len(record) < 12:
        return None
    cid = record[10] if len(record) > 10 and isinstance(record[10], str) else ""
    name = record[11] if len(record) > 11 and isinstance(record[11], str) else ""
    coordinates = record[9] if len(record) > 9 and is_coordinate_list(record[9]) else None
    if not CID_PATTERN.fullmatch(cid) or not name or coordinates is None:
        return None
    address = ""
    if len(record) > 39 and isinstance(record[39], str):
        address = record[39]
    elif len(record) > 2 and isinstance(record[2], list) and record[2]:
        address = str(record[2][0] or "")
    place_id = record[78] if len(record) > 78 and isinstance(record[78], str) else ""
    categories: list[str] = []
    if len(record) > 13 and isinstance(record[13], list):
        categories = [str(value) for value in record[13] if isinstance(value, str)]
    lat = float(coordinates[2])
    lng = float(coordinates[3])
    query = f"{name} {address}".strip()
    params = {"api": "1", "query": query}
    if place_id.startswith("ChIJ"):
        params["query_place_id"] = place_id
    return {
        "name": name,
        "address": address,
        "lat": lat,
        "lng": lng,
        "cid": cid,
        "place_id": place_id,
        "categories": categories,
        "url": "https://www.google.com/maps/search/?" + urlencode(params),
    }


def parse_candidates(payload: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, list):
            candidate = candidate_from_record(value)
            if candidate is not None and candidate["cid"] not in seen:
                seen.add(candidate["cid"])
                candidates.append(candidate)
                return
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(payload)
    return candidates


def fetch_candidates(
    query: str,
    source_lat: Any,
    source_lng: Any,
    timeout: int,
    retries: int,
) -> list[dict[str, Any]]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            maps_url = (
                f"https://www.google.com/maps/search/{quote(query)}/"
                f"@{float(source_lat)},{float(source_lng)},17z?hl=ko"
            )
            page = requests.get(
                maps_url,
                headers=headers,
                timeout=timeout,
            )
            page.raise_for_status()
            if "unusual traffic" in page.text.lower():
                raise RuntimeError("Google unusual traffic 응답")
            link_match = re.search(r'<link href="([^"]*tbm=map[^"]+)"', page.text)
            if link_match is None:
                raise RuntimeError("Google 지도 데이터 링크를 찾지 못함")
            data_url = "https://www.google.com" + html.unescape(link_match.group(1))
            response = requests.get(data_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            if "unusual traffic" in response.text.lower():
                raise RuntimeError("Google unusual traffic 응답")
            body = response.text.split("\n", 1)[1] if "\n" in response.text else response.text
            payload = json.loads(body)
            return parse_candidates(payload)
        except (requests.RequestException, json.JSONDecodeError, RuntimeError, TypeError, ValueError) as exc:
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Google 지도 데이터 요청 실패: {last_error}")


def distance_score(distance: float | None) -> float:
    if distance is None:
        return 0.0
    if distance <= 100:
        return 1.0
    if distance <= 300:
        return 0.85
    if distance <= 700:
        return 0.55
    if distance <= 1500:
        return 0.20
    return 0.0


def road_address_key(value: Any) -> tuple[str, str] | None:
    text = re.sub(r"\s+", "", str(value or ""))
    match = re.search(r"([가-힣0-9]+(?:대로|로|길))(\d+(?:-\d+)?)", text)
    if match is None:
        return None
    return match.group(1), match.group(2)


def variant_tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    tokens = {
        f"floor:{'b' if match.group(1) else ''}{match.group(2)}"
        for match in re.finditer(r"(b\s*)?(\d+)\s*(?:층|f)", text)
    }
    tokens.update(
        f"branch:{match.group(1)}"
        for match in re.finditer(r"(\d+)\s*호점", text)
    )
    return tokens


def canonical_identity_name(value: Any) -> str:
    """Normalize common chain-name variants without removing branch identity."""
    text = re.sub(r"[^0-9a-z가-힣]", "", str(value or "").lower())
    replacements = (
        ("롯데지알에스", ""),
        ("비알던킨도너츠", "던킨"),
        ("던킨도너츠", "던킨"),
        ("커피빈코리아", "커피빈"),
        ("메가엠지씨커피", "메가커피"),
        ("메가mgc커피", "메가커피"),
        ("메가엠지씨", "메가"),
        ("메가mgc", "메가"),
        ("매머드익스프레스", "매머드"),
        ("매머드커피", "매머드"),
        ("엔제리너스커피", "엔제리너스"),
        ("이디야커피", "이디야"),
        ("컴포즈커피", "컴포즈"),
        ("할리스커피", "할리스"),
        ("스타벅스커피", "스타벅스"),
        ("투썸플레이스", "투썸"),
        ("파리바게트", "파리바게뜨"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


CHAIN_ANCHORS = (
    "메가커피",
    "매머드",
    "엔제리너스",
    "이디야",
    "컴포즈",
    "할리스",
    "스타벅스",
    "투썸",
    "파리바게뜨",
    "커피빈",
    "빽다방",
    "더벤티",
    "폴바셋",
    "탐앤탐스",
    "공차",
    "설빙",
    "던킨",
    "뚜레쥬르",
)


def name_identity_match(source: Any, candidate: Any, road_number_match: bool) -> bool:
    """Return True only when the two names identify the same business with high precision."""
    source_name = canonical_identity_name(source)
    candidate_name = canonical_identity_name(candidate)
    if not source_name or not candidate_name:
        return False
    if source_name == candidate_name:
        return True

    source_anchors = {anchor for anchor in CHAIN_ANCHORS if anchor in source_name}
    candidate_anchors = {anchor for anchor in CHAIN_ANCHORS if anchor in candidate_name}
    if (source_anchors or candidate_anchors) and not (
        source_anchors & candidate_anchors
    ):
        return False

    shorter, longer = sorted((source_name, candidate_name), key=len)
    if shorter in longer:
        if len(shorter) >= 4:
            return True
        return len(shorter) >= 2 and road_number_match

    if source_anchors & candidate_anchors:
        return True

    matcher = SequenceMatcher(None, source_name, candidate_name)
    longest = matcher.find_longest_match(0, len(source_name), 0, len(candidate_name))
    shared_prefix_identity = (
        longest.size >= 3
        and longest.a <= 1
        and longest.b <= 1
    )
    if road_number_match and shared_prefix_identity:
        return True
    return road_number_match and matcher.ratio() >= 0.86


def enrich_candidate(candidate: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    name_score = name_similarity(row[NAME], candidate["name"])
    address_score = address_similarity(row[ROAD_ADDRESS], candidate["address"])
    distance = haversine_m(
        row[SOURCE_LAT],
        row[SOURCE_LNG],
        candidate["lat"],
        candidate["lng"],
    )
    source_normalized = normalize_name(row[NAME])
    candidate_normalized = normalize_name(candidate["name"])
    short_name_alignment = (
        len(source_normalized) >= 2
        and source_normalized not in {"카페", "커피", "베이커리"}
        and (
            candidate_normalized.startswith(source_normalized)
            or candidate_normalized.endswith(source_normalized)
        )
    )
    category_text = " ".join(candidate.get("categories", [])).lower()
    category_match = any(
        token in category_text
        for token in (
            "커피",
            "카페",
            "제과",
            "베이커리",
            "빵",
            "디저트",
            "도넛",
            "아이스크림",
            "초콜릿",
            "찻집",
            "tea house",
            "coffee",
            "cafe",
            "bakery",
            "dessert",
        )
    )
    source_road_key = road_address_key(row[ROAD_ADDRESS])
    candidate_road_key = road_address_key(candidate["address"])
    road_number_match = (
        source_road_key is not None
        and candidate_road_key is not None
        and source_road_key == candidate_road_key
    )
    source_variants = variant_tokens(row[NAME])
    candidate_variants = variant_tokens(candidate["name"])
    variant_conflict = bool(
        source_variants
        and candidate_variants
        and source_variants.isdisjoint(candidate_variants)
    )
    identity_match = name_identity_match(
        row[NAME], candidate["name"], road_number_match
    )
    score = (
        name_score * 0.40
        + address_score * 0.15
        + distance_score(distance) * 0.20
        + (0.25 if identity_match else 0.0)
    )
    return {
        **candidate,
        "name_score": name_score,
        "address_score": address_score,
        "distance": distance,
        "score": score,
        "short_name_alignment": short_name_alignment,
        "category_match": category_match,
        "road_number_match": road_number_match,
        "variant_conflict": variant_conflict,
        "identity_match": identity_match,
    }


def is_verified(candidate: dict[str, Any]) -> bool:
    distance = candidate["distance"]
    if (
        not candidate["category_match"]
        or candidate["variant_conflict"]
        or not candidate["identity_match"]
    ):
        return False
    return (
        distance is not None
        and distance <= 50
        and candidate["name_score"] >= 0.80
        and candidate["address_score"] >= 0.45
    ) or (
        distance is not None
        and distance <= 100
        and candidate["name_score"] >= 0.72
        and candidate["address_score"] >= 0.60
    ) or (
        distance is not None
        and distance <= 200
        and candidate["name_score"] >= 0.90
        and candidate["address_score"] >= 0.75
    ) or (
        distance is not None
        and distance <= 1000
        and candidate["name_score"] >= 0.72
        and candidate["road_number_match"]
    ) or (
        distance is not None
        and distance <= 50
        and candidate["address_score"] >= 0.90
        and candidate["short_name_alignment"]
    )


def classify_action(candidate: dict[str, Any] | None) -> str:
    if candidate is None:
        return "manual_review"
    if is_verified(candidate):
        return "recrawl_corrected"
    if (
        candidate["distance"] is not None
        and candidate["distance"] <= 150
        and candidate["address_score"] >= 0.80
        and candidate["name_score"] < 0.20
    ):
        return "exclude_stale_or_replaced"
    return "manual_review"


def process_target(
    row: dict[str, Any],
    timeout: int,
    retries: int,
    delay_min: float,
    delay_max: float,
) -> dict[str, Any]:
    query = str(row[NAME]).strip()
    result = {
        STORE_ID: str(row[STORE_ID]),
        NAME: row[NAME],
        DISTRICT: row[DISTRICT],
        ROAD_ADDRESS: row[ROAD_ADDRESS],
        "source_lat": row[SOURCE_LAT],
        "source_lng": row[SOURCE_LNG],
        "existing_place_slug": row.get("existing_place_slug", ""),
        "existing_place_name": row.get("existing_place_name", ""),
        "existing_name_similarity": row.get("existing_name_similarity", ""),
        "google_name": "",
        "google_address": "",
        "google_lat": "",
        "google_lng": "",
        "distance_m": "",
        "name_similarity": "",
        "address_similarity": "",
        "match_status": "flagged_existing_slug",
        "correction_action": "manual_review",
        "corrected_google_name": "",
        "corrected_google_address": "",
        "corrected_google_categories": "",
        "corrected_google_lat": "",
        "corrected_google_lng": "",
        "corrected_distance_m": "",
        "corrected_name_similarity": "",
        "corrected_address_similarity": "",
        "corrected_google_url": "",
        "selection_mode": "google_maps_http_search",
        "google_url": "",
        "query": query,
        "error": "",
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        query_variants = [
            f"{query} {row.get(ROAD_ADDRESS, '')}".strip(),
            f"{query} {row.get(DISTRICT, '')}".strip(),
            query,
        ]
        candidates = []
        used_query = query
        for query_variant in dict.fromkeys(query_variants):
            used_query = query_variant
            candidates = fetch_candidates(
                query_variant,
                source_lat=row[SOURCE_LAT],
                source_lng=row[SOURCE_LNG],
                timeout=timeout,
                retries=retries,
            )
            if candidates:
                break
        result["query"] = used_query
        enriched = sorted(
            (enrich_candidate(candidate, row) for candidate in candidates),
            key=lambda candidate: candidate["score"],
            reverse=True,
        )
        selected = enriched[0] if enriched else None
        action = classify_action(selected)
        result["correction_action"] = action
        if selected is not None:
            result.update(
                {
                    "corrected_google_name": selected["name"],
                    "corrected_google_address": selected["address"],
                    "corrected_google_categories": " | ".join(selected.get("categories", [])),
                    "corrected_google_lat": selected["lat"],
                    "corrected_google_lng": selected["lng"],
                    "corrected_distance_m": round(selected["distance"], 1)
                    if selected["distance"] is not None
                    else "",
                    "corrected_name_similarity": round(selected["name_score"], 4),
                    "corrected_address_similarity": round(selected["address_score"], 4),
                    "corrected_google_url": selected["url"],
                    "google_url": selected["url"],
                }
            )
    except Exception as exc:  # noqa: BLE001
        result["match_status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    time.sleep(random.uniform(min(delay_min, delay_max), max(delay_min, delay_max)))
    return result


def apply_collision_guard(output_path: Path) -> int:
    frame = pd.read_csv(output_path, encoding="utf-8-sig", dtype=str).fillna("")
    recrawl = frame[
        frame["correction_action"].eq("recrawl_corrected")
        & frame["corrected_google_url"].ne("")
    ]
    collisions = set(
        recrawl.groupby("corrected_google_url")[STORE_ID]
        .nunique()
        .loc[lambda counts: counts > 1]
        .index
    )
    if not collisions:
        return 0
    mask = frame["corrected_google_url"].isin(collisions)
    frame.loc[mask, "correction_action"] = "manual_review"
    frame.loc[mask, "match_status"] = "collision_manual_review"
    frame.loc[mask, "error"] = "동일 Google 장소가 여러 원본 상가에 매핑되어 자동 처리를 보류함"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(output_path)
    return int(mask.sum())


def main() -> int:
    args = parse_args()
    targets = pd.read_csv(args.targets, encoding="utf-8-sig", low_memory=False).fillna("")
    if args.store_id:
        targets = targets[targets[STORE_ID].astype(str).isin(set(args.store_id))]
    completed = processed_store_ids(args.output)
    targets = targets[~targets[STORE_ID].astype(str).isin(completed)]
    if args.limit is not None:
        targets = targets.head(args.limit)
    rows = targets.to_dict("records")
    print(f"HTTP 재검증 대상: {len(rows):,}곳, 기존 완료: {len(completed):,}곳")
    if not rows:
        guarded = apply_collision_guard(args.output)
        if guarded:
            print(f"충돌 방지 보류: {guarded:,}곳")
        return 0

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(
            lambda row: process_target(
                row,
                timeout=args.timeout,
                retries=args.retries,
                delay_min=args.delay_min,
                delay_max=args.delay_max,
            ),
            rows,
        )
        for index, result in enumerate(results, start=1):
            append_result(args.output, result)
            if index % 50 == 0 or index == len(rows):
                elapsed = time.time() - started
                print(
                    f"진행: {index:,}/{len(rows):,}곳, "
                    f"{index / max(elapsed, 0.001):.2f}곳/초",
                    flush=True,
                )
    guarded = apply_collision_guard(args.output)
    if guarded:
        print(f"충돌 방지 보류: {guarded:,}곳")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
