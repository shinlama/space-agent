from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_SOURCE = Path(r"F:\JW\space-agent\서울시_상권_카페빵.csv")
DEFAULT_REVIEWS = Path(r"F:\JW\space-agent\google_reviews_full_min3_max2000.csv")
DEFAULT_OUTPUT = Path("outputs/google_place_revalidation.csv")

STORE_ID = "상가업소번호"
NAME = "상호명"
DISTRICT = "시군구명"
ROAD_ADDRESS = "도로명주소"
SOURCE_LAT = "위도"
SOURCE_LNG = "경도"

OUTPUT_COLUMNS = [
    STORE_ID,
    NAME,
    DISTRICT,
    ROAD_ADDRESS,
    "source_lat",
    "source_lng",
    "existing_place_slug",
    "existing_place_name",
    "existing_name_similarity",
    "google_name",
    "google_address",
    "google_lat",
    "google_lng",
    "distance_m",
    "name_similarity",
    "address_similarity",
    "match_status",
    "correction_action",
    "corrected_google_name",
    "corrected_google_address",
    "corrected_google_categories",
    "corrected_google_lat",
    "corrected_google_lng",
    "corrected_distance_m",
    "corrected_name_similarity",
    "corrected_address_similarity",
    "corrected_google_url",
    "selection_mode",
    "google_url",
    "query",
    "error",
    "checked_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Google Maps 장소 매칭을 재검증합니다.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--targets-input",
        type=Path,
        default=None,
        help="미리 생성한 재검증 대상 CSV를 사용합니다.",
    )
    parser.add_argument(
        "--prepare-targets",
        type=Path,
        default=None,
        help="재검증 대상 CSV만 생성하고 종료합니다.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("flagged", "all"),
        default="flagged",
        help="flagged는 기존 URL 장소명과 상호명이 단순 일치하지 않는 장소만 조회합니다.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--delay-min", type=float, default=1.2)
    parser.add_argument("--delay-max", type=float, default=2.2)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--restart-every", type=int, default=250)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def normalize_name(value: Any) -> str:
    text = str(value or "").lower()
    replacements = {
        "coffee": "커피",
        "cafe": "카페",
        "café": "카페",
        "bakery": "베이커리",
        "&": "앤",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def core_name(value: Any) -> str:
    text = normalize_name(value)
    for token in ("주식회사", "유한회사", "카페", "커피", "베이커리", "로스터스", "로스터리"):
        text = text.replace(token, "")
    return re.sub(r"(?:본점|[가-힣a-z0-9]+점)$", "", text)


def name_similarity(left: Any, right: Any) -> float:
    left_normalized = normalize_name(left)
    right_normalized = normalize_name(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    if left_normalized in right_normalized:
        basic = len(left_normalized) / len(right_normalized)
        return max(0.82, basic) if len(left_normalized) >= 4 else basic
    if right_normalized in left_normalized:
        return len(right_normalized) / len(left_normalized)
    basic = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    left_core = core_name(left)
    right_core = core_name(right)
    core = SequenceMatcher(None, left_core, right_core).ratio() if left_core and right_core else 0.0
    return max(basic, core)


def normalize_address(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("서울특별시", "서울").replace("대한민국", "")
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def address_similarity(left: Any, right: Any) -> float:
    left_normalized = normalize_address(left)
    right_normalized = normalize_address(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized in right_normalized or right_normalized in left_normalized:
        return min(len(left_normalized), len(right_normalized)) / max(
            len(left_normalized), len(right_normalized)
        )
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def haversine_m(lat1: Any, lng1: Any, lat2: Any, lng2: Any) -> float | None:
    try:
        values = [float(value) for value in (lat1, lng1, lat2, lng2)]
    except (TypeError, ValueError):
        return None
    lat1_f, lng1_f, lat2_f, lng2_f = values
    radius = 6_371_000.0
    phi1 = math.radians(lat1_f)
    phi2 = math.radians(lat2_f)
    delta_phi = math.radians(lat2_f - lat1_f)
    delta_lambda = math.radians(lng2_f - lng1_f)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        delta_lambda / 2
    ) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def extract_coordinates(url: str) -> tuple[float | None, float | None]:
    patterns = (
        r"@(-?\d+\.\d+),(-?\d+\.\d+)",
        r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None


def decode_place_slug(slug: Any) -> str:
    return unquote(str(slug or "")).replace("+", " ").strip()


def init_driver(headless: bool) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=ko-KR")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    return webdriver.Chrome(options=options)


def wait_until_loaded(driver: webdriver.Chrome, timeout: int) -> None:
    WebDriverWait(driver, timeout).until(
        lambda current: "/place/" in current.current_url
        or bool(current.find_elements(By.CSS_SELECTOR, "div.Nv2PK, div[role='article']"))
    )


def read_text(driver: webdriver.Chrome, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                text = (element.text or element.get_attribute("aria-label") or "").strip()
                if text:
                    return text
        except WebDriverException:
            continue
    return ""


def parse_place_page(driver: webdriver.Chrome) -> dict[str, Any]:
    url = driver.current_url
    name = read_text(driver, ("h1.DUwDvf", "h1", "div.fontHeadlineSmall"))
    if not name:
        slug_match = re.search(r"/place/([^/]+)", url)
        if slug_match:
            name = decode_place_slug(slug_match.group(1))
    address = read_text(
        driver,
        (
            "button[data-item-id='address']",
            "button[data-tooltip='주소 복사']",
            "div.Io6YTe.fontBodyMedium.kR99db.fdkmkc",
        ),
    )
    address = re.sub(r"[\ue000-\uf8ff]", " ", address)
    address = re.sub(r"^(?:주소\s*:\s*|주소\s*)", "", address)
    address = re.sub(r"\s+", " ", address).strip()
    lat, lng = extract_coordinates(url)
    return {"name": name, "address": address, "lat": lat, "lng": lng, "url": url}


def search_result_candidates(driver: webdriver.Chrome) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    cards = driver.find_elements(By.CSS_SELECTOR, "div.Nv2PK, div[role='article']")
    seen_urls: set[str] = set()
    for card in cards[:8]:
        try:
            links = card.find_elements(By.CSS_SELECTOR, "a.hfpxzc, a[href*='/maps/place/'], a[href*='/place/']")
            if not links:
                continue
            link = links[0]
            href = (link.get_attribute("href") or "").strip()
            if not href or href in seen_urls:
                continue
            seen_urls.add(href)
            name = (link.get_attribute("aria-label") or "").strip()
            if not name:
                title_elements = card.find_elements(By.CSS_SELECTOR, "div.qBF1Pd, div.fontHeadlineSmall")
                name = title_elements[0].text.strip() if title_elements else ""
            lat, lng = extract_coordinates(href)
            candidates.append(
                {
                    "name": name,
                    "context": card.text.strip(),
                    "lat": lat,
                    "lng": lng,
                    "url": href,
                }
            )
        except WebDriverException:
            continue
    return candidates


def candidate_score(candidate: dict[str, Any], row: pd.Series) -> float:
    name_score = name_similarity(row[NAME], candidate.get("name"))
    distance = haversine_m(
        row.get(SOURCE_LAT),
        row.get(SOURCE_LNG),
        candidate.get("lat"),
        candidate.get("lng"),
    )
    if distance is None:
        distance_score = 0.0
    elif distance <= 100:
        distance_score = 1.0
    elif distance <= 300:
        distance_score = 0.85
    elif distance <= 700:
        distance_score = 0.55
    elif distance <= 1500:
        distance_score = 0.2
    else:
        distance_score = 0.0
    context_score = address_similarity(row[ROAD_ADDRESS], candidate.get("context"))
    return name_score * 0.75 + distance_score * 0.20 + context_score * 0.05


def choose_search_result(driver: webdriver.Chrome, row: pd.Series) -> dict[str, Any] | None:
    candidates = search_result_candidates(driver)
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda item: candidate_score(item, row), reverse=True)
    return ranked[0]


def classify_match(
    name_score: float,
    address_score: float,
    distance: float | None,
) -> str:
    if distance is not None:
        if distance <= 200 and name_score >= 0.72:
            return "matched"
        if distance <= 100 and name_score >= 0.55 and address_score >= 0.50:
            return "matched"
        if distance <= 300 and name_score >= 0.65 and address_score >= 0.45:
            return "probable"
        if distance >= 1000:
            return "mismatch"
        if distance <= 300 and name_score < 0.25:
            return "mismatch"
    if name_score >= 0.82 and address_score >= 0.45:
        return "matched"
    if name_score >= 0.72:
        return "probable"
    if name_score < 0.25 and address_score < 0.25:
        return "mismatch"
    return "uncertain"


def build_query(row: pd.Series, query_name: str | None = None) -> str:
    return " ".join(
        part
        for part in (
            str(query_name or "").strip()
            or str(row.get("existing_place_name", "")).strip()
            or str(row.get(NAME, "")).strip(),
            str(row.get(ROAD_ADDRESS, "")).strip(),
        )
        if part
    )


def inspect_place(
    driver: webdriver.Chrome,
    row: pd.Series,
    timeout: int,
    query_name: str | None = None,
) -> dict[str, Any]:
    query = build_query(row, query_name=query_name)
    try:
        source_lat = float(row.get(SOURCE_LAT))
        source_lng = float(row.get(SOURCE_LNG))
        search_url = (
            f"https://www.google.com/maps/search/{quote(query)}/"
            f"@{source_lat},{source_lng},17z?hl=ko"
        )
    except (TypeError, ValueError):
        search_url = f"https://www.google.com/maps/search/{quote(query)}?hl=ko"
    driver.set_page_load_timeout(max(timeout + 15, 30))
    try:
        driver.get(search_url)
    except TimeoutException:
        driver.execute_script("window.stop();")
    wait_until_loaded(driver, timeout)

    selection_mode = "direct"
    if "/place/" not in driver.current_url:
        selected = choose_search_result(driver, row)
        if selected is None:
            raise RuntimeError("검색 결과에서 장소 후보를 찾지 못했습니다.")
        selection_mode = "ranked_search_result"
        driver.get(selected["url"])
        WebDriverWait(driver, timeout).until(lambda current: "/place/" in current.current_url)

    place = parse_place_page(driver)
    place["selection_mode"] = selection_mode
    place["query"] = query
    return place


def load_targets(source_path: Path, reviews_path: Path, candidate_mode: str) -> pd.DataFrame:
    source = pd.read_csv(source_path, encoding="utf-8-sig", low_memory=False)
    required_source = {STORE_ID, NAME, DISTRICT, ROAD_ADDRESS, SOURCE_LAT, SOURCE_LNG}
    missing_source = required_source - set(source.columns)
    if missing_source:
        raise ValueError(f"상가 원본에 필요한 컬럼이 없습니다: {sorted(missing_source)}")

    reviews = pd.read_csv(
        reviews_path,
        encoding="utf-8-sig",
        dtype=str,
        usecols=[STORE_ID, "place_id"],
    ).drop_duplicates(STORE_ID)
    reviews = reviews.rename(columns={"place_id": "existing_place_slug"})
    targets = source.merge(reviews, on=STORE_ID, how="inner")
    targets["existing_place_name"] = targets["existing_place_slug"].map(decode_place_slug)
    targets["existing_name_similarity"] = targets.apply(
        lambda row: name_similarity(row[NAME], row["existing_place_name"]), axis=1
    )

    if candidate_mode == "flagged":
        targets = targets[
            ~targets.apply(
                lambda row: normalize_name(row[NAME]) in normalize_name(row["existing_place_name"])
                or normalize_name(row["existing_place_name"]) in normalize_name(row[NAME]),
                axis=1,
            )
        ]
        targets = targets.sort_values("existing_name_similarity", ascending=True)
    return targets.reset_index(drop=True)


def correction_action(
    initial_status: str,
    initial_distance: float | None,
    corrected_name_score: float,
    corrected_address_score: float,
    corrected_distance: float | None,
) -> str:
    if initial_status == "matched":
        return "keep_existing"
    corrected_is_valid = (
        corrected_distance is not None
        and corrected_distance <= 200
        and corrected_name_score >= 0.72
        and corrected_address_score >= 0.45
    ) or (
        corrected_distance is not None
        and corrected_distance <= 500
        and corrected_name_score >= 0.82
        and corrected_address_score >= 0.60
    )
    if corrected_is_valid:
        return "recrawl_corrected"
    if initial_distance is not None and initial_distance <= 300:
        return "exclude_stale_or_replaced"
    return "manual_review"


def processed_store_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    try:
        frame = pd.read_csv(output_path, encoding="utf-8-sig", dtype=str, usecols=[STORE_ID])
    except (ValueError, pd.errors.EmptyDataError):
        return set()
    return set(frame[STORE_ID].dropna().astype(str))


def append_result(output_path: Path, result: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def main() -> int:
    args = parse_args()
    if args.targets_input is not None:
        targets = pd.read_csv(args.targets_input, encoding="utf-8-sig", low_memory=False)
    else:
        targets = load_targets(args.source, args.reviews, args.candidate_mode)
    if args.prepare_targets is not None:
        args.prepare_targets.parent.mkdir(parents=True, exist_ok=True)
        targets.to_csv(args.prepare_targets, index=False, encoding="utf-8-sig")
        print(f"재검증 대상 목록 저장: {args.prepare_targets} ({len(targets):,}곳)")
        return 0
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index는 0 이상 shard-count 미만이어야 합니다.")
    targets = targets.iloc[args.shard_index :: args.shard_count].reset_index(drop=True)
    completed = processed_store_ids(args.output)
    targets = targets[~targets[STORE_ID].astype(str).isin(completed)]
    if args.limit is not None:
        targets = targets.head(args.limit)

    print(f"재검증 대상: {len(targets):,}곳, 기존 완료: {len(completed):,}곳")
    if targets.empty:
        return 0

    driver: webdriver.Chrome | None = None
    processed = 0
    try:
        for _, row in targets.iterrows():
            if driver is None or (args.restart_every and processed and processed % args.restart_every == 0):
                if driver is not None:
                    driver.quit()
                driver = init_driver(headless=not args.no_headless)

            result = {
                STORE_ID: str(row[STORE_ID]),
                NAME: str(row[NAME]),
                DISTRICT: str(row[DISTRICT]),
                ROAD_ADDRESS: str(row[ROAD_ADDRESS]),
                "source_lat": row[SOURCE_LAT],
                "source_lng": row[SOURCE_LNG],
                "existing_place_slug": row["existing_place_slug"],
                "existing_place_name": row["existing_place_name"],
                "existing_name_similarity": round(float(row["existing_name_similarity"]), 4),
                "google_name": "",
                "google_address": "",
                "google_lat": "",
                "google_lng": "",
                "distance_m": "",
                "name_similarity": "",
                "address_similarity": "",
                "match_status": "error",
                "correction_action": "manual_review",
                "corrected_google_name": "",
                "corrected_google_address": "",
                "corrected_google_lat": "",
                "corrected_google_lng": "",
                "corrected_distance_m": "",
                "corrected_name_similarity": "",
                "corrected_address_similarity": "",
                "corrected_google_url": "",
                "selection_mode": "",
                "google_url": "",
                "query": build_query(row),
                "error": "",
                "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            try:
                place = inspect_place(
                    driver,
                    row,
                    args.timeout,
                    query_name=str(row.get("existing_place_name", "")).strip(),
                )
                distance = haversine_m(
                    row[SOURCE_LAT], row[SOURCE_LNG], place["lat"], place["lng"]
                )
                name_score = name_similarity(row[NAME], place["name"])
                address_score = address_similarity(row[ROAD_ADDRESS], place["address"])
                initial_status = classify_match(name_score, address_score, distance)
                result.update(
                    {
                        "google_name": place["name"],
                        "google_address": place["address"],
                        "google_lat": place["lat"] if place["lat"] is not None else "",
                        "google_lng": place["lng"] if place["lng"] is not None else "",
                        "distance_m": round(distance, 1) if distance is not None else "",
                        "name_similarity": round(name_score, 4),
                        "address_similarity": round(address_score, 4),
                        "match_status": initial_status,
                        "correction_action": "keep_existing"
                        if initial_status == "matched"
                        else "manual_review",
                        "selection_mode": place["selection_mode"],
                        "google_url": place["url"],
                        "query": place["query"],
                    }
                )

                if initial_status in {"mismatch", "uncertain", "probable"}:
                    corrected = inspect_place(
                        driver,
                        row,
                        args.timeout,
                        query_name=str(row[NAME]).strip(),
                    )
                    corrected_distance = haversine_m(
                        row[SOURCE_LAT],
                        row[SOURCE_LNG],
                        corrected["lat"],
                        corrected["lng"],
                    )
                    corrected_name_score = name_similarity(row[NAME], corrected["name"])
                    corrected_address_score = address_similarity(
                        row[ROAD_ADDRESS], corrected["address"]
                    )
                    result.update(
                        {
                            "correction_action": correction_action(
                                initial_status,
                                distance,
                                corrected_name_score,
                                corrected_address_score,
                                corrected_distance,
                            ),
                            "corrected_google_name": corrected["name"],
                            "corrected_google_address": corrected["address"],
                            "corrected_google_lat": corrected["lat"]
                            if corrected["lat"] is not None
                            else "",
                            "corrected_google_lng": corrected["lng"]
                            if corrected["lng"] is not None
                            else "",
                            "corrected_distance_m": round(corrected_distance, 1)
                            if corrected_distance is not None
                            else "",
                            "corrected_name_similarity": round(corrected_name_score, 4),
                            "corrected_address_similarity": round(corrected_address_score, 4),
                            "corrected_google_url": corrected["url"],
                        }
                    )
            except Exception as exc:  # Resume-safe row-level failure logging.
                result["error"] = f"{type(exc).__name__}: {exc}"[:500]

            append_result(args.output, result)
            processed += 1
            print(
                f"[{processed:,}/{len(targets):,}] {result[NAME]} -> "
                f"{result['google_name'] or '-'} ({result['match_status']}, "
                f"{result['correction_action']})"
            )
            time.sleep(random.uniform(min(args.delay_min, args.delay_max), max(args.delay_min, args.delay_max)))
    finally:
        if driver is not None:
            driver.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
