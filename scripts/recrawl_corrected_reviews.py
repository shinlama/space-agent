from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_REVALIDATION = Path(r"F:\JW\space-agent\outputs\google_place_revalidation_v2.csv")
DEFAULT_REVIEWS = Path(r"F:\JW\space-agent\google_reviews_full_min3_max2000.csv")
DEFAULT_OUTPUT = Path(r"F:\JW\space-agent\outputs\google_reviews_recrawled_corrected.csv")
DEFAULT_PROGRESS = Path(r"F:\JW\space-agent\outputs\google_reviews_recrawled_corrected_progress.csv")

STORE_ID = "상가업소번호"
NAME = "상호명"

REVIEW_COLUMNS = [
    "상가업소번호",
    "상호명",
    "시군구명",
    "행정동명",
    "도로명주소",
    "place_id",
    "lat",
    "lng",
    "작성자",
    "평점",
    "리뷰",
    "작성일",
    "언어",
    "review_id",
    "google_url",
    "collected_at",
]

PROGRESS_COLUMNS = [
    STORE_ID,
    NAME,
    "google_name",
    "status",
    "review_cards",
    "text_reviews",
    "rating_only_cards",
    "google_url",
    "error",
    "checked_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="재검증에서 확정된 Google Maps 장소의 텍스트 리뷰만 다시 수집합니다."
    )
    parser.add_argument("--revalidation", type=Path, default=DEFAULT_REVALIDATION)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument(
        "--resume-progress", type=Path, action="append", default=[]
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-reviews", type=int, default=2000)
    parser.add_argument("--max-scrolls", type=int, default=350)
    parser.add_argument("--scroll-pause", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--no-headless", action="store_true")
    return parser.parse_args()


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


def append_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def completed_store_ids(progress_path: Path) -> set[str]:
    if not progress_path.exists() or progress_path.stat().st_size == 0:
        return set()
    progress = pd.read_csv(progress_path, encoding="utf-8-sig", dtype=str)
    completed = progress[progress["status"] == "completed"]
    return set(completed[STORE_ID].dropna().astype(str))


def load_source_metadata(review_path: Path, store_ids: set[str]) -> dict[str, dict[str, Any]]:
    columns = ["상가업소번호", "상호명", "시군구명", "행정동명", "도로명주소"]
    found: dict[str, dict[str, Any]] = {}
    for chunk in pd.read_csv(
        review_path,
        encoding="utf-8-sig",
        dtype=str,
        usecols=columns,
        chunksize=100_000,
    ):
        selected = chunk[chunk[STORE_ID].isin(store_ids)].drop_duplicates(STORE_ID)
        for row in selected.to_dict("records"):
            found[str(row[STORE_ID])] = row
        if len(found) == len(store_ids):
            break
    return found


def dismiss_consent(driver: webdriver.Chrome) -> None:
    labels = ("모두 수락", "Accept all", "동의", "I agree")
    for label in labels:
        try:
            buttons = driver.find_elements(By.XPATH, f"//button[contains(., '{label}')]")
            if buttons:
                driver.execute_script("arguments[0].click();", buttons[0])
                time.sleep(1)
                return
        except WebDriverException:
            continue


def normalize_name(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("coffee", "커피").replace("cafe", "카페")
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def navigate_to_place(driver: webdriver.Chrome, row: dict[str, Any], timeout: int) -> None:
    name = str(row.get("corrected_google_name", "")).strip()
    address = str(row.get("corrected_google_address", "")).strip()
    lat = str(row.get("corrected_google_lat", "")).strip()
    lng = str(row.get("corrected_google_lng", "")).strip()
    corrected_url = str(row.get("corrected_google_url", "")).strip()
    if corrected_url:
        separator = "&" if "?" in corrected_url else "?"
        driver.get(f"{corrected_url}{separator}hl=ko")
    else:
        query = quote(f"{name} {address}".strip())
        center = f"/@{lat},{lng},17z" if lat and lng else ""
        driver.get(f"https://www.google.com/maps/search/{query}{center}?hl=ko")
    dismiss_consent(driver)
    WebDriverWait(driver, timeout).until(
        lambda current: "/place/" in current.current_url
        or bool(current.find_elements(By.CSS_SELECTOR, "a.hfpxzc, div.Nv2PK"))
    )
    if "/place/" not in driver.current_url:
        links = driver.find_elements(
            By.CSS_SELECTOR, "a.hfpxzc, div.Nv2PK a[href*='/maps/place/']"
        )
        expected = normalize_name(name)
        chosen = None
        for link in links:
            candidate = normalize_name(link.get_attribute("aria-label") or link.text)
            if expected and (expected in candidate or candidate in expected):
                chosen = link
                break
        if chosen is None and links:
            chosen = links[0]
        if chosen is None:
            raise RuntimeError("확정 장소 검색 결과를 찾지 못했습니다.")
        driver.execute_script("arguments[0].click();", chosen)
    WebDriverWait(driver, timeout).until(
        lambda current: bool(current.find_elements(By.CSS_SELECTOR, "h1.DUwDvf, h1"))
    )
    loaded_name = first_text(driver, ("h1.DUwDvf", "h1"))
    expected = normalize_name(name)
    loaded = normalize_name(loaded_name)
    if expected and loaded and expected not in loaded and loaded not in expected:
        raise RuntimeError(f"확정 장소와 다른 페이지가 열렸습니다: {loaded_name}")


def open_reviews(driver: webdriver.Chrome, timeout: int) -> bool:
    selectors = (
        "button[role='tab'][aria-label*='리뷰']",
        "button[role='tab'][aria-label*='Reviews']",
        "button[aria-label*='리뷰'][class*='hh2c6']",
        "button[aria-label*='Reviews'][class*='hh2c6']",
        "button[data-value='리뷰']",
        "button[data-value='Reviews']",
        "button[data-tab-index='1']",
    )
    for selector in selectors:
        try:
            buttons = driver.find_elements(By.CSS_SELECTOR, selector)
            for button in buttons:
                label = (button.get_attribute("aria-label") or "").strip()
                if "더보기" in label or "more" in label.lower():
                    continue
                driver.execute_script("arguments[0].click();", button)
                WebDriverWait(driver, timeout).until(
                    lambda current: bool(current.find_elements(By.CSS_SELECTOR, "div.jftiEf"))
                    or bool(current.find_elements(By.CSS_SELECTOR, "div[role='feed']"))
                )
                return True
        except WebDriverException:
            continue
    xpath_selectors = (
        "//button[contains(., '리뷰') and not(contains(., '더보기'))]",
        "//button[contains(., 'Reviews') and not(contains(., 'More'))]",
        "//*[@role='tab'][contains(., '리뷰') or contains(@aria-label, '리뷰')]",
        "//*[@role='tab'][contains(., 'Reviews') or contains(@aria-label, 'Reviews')]",
    )
    for selector in xpath_selectors:
        try:
            buttons = driver.find_elements(By.XPATH, selector)
            for button in buttons:
                driver.execute_script("arguments[0].click();", button)
                WebDriverWait(driver, timeout).until(
                    lambda current: bool(current.find_elements(By.CSS_SELECTOR, "div.jftiEf"))
                    or bool(current.find_elements(By.CSS_SELECTOR, "div[role='feed']"))
                )
                return True
        except WebDriverException:
            continue
    if not driver.find_elements(By.CSS_SELECTOR, "div.jftiEf"):
        diagnostics: list[str] = []
        for button in driver.find_elements(By.CSS_SELECTOR, "button")[:80]:
            text = re.sub(r"\s+", " ", (button.text or "").strip())
            aria = (button.get_attribute("aria-label") or "").strip()
            if text or aria:
                diagnostics.append(f"text={text!r}, aria={aria!r}")
        has_place_overview = bool(
            driver.find_elements(
                By.CSS_SELECTOR,
                "button[aria-label*='개요'], button[aria-label*='Overview']",
            )
        )
        if has_place_overview or driver.find_elements(By.CSS_SELECTOR, "h1.DUwDvf, h1"):
            return False
        raise RuntimeError(
            "리뷰 탭을 열지 못했습니다. 확인된 버튼: " + " | ".join(diagnostics[:20])
        )
    return True


def review_cards(driver: webdriver.Chrome) -> list[Any]:
    return driver.find_elements(By.CSS_SELECTOR, "div.jftiEf")


def scroll_all_reviews(
    driver: webdriver.Chrome,
    max_reviews: int,
    max_scrolls: int,
    pause: float,
) -> list[Any]:
    unchanged = 0
    previous = -1
    for _ in range(max_scrolls):
        cards = review_cards(driver)
        count = len(cards)
        if count >= max_reviews:
            break
        if count == previous:
            unchanged += 1
        else:
            unchanged = 0
            previous = count
        if unchanged >= 3:
            break
        panel = driver.execute_script(
            """
            const cards = Array.from(document.querySelectorAll('div.jftiEf'));
            let node = cards.length ? cards[cards.length - 1] : null;
            while (node && node !== document.body) {
                if (node.scrollHeight > node.clientHeight + 80) return node;
                node = node.parentElement;
            }
            return document.querySelector("div[role='feed']");
            """
        )
        if panel:
            driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollHeight;", panel
            )
        elif cards:
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'end'});", cards[-1]
            )
        time.sleep(pause)
    return review_cards(driver)[:max_reviews]


def expand_review_texts(driver: webdriver.Chrome, cards: list[Any]) -> None:
    for card in cards:
        try:
            buttons = card.find_elements(
                By.CSS_SELECTOR,
                "button.w8nwRe, button[aria-label='더보기'], button[aria-label='More']",
            )
            for button in buttons:
                driver.execute_script("arguments[0].click();", button)
        except WebDriverException:
            continue


def first_text(element: Any, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        try:
            matches = element.find_elements(By.CSS_SELECTOR, selector)
            for match in matches:
                text = (match.get_attribute("textContent") or match.text or "").strip()
                if text:
                    return re.sub(r"\s+", " ", text)
        except WebDriverException:
            continue
    return ""


def extract_rating(card: Any) -> int | None:
    try:
        elements = card.find_elements(By.CSS_SELECTOR, "span.kvMYJc, span[role='img'][aria-label*='별표']")
        for element in elements:
            label = element.get_attribute("aria-label") or ""
            match = re.search(r"([0-5](?:\.\d+)?)", label)
            if match:
                return int(round(float(match.group(1))))
    except WebDriverException:
        pass
    return None


def detect_language(text: str) -> str:
    letters = re.findall(r"[A-Za-z가-힣]", text)
    if not letters:
        return ""
    korean = len(re.findall(r"[가-힣]", text))
    return "ko" if korean / len(letters) >= 0.25 else "other"


def extract_review_id(card: Any, author: str, date: str, text: str) -> str:
    review_id = (card.get_attribute("data-review-id") or "").strip()
    if review_id:
        return review_id
    digest = hashlib.sha1(f"{author}\n{date}\n{text}".encode("utf-8")).hexdigest()
    return f"sha1:{digest}"


def place_slug(url: str) -> str:
    match = re.search(r"/place/([^/]+)", url)
    return match.group(1) if match else ""


def extract_reviews(
    cards: list[Any],
    metadata: dict[str, Any],
    row: dict[str, Any],
    final_url: str,
) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    rating_only = 0
    seen: set[str] = set()
    for card in cards:
        text = first_text(card, ("span.wiI7pd",))
        if not text:
            rating_only += 1
            continue
        author = first_text(card, ("div.d4r55", "button.WEBjve"))
        date = first_text(card, ("span.rsqaWe", "span.xRkPPb"))
        review_id = extract_review_id(card, author, date, text)
        if review_id in seen:
            continue
        seen.add(review_id)
        output.append(
            {
                **metadata,
                "place_id": place_slug(final_url),
                "lat": row.get("corrected_google_lat", ""),
                "lng": row.get("corrected_google_lng", ""),
                "작성자": author,
                "평점": extract_rating(card),
                "리뷰": text,
                "작성일": date,
                "언어": detect_language(text),
                "review_id": review_id,
                "google_url": final_url,
                "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return output, rating_only


def main() -> int:
    args = parse_args()
    revalidation = pd.read_csv(args.revalidation, encoding="utf-8-sig", dtype=str).fillna("")
    targets = revalidation[revalidation["correction_action"] == "recrawl_corrected"].copy()
    targets = targets.drop_duplicates(STORE_ID).sort_values(STORE_ID).reset_index(drop=True)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be between 0 and shard-count - 1")
    targets = targets.iloc[args.shard_index :: args.shard_count].copy()
    done = completed_store_ids(args.progress)
    for progress_path in args.resume_progress:
        done.update(completed_store_ids(progress_path))
    targets = targets[~targets[STORE_ID].isin(done)]
    if args.limit is not None:
        targets = targets.head(args.limit)
    if targets.empty:
        print("[완료] 새로 수집할 확정 장소가 없습니다.")
        return 0

    target_ids = set(targets[STORE_ID].astype(str))
    metadata = load_source_metadata(args.reviews, target_ids)
    driver = init_driver(headless=not args.no_headless)
    try:
        for index, row in enumerate(targets.to_dict("records"), start=1):
            store_id = str(row[STORE_ID])
            source = metadata.get(
                store_id,
                {
                    "상가업소번호": store_id,
                    "상호명": row.get(NAME, ""),
                    "시군구명": row.get("시군구명", ""),
                    "행정동명": "",
                    "도로명주소": row.get("도로명주소", ""),
                },
            )
            url = row["corrected_google_url"]
            progress = {
                STORE_ID: store_id,
                NAME: source.get(NAME, ""),
                "google_name": row.get("corrected_google_name", ""),
                "status": "error",
                "review_cards": 0,
                "text_reviews": 0,
                "rating_only_cards": 0,
                "google_url": url,
                "error": "",
                "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            try:
                navigate_to_place(driver, row, args.timeout)
                has_reviews = open_reviews(driver, args.timeout)
                cards = (
                    scroll_all_reviews(
                        driver,
                        max_reviews=args.max_reviews,
                        max_scrolls=args.max_scrolls,
                        pause=args.scroll_pause,
                    )
                    if has_reviews
                    else []
                )
                expand_review_texts(driver, cards)
                rows, rating_only = extract_reviews(cards, source, row, driver.current_url)
                append_rows(args.output, REVIEW_COLUMNS, rows)
                progress.update(
                    {
                        "status": "completed",
                        "review_cards": len(cards),
                        "text_reviews": len(rows),
                        "rating_only_cards": rating_only,
                        "google_url": driver.current_url,
                    }
                )
                print(
                    f"[{index}/{len(targets)}] {source.get(NAME, '')}: "
                    f"본문 리뷰 {len(rows)}개, 별점만 있는 카드 {rating_only}개"
                )
            except Exception as exc:  # noqa: BLE001
                progress["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{index}/{len(targets)}] {source.get(NAME, '')}: 오류 - {exc}")
            append_rows(args.progress, PROGRESS_COLUMNS, [progress])
    finally:
        driver.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
