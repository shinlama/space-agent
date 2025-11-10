from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm


DEFAULT_INPUT = Path("서울시_상권_카페빵_표본.csv")
DEFAULT_OUTPUT = Path("naver_reviews_sample.csv")
DEFAULT_MAX_REVIEWS = 10
DEFAULT_HEADLESS = True
DISTRICT_COL = "시군구명"
NAME_COL = "상호명"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="서울시 카페 표본 CSV를 기반으로 네이버 지도 리뷰를 수집합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="카페 표본 CSV 경로")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="리뷰 저장 CSV 경로")
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=DEFAULT_MAX_REVIEWS,
        help="카페당 최대 리뷰 수",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_HEADLESS,
        help="Chrome을 headless 모드로 실행 여부 (기본: 사용)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="카페 간 기본 지연(초) (무작위 ±0.5초가 추가됨)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=12,
        help="요소 대기 타임아웃(초)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="기존 리뷰 CSV가 있다면 이어서 수집 (이미 존재하는 카페는 건너뜀)",
    )
    return parser.parse_args()


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")

    missing_cols = [col for col in (DISTRICT_COL, NAME_COL) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing_cols)}")

    return df.dropna(subset=[NAME_COL]).reset_index(drop=True)


def init_driver(headless: bool) -> webdriver.Chrome:
    options = ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,960")
    options.add_argument("--lang=ko-KR")
    return webdriver.Chrome(options=options)


def wait_and_switch_to_frame(driver: webdriver.Chrome, frame_name: str, timeout: int) -> None:
    WebDriverWait(driver, timeout).until(EC.frame_to_be_available_and_switch_to_it((By.NAME, frame_name)))


def extract_reviews_from_entry(driver: webdriver.Chrome, max_reviews: int, timeout: int) -> List[str]:
    wait = WebDriverWait(driver, timeout)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.place_section")))
    except TimeoutException:
        return []

    # 모든 리뷰 텍스트를 파싱하기 위해 soup 사용
    soup = BeautifulSoup(driver.page_source, "html.parser")

    selectors = [
        "span.WoYOw",  # 2024-2025 현재 리뷰 텍스트
        "span.OXiLu",  # 이전 버전
        "div.dAsGb",  # 간혹 div 형태로 변환
    ]

    reviews: List[str] = []
    for sel in selectors:
        for elem in soup.select(sel):
            text = elem.get_text(strip=True)
            if text:
                reviews.append(text)
            if len(reviews) >= max_reviews:
                break
        if reviews:
            break

    if not reviews:
        # 리뷰 탭이 접혀있는 경우가 있어 한번 더 시도
        try:
            more_btn = driver.find_element(By.CSS_SELECTOR, "a.fvwqf")
            driver.execute_script("arguments[0].click();", more_btn)
            time.sleep(1.2)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            for elem in soup.select("span.WoYOw"):
                text = elem.get_text(strip=True)
                if text:
                    reviews.append(text)
                if len(reviews) >= max_reviews:
                    break
        except NoSuchElementException:
            pass

    return reviews[:max_reviews]


def get_naver_reviews(
    driver: webdriver.Chrome,
    query: str,
    max_reviews: int,
    timeout: int,
) -> List[str]:
    search_url = f"https://map.naver.com/p/search/{query}"
    driver.get(search_url)

    wait = WebDriverWait(driver, timeout)

    try:
        wait_and_switch_to_frame(driver, "searchIframe", timeout)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.E3DF5")))
    except TimeoutException:
        driver.switch_to.default_content()
        return []

    try:
        first_result = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "li.VLTHu a.tzwk0, li.Yw7w4 a.tzwk0"))
        )
    except TimeoutException:
        driver.switch_to.default_content()
        return []

    try:
        first_result.click()
    except ElementClickInterceptedException:
        driver.execute_script("arguments[0].click();", first_result)

    time.sleep(1.5)
    driver.switch_to.default_content()

    try:
        wait_and_switch_to_frame(driver, "entryIframe", timeout)
    except TimeoutException:
        return []

    reviews = extract_reviews_from_entry(driver, max_reviews=max_reviews, timeout=timeout)
    driver.switch_to.default_content()
    return reviews


def iter_targets(df: pd.DataFrame, resume_df: pd.DataFrame | None) -> Iterable[pd.Series]:
    if resume_df is None or resume_df.empty:
        for _, row in df.iterrows():
            yield row
    else:
        done_set = set(zip(resume_df[NAME_COL], resume_df[DISTRICT_COL]))
        for _, row in df.iterrows():
            key = (row[NAME_COL], row[DISTRICT_COL])
            if key in done_set:
                continue
            yield row


def main() -> int:
    args = parse_args()

    cafes_df = load_dataframe(args.input)
    resume_df = None
    if args.resume and args.resume.exists():
        resume_df = load_dataframe(args.resume).dropna(subset=["리뷰"]).reset_index(drop=True)

    driver: webdriver.Chrome | None = None
    collected: List[dict] = []
    if resume_df is not None:
        collected.extend(resume_df.to_dict(orient="records"))

    try:
        driver = init_driver(args.headless)
        targets = list(iter_targets(cafes_df, resume_df))
        if not targets:
            print("이미 모든 카페에 대한 리뷰가 존재합니다. 작업을 종료합니다.")
            return 0

        for row in tqdm(targets, desc="카페 리뷰 수집"):
            name = str(row[NAME_COL]).strip()
            district = str(row[DISTRICT_COL]).strip()
            query = f"{district} {name}"

            try:
                reviews = get_naver_reviews(driver, query, args.max_reviews, args.timeout)
            except WebDriverException as exc:
                print(f"[오류] {query} → WebDriver 오류: {exc}")
                reviews = []

            if reviews:
                for review in reviews:
                    collected.append(
                        {
                            NAME_COL: name,
                            DISTRICT_COL: district,
                            "리뷰": review,
                        }
                    )
            else:
                collected.append(
                    {
                        NAME_COL: name,
                        DISTRICT_COL: district,
                        "리뷰": "",
                    }
                )

            pause = max(args.delay + random.uniform(-0.5, 0.5), 0.5)
            time.sleep(pause)

    finally:
        if driver is not None:
            driver.quit()

    if collected:
        output_df = pd.DataFrame(collected)
        output_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"✅ 리뷰 수집 완료: 총 {len(output_df)}행 → {args.output}")
    else:
        print("❌ 수집된 리뷰가 없습니다.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

