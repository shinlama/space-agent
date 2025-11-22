from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

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
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm


DEFAULT_INPUT = Path("서울시_상권_카페빵_표본.csv")
DEFAULT_OUTPUT = Path("google_reviews_scraped.csv")
DEFAULT_MAX_REVIEWS = 100
DEFAULT_HEADLESS = True
NAME_COL = "상호명"
DISTRICT_COL = "시군구명"
EUPMYEON_COL = "행정동명"


def parse_args() -> argparse.Namespace:
    # --headless=false 형식 지원을 위한 전처리
    processed_args = []
    for arg in sys.argv[1:]:
        if arg.startswith("--headless="):
            value = arg.split("=", 1)[1].lower()
            if value in ("false", "0", "no", "off"):
                processed_args.append("--no-headless")
            else:
                processed_args.append("--headless")
        else:
            processed_args.append(arg)
    sys.argv[1:] = processed_args
    
    parser = argparse.ArgumentParser(
        description="서울시 카페 표본 데이터를 기반으로 Google Maps 리뷰를 웹 스크래핑으로 수집합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="카페 표본 CSV 경로")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="리뷰 저장 CSV 경로")
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=DEFAULT_MAX_REVIEWS,
        help="카페당 최대 리뷰 수 (기본: 50)",
    )
    parser.add_argument(
        "--headless",
        nargs="?",
        const=True,
        type=lambda x: str(x).lower() in ("true", "1", "yes", "on") if x else True,
        default=DEFAULT_HEADLESS,
        help="Chrome을 headless 모드로 실행 여부 (--headless, --headless=true, --headless=false, 또는 --no-headless, 기본: true)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Chrome을 headless 모드로 실행하지 않음 (--no-headless 사용)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="카페 간 기본 지연(초) (무작위 ±0.5초가 추가됨)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="요소 대기 타임아웃(초)",
    )
    parser.add_argument(
        "--scroll-pause",
        type=float,
        default=1.5,
        help="리뷰 스크롤 간 대기 시간(초)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="기존 수집 CSV 경로 (있다면 이어서 수집, 이미 처리한 카페는 건너뜀)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="테스트용: 처리할 카페 수 제한",
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


def init_driver(headless: bool) -> webdriver.Chrome:
    options = ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=ko-KR")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # 팝업 및 새 창 차단
    options.add_argument("--disable-popup-blocking")
    prefs = {
        "profile.default_content_setting_values": {
            "notifications": 2,  # 알림 차단
            "geolocation": 2,  # 위치 정보 차단
        },
        "profile.default_content_settings.popups": 0,  # 팝업 차단
    }
    options.add_experimental_option("prefs", prefs)
    
    # Google Maps 차단 방지
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    
    driver = webdriver.Chrome(options=options)
    
    # 새 창이 열리면 자동으로 닫기
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            window.open = function() { return null; };
            window.alert = function() { return true; };
            window.confirm = function() { return true; };
            window.prompt = function() { return null; };
        """
    })
    
    return driver


def close_login_dialog(driver: webdriver.Chrome) -> bool:
    """로그인 다이얼로그가 실제로 있으면 닫기 (더 정확한 확인)"""
    try:
        # 로그인 다이얼로그의 특징적인 텍스트나 구조로 정확하게 찾기
        dialog_indicators = [
            # "리뷰를 작성하려면 Google 계정으로 로그인하세요" 같은 텍스트가 있는 다이얼로그
            "//div[contains(text(), '리뷰를 작성하려면') and contains(text(), '로그인')]",
            "//div[contains(text(), 'Google 계정으로 로그인')]",
            # role='dialog'이고 로그인 관련 텍스트가 있는 경우
            "//div[@role='dialog'][.//text()[contains(., '로그인')] or .//text()[contains(., 'Login')]]",
        ]
        
        dialog_found = False
        for indicator in dialog_indicators:
            try:
                dialogs = driver.find_elements(By.XPATH, indicator)
                for dialog in dialogs:
                    if dialog.is_displayed():
                        dialog_found = True
                        # 취소 버튼 찾기
                        try:
                            cancel_btn = dialog.find_element(By.XPATH, ".//button[contains(text(), '취소') or contains(text(), 'Cancel')]")
                            if cancel_btn and cancel_btn.is_displayed():
                                driver.execute_script("arguments[0].click();", cancel_btn)
                                print("[정보] 로그인 다이얼로그 닫기 완료")
                                time.sleep(0.3)
                                return True
                        except:
                            pass
            except:
                continue
        
        # 다이얼로그를 찾았지만 취소 버튼이 없으면 ESC 키 사용
        if dialog_found:
            try:
                # 다이얼로그 내부에 포커스가 있는지 확인하고 ESC 키 전송
                active_element = driver.switch_to.active_element
                active_element.send_keys(Keys.ESCAPE)
                time.sleep(0.3)
                return True
            except:
                pass
        
        # 다이얼로그가 없으면 아무것도 하지 않음 (ESC 키를 무조건 누르지 않음)
        return False
    except Exception:
        return False


def handle_new_windows(driver: webdriver.Chrome) -> None:
    """새로 열린 창이 있으면 모두 닫고 원래 창으로 돌아가기"""
    try:
        main_window = driver.current_window_handle
        all_windows = driver.window_handles
        
        if len(all_windows) > 1:
            print(f"[정보] 새 창 {len(all_windows) - 1}개 감지, 닫는 중...")
            for window in all_windows:
                if window != main_window:
                    driver.switch_to.window(window)
                    driver.close()
            
            # 원래 창으로 돌아가기
            driver.switch_to.window(main_window)
            time.sleep(0.5)
    except Exception:
        pass


def build_search_query(name: str, district: str, eupmyeon: str = "") -> str:
    """검색 쿼리 생성"""
    parts = [name]
    if eupmyeon:
        parts.append(eupmyeon)
    parts.append(district)
    return " ".join(parts)


def click_more_reviews_button(driver: webdriver.Chrome) -> bool:
    """'리뷰 더보기(n)' 버튼 찾기 및 클릭 (숫자 포함 가능)"""
    try:
        # 다양한 선택자로 "리뷰 더보기" 버튼 찾기 (숫자 포함 가능)
        more_reviews_selectors = [
            # 한국어: "리뷰 더보기", "리뷰 더보기(1)", "리뷰 더보기(7)" 등
            "//button[contains(text(), '리뷰 더보기')]",
            "//button[starts-with(text(), '리뷰 더보기')]",  # "리뷰 더보기"로 시작
            "//button[contains(text(), '리뷰') and contains(text(), '더보기')]",
            # span 내부에 있는 경우
            "//span[contains(text(), '리뷰 더보기')]/parent::button",
            "//span[starts-with(text(), '리뷰 더보기')]/parent::button",
            "//span[contains(text(), '리뷰') and contains(text(), '더보기')]/parent::button",
            # div 내부에 있는 경우
            "//div[contains(text(), '리뷰 더보기')]/parent::button",
            "//div[contains(text(), '리뷰') and contains(text(), '더보기')]/parent::button",
            # 영어: "More reviews", "More reviews (1)" 등
            "//button[contains(text(), 'More reviews')]",
            "//button[starts-with(text(), 'More reviews')]",
            "//button[contains(text(), 'Reviews') and contains(text(), 'more')]",
            # aria-label 속성도 확인
            "//button[contains(@aria-label, '리뷰 더보기')]",
            "//button[contains(@aria-label, 'More reviews')]",
            # CSS 선택자도 추가
            "button[aria-label*='리뷰 더보기']",
            "button[aria-label*='More reviews']",
            "button:has(span:contains('리뷰 더보기'))",
        ]
        
        # 모든 버튼을 먼저 찾기
        all_buttons = []
        for selector in more_reviews_selectors:
            try:
                if selector.startswith("//"):
                    # XPath
                    buttons = driver.find_elements(By.XPATH, selector)
                else:
                    # CSS 선택자
                    buttons = driver.find_elements(By.CSS_SELECTOR, selector)
                all_buttons.extend(buttons)
            except:
                continue
        
        # 중복 제거 (같은 요소가 여러 선택자로 찾아질 수 있음)
        seen = set()
        unique_buttons = []
        for btn in all_buttons:
            try:
                btn_id = btn.id
                if btn_id not in seen:
                    seen.add(btn_id)
                    unique_buttons.append(btn)
            except:
                unique_buttons.append(btn)
        
        # 각 버튼에 대해 클릭 시도
        for btn in unique_buttons:
            try:
                # 버튼이 보이는지 확인 (여러 방법으로)
                is_visible = False
                try:
                    is_visible = btn.is_displayed()
                except:
                    pass
                
                # JavaScript로도 확인
                try:
                    js_visible = driver.execute_script(
                        "return arguments[0].offsetParent !== null && "
                        "window.getComputedStyle(arguments[0]).display !== 'none' && "
                        "window.getComputedStyle(arguments[0]).visibility !== 'hidden';",
                        btn
                    )
                    if js_visible:
                        is_visible = True
                except:
                    pass
                
                if not is_visible:
                    continue
                
                btn_text = btn.text.strip()
                btn_aria = btn.get_attribute("aria-label") or ""
                
                # "리뷰 더보기" 또는 "리뷰 더보기(숫자)" 패턴 확인
                if re.search(r'리뷰\s*더보기|More\s*reviews', btn_text + " " + btn_aria, re.IGNORECASE):
                    # 버튼을 화면 중앙으로 스크롤
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", btn)
                    time.sleep(0.8)  # 스크롤 대기 시간 증가
                    
                    # 여러 방법으로 클릭 시도
                    clicked = False
                    
                    # 방법 1: JavaScript 클릭
                    try:
                        driver.execute_script("arguments[0].click();", btn)
                        clicked = True
                        print(f"[정보] '리뷰 더보기' 버튼 클릭 (JavaScript): {btn_text or btn_aria}")
                    except Exception as e1:
                        # 방법 2: 일반 클릭
                        try:
                            btn.click()
                            clicked = True
                            print(f"[정보] '리뷰 더보기' 버튼 클릭 (일반): {btn_text or btn_aria}")
                        except Exception as e2:
                            # 방법 3: ActionChains 사용
                            try:
                                ActionChains(driver).move_to_element(btn).click().perform()
                                clicked = True
                                print(f"[정보] '리뷰 더보기' 버튼 클릭 (ActionChains): {btn_text or btn_aria}")
                            except Exception as e3:
                                print(f"[경고] 버튼 클릭 실패: {e1}, {e2}, {e3}")
                    
                    if clicked:
                        time.sleep(2.5)  # 리뷰 로드 대기 시간 증가
                        return True
            except Exception as e:
                continue
    except Exception as e:
        print(f"[경고] '리뷰 더보기' 버튼 찾기 중 오류: {e}")
    
    return False


def scroll_reviews_section(driver: webdriver.Chrome, max_scrolls: int = 10, pause: float = 1.5) -> None:
    """리뷰 섹션을 스크롤하여 더 많은 리뷰 로드"""
    try:
        # 리뷰 섹션 찾기 (여러 선택자 시도)
        review_panel = None
        selectors = [
            "div.m6QErb.DxyBCb.kA9KIf.dS8AEf",  # 최신 선택자
            "div.m6QErb",  # 간단한 선택자
            "div[role='main'] div[style*='overflow']",  # 스크롤 가능한 컨테이너
        ]
        
        for selector in selectors:
            try:
                panels = driver.find_elements(By.CSS_SELECTOR, selector)
                for panel in panels:
                    # 스크롤 가능한 요소인지 확인
                    scrollable = driver.execute_script(
                        "return arguments[0].scrollHeight > arguments[0].clientHeight;",
                        panel
                    )
                    if scrollable:
                        review_panel = panel
                        break
                if review_panel:
                    break
            except Exception:
                continue
        
        if not review_panel:
            # 리뷰 섹션을 찾지 못한 경우, 리뷰 요소가 있는 영역으로 스크롤
            print("[경고] 리뷰 패널을 찾을 수 없어 리뷰 요소가 있는 영역으로 스크롤합니다.")
            try:
                # 리뷰 요소 찾기
                review_elements = driver.find_elements(By.CSS_SELECTOR, "div.jftiEf")
                if review_elements:
                    # 마지막 리뷰 요소로 스크롤 (더 적극적으로)
                    last_review = review_elements[-1]
                    no_new_reviews = 0
                    for scroll_idx in range(max_scrolls * 2):  # 스크롤 횟수 2배 증가
                        # 마지막 리뷰로 스크롤
                        driver.execute_script("arguments[0].scrollIntoView({block: 'end', behavior: 'smooth'});", last_review)
                        time.sleep(pause * 1.5)  # 대기 시간 증가
                        
                        # 추가로 페이지 스크롤 (리뷰 섹션이 페이지 내부에 있을 수 있음)
                        driver.execute_script("window.scrollBy(0, 800);")
                        time.sleep(pause * 0.5)
                        
                        # "리뷰 더보기" 버튼 클릭 시도
                        if click_more_reviews_button(driver):
                            time.sleep(2)  # 버튼 클릭 후 더 긴 대기
                            # 버튼 클릭 후 다시 스크롤
                            driver.execute_script("window.scrollBy(0, 800);")
                            time.sleep(pause)
                        
                        # 새로운 리뷰 요소 확인
                        new_reviews = driver.find_elements(By.CSS_SELECTOR, "div.jftiEf")
                        if len(new_reviews) > len(review_elements):
                            # 새로운 리뷰가 추가되었음
                            print(f"[정보] 리뷰 개수 증가: {len(review_elements)} → {len(new_reviews)}개 (스크롤 {scroll_idx + 1})")
                            review_elements = new_reviews
                            last_review = review_elements[-1]
                            no_new_reviews = 0
                        else:
                            no_new_reviews += 1
                            if no_new_reviews >= 3:
                                # 3번 연속 새로운 리뷰가 없으면 중단
                                break
                else:
                    # 리뷰 요소도 없으면 페이지 전체 스크롤 (더 적극적으로)
                    for _ in range(max_scrolls * 2):
                        driver.execute_script("window.scrollBy(0, 1000);")
                        time.sleep(pause)
                        # "리뷰 더보기" 버튼 클릭 시도
                        if click_more_reviews_button(driver):
                            time.sleep(2)
            except Exception as e:
                print(f"[경고] 리뷰 요소 스크롤 중 오류: {e}")
                # 오류 발생 시 페이지 전체 스크롤 (더 적극적으로)
                for _ in range(max_scrolls * 2):
                    driver.execute_script("window.scrollBy(0, 1000);")
                    time.sleep(pause)
                    # "리뷰 더보기" 버튼 클릭 시도
                    if click_more_reviews_button(driver):
                        time.sleep(2)
            return
        
        # 리뷰 섹션 내부로 스크롤 (더 적극적으로)
        last_height = 0
        no_change_count = 0
        for i in range(max_scrolls * 2):  # 스크롤 횟수 2배 증가
            # 현재 높이 저장
            current_height = driver.execute_script("return arguments[0].scrollHeight;", review_panel)
            
            # 스크롤을 끝까지 내리기
            driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollHeight;",
                review_panel
            )
            time.sleep(pause * 1.5)  # 리뷰 로드 대기 시간 증가
            
            # 추가로 약간 더 스크롤 (리뷰가 로드되는 것을 확실히 하기 위해)
            driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollHeight + 500;",
                review_panel
            )
            time.sleep(pause * 0.5)
            
            # "리뷰 더보기(n)" 버튼 클릭 시도
            if click_more_reviews_button(driver):
                # 버튼 클릭 후 추가 대기
                time.sleep(2)  # 대기 시간 증가
                # 스크롤 다시 끝까지
                driver.execute_script(
                    "arguments[0].scrollTop = arguments[0].scrollHeight;",
                    review_panel
                )
                time.sleep(pause)
            
            # 새로운 높이 확인
            new_height = driver.execute_script("return arguments[0].scrollHeight;", review_panel)
            if new_height > current_height:
                # 높이가 증가했으면 계속 (메시지는 디버깅 시에만 출력)
                last_height = current_height
                no_change_count = 0
            elif new_height == current_height:
                no_change_count += 1
                if no_change_count >= 5:  # 5번 연속 높이가 변하지 않으면 중단
                    # 메시지 출력 (중단할 때만)
                    break
            else:
                last_height = current_height
            
    except Exception as e:
        print(f"[경고] 리뷰 스크롤 중 오류: {e}")
        # 오류가 발생해도 계속 진행
        pass


def extract_rating_from_element(elem) -> Optional[int]:
    """리뷰 요소에서 평점 추출"""
    # 평점은 aria-label이나 data-value 속성에 있을 수 있음
    aria_label = elem.get("aria-label", "")
    rating_match = re.search(r"(\d+)", aria_label)
    if rating_match:
        return int(rating_match.group(1))
    
    # 별점 이미지의 alt 텍스트에서 추출
    star_elem = elem.find("img", alt=re.compile(r"별표|star|Star"))
    if star_elem:
        alt_text = star_elem.get("alt", "")
        rating_match = re.search(r"(\d+)", alt_text)
        if rating_match:
            return int(rating_match.group(1))
    
    return None


def normalize_review_text(text: str) -> str:
    """리뷰 텍스트에서 메타데이터 제거 (작성자 정보, 작성일, 특수 문자만)"""
    if not text:
        return ""
    
    normalized = text.strip()
    
    # 1. UI 텍스트 제거 ("자세히", "공유", "더보기" 등)
    ui_texts = ["자세히", "공유", "더보기", "… 자세히", "…"]
    for ui_text in ui_texts:
        normalized = normalized.replace(ui_text, "").strip()
    
    # 2. 작성자 이름 + "리뷰 N개 · 사진 N장" 패턴 제거
    # 예: "HJ리뷰 2개 · 사진 3장" 또는 "작성자명리뷰 2개 · 사진 3장"
    # 패턴: 임의의 문자 + "리뷰" + 숫자 + "개" + "·" + "사진" + 숫자 + "장"
    normalized = re.sub(r"[^\s]*리뷰\s*\d+\s*개\s*·\s*사진\s*\d+\s*장\s*", "", normalized, flags=re.IGNORECASE)
    
    # 3. 작성일 정보 제거 (리뷰 텍스트 시작 부분에 있는 경우)
    # 예: "1년 전", "3달 전", "2주 전" 등
    normalized = re.sub(r"^\d+\s*(년|달|주|일|시간|분)\s*전\s*", "", normalized)
    
    # 4. 특수 문자 패턴 제거 (예: "", "" 등 - 유니코드 제어 문자)
    # 하지만 일반적인 구두점은 유지
    normalized = re.sub(r"[\u2000-\u206F\u2E00-\u2E7F]", "", normalized)  # 일반 구두점 범위 제외
    
    # 5. 연속된 공백 정리
    normalized = re.sub(r"\s+", " ", normalized).strip()
    
    return normalized


def expand_all_review_texts(driver: webdriver.Chrome, review_elements: List) -> None:
    """모든 리뷰의 '자세히' 버튼을 클릭하여 전체 텍스트 표시"""
    expanded_count = 0
    for idx, review_elem in enumerate(review_elements):
        try:
            # 클릭 전 텍스트 길이 확인
            try:
                before_text = review_elem.find_element(By.CSS_SELECTOR, "span.wiI7pd").text.strip()
                before_length = len(before_text)
            except:
                before_length = 0
            
            # 여러 선택자로 "자세히" 또는 "더보기" 버튼 찾기
            more_button = None
            more_selectors = [
                ".//button[contains(@jsaction, 'more')]",
                ".//button[contains(@aria-label, '자세히')]",
                ".//button[contains(@aria-label, '더보기')]",
                ".//button[contains(@aria-label, 'More')]",
                ".//button[contains(@aria-label, 'more')]",
                ".//button[contains(text(), '자세히')]",
                ".//button[contains(text(), '더보기')]",
                ".//button[contains(text(), 'More')]",
                ".//span[contains(text(), '자세히')]",  # span으로도 표시될 수 있음
                ".//a[contains(text(), '자세히')]",  # 링크로도 표시될 수 있음
                ".//span[contains(text(), '…')]/following-sibling::button",  # "…" 다음의 버튼
                ".//span[contains(text(), '…')]/following-sibling::span",  # "…" 다음의 span
            ]
            
            for selector in more_selectors:
                try:
                    more_button = review_elem.find_element(By.XPATH, selector)
                    if more_button:
                        # 버튼이 보이거나 클릭 가능한지 확인
                        is_visible = driver.execute_script(
                            "return arguments[0].offsetParent !== null && window.getComputedStyle(arguments[0]).display !== 'none';",
                            more_button
                        )
                        if is_visible:
                            # 버튼이 보이면 클릭
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", more_button)
                            time.sleep(0.3)
                            driver.execute_script("arguments[0].click();", more_button)
                            time.sleep(0.5)  # 텍스트 확장 대기 시간 증가
                            
                            # 클릭 후 텍스트가 확장되었는지 확인
                            try:
                                after_text = review_elem.find_element(By.CSS_SELECTOR, "span.wiI7pd").text.strip()
                                after_length = len(after_text)
                                if after_length > before_length:
                                    expanded_count += 1
                                    break
                            except:
                                expanded_count += 1  # 확인 실패해도 클릭은 했으므로 카운트
                                break
                except (NoSuchElementException, ElementClickInterceptedException):
                    continue
            
            # "…" 또는 "… 자세히" 텍스트가 있는 경우 추가 시도
            if more_button is None:
                try:
                    # "… 자세히"가 포함된 요소 찾기
                    ellipsis_with_more = review_elem.find_elements(By.XPATH, ".//*[contains(text(), '…') and contains(text(), '자세히')]")
                    if not ellipsis_with_more:
                        ellipsis_with_more = review_elem.find_elements(By.XPATH, ".//*[contains(text(), '…')]")
                    
                    for elem in ellipsis_with_more:
                        try:
                            # 부모 요소나 다음 형제 요소 클릭 시도
                            parent = elem.find_element(By.XPATH, "./..")
                            is_visible = driver.execute_script(
                                "return arguments[0].offsetParent !== null && window.getComputedStyle(arguments[0]).display !== 'none';",
                                parent
                            )
                            if is_visible:
                                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", parent)
                                time.sleep(0.2)
                                driver.execute_script("arguments[0].click();", parent)
                                time.sleep(0.5)
                                expanded_count += 1
                                break
                        except:
                            continue
                except:
                    pass
        except Exception:
            continue
    
    if expanded_count > 0:
        print(f"[정보] {expanded_count}개 리뷰의 '자세히' 버튼 클릭 완료")
    time.sleep(1)  # 모든 확장 완료 대기 시간 증가


def extract_reviews_from_page(driver: webdriver.Chrome, max_reviews: int) -> List[Dict[str, str]]:
    """현재 페이지에서 리뷰 추출"""
    reviews = []
    seen_texts = set()  # 중복 체크용
    
    # 먼저 Selenium으로 리뷰 요소 찾기 시도 (동적 콘텐츠에 더 효과적)
    try:
        # div.jftiEf가 가장 정확한 리뷰 선택자 (스크롤 중 개수 확인과 동일)
        selenium_reviews = driver.find_elements(
            By.CSS_SELECTOR,
            "div.jftiEf"
        )
        if selenium_reviews:
            print(f"[정보] Selenium으로 {len(selenium_reviews)}개 리뷰 요소 발견")
            # 모든 리뷰의 "자세히" 버튼 클릭 (max_reviews 제한 없이 모든 리뷰 처리)
            expand_all_review_texts(driver, selenium_reviews)
            # 버튼 클릭 후 리뷰 요소 다시 찾기 (DOM이 업데이트되었을 수 있음)
            selenium_reviews = driver.find_elements(
                By.CSS_SELECTOR,
                "div.jftiEf"
            )
            print(f"[정보] '자세히' 클릭 후 {len(selenium_reviews)}개 리뷰 요소 재확인")
    except Exception as e:
        print(f"[경고] 리뷰 요소 찾기 중 오류: {e}")
        selenium_reviews = []
    
    # Selenium으로 직접 추출 시도 (동적 콘텐츠에 더 효과적)
    # 모든 리뷰를 추출한 후 max_reviews로 제한 (중간 과정에서는 제한하지 않음)
    if selenium_reviews and len(selenium_reviews) > 0:
        for review_elem_sel in selenium_reviews:  # max_reviews 제한 제거
            try:
                # 작성자 (이름만 추출, 메타데이터 제외)
                author = ""
                try:
                    author_elem = review_elem_sel.find_element(By.CSS_SELECTOR, "div.d4r55, div.X43Kjb")
                    full_author_text = author_elem.text.strip()
                    # 정규식으로 "지역 가이드", "리뷰 N개", "사진 N장" 등 메타데이터 제거
                    # 패턴: 이름 뒤에 "지역 가이드" 또는 "·" 또는 "리뷰" 또는 "사진"이 오는 경우
                    match = re.match(r"^([^·지리사]*?)(?:지역 가이드|·|리뷰 \d+개|사진 \d+장|리뷰|사진).*$", full_author_text, re.DOTALL)
                    if match:
                        author = match.group(1).strip()
                    else:
                        # 패턴이 매치되지 않으면 첫 줄만 가져오기
                        lines = full_author_text.split('\n')
                        if lines:
                            author = lines[0].strip()
                            # 여전히 메타데이터가 포함되어 있으면 제거
                            author = re.sub(r"(?:지역 가이드|·|리뷰 \d+개|사진 \d+장).*$", "", author).strip()
                        else:
                            author = full_author_text
                except Exception:
                    pass
                
                # 평점
                rating = None
                try:
                    rating_elem = review_elem_sel.find_element(By.CSS_SELECTOR, "span.kvMYJc")
                    aria_label = rating_elem.get_attribute("aria-label") or ""
                    rating_match = re.search(r"(\d+)", aria_label)
                    if rating_match:
                        rating = int(rating_match.group(1))
                except:
                    pass
                
                # 각 리뷰마다 "자세히" 버튼 확인 및 클릭 (추가 확인 - 누락 방지)
                try:
                    more_selectors = [
                        ".//button[contains(@jsaction, 'more')]",
                        ".//button[contains(text(), '자세히')]",
                        ".//span[contains(text(), '자세히')]",
                        ".//span[contains(text(), '…')]/following-sibling::button",
                        ".//*[contains(text(), '… 자세히')]",  # "… 자세히" 패턴
                        ".//*[contains(text(), '…') and contains(text(), '자세히')]",  # "…"와 "자세히" 모두 포함
                    ]
                    for selector in more_selectors:
                        try:
                            more_btn = review_elem_sel.find_element(By.XPATH, selector)
                            if more_btn:
                                is_visible = driver.execute_script(
                                    "return arguments[0].offsetParent !== null && window.getComputedStyle(arguments[0]).display !== 'none';",
                                    more_btn
                                )
                                if is_visible:
                                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", more_btn)
                                    time.sleep(0.2)
                                    driver.execute_script("arguments[0].click();", more_btn)
                                    time.sleep(0.5)  # 텍스트 확장 대기 시간 증가
                                    break
                        except:
                            continue
                    
                    # "…" 텍스트가 있는 경우 추가 시도
                    try:
                        ellipsis_elems = review_elem_sel.find_elements(By.XPATH, ".//*[contains(text(), '…')]")
                        for ellipsis in ellipsis_elems:
                            try:
                                # "자세히"가 근처에 있는지 확인
                                parent = ellipsis.find_element(By.XPATH, "./..")
                                parent_text = parent.text
                                if "자세히" in parent_text or "…" in parent_text:
                                    is_visible = driver.execute_script(
                                        "return arguments[0].offsetParent !== null && window.getComputedStyle(arguments[0]).display !== 'none';",
                                        parent
                                    )
                                    if is_visible:
                                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", parent)
                                        time.sleep(0.2)
                                        driver.execute_script("arguments[0].click();", parent)
                                        time.sleep(0.5)
                                        break
                            except:
                                continue
                    except:
                        pass
                except:
                    pass
                
                # 리뷰 텍스트 (이미 expand_all_review_texts에서 "자세히" 클릭 완료, 추가로 다시 확인)
                # 리뷰 본문 선택자를 우선 사용 (하트 버튼이 포함되지 않음)
                review_text = ""
                text_selectors = [
                    "span.wiI7pd",  # 최신 선택자 (리뷰 본문만, 하트 버튼 제외)
                    "div.MyEned span.wiI7pd",  # 대체 선택자
                    "span[jslog*='review']",  # jslog 속성 기반
                ]
                
                for text_sel in text_selectors:
                    try:
                        text_elem = review_elem_sel.find_element(By.CSS_SELECTOR, text_sel)
                        review_text = text_elem.text.strip()
                        if review_text and len(review_text) > 5:  # 최소 길이 완화 (10 → 5, 영어 리뷰도 포함)
                            break
                    except:
                        continue
                
                # 여전히 텍스트를 찾지 못한 경우, 하트 버튼을 제외하고 텍스트 가져오기
                if not review_text or len(review_text) < 5:
                    try:
                        # JavaScript로 하트 버튼을 제외한 텍스트 추출
                        review_text = driver.execute_script("""
                            var elem = arguments[0];
                            var clone = elem.cloneNode(true);
                            
                            // 하트 버튼(좋아요 버튼) 요소 제거
                            var likeButtons = clone.querySelectorAll(
                                'button[aria-label*="좋아요"], ' +
                                'button[aria-label*="좋아요"], ' +
                                'button[jsaction*="like"], ' +
                                'button[jsaction*="helpful"]'
                            );
                            likeButtons.forEach(function(btn) { btn.remove(); });
                            
                            // 텍스트 추출
                            return clone.innerText || clone.textContent || '';
                        """, review_elem_sel)
                        review_text = review_text.strip() if review_text else ""
                    except:
                        # JavaScript 실패 시 기본 방법 사용
                        try:
                            review_text = review_elem_sel.text.strip()
                        except:
                            pass
                
                # 작성일
                date = ""
                try:
                    date_elem = review_elem_sel.find_element(By.CSS_SELECTOR, "span.rsqaWe, span.xRkPPb")
                    date = date_elem.text.strip()
                except:
                    pass
                
                # 리뷰 추가 조건: 작성자나 평점이 있으면 추가 (텍스트가 없어도)
                if author or rating is not None or review_text:
                    # 리뷰 텍스트 정규화 (메타데이터 제거)
                    normalized_text = normalize_review_text(review_text)
                    
                    # 중복 체크 (작성자 + 정규화된 리뷰 텍스트의 처음 150자)
                    # 평점은 중복 체크에 포함하지 않음 (같은 리뷰가 평점 유무로 중복 수집되는 것 방지)
                    text_key_part = normalized_text[:150].strip() if normalized_text else ""
                    text_key = f"{author}|{text_key_part}"
                    if text_key and text_key not in seen_texts:
                        seen_texts.add(text_key)
                        language = "ko" if any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in review_text) else "en"
                        reviews.append({
                            "작성자": author,
                            "평점": rating if rating else "",
                            "리뷰": normalized_text,  # 정규화된 텍스트 저장
                            "작성일": date,
                            "언어": language,
                        })
            except Exception as e:
                continue
    
    # Selenium으로 리뷰를 찾지 못한 경우에만 BeautifulSoup 사용
    if not reviews:
        # 페이지 소스 파싱
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # 리뷰 컨테이너 찾기 (스크롤 중 개수 확인과 동일한 선택자 사용)
        review_elements = soup.select("div.jftiEf")
        if review_elements:
            print(f"[정보] BeautifulSoup으로 {len(review_elements)}개 리뷰 요소 발견")
        
        if review_elements:
            for review_elem in review_elements:  # max_reviews 제한 제거, 모든 리뷰 처리
                try:
                    # 작성자 이름 (이름만 추출, 메타데이터 제외)
                    author = ""
                    author_elem = review_elem.select_one("div.d4r55, div.X43Kjb")
                    if author_elem:
                        full_author_text = author_elem.get_text(strip=True)
                        # 정규식으로 "지역 가이드", "리뷰 N개", "사진 N장" 등 메타데이터 제거
                        match = re.match(r"^([^·지리사]*?)(?:지역 가이드|·|리뷰 \d+개|사진 \d+장|리뷰|사진).*$", full_author_text, re.DOTALL)
                        if match:
                            author = match.group(1).strip()
                        else:
                            # 패턴이 매치되지 않으면 첫 줄만 가져오기
                            lines = full_author_text.split('\n')
                            if lines:
                                author = lines[0].strip()
                                # 여전히 메타데이터가 포함되어 있으면 제거
                                author = re.sub(r"(?:지역 가이드|·|리뷰 \d+개|사진 \d+장).*$", "", author).strip()
                            else:
                                author = full_author_text
                    
                    # 평점
                    rating = None
                    rating_elem = review_elem.select_one("span.kvMYJc, div.fontBodyMedium")
                    if rating_elem:
                        rating = extract_rating_from_element(rating_elem)
                    
                    # 리뷰 텍스트 (리뷰 본문 선택자 우선 사용 - 하트 버튼 제외)
                    review_text = ""
                    text_selectors = [
                        "span.wiI7pd",  # 최신 선택자 (리뷰 본문만, 하트 버튼 제외)
                        "div.MyEned span.wiI7pd",  # 대체 선택자
                        "div[data-review-id] span.wiI7pd",
                    ]
                    for text_sel in text_selectors:
                        text_elem = review_elem.select_one(text_sel)
                        if text_elem:
                            review_text = text_elem.get_text(strip=True)
                            if review_text and len(review_text) > 10:
                                break
                    
                    # 리뷰 텍스트를 찾지 못한 경우, 하트 버튼을 제외하고 텍스트 추출
                    if not review_text or len(review_text) < 10:
                        # 하트 버튼 요소 제거
                        review_elem_copy = BeautifulSoup(str(review_elem), "html.parser")
                        # 하트 버튼 요소 찾기 및 제거
                        like_buttons = review_elem_copy.select(
                            'button[aria-label*="좋아요"], '
                            'button[aria-label*="좋아요"], '
                            'button[jsaction*="like"], '
                            'button[jsaction*="helpful"]'
                        )
                        for btn in like_buttons:
                            btn.decompose()  # 요소 제거
                        
                        review_text = review_elem_copy.get_text(strip=True)
                    
                    # 작성일
                    date_elem = review_elem.select_one("span.rsqaWe, span.xRkPPb")
                    date = date_elem.get_text(strip=True) if date_elem else ""
                    
                    # 언어 감지 (간단히 한국어 여부 확인)
                    language = "ko" if any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in review_text) else "en"
                    
                    # 리뷰 추가 조건: 작성자나 평점이 있으면 추가 (텍스트가 없어도)
                    if author or rating or review_text:
                        # 리뷰 텍스트 정규화 (메타데이터 제거)
                        normalized_text = normalize_review_text(review_text)
                        
                        # 중복 체크 (작성자 + 정규화된 리뷰 텍스트의 처음 150자)
                        # 평점은 중복 체크에 포함하지 않음 (같은 리뷰가 평점 유무로 중복 수집되는 것 방지)
                        text_key_part = normalized_text[:150].strip() if normalized_text else ""
                        text_key = f"{author}|{text_key_part}"
                        if text_key and text_key not in seen_texts:
                            seen_texts.add(text_key)
                            reviews.append({
                                "작성자": author,
                                "평점": rating if rating else "",
                                "리뷰": normalized_text,  # 정규화된 텍스트 저장
                                "작성일": date,
                                "언어": language,
                            })
                except Exception as e:
                    continue
    
    return reviews


def get_google_reviews_scraped(
    driver: webdriver.Chrome,
    name: str,
    district: str,
    eupmyeon: str,
    max_reviews: int,
    timeout: int,
    scroll_pause: float,
) -> List[Dict[str, str]]:
    """Google Maps에서 리뷰 스크래핑"""
    reviews = []
    
    # 검색 쿼리 생성
    query = build_search_query(name, district, eupmyeon)
    search_url = f"https://www.google.com/maps/search/{quote(query)}"
    
    try:
        driver.get(search_url)
        wait = WebDriverWait(driver, timeout)
        
        # 검색 결과 로드 대기
        time.sleep(2)
        
        # 로그인 다이얼로그 닫기 및 새 창 처리
        close_login_dialog(driver)
        handle_new_windows(driver)
        
        # 첫 번째 검색 결과 클릭
        try:
            # 여러 선택자 시도
            first_result = None
            selectors = [
                "div[role='article'] a",
                "div.Nv2PK a",
                "a[data-value='Directions']",  # 방향 버튼 근처의 링크
                "div.m6QErb a",
            ]
            
            for selector in selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        first_result = elements[0]
                        break
                except NoSuchElementException:
                    continue
            
            if not first_result:
                # XPath로 시도
                try:
                    first_result = driver.find_element(By.XPATH, "//a[contains(@href, '/place/')]")
                except NoSuchElementException:
                    pass
            
            if first_result:
                driver.execute_script("arguments[0].scrollIntoView(true);", first_result)
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", first_result)
                time.sleep(4)  # 상세 페이지 로드 대기
                
                # 로그인 다이얼로그 닫기 및 새 창 처리
                close_login_dialog(driver)
                handle_new_windows(driver)
            else:
                print(f"[오류] {name} ({district}) - 검색 결과를 찾을 수 없습니다.")
                return reviews
        except Exception as e:
            print(f"[오류] {name} ({district}) - 검색 결과 클릭 실패: {e}")
            return reviews
        
        # 리뷰 섹션 찾기 및 클릭
        time.sleep(2)  # 페이지 로드 대기
        
        # 먼저 리뷰가 이미 보이는지 확인
        try:
            existing_reviews = driver.find_elements(By.CSS_SELECTOR, "div.jftiEf, div.MyEned")
            if existing_reviews:
                print(f"[정보] {name} ({district}) - 리뷰가 이미 표시되어 있습니다.")
            else:
                # 리뷰 탭/버튼 찾기 및 클릭
                review_button = None
                
                # 다양한 방법으로 리뷰 버튼 찾기
                try:
                    # 방법 1: XPath로 텍스트 기반 검색
                    review_button = driver.find_element(
                        By.XPATH,
                        "//button[contains(text(), '리뷰') or contains(text(), 'Reviews') or contains(@aria-label, '리뷰') or contains(@aria-label, 'Reviews')]"
                    )
                except NoSuchElementException:
                    try:
                        # 방법 2: data-value 속성
                        review_button = driver.find_element(By.CSS_SELECTOR, "button[data-value='리뷰'], button[data-value='Reviews']")
                    except NoSuchElementException:
                        try:
                            # 방법 3: 탭 버튼 찾기
                            tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab'], div.RWPxGd button")
                            for tab in tabs:
                                text = tab.text.lower()
                                if '리뷰' in text or 'review' in text:
                                    review_button = tab
                                    break
                        except NoSuchElementException:
                            pass
                
                if review_button:
                    try:
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", review_button)
                        time.sleep(0.5)
                        driver.execute_script("arguments[0].click();", review_button)
                        time.sleep(3)  # 리뷰 섹션 로드 대기
                        
                        # 로그인 다이얼로그 닫기 및 새 창 처리
                        close_login_dialog(driver)
                        handle_new_windows(driver)
                        
                        print(f"[정보] {name} ({district}) - 리뷰 버튼 클릭 완료")
                    except Exception as e:
                        print(f"[경고] {name} ({district}) - 리뷰 버튼 클릭 실패: {e}")
                else:
                    print(f"[경고] {name} ({district}) - 리뷰 버튼을 찾을 수 없습니다. 리뷰가 이미 표시되어 있을 수 있습니다.")
        except Exception as e:
            print(f"[경고] {name} ({district}) - 리뷰 섹션 확인 중 오류: {e}")
        
        # 리뷰 섹션 스크롤하여 더 많은 리뷰 로드 (반복적으로)
        print(f"[정보] {name} ({district}) - 리뷰 스크롤 시작 (최대 {max_reviews}개 목표)")
        
        # 스크롤하면서 리뷰 개수 확인
        previous_count = 0
        no_change_count = 0
        max_scroll_attempts = 50  # 시도 횟수 더 증가 (45개 리뷰 수집을 위해)
        
        for scroll_attempt in range(max_scroll_attempts):
            # 현재 리뷰 개수 확인 (최종 추출과 동일한 선택자 사용)
            current_count = 0
            try:
                # 최종 추출과 동일한 선택자 사용 (div.jftiEf가 가장 정확)
                current_reviews = driver.find_elements(
                    By.CSS_SELECTOR,
                    "div.jftiEf"
                )
                current_count = len(current_reviews)
                
                # 매 시도마다 현재 리뷰 개수 출력 (디버깅)
                if scroll_attempt == 0:
                    print(f"[정보] 초기 리뷰 개수: {current_count}개")
                elif current_count > previous_count:
                    print(f"[정보] 리뷰 개수 증가: {previous_count} → {current_count}개 (시도 {scroll_attempt + 1})")
                    previous_count = current_count
                    no_change_count = 0
                else:
                    no_change_count += 1
                    if scroll_attempt % 5 == 0:  # 5번마다 현재 상태 출력
                        print(f"[정보] 리뷰 개수 유지: {current_count}개 (시도 {scroll_attempt + 1}, {no_change_count}번 연속 동일)")
                
                # 목표 개수에 도달했거나 더 이상 증가하지 않으면 중단
                if current_count >= max_reviews:
                    print(f"[정보] 목표 리뷰 개수({max_reviews}개)에 도달했습니다.")
                    break
                
            except Exception as e:
                print(f"[경고] 리뷰 개수 확인 중 오류: {e}")
            
            # 로그인 다이얼로그 닫기 및 새 창 처리 (드물게, 10번마다 한 번씩만)
            # 너무 자주 호출하면 리뷰 로딩을 방해할 수 있음
            if scroll_attempt > 0 and scroll_attempt % 10 == 0:
                handle_new_windows(driver)  # 새 창만 확인 (로그인 다이얼로그는 페이지 로드 시에만)
            
            # "리뷰 더보기" 버튼 클릭 시도 (스크롤 전에 먼저 확인)
            button_clicked = click_more_reviews_button(driver)
            
            # 리뷰 개수가 변하지 않고 버튼도 없으면 스크롤 횟수 줄이기
            if no_change_count >= 2 and not button_clicked:
                # 이미 충분히 시도했고 변화가 없으면 스크롤 횟수 줄임
                scroll_reviews_section(driver, max_scrolls=3, pause=scroll_pause)
            else:
                # 스크롤 실행 (더 적극적으로)
                scroll_reviews_section(driver, max_scrolls=8, pause=scroll_pause)  # 스크롤 횟수 조정
            
            # 추가로 페이지 전체 스크롤 (리뷰 섹션이 페이지 내부에 있을 수 있음)
            if no_change_count < 2:  # 변화가 있을 때만 추가 스크롤
                try:
                    for _ in range(2):
                        driver.execute_script("window.scrollBy(0, 1000);")
                        time.sleep(scroll_pause * 0.2)
                except:
                    pass
            
            # 버튼을 클릭했다면 추가 대기
            if button_clicked:
                time.sleep(2)  # 버튼 클릭 후 대기 시간 줄임
                # 다시 스크롤 (버튼 클릭했을 때만)
                scroll_reviews_section(driver, max_scrolls=3, pause=scroll_pause)
                # 추가로 페이지 전체 스크롤
                try:
                    for _ in range(2):
                        driver.execute_script("window.scrollBy(0, 1000);")
                        time.sleep(scroll_pause * 0.2)
                except:
                    pass
            
            # 리뷰 개수 다시 확인 (버튼 클릭 및 스크롤 후)
            try:
                new_reviews = driver.find_elements(
                    By.CSS_SELECTOR,
                    "div.jftiEf"
                )
                new_count = len(new_reviews)
                if new_count > current_count:
                    print(f"[정보] 리뷰 개수 증가: {current_count} → {new_count}개 (시도 {scroll_attempt + 1})")
                    current_count = new_count
                    previous_count = new_count
                    no_change_count = 0
                elif new_count == current_count:
                    # 개수가 변하지 않음
                    no_change_count += 1
                else:
                    # 개수가 줄어든 경우 (이상하지만)
                    current_count = new_count
                    previous_count = new_count
            except:
                pass
            
            # 목표 개수에 도달했으면 중단
            if current_count >= max_reviews:
                print(f"[정보] 목표 리뷰 개수({max_reviews}개)에 도달했습니다. (현재 {current_count}개)")
                break
            
            # "리뷰 더보기" 버튼이 없고 리뷰 개수가 일정 횟수 연속 변하지 않으면 중단
            if no_change_count >= 3:  # 3번 연속 변하지 않으면 빠르게 중단
                # 마지막으로 "리뷰 더보기" 버튼이 있는지 확인
                final_button_check = click_more_reviews_button(driver)
                if not final_button_check:
                    # 버튼도 없고 개수도 안 늘면 중단
                    print(f"[정보] 리뷰 개수가 더 이상 증가하지 않습니다. (현재 {current_count}개, {no_change_count}번 연속 동일, '리뷰 더보기' 버튼 없음)")
                    break
                else:
                    # 버튼이 있었으면 카운트 리셋하고 계속
                    print(f"[정보] '리뷰 더보기' 버튼 발견, 계속 진행...")
                    no_change_count = 0
                    time.sleep(1)  # 대기 시간 줄임
                    scroll_reviews_section(driver, max_scrolls=3, pause=scroll_pause)  # 스크롤 횟수 줄임
                    # 리뷰 개수 다시 확인
                    try:
                        final_reviews = driver.find_elements(By.CSS_SELECTOR, "div.jftiEf")
                        final_count = len(final_reviews)
                        if final_count > current_count:
                            current_count = final_count
                            previous_count = final_count
                            no_change_count = 0
                    except:
                        pass
            
            time.sleep(scroll_pause * 0.3)  # 리뷰 로드 대기 시간 더 줄임
        
        # 최종 리뷰 추출 전에 한 번 더 스크롤하여 모든 리뷰 로드 확인
        print(f"[정보] {name} ({district}) - 최종 리뷰 로드 확인")
        # "리뷰 더보기" 버튼이 사라질 때까지 반복 클릭
        more_reviews_clicked = 0
        consecutive_no_button = 0
        for attempt in range(10):  # 최대 10번 시도 (줄임)
            if click_more_reviews_button(driver):
                more_reviews_clicked += 1
                consecutive_no_button = 0
                scroll_reviews_section(driver, max_scrolls=5, pause=scroll_pause)  # 스크롤 횟수 줄임
                # 추가로 페이지 전체 스크롤
                try:
                    for _ in range(2):
                        driver.execute_script("window.scrollBy(0, 1000);")
                        time.sleep(scroll_pause * 0.2)
                except:
                    pass
                time.sleep(1.5)  # 버튼 클릭 후 대기 시간 줄임
                
                # 리뷰 개수 확인
                try:
                    final_reviews = driver.find_elements(
                        By.CSS_SELECTOR,
                        "div.jftiEf"
                    )
                    print(f"[정보] 최종 확인 중 리뷰 개수: {len(final_reviews)}개")
                except:
                    pass
            else:
                consecutive_no_button += 1
                if consecutive_no_button >= 2:
                    # 2번 연속 버튼이 없으면 중단
                    break
                # 버튼이 없어도 스크롤은 계속
                scroll_reviews_section(driver, max_scrolls=2, pause=scroll_pause)
                time.sleep(1)
        
        if more_reviews_clicked > 0:
            print(f"[정보] 최종 확인 중 '리뷰 더보기' 버튼 {more_reviews_clicked}번 클릭")
        
        # 최종 스크롤
        scroll_reviews_section(driver, max_scrolls=3, pause=scroll_pause)
        time.sleep(1.5)  # 최종 로드 대기 시간 줄임
        
        # 최종 리뷰 개수 확인
        try:
            final_reviews = driver.find_elements(
                By.CSS_SELECTOR,
                "div.jftiEf"
            )
            print(f"[정보] 최종 수집 가능한 리뷰 개수: {len(final_reviews)}개")
        except:
            pass
        
        # 최종 리뷰 추출
        print(f"[정보] {name} ({district}) - 리뷰 추출 시작")
        reviews = extract_reviews_from_page(driver, max_reviews)
        
        # 최종 중복 제거 (리뷰 텍스트 + 작성자 조합으로, UI 텍스트 정규화)
        seen = set()
        unique_reviews = []
        for review in reviews:
            review_text = review.get("리뷰", "")
            author = review.get("작성자", "")
            rating = review.get("평점", "")
            
            # 리뷰 텍스트 정규화 (메타데이터 제거)
            normalized = normalize_review_text(review_text)
            
            # 정규화된 텍스트의 처음 150자 + 작성자로 중복 체크
            # 평점은 중복 체크에 포함하지 않음 (같은 리뷰가 평점 유무로 중복 수집되는 것 방지)
            text_key = normalized[:150].strip() if normalized else ""
            key = f"{author}|{text_key}"
            if key and key not in seen:
                seen.add(key)
                # 정규화된 텍스트로 업데이트
                review["리뷰"] = normalized
                unique_reviews.append(review)
        
        if len(reviews) != len(unique_reviews):
            print(f"[정보] 최종 중복 제거: {len(reviews)}개 → {len(unique_reviews)}개")
        
        reviews = unique_reviews[:max_reviews]  # 최대 개수 제한
        
        # place_id와 좌표 추출 시도
        place_id = None
        lat = None
        lng = None
        
        try:
            # URL에서 place_id 추출
            current_url = driver.current_url
            place_match = re.search(r"place/([^/]+)", current_url)
            if place_match:
                place_id = place_match.group(1)
            
            # 좌표는 페이지 소스나 JavaScript 변수에서 추출 가능
            page_source = driver.page_source
            coord_match = re.search(r'\[(-?\d+\.\d+),(-?\d+\.\d+)\]', page_source)
            if coord_match:
                lat = float(coord_match.group(1))
                lng = float(coord_match.group(2))
        except Exception:
            pass
        
        # 각 리뷰에 메타데이터 추가
        for review in reviews:
            review[NAME_COL] = name
            review[DISTRICT_COL] = district
            review["행정동명"] = eupmyeon
            review["place_id"] = place_id if place_id else ""
            review["lat"] = lat if lat else ""
            review["lng"] = lng if lng else ""
        
    except WebDriverException as e:
        print(f"[WebDriver 오류] {name} ({district}) → {e}")
    except Exception as e:
        print(f"[예외] {name} ({district}) → {e}")
    
    return reviews


def find_last_processed_index(cafes_df: pd.DataFrame, resume_df: pd.DataFrame) -> int:
    """
    재개 CSV에서 마지막으로 처리된 카페의 인덱스를 찾아 반환합니다.
    해당 인덱스의 다음부터 시작하도록 합니다.
    
    Args:
        cafes_df: 원본 카페 데이터프레임
        resume_df: 재개용 CSV 데이터프레임
    
    Returns:
        마지막 처리된 카페의 인덱스 (다음부터 시작하려면 +1 필요)
    """
    if resume_df is None or len(resume_df) == 0:
        return -1
    
    # 재개 CSV의 마지막 행에서 카페 정보 추출
    last_row = resume_df.iloc[-1]
    last_name = str(last_row[NAME_COL]).strip()
    last_district = str(last_row[DISTRICT_COL]).strip()
    
    # 원본 데이터프레임에서 해당 카페의 인덱스 찾기
    for idx, row in cafes_df.iterrows():
        name = str(row[NAME_COL]).strip()
        district = str(row[DISTRICT_COL]).strip()
        if name == last_name and district == last_district:
            return idx
    
    # 찾지 못한 경우 -1 반환 (처음부터 시작)
    return -1


def main() -> int:
    args = parse_args()
    
    cafes_df = filter_targets(load_dataframe(args.input))
    if args.limit:
        cafes_df = cafes_df.head(args.limit)
    print(f"총 {len(cafes_df)}개 카페 표본 로드 완료")
    
    resume_df = None
    start_index = 0
    if args.resume and args.resume.exists():
        resume_df = filter_targets(load_dataframe(args.resume))
        print(f"기존 리뷰 {len(resume_df)}건을 불러왔습니다.")
        
        # 마지막 처리된 카페의 인덱스 찾기
        last_index = find_last_processed_index(cafes_df, resume_df)
        if last_index >= 0:
            start_index = last_index + 1
            print(f"마지막 처리된 카페 인덱스: {last_index}, 다음부터 시작: {start_index}")
        else:
            print("마지막 처리된 카페를 찾을 수 없습니다. 처음부터 시작합니다.")
    
    driver: webdriver.Chrome | None = None
    collected: List[Dict[str, str]] = []
    if resume_df is not None:
        collected.extend(resume_df.to_dict(orient="records"))
    
    try:
        driver = init_driver(args.headless)
        print("Chrome 드라이버 초기화 완료")
        
        # start_index부터 시작
        cafes_to_process = cafes_df.iloc[start_index:]
        for idx, row in tqdm(cafes_to_process.iterrows(), total=len(cafes_to_process), desc="구글 리뷰 수집"):
            name = str(row[NAME_COL]).strip()
            district = str(row[DISTRICT_COL]).strip()
            eupmyeon = str(row.get(EUPMYEON_COL, "") or "").strip()
            
            reviews = get_google_reviews_scraped(
                driver,
                name=name,
                district=district,
                eupmyeon=eupmyeon,
                max_reviews=args.max_reviews,
                timeout=args.timeout,
                scroll_pause=args.scroll_pause,
            )
            
            if reviews:
                collected.extend(reviews)
                print(f"✅ {name} ({district}): {len(reviews)}개 리뷰 수집")
            else:
                print(f"⚠️  {name} ({district}): 리뷰를 찾을 수 없습니다")
            
            # 지연 시간
            delay = max(args.delay + random.uniform(-0.5, 0.5), 1.0)
            time.sleep(delay)
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"치명적 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if driver is not None:
            driver.quit()
            print("Chrome 드라이버 종료")
    
    if collected:
        output_df = pd.DataFrame(collected)
        # 컬럼 순서 정리
        column_order = [
            NAME_COL,
            DISTRICT_COL,
            "행정동명",
            "place_id",
            "lat",
            "lng",
            "작성자",
            "평점",
            "리뷰",
            "작성일",
            "언어",
        ]
        # 존재하는 컬럼만 선택
        available_cols = [col for col in column_order if col in output_df.columns]
        output_df = output_df[available_cols]
        
        output_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"✅ 총 {len(output_df)}개 리뷰 저장 완료 → {args.output}")
    else:
        print("❌ 리뷰가 수집되지 않았습니다.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

