from __future__ import annotations

"""
카카오 지도 길찾기 URL에서 도보 거리·시간 정보를 추출하는 유틸리티.

사용 예시:
    python scripts/kakao_walking_scraper.py \
        "https://map.kakao.com/link/by/walk/에이치스퀘어,37.402056,127.108212/알파돔타워,37.394245407468,127.110306812433/카카오판교아지트,37.3952969470752,127.110449292622"
"""

import argparse
import re
from typing import Iterable, Optional, Tuple, Union
from urllib.parse import quote
import json

import requests
from bs4 import BeautifulSoup

KAKAO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    )
}

DISTANCE_TIME_PATTERN = re.compile(
    r"(?P<distance>\d+(?:\.\d+)?)\s*(?P<unit>km|킬로미터|m|미터)"
    r".{0,60}?"
    r"(?:(?P<hours>\d+)\s*(?:시간|hour)s?)?\s*"
    r"(?P<minutes>\d+(?:\.\d+)?)\s*(?:분|minute)s?",
    flags=re.IGNORECASE,
)


def get_kakao_walking_time_distance(url: str, *, debug: bool = False) -> Tuple[Optional[float], Optional[int]]:
    """
    카카오맵 길찾기 URL에서 도보 경로 거리(km)와 시간(분)을 추출합니다.

    Args:
        url: https://map.kakao.com/link/by/walk/... 형태의 길찾기 URL.

    Returns:
        (distance_km, duration_min) 또는 (None, None)
    """
    if not url.startswith("http"):
        raise ValueError("카카오맵 절대경로 URL을 넣어주세요.")

    safe_url = quote(url, safe=":/?&=,.")
    try:
        response = requests.get(safe_url, headers=KAKAO_HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        if debug:
            print(f"[ERROR] URL 요청 실패: {exc}")
        raise
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding

    soup = BeautifulSoup(response.text, "html.parser")

    # 0) title에 요약이 들어올 때(모바일/PC UI 차이) 대비
    if soup.title and soup.title.string:
        match = DISTANCE_TIME_PATTERN.search(soup.title.string)
        if match:
            return _normalize_distance_time(match)
        if debug:
            print("[DEBUG] 타이틀 문자열에 거리/시간 정보 없음")

    # 1) data-react-props 형태의 JSON에서 정보 찾기
    props_checked = False
    for tag in soup.find_all(attrs={"data-react-props": True}):
        props_raw = tag.get("data-react-props")
        if not props_raw:
            continue
        props_checked = True
        results = _search_in_serialized_props(props_raw, debug=debug)
        if results:
            return results
    if debug:
        if props_checked:
            print("[DEBUG] data-react-props JSON에서 거리/시간 정보를 찾지 못했습니다.")
        else:
            print("[DEBUG] data-react-props 속성을 가진 노드가 없습니다.")

    # 2) aria-label 등 접근성 텍스트 탐색
    aria_checked = False
    for tag in soup.find_all(attrs={"aria-label": True}):
        label = tag.get("aria-label") or ""
        aria_checked = True
        match = DISTANCE_TIME_PATTERN.search(label)
        if match:
            if debug:
                _debug_match("aria-label", label, match)
            return _normalize_distance_time(match)
    if debug:
        if aria_checked:
            print("[DEBUG] aria-label 속성에서 거리/시간 정보를 찾지 못했습니다.")
        else:
            print("[DEBUG] aria-label 속성을 가진 노드가 없습니다.")

    # 3) 일반 텍스트 탐색
    text_blob = soup.get_text(" ", strip=True)
    match = DISTANCE_TIME_PATTERN.search(text_blob)
    if match:
        if debug:
            _debug_match("page-text", text_blob, match)
        return _normalize_distance_time(match)
    elif debug:
        print("[DEBUG] 페이지 전체 텍스트에서 거리/시간 패턴을 찾지 못했습니다.")

    sections_checked = False
    for section in soup.select("div, span, li"):
        section_text = section.get_text(" ", strip=True)
        if not section_text:
            continue
        sections_checked = True
        match = DISTANCE_TIME_PATTERN.search(section_text)
        if match:
            if debug:
                _debug_match("section", section_text, match)
            return _normalize_distance_time(match)
    if debug:
        if sections_checked:
            print("[DEBUG] div/span/li 텍스트에서도 패턴을 찾지 못했습니다.")
        else:
            print("[DEBUG] div/span/li 텍스트가 비어 있어 추가 탐색을 건너뜁니다.")

    # 4) script 태그에서 JSON 형태의 경로 요약 텍스트를 찾는 시도
    scripts_checked = False
    for script in soup.find_all("script"):
        script_text = script.get_text(strip=True)
        if not script_text:
            continue
        scripts_checked = True
        match = DISTANCE_TIME_PATTERN.search(script_text)
        if match:
            if debug:
                _debug_match("script", script_text, match)
            return _normalize_distance_time(match)
    if debug:
        if scripts_checked:
            print("[DEBUG] script 태그 내부에서도 거리/시간 정보를 찾지 못했습니다.")
        else:
            print("[DEBUG] script 태그 텍스트가 비어 있거나 존재하지 않습니다.")

        sample = soup.get_text(" ", strip=True)[:200].replace("\n", " ")
        print(f"[DEBUG] 페이지 텍스트 앞 200자: {sample!r}")

    return None, None


def _search_in_serialized_props(props_raw: str, *, debug: bool = False) -> Optional[Tuple[float, int]]:
    """
    data-react-props 속성 내부의 JSON 문자열을 파싱해 거리/시간 텍스트를 탐색합니다.
    """
    try:
        props_json = json.loads(props_raw)
    except json.JSONDecodeError:
        match = DISTANCE_TIME_PATTERN.search(props_raw)
        if match:
            if debug:
                _debug_match("data-react-props(raw)", props_raw, match)
            return _normalize_distance_time(match)
        return None

    for value in _iter_nested_strings(props_json):
        match = DISTANCE_TIME_PATTERN.search(value)
        if match:
            if debug:
                _debug_match("data-react-props(json)", value, match)
            return _normalize_distance_time(match)
    return None


def _iter_nested_strings(data: Union[dict, list, tuple, str]) -> Iterable[str]:
    if isinstance(data, str):
        yield data
    elif isinstance(data, dict):
        for value in data.values():
            yield from _iter_nested_strings(value)
    elif isinstance(data, (list, tuple)):
        for item in data:
            yield from _iter_nested_strings(item)


def _normalize_distance_time(match: re.Match) -> Tuple[float, int]:
    raw_distance = float(match.group("distance"))
    unit = match.group("unit").lower()
    if unit in {"m", "미터"}:
        distance_km = raw_distance / 1000.0
    else:
        distance_km = raw_distance

    hours = match.group("hours")
    minutes = float(match.group("minutes"))
    total_minutes = minutes
    if hours:
        total_minutes += int(hours) * 60

    return distance_km, int(round(total_minutes))


def _debug_match(source: str, context: str, match: re.Match, *, width: int = 120) -> None:
    start, end = match.span()
    snippet_start = max(0, start - 40)
    snippet_end = min(len(context), end + 40)
    snippet = context[snippet_start:snippet_end].replace("\n", " ")
    if len(snippet) > width:
        snippet = snippet[:width] + "…"
    print(f"[DEBUG] matched in {source}: '{match.group(0)}' | snippet: {snippet}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="카카오맵 길찾기 URL에서 도보 거리와 소요 시간을 추출합니다."
    )
    parser.add_argument("url", type=str, help="https://map.kakao.com/link/by/walk/... 형태의 URL")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="추출 과정에서 매치 후보 텍스트를 표준 출력으로 보여줍니다.",
    )
    args = parser.parse_args()

    distance_km, duration_min = get_kakao_walking_time_distance(args.url, debug=args.debug)
    if distance_km is None or duration_min is None:
        print("❌ 거리/시간 정보를 찾지 못했습니다. (경로가 없는 URL이거나, 페이지 구조가 변경되었을 수 있습니다.)")
        if not args.debug:
            print("   → 원인을 확인하려면 --debug 옵션을 함께 실행해보세요.")
        return 1

    print(f"거리: {distance_km} km, 예상 도보 시간: {duration_min} 분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

