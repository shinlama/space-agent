from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# This lexicon is a high-recall prefilter. A match only selects a review for
# contextual LLM analysis; it does not assign a placeness factor by itself.
FACTOR_CUES: dict[str, tuple[str, ...]] = {
    "심미성": (
        "인테리어", "디자인", "분위기", "감성", "예쁘", "아름답", "멋지",
        "세련", "촌스럽", "장식", "소품", "외관", "건물", "공간", "뷰",
        "전망", "창밖", "포토존", "사진 찍", "사진찍",
    ),
    "개방성": (
        "개방", "탁 트", "트여", "넓", "좁", "협소", "답답", "층고",
        "천장", "통창", "창문", "채광", "햇빛", "시야", "좌석 간격",
        "좌석간격", "테이블 간격", "테이블간격", "여유로운 공간",
    ),
    "감각적 경험": (
        "조명", "밝", "어둡", "음악", "노래", "소리", "소음", "시끄",
        "조용", "향기", "냄새", "악취", "온도", "더워", "덥다", "추워",
        "춥다", "따뜻", "시원", "아늑", "포근", "감각", "촉감",
    ),
    "접근성": (
        "접근", "위치", "지하철", "버스", "정류장", "역에서", "역 근처",
        "역근처", "주차", "발렛", "도보", "걸어서", "찾기 쉽", "찾기 어렵",
        "찾아가기", "교통", "골목", "계단", "엘리베이터", "출입구", "입구",
        "휠체어", "유모차",
    ),
    "쾌적성": (
        "쾌적", "깨끗", "청결", "위생", "더럽", "화장실", "편안", "불편",
        "좌석", "의자", "테이블", "콘센트", "냉방", "난방", "에어컨",
        "환기", "먼지", "벌레", "냄새나", "정돈",
    ),
    "활동성": (
        "공부", "업무", "작업", "노트북", "독서", "회의", "모임", "데이트",
        "휴식", "쉬기", "쉬어", "머물", "체류", "혼자", "아이와", "가족과",
        "반려견", "반려동물", "산책", "대화하기", "책 읽",
    ),
    "상호작용성": (
        "친절", "불친절", "직원", "사장", "응대", "서비스", "소통", "교류",
        "함께", "커뮤니티", "행사", "이벤트", "체험", "워크숍", "클래스",
    ),
    "상징성": (
        "상징", "랜드마크", "대표하는", "명물", "유명한 곳", "역사", "문화",
        "전통", "콘셉트", "컨셉", "테마", "독특", "개성", "특별한 공간",
        "시그니처 공간",
    ),
    "기억 및 선호": (
        "재방문", "다시 방문", "다시방문", "또 오", "또오", "단골", "최애",
        "애정", "좋아하는 공간", "기억", "추억", "인생 카페", "인생카페",
        "취향", "선호", "아끼는", "추천하고 싶", "다시 찾",
    ),
    "지역 정체성": (
        "동네", "지역", "로컬", "마을", "주변 환경", "지역색", "정체성",
        "한옥", "옛 건물", "오래된 건물", "골목 분위기", "동네 분위기",
        "지역 분위기", "역사적", "문화적", "지역 특색",
    ),
}

METADATA_ONLY_PATTERNS = (
    re.compile(r"^\s*[0-5](?:\.0)?\s*$"),
    re.compile(r"^\s*(?:지역\s*가이드|리뷰\s*[\d,]+개|사진\s*[\d,]+장|·|\s)+\s*$"),
)

FACTOR_PATTERNS: dict[str, re.Pattern[str]] = {
    factor: re.compile("|".join(re.escape(cue.lower()) for cue in cues))
    for factor, cues in FACTOR_CUES.items()
}
ALL_CUES_PATTERN = re.compile(
    "|".join(
        re.escape(cue.lower())
        for cues in FACTOR_CUES.values()
        for cue in cues
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select high-recall spatial-experience review candidates for contextual "
            "placeness mapping. Cue matches are not final factor labels."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "google_reviews_validated_v5.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "spatial_review_candidates_v5.csv",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=4,
        help="Minimum number of non-whitespace characters in a candidate review.",
    )
    return parser.parse_args()


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def is_metadata_only(text: str) -> bool:
    return any(pattern.fullmatch(text) for pattern in METADATA_ONLY_PATTERNS)


def match_factor_cues(text: str) -> tuple[list[str], list[str]]:
    normalized = re.sub(r"\s+", " ", text.strip()).lower()
    matched_factors: list[str] = []
    matched_terms: list[str] = []
    for factor, cues in FACTOR_CUES.items():
        if not FACTOR_PATTERNS[factor].search(normalized):
            continue
        factor_terms = [cue for cue in cues if cue.lower() in normalized]
        if factor_terms:
            matched_factors.append(factor)
            matched_terms.extend(factor_terms)
    return matched_factors, list(dict.fromkeys(matched_terms))


def main() -> None:
    configure_console_encoding()
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    temporary = Path(f"{args.output}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    input_rows = 0
    candidate_rows = 0
    candidate_places: set[str] = set()
    factor_counts: Counter[str] = Counter()
    cue_counts: Counter[str] = Counter()
    first = True
    store_column = "상가업소번호"

    for chunk in pd.read_csv(
        args.input,
        encoding="utf-8-sig",
        dtype=str,
        chunksize=args.chunksize,
        on_bad_lines="skip",
    ):
        if "리뷰" not in chunk.columns:
            raise ValueError(f"'리뷰' column is missing: {list(chunk.columns)}")
        input_rows += len(chunk)
        text = chunk["리뷰"].fillna("").astype(str).str.strip()
        compact_length = text.str.replace(r"\s+", "", regex=True).str.len()
        broad_mask = (
            compact_length.ge(args.min_text_chars)
            & text.str.lower().str.contains(ALL_CUES_PATTERN, regex=True, na=False)
        )
        selected_rows: list[dict[str, str]] = []
        for row in chunk.loc[broad_mask].to_dict(orient="records"):
            review_text = str(row.get("리뷰") or "").strip()
            if is_metadata_only(review_text):
                continue
            factors, terms = match_factor_cues(review_text)
            row["prefilter_factors"] = "|".join(factors)
            row["prefilter_terms"] = "|".join(terms)
            selected_rows.append(row)
            factor_counts.update(factors)
            cue_counts.update(terms)
            if store_column in row and row[store_column]:
                candidate_places.add(str(row[store_column]))

        if selected_rows:
            selected = pd.DataFrame(selected_rows)
            selected.to_csv(
                temporary,
                mode="w" if first else "a",
                header=first,
                index=False,
                encoding="utf-8-sig" if first else "utf-8",
            )
            first = False
            candidate_rows += len(selected)

    if first:
        header = pd.read_csv(args.input, encoding="utf-8-sig", nrows=0)
        header["prefilter_factors"] = pd.Series(dtype=str)
        header["prefilter_terms"] = pd.Series(dtype=str)
        header.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(args.output)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "method": "high-recall lexical prefilter before contextual LLM mapping",
        "input_rows": input_rows,
        "candidate_rows": candidate_rows,
        "candidate_ratio": candidate_rows / input_rows if input_rows else 0.0,
        "candidate_places": len(candidate_places),
        "factor_cue_hits": dict(factor_counts.most_common()),
        "top_cue_hits": dict(cue_counts.most_common(50)),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
