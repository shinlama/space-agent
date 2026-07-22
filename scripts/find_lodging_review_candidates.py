from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote_plus


CORE_KEYWORDS = ("숙소", "호텔", "모텔")
LODGING_CONTEXT_KEYWORDS = (
    "객실",
    "체크인",
    "체크아웃",
    "투숙",
    "숙박",
    "침대",
    "프런트",
    "프론트",
    "리셉션",
    "컨시어지",
    "룸서비스",
    "어메니티",
    "욕실",
    "샤워",
    "수건",
    "조식",
    "방음",
    "스위트룸",
    "하룻밤",
    "묵었",
    "묵고",
    "묵는",
    "묵기",
)
CAFE_CONTEXT_KEYWORDS = (
    "카페",
    "커피",
    "음료",
    "디저트",
    "베이커리",
    "케이크",
    "빵",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find places whose reviews contain lodging-related language."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--sample-count", type=int, default=5)
    return parser.parse_args()


def matched_keywords(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword in text]


def classify_risk(
    lodging_only_reviews: int,
    lodging_context_reviews: int,
    keyword_reviews: int,
    total_reviews: int,
) -> str:
    total_ratio = lodging_only_reviews / total_reviews if total_reviews else 0.0
    keyword_ratio = (
        lodging_only_reviews / keyword_reviews if keyword_reviews else 0.0
    )
    if lodging_only_reviews >= 5 or (
        lodging_only_reviews >= 3 and (total_ratio >= 0.05 or keyword_ratio >= 0.5)
    ):
        return "높음"
    if lodging_only_reviews >= 2 or lodging_context_reviews >= 3:
        return "중간"
    return "검토"


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)

    total_by_store: Counter[str] = Counter()
    candidates: dict[str, dict[str, object]] = {}
    total_keyword_rows = 0

    with args.input.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "상가업소번호",
            "상호명",
            "시군구명",
            "행정동명",
            "도로명주소",
            "리뷰",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            store_id = (row.get("상가업소번호") or "").strip()
            total_by_store[store_id] += 1
            review = re.sub(r"\s+", " ", row.get("리뷰") or "").strip()
            core_hits = matched_keywords(review, CORE_KEYWORDS)
            if not core_hits:
                continue

            total_keyword_rows += 1
            lodging_hits = matched_keywords(review, LODGING_CONTEXT_KEYWORDS)
            cafe_hits = matched_keywords(review, CAFE_CONTEXT_KEYWORDS)
            record = candidates.setdefault(
                store_id,
                {
                    "상가업소번호": store_id,
                    "상호명": row.get("상호명") or "",
                    "시군구명": row.get("시군구명") or "",
                    "행정동명": row.get("행정동명") or "",
                    "도로명주소": row.get("도로명주소") or "",
                    "실제_리뷰수집대상": unquote_plus(row.get("place_id") or ""),
                    "키워드_리뷰수": 0,
                    "숙박맥락_리뷰수": 0,
                    "숙박중심_리뷰수": 0,
                    "핵심키워드": Counter(),
                    "보조키워드": Counter(),
                    "대표리뷰": [],
                },
            )
            record["키워드_리뷰수"] = int(record["키워드_리뷰수"]) + 1
            record["핵심키워드"].update(core_hits)
            record["보조키워드"].update(lodging_hits)
            if lodging_hits:
                record["숙박맥락_리뷰수"] = int(record["숙박맥락_리뷰수"]) + 1
            if lodging_hits and not cafe_hits:
                record["숙박중심_리뷰수"] = int(record["숙박중심_리뷰수"]) + 1

            samples = record["대표리뷰"]
            samples.append(
                {
                    "score": len(lodging_hits) * 2 + int(bool(lodging_hits and not cafe_hits)),
                    "text": review[:500],
                }
            )
            samples.sort(key=lambda sample: int(sample["score"]), reverse=True)
            del samples[args.sample_count :]

    output_rows: list[dict[str, object]] = []
    for store_id, record in candidates.items():
        total_reviews = total_by_store[store_id]
        keyword_reviews = int(record["키워드_리뷰수"])
        lodging_context_reviews = int(record["숙박맥락_리뷰수"])
        lodging_only_reviews = int(record["숙박중심_리뷰수"])
        output_rows.append(
            {
                "위험도": classify_risk(
                    lodging_only_reviews,
                    lodging_context_reviews,
                    keyword_reviews,
                    total_reviews,
                ),
                "상가업소번호": record["상가업소번호"],
                "상호명": record["상호명"],
                "시군구명": record["시군구명"],
                "행정동명": record["행정동명"],
                "도로명주소": record["도로명주소"],
                "실제_리뷰수집대상": record["실제_리뷰수집대상"],
                "전체_리뷰수": total_reviews,
                "키워드_리뷰수": keyword_reviews,
                "숙박맥락_리뷰수": lodging_context_reviews,
                "숙박중심_리뷰수": lodging_only_reviews,
                "숙박중심_비율": round(lodging_only_reviews / total_reviews, 4),
                "핵심키워드": ", ".join(
                    f"{key}:{value}"
                    for key, value in record["핵심키워드"].most_common()
                ),
                "보조키워드": ", ".join(
                    f"{key}:{value}"
                    for key, value in record["보조키워드"].most_common()
                ),
                "대표리뷰": " || ".join(
                    str(sample["text"]) for sample in record["대표리뷰"]
                ),
            }
        )

    risk_order = {"높음": 0, "중간": 1, "검토": 2}
    output_rows.sort(
        key=lambda row: (
            risk_order[str(row["위험도"])],
            -int(row["숙박중심_리뷰수"]),
            -int(row["숙박맥락_리뷰수"]),
            -int(row["키워드_리뷰수"]),
        )
    )

    fieldnames = list(output_rows[0]) if output_rows else []
    with args.output.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "matched_review_rows": total_keyword_rows,
        "matched_places": len(output_rows),
        "risk_counts": dict(Counter(row["위험도"] for row in output_rows)),
        "top_candidates": output_rows[:30],
    }
    if args.summary:
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
