from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORE_ID_COLUMN = "상가업소번호"

CONFIRMED_MISMATCHES = {
    "MA0101202505A0052865": "카페플로리안",
    "MA010120220803913346": "오타르DDP점",
    "MA0101202303A0000033": "버블에그",
    "MA0101202501A0039340": "북서울스넥",
    "MA0106202201A2364743": "서울아트센터",
    "MA010120220810440902": "바스켓12계동점",
    "MA0101202310A0093029": "카페소림",
    "MA0101202305A0103250": "부스럭",
    "MA0101202305A0124638": "태린",
    "MA010120220812650776": "세종문화회관카페베네",
    # District-list audit: the Google review target did not match the source place.
    "MA010120220801039580": "주스집 (흑우정 리뷰)",
    "MA010120220812948192": "서록다원 (선운각 리뷰)",
    "MA010120220813256047": "손만동제과 (연남동 만동제과 리뷰)",
    "MA0101202303A0040122": "브니엘베이커리 (타 제과점 리뷰)",
    "MA010120220800920071": "친구네과자집 (고깃집 리뷰)",
    "MA0106202201A2364794": "월계트레이더스 (대형마트 리뷰)",
    "MA0101202412A0047033": "라라플로아창동하나로마트점 (하나로마트 리뷰)",
    "MA010120220813015565": "유이네 (타 대형카페 리뷰)",
    "MA0101202307A0034435": "태극당롯데월드몰점 (롯데월드몰 리뷰)",
    "MA0106202501A0499854": "디자인플라자 (DDP 리뷰)",
    "MA010120220805418966": "목 (해장국집 리뷰)",
    "MA0101202505A0036594": "설고단보타닉파크 (웨딩홀 리뷰)",
    "MA0101202309A0047667": "슬레이케이크 (타 제과점 리뷰)",
    "MA0101202406A0496262": "서울휴게소 (고속도로 휴게소 리뷰)",
    "MA010120220805042101": "카페공간학 (교보문고 리뷰)",
    "MA010120220809871666": "카페이티씨. (복합쇼핑몰 리뷰)",
    "MA0101202211A0000781": "Myohyae (Megabox cinema reviews)",
    "MA010120220804268801": "The Cafe Garden Five VIP (Garden Five mall reviews)",
    "MA010120220804298640": "Hanwoori Distribution (Garak seafood market reviews)",
    "MA010120220804302534": "Bonifacio Store (Garak seafood market reviews)",
    "MA010120220804301640": "Hamchorom Distribution (Garak seafood market reviews)",
    "MA010120220802433986": "Vintage (NC department store reviews)",
    "MA010120220808564122": "Mammas (NC department store reviews)",
    "MA0101202406A0140046": "Pain et Moi (fashion showroom reviews)",
    "MA010120220800252042": "Lion King (Cartoon Plus comic cafe reviews)",
}

SOURCE_REVIEWS = PROJECT_ROOT / "data" / "google_reviews_clear_mismatch_v6.csv"
CANDIDATE_REVIEWS = PROJECT_ROOT / "data" / "spatial_review_candidates_v6.csv"
BATCH_DIR = PROJECT_ROOT / "outputs" / "openai_batch_full_gpt54nano_v6"
MAPPING_REVIEWS = BATCH_DIR / "placeness_mapping_reviews.csv"
MAPPING_SENTENCES = BATCH_DIR / "placeness_mapping_sentences.csv"


def filter_csv_in_place(path: Path, chunksize: int = 100_000) -> dict[str, object]:
    temp_path = path.with_suffix(path.suffix + ".postclean.tmp")
    temp_path.unlink(missing_ok=True)

    before_rows = 0
    after_rows = 0
    removed_by_store: Counter[str] = Counter()
    before_places: set[str] = set()
    after_places: set[str] = set()
    wrote_header = False

    try:
        reader = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype={STORE_ID_COLUMN: "string"},
            chunksize=chunksize,
            low_memory=False,
        )
        with reader:
            for chunk in reader:
                if STORE_ID_COLUMN not in chunk.columns:
                    raise ValueError(f"{path} does not contain {STORE_ID_COLUMN}")

                store_ids = chunk[STORE_ID_COLUMN].fillna("").astype(str)
                remove_mask = store_ids.isin(CONFIRMED_MISMATCHES)
                before_rows += len(chunk)
                before_places.update(store_ids[store_ids.ne("")].unique().tolist())
                removed_by_store.update(store_ids[remove_mask].tolist())

                kept = chunk.loc[~remove_mask].copy()
                kept_ids = kept[STORE_ID_COLUMN].fillna("").astype(str)
                after_rows += len(kept)
                after_places.update(kept_ids[kept_ids.ne("")].unique().tolist())
                kept.to_csv(
                    temp_path,
                    mode="w" if not wrote_header else "a",
                    header=not wrote_header,
                    index=False,
                    encoding="utf-8-sig" if not wrote_header else "utf-8",
                    lineterminator="\n",
                )
                wrote_header = True

        if before_rows - after_rows != sum(removed_by_store.values()):
            raise RuntimeError(f"Row-count verification failed for {path}")

        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return {
        "path": str(path),
        "before_rows": before_rows,
        "after_rows": after_rows,
        "removed_rows": before_rows - after_rows,
        "before_places": len(before_places),
        "after_places": len(after_places),
        "removed_places": len(before_places - after_places),
        "removed_rows_by_store": {
            store_id: removed_by_store.get(store_id, 0)
            for store_id in CONFIRMED_MISMATCHES
        },
    }


def count_exact_matches(path: Path) -> tuple[int, int]:
    total = 0
    exact = 0
    for chunk in pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=["evidence_exact_match"],
        chunksize=100_000,
        low_memory=False,
    ):
        values = chunk["evidence_exact_match"].astype(str).str.lower()
        total += len(values)
        exact += values.eq("true").sum()
    return total, int(exact)


def update_json(path: Path, updates: dict[str, object]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(updates)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    targets = [SOURCE_REVIEWS, CANDIDATE_REVIEWS, MAPPING_REVIEWS, MAPPING_SENTENCES]
    missing = [str(path) for path in targets if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    source_summary_path = PROJECT_ROOT / "data" / "google_reviews_clear_mismatch_v6.summary.json"
    candidate_summary_path = PROJECT_ROOT / "data" / "spatial_review_candidates_v6.summary.json"
    finalize_summary_path = BATCH_DIR / "finalize_summary.json"
    source_baseline = json.loads(source_summary_path.read_text(encoding="utf-8"))
    candidate_baseline = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    finalize_baseline = json.loads(finalize_summary_path.read_text(encoding="utf-8"))
    original_source_rows = int(source_baseline["output_rows"]) + int(
        source_baseline.get("postclean_removed_rows", 0)
    )
    original_source_places = int(source_baseline["output_places"]) + int(
        source_baseline.get("postclean_removed_places", 0)
    )
    original_candidate_rows = int(candidate_baseline["candidate_rows"]) + int(
        candidate_baseline.get("postclean_removed_rows", 0)
    )
    original_candidate_places = int(candidate_baseline["candidate_places"]) + int(
        candidate_baseline.get("postclean_removed_places", 0)
    )
    original_mapping_reviews = int(finalize_baseline["expected_reviews"]) + int(
        finalize_baseline.get("postclean_removed_reviews", 0)
    )
    original_mapping_rows = int(finalize_baseline["mapping_rows"]) + int(
        finalize_baseline.get("postclean_removed_mapping_rows", 0)
    )

    results = [filter_csv_in_place(path) for path in targets]
    by_path = {Path(result["path"]): result for result in results}

    source = by_path[SOURCE_REVIEWS]
    candidates = by_path[CANDIDATE_REVIEWS]
    mapping_reviews = by_path[MAPPING_REVIEWS]
    mapping_sentences = by_path[MAPPING_SENTENCES]

    mapped_review_frame = pd.read_csv(
        MAPPING_REVIEWS,
        encoding="utf-8-sig",
        usecols=[STORE_ID_COLUMN, "review_index", "mapping_count"],
        dtype={STORE_ID_COLUMN: "string", "review_index": "string"},
        low_memory=False,
    )
    mapped_mask = pd.to_numeric(
        mapped_review_frame["mapping_count"], errors="coerce"
    ).fillna(0).gt(0)
    mapped_reviews = int(mapped_review_frame.loc[mapped_mask, "review_index"].nunique())
    mapped_places = int(mapped_review_frame.loc[mapped_mask, STORE_ID_COLUMN].nunique())

    _, exact_rows = count_exact_matches(MAPPING_SENTENCES)
    mapping_rows = int(mapping_sentences["after_rows"])

    update_json(
        source_summary_path,
        {
            "output_places": source["after_places"],
            "output_rows": source["after_rows"],
            "postclean_removed_places": original_source_places - source["after_places"],
            "postclean_removed_rows": original_source_rows - source["after_rows"],
        },
    )
    update_json(
        candidate_summary_path,
        {
            "input_rows": source["after_rows"],
            "candidate_rows": candidates["after_rows"],
            "candidate_ratio": candidates["after_rows"] / source["after_rows"],
            "candidate_places": candidates["after_places"],
            "postclean_removed_places": original_candidate_places - candidates["after_places"],
            "postclean_removed_rows": original_candidate_rows - candidates["after_rows"],
        },
    )
    update_json(
        finalize_summary_path,
        {
            "expected_reviews": mapping_reviews["after_rows"],
            "returned_reviews": mapping_reviews["after_rows"],
            "mapping_rows": mapping_rows,
            "mapped_reviews": mapped_reviews,
            "mapped_places": mapped_places,
            "unmapped_reviews": mapping_reviews["after_rows"] - mapped_reviews,
            "evidence_exact_substring_rows": exact_rows,
            "evidence_exact_substring_ratio": exact_rows / mapping_rows,
            "postclean_removed_places": original_candidate_places - mapping_reviews["after_places"],
            "postclean_removed_reviews": original_mapping_reviews - mapping_reviews["after_rows"],
            "postclean_removed_mapping_rows": original_mapping_rows - mapping_sentences["after_rows"],
        },
    )

    audit = {
        "removed_store_ids": CONFIRMED_MISMATCHES,
        "files": results,
        "final": {
            "source_review_places": source["after_places"],
            "source_reviews": source["after_rows"],
            "candidate_places": candidates["after_places"],
            "candidate_reviews": candidates["after_rows"],
            "mapping_review_rows": mapping_reviews["after_rows"],
            "mapped_reviews": mapped_reviews,
            "mapped_places": mapped_places,
            "mapping_sentence_rows": mapping_rows,
        },
        "cumulative_removed": {
            "source_review_places": original_source_places - source["after_places"],
            "source_reviews": original_source_rows - source["after_rows"],
            "candidate_places": original_candidate_places - candidates["after_places"],
            "candidate_reviews": original_candidate_rows - candidates["after_rows"],
            "mapping_review_rows": original_mapping_reviews - mapping_reviews["after_rows"],
            "mapping_sentence_rows": original_mapping_rows - mapping_rows,
        },
    }
    audit_path = BATCH_DIR / "confirmed_mismatch_removal_summary.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
