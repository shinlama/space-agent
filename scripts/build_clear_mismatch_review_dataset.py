from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from clean_review_text_rows import classify_nontext_rows


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OBVIOUS_COLLISION_MIN_ASSIGNMENTS = 20

FACILITY_KEYWORDS = (
    "병원",
    "의료원",
    "클리닉",
    "호텔",
    "모텔",
    "리조트",
    "대학교",
    "대학",
    "고등학교",
    "중학교",
    "초등학교",
    "학교",
    "이마트",
    "롯데마트",
    "홈플러스",
    "코스트코",
    "백화점",
    "면세점",
    "쇼핑몰",
    "아울렛",
    "타임스퀘어",
    "스타시티몰",
    "아이파크몰",
    "테크노마트",
    "코엑스",
    "공항",
    "터미널",
    "아파트",
    "오피스",
    "방송국",
    "신문",
    "박물관",
    "미술관",
    "도서관",
    "공원",
    "체육관",
    "경기장",
    "교회",
    "성당",
    "사찰",
    "구청",
    "주민센터",
    "복지관",
    "장례식장",
)

CHAIN_ALIASES = {
    "스타벅스": ("스타벅스", "starbucks"),
    "커피빈": ("커피빈", "coffeebean", "thecoffeebean"),
    "이디야": ("이디야", "이디아", "ediya"),
    "메가MGC커피": ("메가mgc", "메가엠지씨", "megacoffee"),
    "투썸플레이스": ("투썸", "atwosomeplace"),
    "파리바게뜨": ("파리바게뜨", "파리바게트", "parisbaguette"),
    "뚜레쥬르": ("뚜레쥬르", "touslesjours"),
    "컴포즈커피": ("컴포즈", "composecoffee"),
    "매머드커피": ("매머드", "mammothcoffee"),
    "공차": ("공차", "gongcha"),
    "던킨": ("던킨", "dunkin"),
    "빽다방": ("빽다방", "paikdabang"),
    "폴바셋": ("폴바셋", "paulbassett"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an expanded review dataset that excludes only clear place-link "
            "mismatches and preserves unresolved unique links."
        )
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "google_reviews_full_corrected_textclean_v4.csv"
        ),
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "place_revalidation_classified_v5.csv",
    )
    parser.add_argument(
        "--recrawled",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "google_reviews_recrawled_corrected_v4.csv",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "google_reviews_recrawled_corrected_v4_progress.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "google_reviews_clear_mismatch_v6.csv",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=(
            PROJECT_ROOT / "outputs" / "place_revalidation_clear_mismatch_v6.csv"
        ),
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def normalize_name(value: object) -> str:
    return "".join(character.lower() for character in str(value or "") if character.isalnum())


def facility_target(name: object) -> bool:
    text = normalize_name(name)
    return any(normalize_name(keyword) in text for keyword in FACILITY_KEYWORDS)


def mismatched_chain(source_name: object, linked_name: object) -> str:
    source = normalize_name(source_name)
    linked = normalize_name(linked_name)
    for chain, aliases in CHAIN_ALIASES.items():
        normalized_aliases = tuple(normalize_name(alias) for alias in aliases)
        linked_matches = any(alias in linked for alias in normalized_aliases)
        source_matches = any(alias in source for alias in normalized_aliases)
        if linked_matches and not source_matches:
            return chain
    return ""


def classify_expanded_actions(classification: pd.DataFrame) -> pd.DataFrame:
    frame = classification.copy()
    store_column = frame.columns[0]
    source_name_column = frame.columns[1]
    numeric = lambda column: pd.to_numeric(frame[column], errors="coerce")

    manual = frame["final_action"].eq("manual_review")
    frame["facility_target"] = frame["existing_place_name"].map(facility_target)
    frame["mismatched_chain"] = frame.apply(
        lambda row: mismatched_chain(
            row[source_name_column], row["existing_place_name"]
        ),
        axis=1,
    )

    strict_corroboration = (
        manual
        & frame["category_match"].eq("True")
        & frame["road_number_match"].eq("True")
        & numeric("corrected_distance_m").le(50)
        & numeric("existing_corrected_name_similarity").ge(0.8)
    )
    frame["strict_current_corroboration"] = strict_corroboration

    collision = manual & numeric("current_target_assignment_count").ge(
        OBVIOUS_COLLISION_MIN_ASSIGNMENTS
    )
    corroborated_winner = pd.Series(False, index=frame.index)
    candidates = frame.loc[collision & strict_corroboration].copy()
    if not candidates.empty:
        candidates["_source_existing_similarity"] = numeric(
            "existing_name_similarity"
        ).loc[candidates.index]
        candidates["_corrected_distance"] = numeric("corrected_distance_m").loc[
            candidates.index
        ]
        winners = (
            candidates.sort_values(
                [
                    "current_place_slug",
                    "_source_existing_similarity",
                    "_corrected_distance",
                    store_column,
                ],
                ascending=[True, False, True, True],
            )
            .drop_duplicates("current_place_slug", keep="first")
            .index
        )
        corroborated_winner.loc[winners] = True
    frame["collision_corroborated_winner"] = corroborated_winner

    reasons: list[list[str]] = [[] for _ in range(len(frame))]
    reason_by_index = dict(zip(frame.index, reasons))

    def add_reason(mask: pd.Series, reason: str) -> None:
        for index in frame.index[mask]:
            reason_by_index[index].append(reason)

    add_reason(collision & ~corroborated_winner, "shared_google_target")
    add_reason(manual & frame["facility_target"], "non_cafe_parent_facility")
    add_reason(manual & frame["mismatched_chain"].ne(""), "different_chain_brand")

    frame["clear_mismatch_reason"] = [
        "|".join(reason_by_index[index]) for index in frame.index
    ]
    clear_manual_mismatch = manual & frame["clear_mismatch_reason"].ne("")

    frame["expanded_action"] = "exclude"
    frame.loc[
        frame["final_action"].eq("keep_existing"), "expanded_action"
    ] = "keep_existing_verified"
    frame.loc[
        frame["final_action"].eq("recrawl_corrected"), "expanded_action"
    ] = "use_recrawled_corrected"
    frame.loc[
        manual & ~clear_manual_mismatch & strict_corroboration,
        "expanded_action",
    ] = "keep_existing_recovered"
    frame.loc[
        manual & ~clear_manual_mismatch & ~strict_corroboration,
        "expanded_action",
    ] = "keep_existing_unresolved"
    frame.loc[clear_manual_mismatch, "expanded_action"] = "exclude_clear_mismatch"
    frame.loc[
        frame["final_action"].eq("exclude_non_cafe"),
        "clear_mismatch_reason",
    ] = "verified_non_cafe_target"
    frame.loc[
        frame["final_action"].eq("exclude_duplicate_target"),
        "clear_mismatch_reason",
    ] = "duplicate_corrected_target"
    return frame


def load_recrawled_replacements(
    args: argparse.Namespace,
    recrawl_ids: set[str],
    store_column: str,
    current_columns: list[str],
) -> tuple[pd.DataFrame, set[str]]:
    progress = pd.read_csv(args.progress, encoding="utf-8-sig", dtype=str).fillna("")
    if "checked_at" in progress.columns:
        progress["_checked_at"] = pd.to_datetime(
            progress["checked_at"], errors="coerce"
        )
        progress = progress.sort_values("_checked_at", kind="stable").drop(
            columns="_checked_at"
        )
    progress = progress.drop_duplicates(store_column, keep="last")
    completed_ids = set(
        progress.loc[progress["status"].eq("completed"), store_column]
    ) & recrawl_ids

    recrawled = pd.read_csv(args.recrawled, encoding="utf-8-sig", dtype=str)
    recrawled = recrawled[recrawled[store_column].isin(completed_ids)].copy()
    recrawled = recrawled.dropna(subset=["리뷰"])
    recrawled["리뷰"] = recrawled["리뷰"].astype(str).str.strip()
    recrawled = recrawled[recrawled["리뷰"].ne("")]
    dedupe_columns = [store_column]
    if "review_id" in recrawled.columns:
        dedupe_columns.append("review_id")
    else:
        dedupe_columns.extend(["작성자", "리뷰"])
    recrawled = recrawled.drop_duplicates(dedupe_columns, keep="last")
    return recrawled.reindex(columns=current_columns), completed_ids


def main() -> None:
    configure_console_encoding()
    args = parse_args()
    args.output = args.output.resolve()
    args.audit_output = args.audit_output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)

    classification = pd.read_csv(
        args.classification, encoding="utf-8-sig", dtype=str
    ).fillna("")
    audit = classify_expanded_actions(classification)
    audit.to_csv(args.audit_output, index=False, encoding="utf-8-sig")

    store_column = classification.columns[0]
    keep_actions = {
        "keep_existing_verified",
        "keep_existing_recovered",
        "keep_existing_unresolved",
    }
    keep_ids = set(
        audit.loc[audit["expanded_action"].isin(keep_actions), store_column]
    )
    recrawl_ids = set(
        audit.loc[
            audit["expanded_action"].eq("use_recrawled_corrected"), store_column
        ]
    )

    current_columns = list(
        pd.read_csv(args.current, encoding="utf-8-sig", nrows=0).columns
    )
    replacement, completed_ids = load_recrawled_replacements(
        args, recrawl_ids, store_column, current_columns
    )

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    input_rows = 0
    kept_current_rows = 0
    first = True
    for chunk in pd.read_csv(
        args.current,
        encoding="utf-8-sig",
        dtype=str,
        chunksize=args.chunksize,
        on_bad_lines="skip",
    ):
        input_rows += len(chunk)
        kept = chunk[chunk[store_column].isin(keep_ids)]
        kept_current_rows += len(kept)
        kept.to_csv(
            temporary,
            mode="w" if first else "a",
            header=first,
            index=False,
            encoding="utf-8-sig" if first else "utf-8",
        )
        first = False

    replacement.to_csv(
        temporary,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8",
    )
    os.replace(temporary, args.output)

    merged = pd.read_csv(args.output, encoding="utf-8-sig", dtype=str).fillna("")
    nontext_reason = classify_nontext_rows(merged)
    nontext_rows_removed = int(nontext_reason.ne("").sum())
    merged = merged.loc[nontext_reason.eq("")].copy()
    dedupe_columns = [store_column, "작성자", "리뷰"]
    duplicate_rows_removed = int(
        merged.duplicated(dedupe_columns, keep="last").sum()
    )
    merged = merged.drop_duplicates(dedupe_columns, keep="last")
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")

    summary = {
        "current": str(args.current.resolve()),
        "classification": str(args.classification.resolve()),
        "recrawled": str(args.recrawled.resolve()),
        "progress": str(args.progress.resolve()),
        "output": str(args.output),
        "audit_output": str(args.audit_output),
        "policy": "exclude_only_clear_place_link_mismatches",
        "input_rows": input_rows,
        "expanded_actions": {
            key: int(value)
            for key, value in audit["expanded_action"].value_counts().items()
        },
        "clear_mismatch_reasons": {
            key: int(value)
            for key, value in (
                audit.loc[
                    audit["clear_mismatch_reason"].ne(""),
                    "clear_mismatch_reason",
                ].value_counts()
            ).items()
        },
        "kept_current_places": len(keep_ids),
        "kept_current_rows": kept_current_rows,
        "recrawl_target_places": len(recrawl_ids),
        "recrawl_completed_places": len(completed_ids),
        "recrawl_places_with_text": int(replacement[store_column].nunique()),
        "recrawled_text_rows": len(replacement),
        "nontext_rows_removed_after_merge": nontext_rows_removed,
        "duplicate_rows_removed_after_merge": duplicate_rows_removed,
        "output_places": int(merged[store_column].nunique()),
        "output_rows": len(merged),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
