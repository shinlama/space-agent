from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


DEFAULT_REVALIDATION = Path(
    r"F:\JW\space-agent\outputs\google_place_revalidation_http_v4.csv"
)
DEFAULT_REVIEWS = Path(r"F:\JW\space-agent\google_reviews_full_min3_max2000.csv")
DEFAULT_OUTPUT_DIR = Path(r"F:\JW\space-agent\outputs\review_recrawl_v4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run corrected Google review recrawling in parallel shards."
    )
    parser.add_argument("--revalidation", type=Path, default=DEFAULT_REVALIDATION)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--max-reviews", type=int, default=2000)
    parser.add_argument("--max-scrolls", type=int, default=350)
    parser.add_argument("--scroll-pause", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def merge_csvs(paths: list[Path], output: Path, dedupe_progress: bool = False) -> int:
    frames = [
        pd.read_csv(path, encoding="utf-8-sig", dtype=str)
        for path in paths
        if path.exists() and path.stat().st_size > 0
    ]
    if not frames:
        return 0
    merged = pd.concat(frames, ignore_index=True)
    if dedupe_progress:
        store_col = merged.columns[0]
        if "checked_at" in merged.columns:
            merged["_checked_at"] = pd.to_datetime(
                merged["checked_at"], errors="coerce"
            )
            merged = merged.sort_values("_checked_at", kind="stable").drop(
                columns="_checked_at"
            )
        merged = merged.drop_duplicates(store_col, keep="last")
    else:
        store_col = merged.columns[0]
        if "review_id" in merged.columns:
            merged = merged.drop_duplicates([store_col, "review_id"], keep="last")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False, encoding="utf-8-sig")
    return len(merged)


def main() -> int:
    args = parse_args()
    if args.shards < 1:
        raise ValueError("shards must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("recrawl_corrected_reviews.py")
    resume_progresses = sorted(args.output_dir.glob("progress_shard_*.csv"))
    processes: list[tuple[int, subprocess.Popen[str], object]] = []
    outputs: list[Path] = []
    progresses: list[Path] = []

    for shard in range(args.shards):
        output = args.output_dir / f"reviews_shard_{shard:02d}.csv"
        progress = args.output_dir / f"progress_shard_{shard:02d}.csv"
        log = args.output_dir / f"shard_{shard:02d}.log"
        outputs.append(output)
        progresses.append(progress)
        command = [
            sys.executable,
            str(script),
            "--revalidation",
            str(args.revalidation),
            "--reviews",
            str(args.reviews),
            "--output",
            str(output),
            "--progress",
            str(progress),
            "--shard-index",
            str(shard),
            "--shard-count",
            str(args.shards),
            "--max-reviews",
            str(args.max_reviews),
            "--max-scrolls",
            str(args.max_scrolls),
            "--scroll-pause",
            str(args.scroll_pause),
            "--timeout",
            str(args.timeout),
        ]
        for resume_progress in resume_progresses:
            command.extend(["--resume-progress", str(resume_progress)])
        log_handle = log.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard, process, log_handle))

    while any(process.poll() is None for _, process, _ in processes):
        completed = sum(csv_rows(path) for path in progresses)
        reviews = sum(csv_rows(path) for path in outputs)
        running = sum(process.poll() is None for _, process, _ in processes)
        print(
            f"[progress] completed-place rows={completed}, text reviews={reviews}, "
            f"running shards={running}",
            flush=True,
        )
        time.sleep(30)

    return_codes: list[int] = []
    for _, process, log_handle in processes:
        return_codes.append(int(process.returncode or 0))
        log_handle.close()

    merged_reviews = merge_csvs(
        outputs,
        args.output_dir.parent / "google_reviews_recrawled_corrected_v4.csv",
    )
    merged_progress = merge_csvs(
        progresses,
        args.output_dir.parent / "google_reviews_recrawled_corrected_v4_progress.csv",
        dedupe_progress=True,
    )
    print(
        f"[done] merged text reviews={merged_reviews}, progress rows={merged_progress}, "
        f"return codes={return_codes}",
        flush=True,
    )
    return 1 if any(return_codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
