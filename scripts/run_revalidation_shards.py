from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Google 장소 재검증 작업을 여러 Chrome 작업으로 나눠 실행합니다.")
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-per-shard", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--delay-min", type=float, default=0.1)
    parser.add_argument("--delay-max", type=float, default=0.3)
    return parser.parse_args()


def csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[str], object]] = []
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for shard in range(args.workers):
        output = args.output_dir / f"google_place_revalidation_shard_{shard:02d}.csv"
        log_path = args.output_dir / f"google_place_revalidation_shard_{shard:02d}.log"
        command = [
            sys.executable,
            str(args.script),
            "--targets-input",
            str(args.targets),
            "--output",
            str(output),
            "--shard-index",
            str(shard),
            "--shard-count",
            str(args.workers),
            "--timeout",
            str(args.timeout),
            "--delay-min",
            str(args.delay_min),
            "--delay-max",
            str(args.delay_max),
        ]
        if args.limit_per_shard is not None:
            command.extend(["--limit", str(args.limit_per_shard)])
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        processes.append((shard, process, log_handle))

    started = time.time()
    try:
        while True:
            active = sum(process.poll() is None for _, process, _ in processes)
            rows = sum(
                csv_rows(args.output_dir / f"google_place_revalidation_shard_{shard:02d}.csv")
                for shard in range(args.workers)
            )
            elapsed = int(time.time() - started)
            print(f"진행: {rows:,}곳 완료, {active}개 작업 실행 중, {elapsed:,}초 경과", flush=True)
            if active == 0:
                break
            time.sleep(30)
    finally:
        for _, process, log_handle in processes:
            if process.poll() is None:
                process.terminate()
            log_handle.close()

    failures = [(shard, process.returncode) for shard, process, _ in processes if process.returncode]
    if failures:
        print(f"실패 작업: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
