from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from map_factors_openai import (
    FACTOR_GUIDE,
    FACTOR_NAMES,
    MAPPING_CONFIDENCE_THRESHOLD,
    SYSTEM_PROMPT,
    split_into_sentences,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "spatial_review_candidates_v6.csv"
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "openai_batch_v6"


RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "placeness_factor_mapping",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "review_index": {"type": "integer"},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sentence_id": {"type": "integer"},
                                    "mappings": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "factor": {
                                                    "type": "string",
                                                    "enum": FACTOR_NAMES,
                                                },
                                                "evidence": {"type": "string"},
                                                "sentiment_hint": {
                                                    "type": "string",
                                                    "enum": [
                                                        "positive",
                                                        "negative",
                                                        "neutral",
                                                        "mixed",
                                                    ],
                                                },
                                            },
                                            "required": [
                                                "factor",
                                                "evidence",
                                                "sentiment_hint",
                                            ],
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                                "required": ["sentence_id", "mappings"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["review_index", "items"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["reviews"],
        "additionalProperties": False,
    },
}


BATCH_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "8. confidence는 0~1 사이로, 해당 요인 언급이라고 판단하는 확신입니다.",
    "8. 요인 정의와 직접적인 원문 근거가 명확할 때만 매핑하세요.",
).replace(
    "15. 원문에 직접적인 근거가 있고 confidence가 0.65 이상일 때만 매핑하세요. 근거가 간접적이거나 추측이 필요하면 매핑하지 마세요.",
    "15. 원문에 직접적인 근거가 있을 때만 매핑하세요. 근거가 간접적이거나 추측이 필요하면 매핑하지 마세요.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, run, submit, retrieve, and validate OpenAI mapping jobs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    prepare.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--reviews-per-request", type=int, default=20)
    prepare.add_argument("--max-shard-mb", type=int, default=150)
    prepare.add_argument("--max-requests-per-shard", type=int, default=600)
    prepare.add_argument("--start-row", type=int, default=0)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--max-sentence-chars", type=int, default=500)
    prepare.add_argument("--max-completion-tokens", type=int, default=8000)
    prepare.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="none",
        help="Applied only to GPT-5-family models.",
    )

    live = subparsers.add_parser("run-live")
    live.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    live.add_argument("--concurrency", type=int, default=10)
    live.add_argument("--max-retries", type=int, default=6)
    live.add_argument("--shard", type=int, action="append")

    submit = subparsers.add_parser("submit")
    submit.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    submit.add_argument("--shard", type=int, action="append")

    status = subparsers.add_parser("status")
    status.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)

    download = subparsers.add_parser("download")
    download.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    download.add_argument("--shard", type=int, action="append")

    retry_failed_parser = subparsers.add_parser("retry-failed")
    retry_failed_parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    retry_failed_parser.add_argument(
        "--max-requests-per-shard", type=int, default=600
    )
    retry_failed_parser.add_argument(
        "--reviews-per-retry-request", type=int, default=5
    )

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)

    return parser.parse_args()


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def factor_guide_text() -> str:
    lines: list[str] = []
    for factor, guide in FACTOR_GUIDE.items():
        lines.append(f"- {factor} ({guide['category']}): {guide['definition']}")
        lines.append(f"  키워드: {', '.join(guide['keywords'])}")
        lines.append(f"  긍정 예시: {guide['positive_examples'][0]}")
        lines.append(f"  부정 예시: {guide['negative_examples'][0]}")
    return "\n".join(lines)


def build_user_prompt(review_payload: list[dict[str, Any]]) -> str:
    return (
        "아래 장소성 요인 가이드를 기준으로 리뷰 문장들을 다중 라벨로 매핑하세요.\n"
        "각 입력 review_index와 각 sentence_id를 누락 없이 정확히 한 번씩 그대로 반환하세요. 관련 요인이 없는 "
        "문장도 items에 포함하되 mappings를 빈 배열로 반환하세요. evidence는 반드시 해당 문장의 "
        "원문 일부를 그대로 사용하세요. confidence와 reason은 출력하지 마세요.\n\n"
        "[장소성 요인 가이드]\n"
        f"{factor_guide_text()}\n\n"
        "[입력 리뷰]\n"
        + json.dumps(review_payload, ensure_ascii=False, separators=(",", ":"))
    )


def request_body(
    model: str,
    payload: list[dict[str, Any]],
    max_completion_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "max_completion_tokens": max_completion_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": RESPONSE_SCHEMA,
        },
        "messages": [
            {"role": "system", "content": BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(payload)},
        ],
    }
    if model.startswith("gpt-5"):
        body["reasoning_effort"] = reasoning_effort
    else:
        body["temperature"] = 0
    return body


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Run manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    path = run_dir / "run_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def selected_frame(manifest: dict[str, Any]) -> pd.DataFrame:
    input_path = Path(manifest["input"])
    current_hash = file_sha256(input_path)
    if current_hash != manifest["input_sha256"]:
        raise RuntimeError("Input CSV changed after preparation; refusing to merge results.")
    frame = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str).fillna("")
    frame["review_index"] = frame.index
    start = int(manifest["start_row"])
    end = start + int(manifest["selected_reviews"])
    return frame.iloc[start:end].copy()


def prepare(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    run_dir = args.run_dir.resolve()
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.reviews_per_request < 1:
        raise ValueError("reviews-per-request must be positive")
    if args.max_requests_per_shard < 1:
        raise ValueError("max-requests-per-shard must be positive")
    if args.reviews_per_retry_request < 1:
        raise ValueError("reviews-per-retry-request must be positive")

    for old_file in input_dir.glob("batch_input_*.jsonl"):
        old_file.unlink()

    frame = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str).fillna("")
    if "리뷰" not in frame.columns or "상호명" not in frame.columns:
        raise ValueError(f"Required columns are missing: {list(frame.columns)}")
    frame["review_index"] = frame.index
    start = max(0, args.start_row)
    end = len(frame) if args.limit is None else min(len(frame), start + args.limit)
    selected = frame.iloc[start:end]
    if selected.empty:
        raise ValueError("No reviews selected")

    max_bytes = max(1, args.max_shard_mb) * 1024 * 1024
    shard_no = 0
    shard_path = input_dir / f"batch_input_{shard_no:03d}.jsonl"
    shard_handle = shard_path.open("w", encoding="utf-8", newline="\n")
    shard_bytes = 0
    shard_requests = 0
    total_requests = 0
    shards: list[dict[str, Any]] = []

    def close_shard() -> None:
        nonlocal shard_handle, shard_bytes, shard_requests, shard_path
        shard_handle.close()
        if shard_requests:
            shards.append(
                {
                    "shard": len(shards),
                    "input_path": str(shard_path),
                    "requests": shard_requests,
                    "bytes": shard_bytes,
                    "input_file_id": None,
                    "batch_id": None,
                    "status": "prepared",
                    "output_file_id": None,
                    "error_file_id": None,
                }
            )

    rows = list(selected.to_dict(orient="records"))
    for offset in range(0, len(rows), args.reviews_per_request):
        review_rows = rows[offset : offset + args.reviews_per_request]
        payload: list[dict[str, Any]] = []
        for row in review_rows:
            sentences = []
            for sentence_id, sentence in enumerate(split_into_sentences(row["리뷰"])):
                sentence = sentence.strip()
                if len(sentence) > args.max_sentence_chars:
                    sentence = sentence[: args.max_sentence_chars].rstrip() + "..."
                sentences.append({"sentence_id": sentence_id, "text": sentence})
            payload.append(
                {
                    "review_index": int(row["review_index"]),
                    "cafe_name": row["상호명"],
                    "sentences": sentences,
                }
            )

        request = {
            "custom_id": f"placeness-{int(review_rows[0]['review_index']):09d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request_body(
                args.model,
                payload,
                args.max_completion_tokens,
                args.reasoning_effort,
            ),
        }
        encoded = (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
        if shard_requests and (
            shard_bytes + len(encoded) > max_bytes
            or shard_requests >= args.max_requests_per_shard
        ):
            close_shard()
            shard_no += 1
            shard_path = input_dir / f"batch_input_{shard_no:03d}.jsonl"
            shard_handle = shard_path.open("w", encoding="utf-8", newline="\n")
            shard_bytes = 0
            shard_requests = 0
        shard_handle.write(encoded.decode("utf-8"))
        shard_bytes += len(encoded)
        shard_requests += 1
        total_requests += 1
    close_shard()

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "run_dir": str(run_dir),
        "model": args.model,
        "temperature": None if args.model.startswith("gpt-5") else 0,
        "reasoning_effort": (
            args.reasoning_effort if args.model.startswith("gpt-5") else None
        ),
        "max_completion_tokens": args.max_completion_tokens,
        "response_format": "strict_json_schema",
        "start_row": start,
        "selected_reviews": len(selected),
        "reviews_per_request": args.reviews_per_request,
        "max_requests_per_shard": args.max_requests_per_shard,
        "max_sentence_chars": args.max_sentence_chars,
        "total_requests": total_requests,
        "shards": shards,
    }
    save_manifest(run_dir, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def selected_shards(
    manifest: dict[str, Any], shard_numbers: list[int] | None
) -> list[dict[str, Any]]:
    if not shard_numbers:
        return manifest["shards"]
    selected = set(shard_numbers)
    return [shard for shard in manifest["shards"] if shard["shard"] in selected]


async def live_request(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    request: dict[str, Any],
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(**request["body"])
            return {
                "id": f"live-{request['custom_id']}",
                "custom_id": request["custom_id"],
                "response": {
                    "status_code": 200,
                    "request_id": getattr(response, "_request_id", None),
                    "body": response.model_dump(mode="json"),
                },
                "error": None,
            }
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max_retries:
                break
            retry_after = min(60.0, (2**attempt) + random.random())
            await asyncio.sleep(retry_after)
    return {
        "id": f"live-{request['custom_id']}",
        "custom_id": request["custom_id"],
        "response": None,
        "error": {"message": str(last_error)},
    }


async def run_live_async(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    shards = selected_shards(manifest, args.shard)
    for shard in shards:
        input_path = Path(shard["input_path"])
        output_path = run_dir / "output" / f"live_output_{shard['shard']:03d}.jsonl"
        completed: set[str] = set()
        successful_records: list[dict[str, Any]] = []
        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                        if record.get("response") and not record.get("error"):
                            completed.add(record["custom_id"])
                            successful_records.append(record)
                    except (json.JSONDecodeError, KeyError):
                        continue
        requests: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                request = json.loads(line)
                if request["custom_id"] not in completed:
                    requests.append(request)
        print(
            f"Shard {shard['shard']:03d}: {len(requests):,} pending requests "
            f"at concurrency {args.concurrency}."
        )
        tasks = [
            asyncio.create_task(
                live_request(client, semaphore, request, args.max_retries)
            )
            for request in requests
        ]
        done = 0
        with output_path.open("w", encoding="utf-8") as output_handle:
            for record in successful_records:
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            for future in asyncio.as_completed(tasks):
                result = await future
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_handle.flush()
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"Completed {done:,}/{len(tasks):,} requests")
        shard["status"] = "live_completed"
        shard["local_output_path"] = str(output_path)
        save_manifest(run_dir, manifest)
    await client.close()


def run_live(args: argparse.Namespace) -> None:
    asyncio.run(run_live_async(args))


def submit(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    client = OpenAI()
    for shard in selected_shards(manifest, args.shard):
        if shard.get("batch_id"):
            print(
                f"Shard {shard['shard']:03d} already submitted: {shard['batch_id']}"
            )
            continue
        input_path = Path(shard["input_path"])
        with input_path.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "placeness-v6-factor-mapping",
                "shard": str(shard["shard"]),
            },
        )
        shard["input_file_id"] = uploaded.id
        shard["batch_id"] = batch.id
        shard["status"] = batch.status
        save_manifest(run_dir, manifest)
        print(
            f"Submitted shard {shard['shard']:03d}: file={uploaded.id}, "
            f"batch={batch.id}, status={batch.status}"
        )


def status(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    client = OpenAI()
    for shard in manifest["shards"]:
        batch_id = shard.get("batch_id")
        if not batch_id:
            print(f"Shard {shard['shard']:03d}: not submitted")
            continue
        batch = client.batches.retrieve(batch_id)
        shard["status"] = batch.status
        shard["output_file_id"] = batch.output_file_id
        shard["error_file_id"] = batch.error_file_id
        request_counts = (
            batch.request_counts.model_dump()
            if batch.request_counts is not None
            else None
        )
        shard["request_counts"] = request_counts
        print(
            f"Shard {shard['shard']:03d}: {batch.status}, "
            f"counts={request_counts}"
        )
    save_manifest(run_dir, manifest)


def download(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    client = OpenAI()
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for shard in selected_shards(manifest, args.shard):
        batch_id = shard.get("batch_id")
        if not batch_id:
            continue
        batch = client.batches.retrieve(batch_id)
        shard["status"] = batch.status
        shard["output_file_id"] = batch.output_file_id
        shard["error_file_id"] = batch.error_file_id
        if batch.output_file_id:
            output_path = output_dir / f"batch_output_{shard['shard']:03d}.jsonl"
            response = client.files.content(batch.output_file_id)
            output_path.write_bytes(response.content)
            shard["local_output_path"] = str(output_path)
            print(f"Downloaded shard {shard['shard']:03d} to {output_path}")
        if batch.error_file_id:
            error_path = output_dir / f"batch_error_{shard['shard']:03d}.jsonl"
            response = client.files.content(batch.error_file_id)
            error_path.write_bytes(response.content)
            shard["local_error_path"] = str(error_path)
            print(f"Downloaded errors for shard {shard['shard']:03d}")
    save_manifest(run_dir, manifest)


def retry_failed(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    output_dir = run_dir / "output"
    input_dir = run_dir / "input"
    if args.max_requests_per_shard < 1:
        raise ValueError("max-requests-per-shard must be positive")

    stale_prepared = [
        shard
        for shard in manifest["shards"]
        if shard.get("kind") == "retry" and not shard.get("batch_id")
    ]
    for shard in stale_prepared:
        Path(shard["input_path"]).unlink(missing_ok=True)
    if stale_prepared:
        stale_numbers = {int(shard["shard"]) for shard in stale_prepared}
        manifest["shards"] = [
            shard
            for shard in manifest["shards"]
            if int(shard["shard"]) not in stale_numbers
        ]

    frame = selected_frame(manifest)
    review_lookup = frame.set_index("review_index").to_dict(orient="index")
    expected_indices = set(review_lookup)
    returned_by_id: dict[str, set[int]] = {}
    candidate_ids: set[str] = set()

    def canonical_id(raw_custom_id: str) -> str | None:
        match = re.match(r"placeness-(\d+)", raw_custom_id)
        if match is None:
            return None
        return f"placeness-{int(match.group(1)):09d}"

    for output_path in output_dir.glob("batch_output_*.jsonl"):
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                custom_id = canonical_id(str(record.get("custom_id", "")))
                if custom_id is None:
                    continue
                candidate_ids.add(custom_id)
                result, _ = response_content(record)
                if result is None:
                    continue
                returned_request_indices = returned_by_id.setdefault(custom_id, set())
                for review in result.get("reviews", []):
                    try:
                        returned_request_indices.add(int(review["review_index"]))
                    except (KeyError, TypeError, ValueError):
                        continue

    for error_path in output_dir.glob("batch_error_*.jsonl"):
        with error_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                custom_id = canonical_id(str(record.get("custom_id", "")))
                if custom_id is not None:
                    candidate_ids.add(custom_id)

    missing_by_id: dict[str, list[int]] = {}
    for custom_id in candidate_ids:
        request_start = int(custom_id.removeprefix("placeness-"))
        expected_request_indices = {
            index
            for index in range(
                request_start,
                request_start + int(manifest["reviews_per_request"]),
            )
            if index in expected_indices
        }
        missing = sorted(expected_request_indices - returned_by_id.get(custom_id, set()))
        if missing:
            missing_by_id[custom_id] = missing

    active_retry_ids: set[str] = set()
    active_statuses = {
        "prepared",
        "validating",
        "in_progress",
        "finalizing",
        "cancelling",
    }
    for shard in manifest["shards"]:
        if shard.get("kind") != "retry" or shard.get("status") not in active_statuses:
            continue
        with Path(shard["input_path"]).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    custom_id = canonical_id(
                        str(json.loads(line).get("custom_id", ""))
                    )
                    if custom_id is not None:
                        active_retry_ids.add(custom_id)
    for custom_id in active_retry_ids:
        missing_by_id.pop(custom_id, None)

    if not missing_by_id:
        save_manifest(run_dir, manifest)
        print("No unresolved failed requests to retry.")
        return

    requests: list[dict[str, Any]] = []
    missing_review_count = 0
    for custom_id in sorted(missing_by_id):
        missing_indices = missing_by_id[custom_id]
        for part_no, offset in enumerate(
            range(0, len(missing_indices), args.reviews_per_retry_request)
        ):
            part_indices = missing_indices[
                offset : offset + args.reviews_per_retry_request
            ]
            payload: list[dict[str, Any]] = []
            for review_index in part_indices:
                metadata = review_lookup[review_index]
                sentences = []
                for sentence_id, sentence in enumerate(
                    split_into_sentences(metadata["리뷰"])
                ):
                    sentence = sentence.strip()
                    if len(sentence) > int(manifest["max_sentence_chars"]):
                        sentence = (
                            sentence[: int(manifest["max_sentence_chars"])].rstrip()
                            + "..."
                        )
                    sentences.append({"sentence_id": sentence_id, "text": sentence})
                payload.append(
                    {
                        "review_index": int(review_index),
                        "cafe_name": metadata["상호명"],
                        "sentences": sentences,
                    }
                )
            missing_review_count += len(payload)
            requests.append(
                {
                    "custom_id": f"{custom_id}-part-{part_no:03d}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request_body(
                        manifest["model"],
                        payload,
                        int(manifest["max_completion_tokens"]),
                        manifest.get("reasoning_effort") or "none",
                    ),
                }
            )

    next_shard = max(
        (int(shard["shard"]) for shard in manifest["shards"]), default=-1
    ) + 1
    created = 0
    for offset in range(0, len(requests), args.max_requests_per_shard):
        chunk = requests[offset : offset + args.max_requests_per_shard]
        shard_no = next_shard + created
        input_path = input_dir / f"retry_input_{shard_no:03d}.jsonl"
        encoded_lines = [
            json.dumps(request, ensure_ascii=False) + "\n" for request in chunk
        ]
        input_path.write_text("".join(encoded_lines), encoding="utf-8", newline="\n")
        manifest["shards"].append(
            {
                "shard": shard_no,
                "kind": "retry",
                "input_path": str(input_path),
                "requests": len(chunk),
                "bytes": input_path.stat().st_size,
                "input_file_id": None,
                "batch_id": None,
                "status": "prepared",
                "output_file_id": None,
                "error_file_id": None,
            }
        )
        created += 1

    manifest["retry_requests"] = int(manifest.get("retry_requests", 0)) + len(
        requests
    )
    save_manifest(run_dir, manifest)
    print(
        f"Prepared {len(requests)} failed requests covering "
        f"{missing_review_count} missing reviews in {created} retry shard(s)."
    )


def response_content(record: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    response = record.get("response")
    if not response or response.get("status_code") != 200:
        return None, {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None, body.get("usage") or {}
    message = choices[0].get("message") or {}
    content = message.get("content") or "{}"
    try:
        return json.loads(content), body.get("usage") or {}
    except json.JSONDecodeError:
        return None, body.get("usage") or {}


def iter_result_mappings(result: dict[str, Any]) -> Any:
    flat_mappings = result.get("mappings")
    if isinstance(flat_mappings, list):
        yield from flat_mappings
        return
    for review in result.get("reviews", []):
        review_index = review.get("review_index")
        for sentence_item in review.get("items", []):
            sentence_id = sentence_item.get("sentence_id")
            for mapping in sentence_item.get("mappings", []):
                yield {
                    **mapping,
                    "review_index": review_index,
                    "sentence_id": sentence_id,
                }


def inspect_results(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    frame = selected_frame(manifest)
    review_lookup = frame.set_index("review_index").to_dict(orient="index")
    expected_indices = set(review_lookup)
    output_files = sorted((run_dir / "output").glob("batch_output_*.jsonl"))
    if not output_files:
        raise FileNotFoundError(f"No output JSONL files found under {run_dir / 'output'}")

    response_records = 0
    parsed_responses = 0
    invalid_responses = 0
    valid_mappings = 0
    invalid_factor_rows = 0
    invalid_sentiment_rows = 0
    invalid_reference_rows = 0
    evidence_exact_rows = 0
    returned_indices: set[int] = set()
    usage_totals: dict[str, int] = {}

    for output_path in output_files:
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                response_records += 1
                record = json.loads(line)
                result, usage = response_content(record)
                for key, value in usage.items():
                    if isinstance(value, int):
                        usage_totals[key] = usage_totals.get(key, 0) + value
                if result is None:
                    invalid_responses += 1
                    continue
                parsed_responses += 1
                for review in result.get("reviews", []):
                    try:
                        review_index = int(review["review_index"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if review_index in expected_indices:
                        returned_indices.add(review_index)

                for mapping in iter_result_mappings(result):
                    try:
                        review_index = int(mapping["review_index"])
                        sentence_id = int(mapping["sentence_id"])
                    except (KeyError, TypeError, ValueError):
                        invalid_reference_rows += 1
                        continue
                    factor = str(mapping.get("factor", ""))
                    sentiment = str(mapping.get("sentiment_hint", ""))
                    if factor not in FACTOR_NAMES:
                        invalid_factor_rows += 1
                        continue
                    if sentiment not in {"positive", "negative", "neutral", "mixed"}:
                        invalid_sentiment_rows += 1
                        continue
                    metadata = review_lookup.get(review_index)
                    if metadata is None:
                        invalid_reference_rows += 1
                        continue
                    sentences = split_into_sentences(metadata["리뷰"])
                    if sentence_id < 0 or sentence_id >= len(sentences):
                        invalid_reference_rows += 1
                        continue
                    valid_mappings += 1
                    evidence = str(mapping.get("evidence", "")).strip()
                    if evidence and evidence in sentences[sentence_id]:
                        evidence_exact_rows += 1

    reviews_per_request = int(manifest["reviews_per_request"])
    ordered_indices = list(frame["review_index"].astype(int))
    expected_request_groups = [
        set(ordered_indices[offset : offset + reviews_per_request])
        for offset in range(0, len(ordered_indices), reviews_per_request)
    ]
    complete_original_requests = sum(
        request_indices <= returned_indices
        for request_indices in expected_request_groups
    )
    total_requests = len(expected_request_groups)
    summary = {
        "output_files": len(output_files),
        "response_records": response_records,
        "unique_complete_requests": complete_original_requests,
        "incomplete_requests": total_requests - complete_original_requests,
        "total_original_requests": total_requests,
        "successful_request_progress_ratio": (
            complete_original_requests / total_requests if total_requests else 0.0
        ),
        "parsed_responses": parsed_responses,
        "invalid_responses": invalid_responses,
        "expected_reviews": len(expected_indices),
        "returned_reviews": len(returned_indices),
        "returned_review_ratio": (
            len(returned_indices) / len(expected_indices) if expected_indices else 0.0
        ),
        "valid_mapping_rows": valid_mappings,
        "invalid_factor_rows": invalid_factor_rows,
        "invalid_sentiment_rows": invalid_sentiment_rows,
        "invalid_reference_rows": invalid_reference_rows,
        "evidence_exact_substring_rows": evidence_exact_rows,
        "evidence_exact_substring_ratio": (
            evidence_exact_rows / valid_mappings if valid_mappings else 0.0
        ),
        "usage": usage_totals,
    }
    (run_dir / "inspect_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def finalize(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    frame = selected_frame(manifest)
    review_lookup = frame.set_index("review_index").to_dict(orient="index")
    expected_indices = set(review_lookup)
    returned_indices: set[int] = set()
    sentence_rows: list[dict[str, Any]] = []
    historical_failed_response_records = 0
    usage_totals: dict[str, int] = {}
    output_files = sorted((run_dir / "output").glob("*_output_*.jsonl"))
    if not output_files:
        raise FileNotFoundError(f"No output JSONL files found under {run_dir / 'output'}")

    for output_path in output_files:
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                result, usage = response_content(record)
                for key, value in usage.items():
                    if isinstance(value, int):
                        usage_totals[key] = usage_totals.get(key, 0) + value
                if result is None:
                    historical_failed_response_records += 1
                    continue
                for review in result.get("reviews", []):
                    try:
                        review_index = int(review["review_index"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if review_index in expected_indices:
                        returned_indices.add(review_index)
                for mapping in iter_result_mappings(result):
                    try:
                        review_index = int(mapping["review_index"])
                        sentence_id = int(mapping["sentence_id"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    metadata = review_lookup.get(review_index)
                    if metadata is None:
                        continue
                    sentences = split_into_sentences(metadata["리뷰"])
                    if sentence_id < 0 or sentence_id >= len(sentences):
                        continue
                    sentence = sentences[sentence_id]
                    factor = str(mapping.get("factor", ""))
                    sentiment = str(mapping.get("sentiment_hint", ""))
                    if factor not in FACTOR_NAMES:
                        continue
                    if sentiment not in {"positive", "negative", "neutral", "mixed"}:
                        continue
                    if "confidence" in mapping:
                        try:
                            if float(mapping["confidence"]) < MAPPING_CONFIDENCE_THRESHOLD:
                                continue
                        except (TypeError, ValueError):
                            continue
                    raw_evidence = str(mapping.get("evidence", "")).strip()
                    evidence_exact_match = bool(raw_evidence) and raw_evidence in sentence
                    sentence_rows.append(
                        {
                            "review_index": review_index,
                            "상가업소번호": metadata.get("상가업소번호", ""),
                            "상호명": metadata.get("상호명", ""),
                            "시군구명": metadata.get("시군구명", ""),
                            "행정동명": metadata.get("행정동명", ""),
                            "도로명주소": metadata.get("도로명주소", ""),
                            "작성자": metadata.get("작성자", ""),
                            "평점": metadata.get("평점", ""),
                            "리뷰": metadata.get("리뷰", ""),
                            "sentence_id": sentence_id,
                            "sentence": sentence,
                            "factor": factor,
                            "evidence": raw_evidence if evidence_exact_match else sentence,
                            "model_evidence": raw_evidence,
                            "evidence_exact_match": evidence_exact_match,
                            "sentiment_hint": sentiment,
                            "model": manifest["model"],
                        }
                    )

    sentence_frame = pd.DataFrame(sentence_rows)
    if not sentence_frame.empty:
        sentence_frame = sentence_frame.drop_duplicates(
            ["review_index", "sentence_id", "factor", "evidence"], keep="last"
        )
    sentence_path = run_dir / "placeness_mapping_sentences.csv"
    sentence_frame.to_csv(sentence_path, index=False, encoding="utf-8-sig")

    mapped_counts = (
        sentence_frame.groupby("review_index").size()
        if not sentence_frame.empty
        else pd.Series(dtype=int)
    )
    review_frame = frame.copy()
    review_frame["returned_by_model"] = review_frame["review_index"].isin(
        returned_indices
    )
    review_frame["mapping_count"] = (
        review_frame["review_index"].map(mapped_counts).fillna(0).astype(int)
    )
    review_path = run_dir / "placeness_mapping_reviews.csv"
    review_frame.to_csv(review_path, index=False, encoding="utf-8-sig")

    evidence_in_sentence = 0
    if not sentence_frame.empty:
        evidence_in_sentence = int(sentence_frame["evidence_exact_match"].sum())
    summary = {
        "run_dir": str(run_dir),
        "model": manifest["model"],
        "expected_reviews": len(expected_indices),
        "returned_reviews": len(returned_indices),
        "missing_reviews": len(expected_indices - returned_indices),
        "historical_failed_response_records": historical_failed_response_records,
        "unresolved_failed_reviews": len(expected_indices - returned_indices),
        "mapping_rows": len(sentence_frame),
        "mapped_reviews": int(review_frame["mapping_count"].gt(0).sum()),
        "unmapped_reviews": int(review_frame["mapping_count"].eq(0).sum()),
        "evidence_exact_substring_rows": evidence_in_sentence,
        "evidence_exact_substring_ratio": (
            evidence_in_sentence / len(sentence_frame)
            if len(sentence_frame)
            else 0.0
        ),
        "usage": usage_totals,
        "sentence_output": str(sentence_path),
        "review_output": str(review_path),
    }
    (run_dir / "finalize_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    configure_console_encoding()
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    if args.command != "prepare" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run-live":
        run_live(args)
    elif args.command == "submit":
        submit(args)
    elif args.command == "status":
        status(args)
    elif args.command == "download":
        download(args)
    elif args.command == "retry-failed":
        retry_failed(args)
    elif args.command == "inspect":
        inspect_results(args)
    elif args.command == "finalize":
        finalize(args)


if __name__ == "__main__":
    main()
