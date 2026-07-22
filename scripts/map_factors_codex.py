from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from map_factors_openai import (
    MAPPING_CONFIDENCE_THRESHOLD,
    PROJECT_ROOT,
    SYSTEM_PROMPT,
    ReviewItem,
    build_prompt,
    build_review_items,
    chunked,
    load_processed_review_indices,
    normalize_mapping,
    normalize_review_df,
    raw_jsonl_to_frames,
    select_review_slice,
)


DEFAULT_INPUT_CSV = (
    PROJECT_ROOT / "data" / "google_reviews_full_corrected_textclean_v4.csv"
)
DEFAULT_SCHEMA = PROJECT_ROOT / "schemas" / "placeness_mapping.schema.json"
LARGE_RUN_THRESHOLD = 10_000


def configure_console_encoding() -> None:
    """Keep Korean progress messages readable in Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map Korean cafe reviews to placeness factors through Codex CLI. "
            "This uses saved Codex CLI authentication, not OPENAI_API_KEY."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-prefix", default="codex_factor_mapping")
    parser.add_argument("--suffix", default="")
    parser.add_argument(
        "--codex-bin",
        default=os.getenv("CODEX_BIN", "codex"),
        help="Codex CLI executable name or path.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CODEX_MODEL", ""),
        help="Optional Codex model. Empty uses the current CLI default.",
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--start-cafe", type=int, default=None)
    parser.add_argument("--end-cafe", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--random-sample",
        type=int,
        default=None,
        help="Randomly sample this many reviews after applying row/cafe slices.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-sentence-chars", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-large-run",
        action="store_true",
        help=(
            f"Allow more than {LARGE_RUN_THRESHOLD:,} reviews. Large corpora are "
            "usually better processed through a batch API than repeated Codex CLI calls."
        ),
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args()


def output_paths(output_dir: Path, prefix: str, suffix: str) -> dict[str, Path]:
    clean_suffix = suffix.strip()
    if clean_suffix and not clean_suffix.startswith("_"):
        clean_suffix = "_" + clean_suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "raw": output_dir / f"{prefix}_raw{clean_suffix}.jsonl",
        "sentence": output_dir / f"{prefix}_sentences{clean_suffix}.csv",
        "review": output_dir / f"{prefix}_reviews{clean_suffix}.csv",
        "run": output_dir / f"{prefix}_run{clean_suffix}.json",
    }


def resolve_codex_binary(value: str) -> str:
    explicit = Path(value).expanduser()
    if explicit.parent != Path(".") or explicit.is_absolute():
        if not explicit.exists():
            raise FileNotFoundError(f"Codex CLI not found: {explicit}")
        return str(explicit.resolve())

    resolved = shutil.which(value)
    if resolved:
        return resolved
    raise FileNotFoundError(
        "Codex CLI was not found on PATH. Install/login to Codex CLI or pass "
        "--codex-bin with an executable path."
    )


def build_codex_prompt(batch: list[ReviewItem]) -> str:
    return f"""{SYSTEM_PROMPT.strip()}

추가 실행 지침:
- 이 작업은 데이터 주석 작업입니다. 파일을 읽거나 명령을 실행하지 마세요.
- 입력된 모든 review_index와 모든 sentence_id를 출력에 포함하세요.
- 관련 요인이 없는 문장은 mappings를 빈 배열로 출력하세요.
- evidence는 반드시 해당 입력 문장에 실제로 존재하는 짧은 원문 구절이어야 합니다.
- 출력은 제공된 JSON Schema를 따르는 JSON 객체 하나여야 합니다.

{build_prompt(batch).strip()}
"""


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Codex response must be a JSON object.")
    return parsed


def normalize_codex_result(
    result: dict[str, Any], batch: list[ReviewItem]
) -> dict[str, Any]:
    response_reviews: dict[int, dict[str, Any]] = {}
    for review in result.get("reviews", []):
        if not isinstance(review, dict):
            continue
        try:
            review_index = int(review.get("review_index"))
        except (TypeError, ValueError):
            continue
        response_reviews[review_index] = review

    normalized_reviews: list[dict[str, Any]] = []
    for item in batch:
        response_review = response_reviews.get(item.review_index, {})
        response_sentences: dict[int, dict[str, Any]] = {}
        for sentence_item in response_review.get("items", []):
            if not isinstance(sentence_item, dict):
                continue
            try:
                sentence_id = int(sentence_item.get("sentence_id"))
            except (TypeError, ValueError):
                continue
            response_sentences[sentence_id] = sentence_item

        normalized_items: list[dict[str, Any]] = []
        for sentence_id, sentence in enumerate(item.sentences):
            response_sentence = response_sentences.get(sentence_id, {})
            best_by_factor: dict[str, dict[str, Any]] = {}
            for raw_mapping in response_sentence.get("mappings", []):
                if not isinstance(raw_mapping, dict):
                    continue
                mapping = normalize_mapping(raw_mapping)
                if mapping is None:
                    continue
                if not mapping["evidence"] or mapping["evidence"] not in sentence:
                    mapping["evidence"] = sentence
                previous = best_by_factor.get(mapping["factor"])
                if previous is None or mapping["confidence"] > previous["confidence"]:
                    best_by_factor[mapping["factor"]] = mapping
            normalized_items.append(
                {
                    "sentence_id": sentence_id,
                    "mappings": list(best_by_factor.values()),
                }
            )

        normalized_reviews.append(
            {
                "review_index": item.review_index,
                "items": normalized_items,
            }
        )
    return {"reviews": normalized_reviews}


def call_codex_json(
    codex_bin: str,
    schema_path: Path,
    prompt: str,
    model: str,
    timeout_seconds: int,
    max_retries: int,
) -> dict[str, Any]:
    temp_dir = PROJECT_ROOT / "tmp" / "codex_mapping"
    temp_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(max(1, max_retries)):
        response_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="batch_",
                suffix=".json",
                dir=temp_dir,
                delete=False,
            ) as response_file:
                response_path = Path(response_file.name)

            command = [codex_bin, "exec"]
            if model:
                command.extend(["--model", model])
            command.extend(
                [
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--output-schema",
                    str(schema_path.resolve()),
                    "--output-last-message",
                    str(response_path.resolve()),
                    "-",
                ]
            )
            environment = os.environ.copy()
            environment.setdefault("NO_COLOR", "1")
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(1, timeout_seconds),
                check=False,
            )
            if completed.returncode != 0:
                stderr_tail = completed.stderr.strip()[-2000:]
                if "usage limit" in completed.stderr.lower():
                    raise RuntimeError(
                        "Codex usage limit reached. Preserve the raw JSONL and rerun "
                        "with --resume after the limit resets."
                    )
                raise RuntimeError(
                    f"codex exec exited with {completed.returncode}: {stderr_tail}"
                )

            response_text = response_path.read_text(encoding="utf-8").strip()
            if not response_text:
                response_text = completed.stdout.strip()
            return parse_json_response(response_text)
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max(1, max_retries):
                break
            sleep_seconds = min(30, 2**attempt)
            print(
                f"Codex call failed ({attempt + 1}/{max_retries}). "
                f"Retrying in {sleep_seconds}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)
        finally:
            if response_path is not None:
                response_path.unlink(missing_ok=True)

    raise RuntimeError(f"Codex call failed after {max_retries} attempts: {last_error}")


def write_run_metadata(
    path: Path,
    args: argparse.Namespace,
    codex_bin: str,
    selected_reviews: int,
) -> None:
    metadata = {
        "input": str(args.input.resolve()),
        "codex_bin": codex_bin,
        "model": args.model or "codex-cli-default",
        "schema": str(args.schema.resolve()),
        "batch_size": args.batch_size,
        "start_row": args.start_row,
        "end_row": args.end_row,
        "start_cafe": args.start_cafe,
        "end_cafe": args.end_cafe,
        "limit": args.limit,
        "random_sample": args.random_sample,
        "sample_seed": args.sample_seed,
        "max_sentence_chars": args.max_sentence_chars,
        "mapping_confidence_threshold": MAPPING_CONFIDENCE_THRESHOLD,
        "selected_reviews": selected_reviews,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    configure_console_encoding()
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    args.input = args.input.resolve()
    args.output_dir = args.output_dir.resolve()
    args.schema = args.schema.resolve()

    if not args.schema.exists():
        raise FileNotFoundError(f"JSON Schema not found: {args.schema}")
    with args.schema.open("r", encoding="utf-8") as schema_handle:
        json.load(schema_handle)

    df = normalize_review_df(args.input)
    selected_df = select_review_slice(df, args)
    if args.random_sample is not None:
        sample_size = min(max(0, args.random_sample), len(selected_df))
        selected_df = (
            selected_df.sample(n=sample_size, random_state=args.sample_seed)
            .sort_values("review_index")
            .reset_index(drop=True)
        )
    review_items = build_review_items(selected_df, args.max_sentence_chars)
    paths = output_paths(args.output_dir, args.output_prefix, args.suffix)

    print(f"Loaded {len(df):,} cleaned reviews.")
    print(f"Selected {len(review_items):,} reviews for Codex factor mapping.")
    print(f"Raw output: {paths['raw']}")
    if not review_items:
        raise ValueError("No reviews selected.")
    if (
        len(review_items) > LARGE_RUN_THRESHOLD
        and not args.allow_large_run
        and not args.dry_run
    ):
        raise ValueError(
            f"Selected {len(review_items):,} reviews. Run a pilot with --limit first, "
            "or add --allow-large-run after checking runtime and usage."
        )

    first_batch = review_items[: min(args.batch_size, len(review_items))]
    if args.dry_run:
        print(build_codex_prompt(first_batch))
        return

    codex_bin = resolve_codex_binary(args.codex_bin)
    if args.overwrite:
        for key in ("raw", "sentence", "review", "run"):
            paths[key].unlink(missing_ok=True)
    elif paths["raw"].exists() and not args.resume:
        raise FileExistsError(
            f"Raw output already exists: {paths['raw']}. "
            "Use --resume or --overwrite."
        )

    processed = load_processed_review_indices(paths["raw"]) if args.resume else set()
    pending_items = [item for item in review_items if item.review_index not in processed]
    if processed:
        print(f"Resume enabled: skipping {len(processed):,} processed reviews.")

    write_run_metadata(paths["run"], args, codex_bin, len(review_items))
    with paths["raw"].open("a", encoding="utf-8") as raw_handle:
        for batch_no, batch in enumerate(
            chunked(pending_items, args.batch_size), start=1
        ):
            result = call_codex_json(
                codex_bin=codex_bin,
                schema_path=args.schema,
                prompt=build_codex_prompt(batch),
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            normalized = normalize_codex_result(result, batch)
            raw_handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            raw_handle.flush()
            done = min(batch_no * args.batch_size, len(pending_items))
            print(f"Mapped {done:,}/{len(pending_items):,} pending reviews")

    model_label = args.model or "codex-cli-default"
    sentence_df, review_df = raw_jsonl_to_frames(
        paths["raw"], review_items, model_label
    )
    sentence_df.to_csv(paths["sentence"], index=False, encoding="utf-8-sig")
    review_df.to_csv(paths["review"], index=False, encoding="utf-8-sig")
    print(f"Wrote sentence mappings: {paths['sentence']} ({len(sentence_df):,} rows)")
    print(f"Wrote review mappings: {paths['review']} ({len(review_df):,} rows)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise
