from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove review rows for exact place names or store IDs from a CSV."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--place", action="append", default=[])
    parser.add_argument("--store-id", action="append", default=[])
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    if not args.place and not args.store_id:
        parser.error("at least one --place or --store-id is required")
    return args


def main() -> int:
    args = parse_args()
    output = args.output or args.input
    place_targets = set(args.place)
    store_id_targets = set(args.store_id)
    temp_output = output.with_suffix(output.suffix + ".tmp")
    if temp_output.exists():
        temp_output.unlink()

    input_rows = 0
    output_rows = 0
    removed_by_place = {place: 0 for place in sorted(place_targets)}
    removed_by_store_id = {store_id: 0 for store_id in sorted(store_id_targets)}
    remaining_places: set[str] = set()
    first = True
    for chunk in pd.read_csv(
        args.input,
        encoding="utf-8-sig",
        dtype=str,
        chunksize=args.chunksize,
    ):
        store_id_column = chunk.columns[0]
        place_column = chunk.columns[1]
        store_ids = chunk[store_id_column].fillna("").astype(str)
        names = chunk[place_column].fillna("").astype(str)
        remove_mask = names.isin(place_targets) | store_ids.isin(store_id_targets)
        for place in place_targets:
            removed_by_place[place] += int(names.eq(place).sum())
        for store_id in store_id_targets:
            removed_by_store_id[store_id] += int(store_ids.eq(store_id).sum())
        kept = chunk.loc[~remove_mask]
        input_rows += len(chunk)
        output_rows += len(kept)
        remaining_places.update(kept[store_id_column].dropna().astype(str))
        kept.to_csv(
            temp_output,
            mode="w" if first else "a",
            header=first,
            index=False,
            encoding="utf-8-sig" if first else "utf-8",
        )
        first = False

    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_output, output)

    removal_event = {
        "removed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_rows": input_rows,
        "removed_rows": input_rows - output_rows,
        "removed_by_place": removed_by_place,
        "removed_by_store_id": removed_by_store_id,
        "output_rows": output_rows,
        "output_places": len(remaining_places),
    }
    audit_path = output.with_suffix(".manual_removals.json")
    previous_audit: dict[str, object] = {}
    if audit_path.exists():
        previous_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    history = list(previous_audit.get("removal_history", []))
    if not history and previous_audit.get("removed_rows"):
        history.append(
            {
                key: previous_audit.get(key)
                for key in (
                    "removed_at",
                    "input_rows",
                    "removed_rows",
                    "removed_by_place",
                    "removed_by_store_id",
                    "output_rows",
                    "output_places",
                )
            }
        )
    history.append(removal_event)
    audit = {
        "input_file": str(args.input),
        "output_file": str(output),
        "cumulative_removed_rows": sum(
            int(event.get("removed_rows") or 0) for event in history
        ),
        "removal_history": history,
        "output_rows": output_rows,
        "output_places": len(remaining_places),
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_path = output.with_suffix(".summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.setdefault("text_cleaning_output_rows", summary.get("output_rows"))
        summary["manual_removed_rows"] = int(
            summary.get("manual_removed_rows", 0)
        ) + removal_event["removed_rows"]
        cumulative_removed_by_place = dict(
            summary.get("manual_removed_by_place", {})
        )
        for place, count in removed_by_place.items():
            cumulative_removed_by_place[place] = int(
                cumulative_removed_by_place.get(place, 0)
            ) + count
        summary["manual_removed_by_place"] = cumulative_removed_by_place
        cumulative_removed_by_store_id = dict(
            summary.get("manual_removed_by_store_id", {})
        )
        for store_id, count in removed_by_store_id.items():
            cumulative_removed_by_store_id[store_id] = int(
                cumulative_removed_by_store_id.get(store_id, 0)
            ) + count
        summary["manual_removed_by_store_id"] = cumulative_removed_by_store_id
        summary["output_rows"] = output_rows
        summary["output_places"] = len(remaining_places)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
