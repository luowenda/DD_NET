"""Export recovered timestamp records, never inventing radar-array sample indices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path


def export(source: Path):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["inventory_index", "inventory_id", "split", "year", "start_time", "end_time"])
    counters = {"train": 0, "test": 0}
    discarded = 0
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.reader(handle), 1):
            if not row or row[0].strip() in {"test", "train", "starttime", ""}:
                discarded += 1
                continue
            if len(row) != 3:
                raise ValueError(f"Unexpected source CSV schema at line {line}")
            start, end = datetime.fromisoformat(row[0]), datetime.fromisoformat(row[1])
            if start.year != end.year or not 2011 <= start.year <= 2021:
                raise ValueError(f"Unexpected source year at line {line}")
            split = "test" if start.year == 2018 else "train"
            index = counters[split]
            writer.writerow([index, f"timestamp-record-{line}", split, start.year,
                             start.isoformat(sep=" "), end.isoformat(sep=" ")])
            counters[split] += 1
    return buffer.getvalue(), {
        "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "records": counters, "discarded_header_or_marker_rows": discarded,
        "array_index_mapping_established": False,
        "inventory_index_meaning": "Ordinal within each split in the recovered CSV; not a radar-array sample index",
        "timezone": "Preserved exactly as recorded; no timezone offset was invented",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text, metadata = export(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
