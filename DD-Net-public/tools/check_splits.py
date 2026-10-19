"""Check year-level records or a supplied authoritative sequence-time CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_summary(summary):
    if set(summary["train"]["years"]) & set(summary["test"]["years"]):
        raise ValueError("Training and testing years overlap")
    for split in ("train", "test"):
        counts = summary[split]["counts_by_year"]
        if sum(counts.values()) != summary[split]["samples"]:
            raise ValueError(f"Year counts do not add up: {split}")
        if {int(year) for year in counts} != set(summary[split]["years"]):
            raise ValueError(f"Year metadata mismatch: {split}")
    return {"year_counts_consistent": True, "year_sets_disjoint": True,
            "sequence_overlap_verified": False, "sequence_adjacency_verified": False,
            "evidence_level": summary["evidence_level"]}


def check_sequences(rows, summary, *, indexed=True):
    check_summary(summary)
    parsed = {"train": [], "test": []}
    counts = {"train": Counter(), "test": Counter()}
    indices = {"train": set(), "test": set()}
    identifiers = {"train": set(), "test": set()}
    cadence = timedelta(minutes=summary["frame_interval_minutes"])
    tolerance = float(summary.get("endpoint_tolerance_seconds", 0))
    deviations = []
    for row in rows:
        split = row["split"]
        if split not in parsed:
            raise ValueError("Sequence split must be train or test")
        index_key = "sample_index" if indexed else "inventory_index"
        index, year = int(row[index_key]), int(row["year"])
        if index in indices[split] or not 0 <= index < summary[split]["samples"]:
            raise ValueError("Duplicate or invalid sample index")
        indices[split].add(index)
        source_id = row["source_id"] if indexed else row["inventory_id"]
        if not source_id or source_id in identifiers[split] or "/" in source_id or "\\" in source_id:
            raise ValueError("Use unique opaque source-record IDs, not file system paths")
        identifiers[split].add(source_id)
        start = datetime.fromisoformat(row["start_time"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(row["end_time"].replace("Z", "+00:00"))
        if indexed and (start.tzinfo is None or end.tzinfo is None):
            raise ValueError("Times require explicit timezone offsets")
        if year not in summary[split]["years"] or start.year != year or end.year != year:
            raise ValueError("Sequence dates do not match the year-wise split")
        deviation = abs(((end - start) - cadence * (summary["frame_count"] - 1)).total_seconds())
        deviations.append(deviation)
        if deviation > tolerance:
            raise ValueError("Sequence endpoint span does not match 40 six-minute frames")
        parsed[split].append((start, end, source_id))
        counts[split][str(year)] += 1
    for split in parsed:
        if indices[split] != set(range(summary[split]["samples"])):
            raise ValueError(f"Missing sequence indices: {split}")
        if dict(counts[split]) != summary[split]["counts_by_year"]:
            raise ValueError(f"Sequence year counts differ from recorded counts: {split}")
    if identifiers["train"] & identifiers["test"]:
        raise ValueError("Source record occurs in both training and testing")
    minimum_gap = None
    for start, end, _ in parsed["train"]:
        for other_start, other_end, _ in parsed["test"]:
            if start <= other_end + cadence and other_start <= end + cadence:
                raise ValueError("Cross-split sequences overlap or are adjacent within one frame interval")
            gap = other_start - end if end < other_start else start - other_end
            minimum_gap = gap if minimum_gap is None else min(minimum_gap, gap)
    return {"sequence_overlap_verified": indexed, "sequence_adjacency_verified": indexed,
            "inventory_interval_overlap_verified": True,
            "inventory_interval_adjacency_verified": True,
            "array_index_mapping_verified": False,
            "closest_cross_split_gap_minutes": minimum_gap.total_seconds() / 60,
            "endpoint_span_tolerance_seconds": tolerance,
            "maximum_endpoint_span_deviation_seconds": max(deviations),
            "records_with_nonzero_endpoint_span_deviation": sum(value > 0 for value in deviations),
            "date_ranges": {split: {"first_start": min(row[0] for row in values).isoformat(),
                                    "last_end": max(row[1] for row in values).isoformat()}
                            for split, values in parsed.items()},
            "counts_match_record": True, "input_manifest_authority_independently_established": False,
            "note": "Checks supplied interval metadata; authoritative linkage to radar array indices must be established separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "data/split_summary.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--inventory", type=Path,
                        help="Timestamp inventory without an established array-index mapping")
    parser.add_argument("--require-sequences", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    report = check_summary(summary)
    if args.manifest and args.inventory:
        parser.error("Choose --manifest or --inventory")
    source = args.manifest or args.inventory
    if source:
        with source.open(newline="", encoding="utf-8") as handle:
            report.update(check_sequences(list(csv.DictReader(handle)), summary,
                                          indexed=args.manifest is not None))
        report["manifest_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        if args.require_sequences and not args.manifest:
            parser.error("An inventory cannot establish an array-index mapping; provide --manifest")
    elif args.require_sequences:
        parser.error("An authoritative --manifest is required; year counts alone do not prove non-adjacency")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
