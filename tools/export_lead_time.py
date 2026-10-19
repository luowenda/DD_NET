"""Export the shared evaluator's per-lead metrics, preserving seed aggregation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from aggregate_metrics import collect, flatten
from ddnet_revision.config import list_models, load_config


def export(suite, seeds, artifact_root, output):
    collect(suite, seeds, artifact_root)
    rows = []
    for alias in list_models(suite):
        config = load_config(alias)
        deterministic = config["model"]["family"] == "non_trainable"
        selected = [0] if deterministic else seeds
        runs = []
        for seed in selected:
            run = "deterministic" if deterministic else f"seed{seed}"
            metrics = json.loads((artifact_root / config["model"]["name"] / run / "metrics.json").read_text())
            leads = metrics["by_lead_time"]
            if len(leads) != 20 or [lead["minutes"] for lead in leads] != list(range(6, 121, 6)):
                raise ValueError("Expected 20 nominal six-minute forecast leads")
            runs.append([flatten(lead) for lead in leads])
        for lead_index in range(20):
            for name in sorted(runs[0][lead_index]):
                if name in {"frame", "minutes"} or name.rsplit(".", 1)[-1] in {"TP", "FP", "FN"}:
                    continue
                values = [run[lead_index][name] for run in runs]
                rows.append({"model": alias, "lead_minutes": (lead_index + 1) * 6,
                             "metric": name, "mean": statistics.mean(values),
                             "std": statistics.stdev(values) if len(values) > 1 else "",
                             "runs": len(values)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "lead_minutes", "metric", "mean", "std", "runs"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("main", "fine", "coarse"), default="main")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else ([0, 1, 2] if args.suite == "main" else [0])
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("Supply distinct seeds")
    print(export(args.suite, seeds, args.artifact_root, args.output))


if __name__ == "__main__":
    main()
