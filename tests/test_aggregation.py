import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("metric_aggregator", ROOT / "tools/aggregate_metrics.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)
from ddnet_revision.config import load_config


def write_run(root, alias, seed, value, test_hash="same-test", fallback=0):
    config = load_config(alias)
    config["experiment"]["seed"] = seed
    deterministic = config["model"]["family"] == "non_trainable"
    directory = root / config["model"]["name"] / ("deterministic" if deterministic else f"seed{seed}")
    directory.mkdir(parents=True)
    prediction = {"config": config, "datasets": {"test": {"sha256": test_hash}}}
    (directory / "prediction_manifest.json").write_text(json.dumps(prediction))
    (directory / "metrics.json").write_text(json.dumps({
        "samples": 88, "comparison": ">=", "overall": {"MSE": value, "thresholds": {"10": {"CSI": value / 10}}},
        "by_lead_time": [{"frame": frame, "minutes": frame * 6, "MSE": value,
                           "thresholds": {"10": {"CSI": value / 10}}}
                          for frame in range(1, 21)],
    }))
    if deterministic:
        (directory / "pysteps_status.json").write_text(json.dumps({"fallback_count": fallback}))
    else:
        (directory / "training_manifest.json").write_text(json.dumps({
            "config": config, "datasets": {"train": {"sha256": "same-train"}},
        }))
        (directory / "train.jsonl").write_text("".join(json.dumps({"epoch": epoch, "global_step": epoch * 137}) + "\n"
                                                     for epoch in range(1, 101)))
    return directory


class AggregationTests(unittest.TestCase):
    def test_lead_export_preserves_twenty_leads_and_seed_std(self):
        import csv

        spec = importlib.util.spec_from_file_location("lead_export", ROOT / "tools/export_lead_time.py")
        lead = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lead)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, "ddnet", 0, 1)
            write_run(root, "ddnet", 1, 3)
            output = root / "lead.csv"
            with patch.object(lead, "collect"), patch.object(lead, "list_models", return_value=["ddnet"]):
                self.assertEqual(lead.export("main", [0, 1], root, output), 40)
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({int(row["lead_minutes"]) for row in rows}, set(range(6, 121, 6)))
            for row in rows:
                if row["metric"] == "MSE":
                    self.assertEqual(float(row["mean"]), 2)
                    self.assertAlmostEqual(float(row["std"]), math.sqrt(2))

    def test_sample_std_not_sem_or_population_std(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, "ddnet", 0, 1)
            write_run(root, "ddnet", 1, 3)
            with patch.object(aggregate, "list_models", return_value=["ddnet"]):
                result = aggregate.collect("main", [0, 1], root)["ddnet"]["metrics"]["MSE"]
            self.assertEqual(result["mean"], 2)
            self.assertAlmostEqual(result["std"], math.sqrt(2))

    def test_single_seed_has_no_artificial_std(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, "ddnet", 0, 1)
            with patch.object(aggregate, "list_models", return_value=["ddnet"]):
                result = aggregate.collect("fine", [0], root)
            self.assertIsNone(result["ddnet"]["metrics"]["MSE"]["std"])

    def test_mixed_test_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, "ddnet", 0, 1)
            write_run(root, "ddnet", 1, 3, test_hash="different")
            with patch.object(aggregate, "list_models", return_value=["ddnet"]), self.assertRaises(ValueError):
                aggregate.collect("main", [0, 1], root)

    def test_incomplete_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = write_run(root, "ddnet", 0, 1)
            (run / "train.jsonl").write_text(json.dumps({"epoch": 1, "global_step": 137}))
            with patch.object(aggregate, "list_models", return_value=["ddnet"]), self.assertRaises(ValueError):
                aggregate.collect("main", [0], root)

    def test_pysteps_is_once_and_fallbacks_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = write_run(root, "optical_flow", 0, 1)
            with patch.object(aggregate, "list_models", return_value=["optical_flow"]):
                result = aggregate.collect("main", [0, 1, 2], root)["optical_flow"]
                self.assertTrue(result["deterministic"])
                self.assertEqual(result["seeds"], [])
                self.assertIsNone(result["metrics"]["MSE"]["std"])
                (run / "pysteps_status.json").write_text('{"fallback_count": 1}')
                with self.assertRaises(ValueError):
                    aggregate.collect("main", [0, 1, 2], root)


if __name__ == "__main__":
    unittest.main()
