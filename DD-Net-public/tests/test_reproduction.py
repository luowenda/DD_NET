from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddnet_revision.config import list_models, load_config, registry, validate_config
from ddnet_revision import external_sources


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegistryTests(unittest.TestCase):
    def test_groups_and_recorded_recipe_guards(self):
        self.assertEqual((len(list_models("main")), len(list_models("fine")), len(list_models("coarse"))), (10, 8, 4))
        self.assertEqual(len(list_models()), 19)
        for alias in list_models():
            with self.subTest(alias=alias):
                config = load_config(alias)
                self.assertEqual(load_config(config["model"]["name"])["model"], config["model"])
                if config["model"]["family"] == "trainable":
                    self.assertEqual(config["training"]["epochs"], 100)
                    changed = copy.deepcopy(config)
                    changed["training"]["learning_rate"] *= 10
                    with self.assertRaises(ValueError):
                        validate_config(changed)

    def test_native_recipe_differences_are_preserved(self):
        self.assertEqual(load_config("motionrnn")["training"]["scheduler"], "none")
        self.assertEqual(load_config("predrnn")["training"]["scheduler"], "onecycle")
        conv = load_config("convlstm")["training"]
        self.assertEqual((conv["optimizer"], conv["weight_decay"], conv["loss"]), ("adamw", 0.01, "mse"))
        self.assertEqual(load_config("trajgru")["training"]["optimizer_betas"], [0.5, 0.999])
        self.assertEqual(load_config("wast")["model"]["reference_config"], "historical/train_ablation.py")

    def test_every_neural_entry_has_a_dispatch_branch(self):
        tree = ast.parse((ROOT / "src/ddnet_revision/models/adapters.py").read_text())
        factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "build_trainable_adapter")
        supported = set()
        for node in factory.body:
            if isinstance(node, ast.If):
                supported |= {item.value for item in ast.walk(node.test)
                              if isinstance(item, ast.Constant) and isinstance(item.value, str)}
        expected = {load_config(alias)["model"]["name"] for alias in list_models()
                    if load_config(alias)["model"]["family"] == "trainable"}
        self.assertEqual(supported, expected)

    def test_fine_ablation_single_factors_are_not_crossed(self):
        full = load_config()
        for alias, key, value in (
            ("ablation_small", "hypergraph_scale_gates", [1.0, 0.0]),
            ("ablation_no_mask", "modulation_mask_gate", 0.0),
            ("ablation_no_temporal", "temporal_mixer_gate", 0.0),
            ("ablation_no_wavelet", "use_wavelet", False),
            ("coarse_hge_only", "use_dcn_core", False),
        ):
            with self.subTest(alias=alias):
                changed = load_config(alias)
                before = dict(full["model"])
                before[key] = value
                metadata = {"name", "source", "implementation_status", "configuration_status"}
                self.assertEqual({k: v for k, v in before.items() if k not in metadata},
                                 {k: v for k, v in changed["model"].items() if k not in metadata})
                self.assertEqual({k: v for k, v in full["training"].items() if k != "protocol_status"},
                                 {k: v for k, v in changed["training"].items() if k != "protocol_status"})
        mse = load_config("ablation_mse")
        self.assertEqual(mse["training"]["loss"], "mse")
        self.assertEqual(load_config("ablation_uniform")["model"]["hyperedge_weighting"], "corrected_strict_uniform")
        self.assertEqual(load_config("ablation_dual")["model"]["hypergraph_scale_gates"], [1.0, 1.0])

    def test_plans_are_complete_and_do_not_execute(self):
        runner = tool("run_suite")
        self.assertEqual(len(runner.plan("main", [0, 1, 2])), 83)
        self.assertEqual(len(runner.plan("fine", [0])), 24)
        self.assertEqual(len(runner.plan("coarse", [0])), 12)
        commands = runner.plan("main", [0, 1, 2])
        flow = [command for command in commands if "optical_flow" in command]
        self.assertEqual(len(flow), 2)
        self.assertFalse(any("train" in command for command in flow))
        with patch.object(runner.subprocess, "run", side_effect=AssertionError("must not execute")):
            runner.plan("fine", [0])


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.checker = tool("check_splits")
        self.summary = {"evidence_level": "synthetic_test", "frame_count": 40,
                        "frame_interval_minutes": 6, "endpoint_tolerance_seconds": 1,
                        "train": {"years": [2017], "samples": 1, "counts_by_year": {"2017": 1}},
                        "test": {"years": [2018], "samples": 1, "counts_by_year": {"2018": 1}}}
        self.rows = [
            {"split": "train", "sample_index": "0", "year": "2017", "source_id": "a",
             "start_time": "2017-06-01T00:00:00+00:00", "end_time": "2017-06-01T03:54:00+00:00"},
            {"split": "test", "sample_index": "0", "year": "2018", "source_id": "b",
             "start_time": "2018-04-04T00:00:00+00:00", "end_time": "2018-04-04T03:54:00+00:00"},
        ]

    def test_valid_intervals_and_summary_limits(self):
        self.assertTrue(self.checker.check_sequences(self.rows, self.summary)["sequence_overlap_verified"])
        self.assertFalse(self.checker.check_summary(self.summary)["sequence_overlap_verified"])

    def test_different_years_do_not_automatically_prove_nonadjacency(self):
        self.rows[0].update(start_time="2017-12-31T20:00:00+00:00", end_time="2017-12-31T23:54:00+00:00")
        self.rows[1].update(start_time="2018-01-01T00:00:00+00:00", end_time="2018-01-01T03:54:00+00:00")
        with self.assertRaisesRegex(ValueError, "adjacent"):
            self.checker.check_sequences(self.rows, self.summary)

    def test_missing_duplicate_and_shared_source_records_are_rejected(self):
        for rows in (self.rows[:1], [*self.rows, self.rows[0]],
                     [self.rows[0], dict(self.rows[1], source_id="a")]):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                self.checker.check_sequences(rows, self.summary)

    def test_source_time_jitter_is_preserved_and_bounded(self):
        self.rows[0]["end_time"] = "2017-06-01T03:54:01+00:00"
        report = self.checker.check_sequences(self.rows, self.summary)
        self.assertEqual(report["maximum_endpoint_span_deviation_seconds"], 1)
        self.rows[0]["end_time"] = "2017-06-01T03:54:02+00:00"
        with self.assertRaises(ValueError):
            self.checker.check_sequences(self.rows, self.summary)

    def test_public_summary_does_not_claim_sequence_mapping(self):
        summary = json.loads((ROOT / "data/split_summary.json").read_text())
        report = self.checker.check_summary(summary)
        self.assertEqual((summary["train"]["samples"], summary["test"]["samples"]), (551, 88))
        self.assertFalse(summary["sequence_timestamps_available"])
        self.assertFalse(report["sequence_overlap_verified"])
        self.assertFalse((ROOT / "data/sequence_time_inventory.csv").exists())

    def test_inventory_intervals_are_not_falsely_labeled_array_mapping(self):
        rows = [dict(row, inventory_index=row["sample_index"], inventory_id=f"record-{index}")
                for index, row in enumerate(self.rows)]
        report = self.checker.check_sequences(rows, self.summary, indexed=False)
        self.assertTrue(report["inventory_interval_adjacency_verified"])
        self.assertFalse(report["sequence_adjacency_verified"])
        self.assertFalse(report["array_index_mapping_verified"])
        self.assertEqual(report["records_with_nonzero_endpoint_span_deviation"], 0)

    def test_time_export_does_not_guess_array_indices(self):
        exporter = tool("export_time_inventory")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("test\nstarttime,endtime,avg\n2018-01-01 00:00:00,2018-01-01 03:54:00,5\n")
            text, metadata = exporter.export(path)
        self.assertIn("inventory_index", text)
        self.assertNotIn("sample_index", text)
        self.assertFalse(metadata["array_index_mapping_established"])


class ExternalSourceTests(unittest.TestCase):
    def test_nonvendored_source_is_not_bundled(self):
        for model in ("trajgru", "motionrnn"):
            self.assertFalse((ROOT / "src/ddnet_revision/models/vendor" / model).exists())
            with self.assertRaises(PermissionError):
                external_sources.prepare(model, acknowledge=False)
            for item in external_sources.catalog()[model]["files"].values():
                self.assertEqual(len(item["upstream_sha256"]), 64)
                self.assertEqual(len(item["prepared_ast_sha256"]), 64)

    def test_offline_preparation_and_tamper_detection(self):
        text = "VALUE = 1\n"
        fixture = {"motionrnn": {"repository": "test/fixture", "commit": "0" * 40,
                   "files": {"__init__.py": {"upstream_path": "source.py",
                   "upstream_sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "prepared_ast_sha256": external_sources.ast_sha256(text)}},
                   "support_files": {}}}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "source.py").write_text(text)
            with patch.object(external_sources, "catalog", return_value=fixture):
                folder = external_sources.prepare("motionrnn", acknowledge=True, root=base / "cache",
                                                  source_directory=source)
                self.assertEqual(external_sources.verify_prepared("motionrnn", base / "cache"), folder)
                (folder / "__init__.py").write_text("VALUE = 2\n")
                with self.assertRaises(ValueError):
                    external_sources.verify_prepared("motionrnn", base / "cache")

    def test_raw_hash_mismatch_is_rejected_without_installation(self):
        fixture = {"motionrnn": {"repository": "test/fixture", "commit": "0" * 40,
                   "files": {"__init__.py": {"upstream_path": "source.py",
                   "upstream_sha256": "0" * 64, "prepared_ast_sha256": "0" * 64}}, "support_files": {}}}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "source.py").write_text("VALUE = 1\n")
            with patch.object(external_sources, "catalog", return_value=fixture), self.assertRaises(ValueError):
                external_sources.prepare("motionrnn", acknowledge=True, root=base / "cache", source_directory=base)
            self.assertFalse((base / "cache/motionrnn").exists())


if __name__ == "__main__":
    unittest.main()
