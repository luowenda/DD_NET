"""No-dependency checks; these do not run DD-Net on the local machine."""

from __future__ import annotations

import ast
import copy
import fcntl
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddnet_revision.config import FINAL_MODEL, comparable_model, list_models, load_config, validate_config


def engine_helpers(*names):
    tree = ast.parse((ROOT / "src/ddnet_revision/engine.py").read_text())
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"Path": Path, "fcntl": fcntl, "contextmanager": contextmanager,
                 "comparable_model": comparable_model}
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, "engine_helpers", "exec"), namespace)
    return namespace


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_final_model_is_default_in_frozen_registry(self):
        self.assertEqual(list_models()[0], "ddnet")
        self.assertEqual(self.config["model"]["name"], FINAL_MODEL)
        self.assertEqual(load_config(FINAL_MODEL)["model"], self.config["model"])
        self.assertEqual(len(list((ROOT / "configs/models").glob("*.toml"))), 19)

    def test_final_recipe(self):
        t = self.config["training"]
        self.assertEqual((t["epochs"], t["batch_size"], t["drop_last"]), (100, 4, True))
        self.assertEqual((t["optimizer"], t["learning_rate"], t["weight_decay"]),
                         ("adamw", 1e-4, 0.01))
        self.assertEqual((t["scheduler"], t["minimum_learning_rate"]), ("cosine", 1e-6))
        self.assertEqual((t["loss"], t["hffl_weight"]), ("mse_hffl", 0.001))
        self.assertEqual(t["optimizer_betas"], [0.9, 0.999])
        self.assertEqual(t["optimizer_eps"], 1e-8)
        self.assertFalse(t["scheduled_sampling"])
        self.assertEqual(t["gradient_clip_norm"], 0.0)

    def test_default_data_and_runtime(self):
        self.assertEqual(self.config["experiment"]["seed"], 0)
        self.assertEqual(self.config["data"]["train_samples"], 551)
        self.assertEqual(self.config["data"]["test_samples"], 88)
        self.assertEqual(self.config["runtime"]["prediction_batch_size"], 4)
        self.assertEqual(self.config["runtime"]["checkpoint_every"], 200)
        self.assertEqual(Path(self.config["data"]["train_path"]).parent, ROOT / "data")

    def test_incompatible_architectures_are_rejected(self):
        mutations = {
            "hypergraph_scale_gates": [1.0, 1.0],
            "hyperedge_weighting": "uniform",
            "batching_mode": "legacy_broadcast",
            "wavelet_subband_grouping": "axis_consistent",
            "residual_layer_scale_mode": "independent",
            "hypergraph_batching_revision": "legacy_v1",
            "drop_path_rate": 0.25,
            "use_modulation_mask": False,
            "use_temporal_mixer": False,
            "use_wavelet": False,
            "hypergraph_input_stem_channels": 4,
            "name": "ddnet_legacy",
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                changed = copy.deepcopy(self.config)
                changed["model"][key] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)

    def test_evaluation_mutations_are_rejected(self):
        changes = [
            ("experiment", "input_frames", 10),
            ("experiment", "normalization_scale", 1.0),
            ("data", "canonical_layout", "N,T,H,W"),
            ("selection", "test_during_training", True),
            ("selection", "validation_enabled", True),
            ("selection", "strategy", "best_test"),
            ("evaluation", "comparison", ">"),
            ("evaluation", "clip_predictions", False),
            ("training", "scheduled_sampling", True),
            ("training", "loss", "mse"),
            ("runtime", "prediction_batch_size", 0),
            ("runtime", "prediction_batch_size", 1),
            ("runtime", "deterministic_cudnn", False),
            ("data", "test_year", 2021),
        ]
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(self.config)
                changed[section][key] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)

    def test_discarded_candidate_names_are_not_supported(self):
        for name in ("ddnet_legacy", "convlstm_ndrplz_e100", "ddnet_attention_v1_dual_dynamic_e100"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_config(name)

    def test_config_does_not_open_the_datasets(self):
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if path.suffix == ".npy":
                self.fail("Configuration loading must not read radar arrays")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            self.assertEqual(load_config()["model"]["name"], FINAL_MODEL)


class ArtifactGuardTests(unittest.TestCase):
    def setUp(self):
        self.helpers = engine_helpers("_training_run_lock", "_prediction_run_lock",
                                      "_prediction_artifact_names", "_validate_checkpoint_config",
                                      "_validate_resume_config", "load_model_checkpoint")

    def test_legacy_haar_exception_never_ignores_learned_state(self):
        class Fixed:
            def cpu(self):
                return self

        fixed = Fixed()

        class Adapter:
            model = None

            def state_dict(self):
                return {"model.weight": "learned", "hffl.dwt.filters": fixed}

            def load_state_dict(self, state, strict):
                if not strict or set(state) != set(self.state_dict()):
                    raise RuntimeError("strict state mismatch")
                self.loaded = state

        config = load_config("predrnn")
        payload = {"config": config, "model": {"model.weight": "learned"}}
        fake = types.SimpleNamespace(load=lambda *args, **kwargs: payload,
                                     testing=types.SimpleNamespace(assert_close=lambda a, b, **kwargs: self.assertIs(a, b)))
        self.helpers["torch"] = fake
        self.helpers["__package__"] = "ddnet_revision"
        losses = types.ModuleType("ddnet_revision.losses")
        losses.HaarDWT2D = lambda: types.SimpleNamespace(filters=fixed)
        with patch.dict(sys.modules, {"ddnet_revision.losses": losses}):
            adapter = Adapter()
            self.helpers["load_model_checkpoint"](adapter, "trusted-qa-checkpoint", "cpu", config)
            self.assertIs(adapter.loaded["hffl.dwt.filters"], fixed)
            for state in ({}, {"model.weight": "learned", "unexpected": "bad"}):
                payload["model"] = state
                with self.subTest(keys=list(state)), self.assertRaises((RuntimeError, AttributeError)):
                    self.helpers["load_model_checkpoint"](Adapter(), "trusted-qa-checkpoint", "cpu", config)

    def test_existing_training_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "last.pt"
            checkpoint.write_bytes(b"keep")
            with self.assertRaises(FileExistsError), self.helpers["_training_run_lock"](root, None):
                self.fail("must not overwrite")
            self.assertEqual(checkpoint.read_bytes(), b"keep")

    def test_resume_uses_the_same_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "last.pt").write_bytes(b"keep")
            with self.helpers["_training_run_lock"](root, root / "last.pt"):
                pass
            with self.assertRaises(ValueError), self.helpers["_training_run_lock"](root, root / "other.pt"):
                self.fail("must not resume a different checkpoint")

    def test_duplicate_training_is_locked_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.helpers["_training_run_lock"](root, None):
                with self.assertRaises(RuntimeError), self.helpers["_training_run_lock"](root, None):
                    self.fail("must not start twice")

    def test_predictions_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions.npy"
            output.write_bytes(b"keep")
            with self.assertRaises(FileExistsError), self.helpers["_prediction_run_lock"](root):
                self.fail("must not overwrite")
            self.assertEqual(output.read_bytes(), b"keep")

    def test_checkpoint_architecture_mismatch_is_rejected(self):
        config = load_config()
        saved = copy.deepcopy(config)
        saved["model"]["hyperedge_weighting"] = "uniform"
        with self.assertRaises(ValueError):
            self.helpers["_validate_checkpoint_config"](config, saved)

    def test_checkpoint_can_move_between_machines(self):
        config = load_config()
        saved = copy.deepcopy(config)
        saved["data"]["train_path"] = "another-machine/train.npy"
        saved["data"]["test_path"] = "another-machine/test.npy"
        saved["experiment"]["device"] = "cuda:1"
        saved["experiment"]["num_workers"] = 0
        self.helpers["_validate_checkpoint_config"](config, saved)


class SourceAndEntryTests(unittest.TestCase):
    def test_project_license_and_third_party_attribution_are_present(self):
        license_text = (ROOT / "LICENSE").read_text()
        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0, January 2004", license_text)
        self.assertIn("Copyright 2026 DD-Net contributors.", license_text)
        original_license = (ROOT / "licenses/OPENSTL-APACHE-2.0.txt").read_text()
        self.assertIn("Copyright 2022-2023 CAIRI AI Lab.", original_license)
        self.assertIn("Copyright 2022-2023 CAIRI AI Lab.", (ROOT / "NOTICE").read_text())
        self.assertNotIn("not yet been selected", (ROOT / "LICENSING.md").read_text())

    def test_numerical_sources_are_preserved(self):
        spec = importlib.util.spec_from_file_location("release_verifier", ROOT / "tools/verify_release.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.verify()["status"], "passed")

    def test_cli_help_and_config_do_not_require_torch(self):
        script = ROOT / "src/ddnet_revision/cli.py"
        env = dict(__import__("os").environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        for args in (["--help"], ["train", "--help"], ["validate-config"]):
            result = subprocess.run([sys.executable, "-m", "ddnet_revision.cli", *args],
                                    env=env, capture_output=True, text=True, check=True)
            self.assertTrue(result.stdout.strip())
        self.assertTrue(script.is_file())

    def test_core_has_no_nonfinal_standalone_demo(self):
        tree = ast.parse((ROOT / "src/ddnet_revision/models/vendor/ddnet/model.py").read_text())
        self.assertFalse(any(isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
                             for node in tree.body))

    def test_inference_only_receives_observed_input(self):
        path = ROOT / "src/ddnet_revision/engine.py"
        tree = ast.parse(path.read_text())
        predict = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "predict_trainable")
        calls = [node for node in ast.walk(predict) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "predict"]
        self.assertEqual(len(calls), 1)
        self.assertIn("input_dbz", ast.unparse(calls[0]))
        self.assertNotIn("target_dbz", ast.unparse(calls[0]))

    def test_training_loader_is_training_only(self):
        tree = ast.parse((ROOT / "src/ddnet_revision/engine.py").read_text())
        train = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_train_run")
        calls = [node for node in ast.walk(train) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "_loader"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.literal_eval(calls[0].args[1]), "train")

    def test_all_python_sources_compile(self):
        for directory in ("src", "tools", "tests"):
            for path in (ROOT / directory).rglob("*.py"):
                with self.subTest(path=path.relative_to(ROOT)):
                    compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_no_results_or_private_data_are_shipped(self):
        for name in ("publication", "results"):
            self.assertFalse((ROOT / name).exists())
        rules = (ROOT / ".gitignore").read_text().splitlines()
        for pattern in ("data/*", "*.npy", "*.pt", "*.pth", "*.ckpt", "artifacts/"):
            self.assertIn(pattern, rules)


if __name__ == "__main__":
    unittest.main()
