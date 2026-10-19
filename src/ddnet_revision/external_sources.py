"""Explicit, verified preparation of non-vendored upstream baseline code."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from .config import PROJECT_ROOT

CATALOG_PATH = PROJECT_ROOT / "EXTERNAL_SOURCES.json"
CACHE_ROOT = PROJECT_ROOT / "artifacts" / "external_sources"


def ast_sha256(text: str) -> str:
    return hashlib.sha256(ast.dump(ast.parse(text), include_attributes=False).encode()).hexdigest()


def _replace(text: str, old: str, new: str, count: int = 1) -> str:
    if text.count(old) != count:
        raise ValueError(f"Upstream patch anchor changed: {old[:70]!r}")
    return text.replace(old, new)


def transform(model: str, filename: str, text: str) -> str:
    if model == "motionrnn" and filename == "model.py":
        text = _replace(text, "from core.layers.SpatioTemporalLSTMCell_Motion_Highway import SpatioTemporalLSTMCell", "from .modules import SpatioTemporalLSTMCell")
        text = _replace(text, "from core.layers.MotionGRU import MotionGRU", "from .motion_gru import MotionGRU")
        text = _replace(text, "    def forward(self, all_frames, mask_true):\n", "    def forward(self, all_frames, mask_true):\n        batch_size = all_frames.size(0)\n")
        text = _replace(text, "self.configs.batch_size", "batch_size", 5)
    elif model == "trajgru":
        if filename in {"encoder.py", "forecaster.py"}:
            text = _replace(text, "from nowcasting.config import cfg\n", "")
        if filename in {"encoder.py", "forecaster.py", "model.py"}:
            text = _replace(text, "from nowcasting.utils import make_layers", "from .layers import make_layers")
        if filename == "forecaster.py":
            text = _replace(text, "def __init__(self, subnets, rnns):", "def __init__(self, subnets, rnns, output_frames):")
            text = _replace(text, "        self.blocks = len(subnets)\n", "        self.blocks = len(subnets)\n        self.output_frames = output_frames\n")
            text = _replace(text, "seq_len=cfg.HKO.BENCHMARK.OUT_LEN", "seq_len=self.output_frames")
        if filename == "trajgru.py":
            text = _replace(text, "from nowcasting.config import cfg\n", "")
            text = _replace(text, "from nowcasting.models.model import activation", "from .model import activation")
            text = _replace(text, "torch.arange(0, W).view(1, -1).repeat(H, 1).to(cfg.GLOBAL.DEVICE)", "torch.arange(0, W, device=input.device).view(1, -1).repeat(H, 1)")
            text = _replace(text, "torch.arange(0, H).view(-1, 1).repeat(1, W).to(cfg.GLOBAL.DEVICE)", "torch.arange(0, H, device=input.device).view(-1, 1).repeat(1, W)")
            text = _replace(text, "torch.nn.functional.grid_sample(input, vgrid)", "torch.nn.functional.grid_sample(input, vgrid, align_corners=True)")
            text = _replace(text, "                 act_type=cfg.MODEL.RNN_ACT_TYPE):\n", "                 act_type=None):\n        if act_type is None:\n            act_type = activation('leaky', negative_slope=0.2, inplace=True)\n")
            text = _replace(text, "seq_len=cfg.HKO.BENCHMARK.IN_LEN", "seq_len=20")
            text = _replace(text, "dtype=torch.float).to(cfg.GLOBAL.DEVICE)", "dtype=torch.float, device=inputs.device)")
    return text


def catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["models"]


def verify_prepared(model: str, root: Path = CACHE_ROOT) -> Path:
    spec = catalog()[model]
    folder = root / model
    manifest = json.loads((folder / "prepared_manifest.json").read_text())
    if manifest["commit"] != spec["commit"]:
        raise ValueError("Prepared upstream revision mismatch")
    expected_names = set(spec["files"]) | set(spec["support_files"])
    if set(manifest["files"]) != expected_names:
        raise ValueError("Prepared source file manifest is incomplete")
    for filename, expected in manifest["files"].items():
        text = (folder / filename).read_text(encoding="utf-8")
        if hashlib.sha256(text.encode()).hexdigest() != expected:
            raise ValueError(f"Prepared source changed: {model}/{filename}")
        reference = spec["files"].get(filename, spec["support_files"].get(filename))
        if ast_sha256(text) != reference["prepared_ast_sha256"]:
            raise ValueError(f"Prepared source does not match the trained integration: {filename}")
    return folder


def prepare(model: str, *, acknowledge: bool, root: Path = CACHE_ROOT,
            source_directory: Path | None = None) -> Path:
    if not acknowledge:
        raise PermissionError("Read docs/EXTERNAL_BASELINES.md and explicitly acknowledge upstream licensing status")
    spec = catalog()[model]
    root.mkdir(parents=True, exist_ok=True)
    target = root / model
    if target.exists():
        return verify_prepared(model, root)
    temporary = Path(tempfile.mkdtemp(prefix=f".{model}-", dir=root))
    try:
        hashes = {}
        for filename, item in spec["files"].items():
            if source_directory is None:
                url = f"https://raw.githubusercontent.com/{spec['repository']}/{spec['commit']}/{item['upstream_path']}"
                request = urllib.request.Request(url, headers={"User-Agent": "DDNet-fixed-source-preparation"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    raw = response.read(1024 * 1024 + 1)
            else:
                raw = (source_directory / item["upstream_path"]).read_bytes()
            if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != item["upstream_sha256"]:
                raise ValueError(f"Upstream source checksum mismatch: {model}/{filename}")
            text = transform(model, filename, raw.decode("utf-8"))
            if ast_sha256(text) != item["prepared_ast_sha256"]:
                raise ValueError(f"Integration differs from the trained numerical code: {model}/{filename}")
            (temporary / filename).write_text(text, encoding="utf-8")
            hashes[filename] = hashlib.sha256(text.encode()).hexdigest()
        for filename, item in spec["support_files"].items():
            text = (PROJECT_ROOT / item["public_path"]).read_text(encoding="utf-8")
            if ast_sha256(text) != item["prepared_ast_sha256"]:
                raise ValueError(f"Integration helper changed: {filename}")
            (temporary / filename).write_text(text, encoding="utf-8")
            hashes[filename] = hashlib.sha256(text.encode()).hexdigest()
        (temporary / "prepared_manifest.json").write_text(json.dumps({
            "commit": spec["commit"], "repository": spec["repository"],
            "files": hashes, "license_permission_granted_by_this_tool": False,
        }, indent=2), encoding="utf-8")
        os.rename(temporary, target)
        return verify_prepared(model, root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
