"""Standard-library-only checks for the code-only final DD-Net release."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def node_hash(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def symbols(path: Path) -> dict[str, ast.AST]:
    return {node.name: node for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}


def model_constructor(path: Path) -> ast.Call:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "model_type"]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one final numerical model constructor")
    return candidates[0]


def verify(root: Path = ROOT) -> dict:
    from ddnet_revision.config import list_models, load_config

    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    checked = 0
    for record in manifest["unchanged_files"]:
        actual = hashlib.sha256((root / record["file"]).read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"Numerical/environment source changed: {record['file']}")
        checked += 1
    for relative, expected in manifest["preserved_symbols"].items():
        actual = symbols(root / relative)
        mutable = set(manifest.get("mutable_integration_symbols", {}).get(relative, []))
        if set(actual) - mutable != set(expected):
            raise ValueError(f"Unexpected/missing numerical definitions: {relative}")
        for name, digest in expected.items():
            if node_hash(actual[name]) != digest:
                raise ValueError(f"Numerical definition changed: {relative}:{name}")
            checked += 1
    adapter = root / "src/ddnet_revision/models/adapters.py"
    for name, digest in manifest["adapter_preserved_symbols"].items():
        if node_hash(symbols(adapter)[name]) != digest:
            raise ValueError(f"Shared sequence/loss adapter changed: {name}")
        checked += 1
    trajgru = symbols(adapter)["_build_trajgru"]
    if node_hash(ast.Module(body=trajgru.body[3:], type_ignores=[])) != manifest["trajgru_constructor_tail_sha256"]:
        raise ValueError("TrajGRU numerical constructor changed beyond import integration")
    if node_hash(symbols(adapter)["DirectSequenceAdapter"]) != manifest["adapter_class_sha256"]:
        raise ValueError("The final prediction/loss adapter changed")
    if node_hash(model_constructor(adapter)) != manifest["constructor_call_sha256"]:
        raise ValueError("The final model's constructor arguments/order changed")
    config = load_config()
    if config["model"] != manifest["final_configuration"]["model"]:
        raise ValueError("The final architecture configuration changed")
    if config["training"] != manifest["final_configuration"]["training"]:
        raise ValueError("The recorded training recipe changed")
    expected_experiment = manifest["final_configuration"]["experiment"]
    if config["experiment"] != expected_experiment:
        raise ValueError("The default experimental configuration changed")
    for alias in list_models():
        load_config(alias)
        record = manifest["recorded_configurations"][alias]
        if hashlib.sha256((root / record["file"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Recorded experiment configuration changed: {alias}")
        checked += 1
    ignored = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "artifacts", ".DS_Store"}
    blocked_suffixes = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".zip", ".gz", ".pdf", ".png"}
    forbidden = ("/Users/", "/root/autodl", "connect.bjb2.", "BEGIN OPENSSH PRIVATE KEY",
                 "BEGIN RSA PRIVATE KEY")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if set(relative.parts) & ignored:
            continue
        public_data = {"data/README.md", "data/split_summary.json"}
        if relative.parts[0] == "data" and relative.as_posix() not in public_data:
            continue
        if path.is_symlink():
            raise ValueError(f"Do not ship symlinks: {relative}")
        if not path.is_file():
            continue
        if path.suffix.lower() in blocked_suffixes or path.stat().st_size > 2 * 1024**2:
            raise ValueError(f"Unexpected result, data, checkpoint, or large file: {relative}")
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in forbidden):
            raise ValueError(f"Private path or server material: {relative}")
    return {"status": "passed", "preservation_checks": checked + 6,
            "model": config["model"]["name"], "runtime_model_execution": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
