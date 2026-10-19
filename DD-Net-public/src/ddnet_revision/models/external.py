from __future__ import annotations

import importlib.util
import sys

from ..external_sources import verify_prepared


def load_external_module(name: str):
    try:
        folder = verify_prepared(name)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Prepare {name} with tools/prepare_external_baselines.py first; "
            "see docs/EXTERNAL_BASELINES.md"
        ) from error
    package_name = f"_ddnet_external_{name}"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name, folder / "__init__.py", submodule_search_locations=[str(folder)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[package_name]
            raise
    return sys.modules[package_name]
