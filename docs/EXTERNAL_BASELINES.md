# Non-Vendored MotionRNN And TrajGRU

At the audited commits, the MotionRNN and Hzzone PyTorch-port repositories did
not contain a verified redistribution license. This repository therefore does
not bundle those source files and does not assign Apache-2.0 to them. A citation
or public GitHub URL is not a redistribution grant.

The public integration includes model-specific recipes, adapters and a source
preparation tool. Use it only after reviewing the upstream terms and obtaining
any permission needed for your use. The acknowledgment flag records awareness;
it does **not** grant a license or assert that permission exists.

```bash
python3 tools/prepare_external_baselines.py --model motionrnn \
  --acknowledge-unlicensed-upstream
python3 tools/prepare_external_baselines.py --model trajgru \
  --acknowledge-unlicensed-upstream
```

The tool fetches only listed files at fixed commit hashes. It checks each raw
SHA256, applies the exact integration transformations, and checks the prepared
Python AST against the code used by the experiments. The raw source itself is
not included in this repository or embedded in a patch/archive here.

For offline preparation, supply `--source-directory /path/to/checkout` at the
pinned revision. The same raw checksums apply. Downloads are never triggered
by importing a model, listing configurations, running tests, or printing a plan.

Prepared packages and their checksums live in `artifacts/external_sources/`,
which is excluded from Git. Runtime imports verify them before executing any
external model code. Do not add that directory to the public repository.

Changes are restricted to import integration, device/config independence,
runtime batch-size handling, the 20-frame output interface, and TrajGRU's
explicit `align_corners=True`. The TrajGRU 64x64 encoder/forecaster adapter is
the actual radar-specific port integration, not an official PyTorch release.

`EXTERNAL_SOURCES.json` records the upstream repositories, fixed commits,
upstream file paths and SHA256, and prepared AST hashes. It contains provenance
and transformation checks, not a copy of the unlicensed model source.
