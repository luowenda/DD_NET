# DD-Net

**Dual-Dynamic Adaptive Network for Precipitation Nowcasting**

Training and evaluation code for DD-Net and nine comparison baselines.
DD-Net combines a WaST-derived backbone, a hypergraph
encoder with adaptive incidence attention, and a deformable convolution core.
The released model uses a single 5x5x5 neighborhood, fixed topology, and
sample-independent batching. Deformable sampling is spatial; temporal modeling
uses folded channel mixing.

## Installation

Neural models require Linux and an NVIDIA GPU. The research environment used
Python 3.12, PyTorch 2.12.1/CUDA 13.0 and torchvision 0.27.1 on an RTX 5090.
The dependency specification and lockfile are included.

Run the following commands from the cloned repository. This release supports
the editable source-checkout workflow (`uv sync`); standalone wheel installation
is not supported because configurations and registries live at the repository root.

```bash
uv sync --locked --extra dev
uv run ddnet-revision list-models --group main
uv run ddnet-revision validate-config --model ddnet
```

MotionRNN and TrajGRU require separate, explicit source preparation because
redistribution permission was not verified. See
[external baseline setup](docs/EXTERNAL_BASELINES.md).

## Data

Radar data and pretrained weights are not distributed with this repository.
Provide authorized preprocessed NumPy arrays and set their paths in
`configs/experiment.toml`:

```text
data/train_data_new.npy    # T,N,H,W = 40,551,64,64
data/test_data_new.npy     # T,N,H,W = 40,88,64,64
```

Values must be finite reflectivity in 0-70 dBZ, not pre-normalized to [0,1].
The first 20 frames are context and the next 20 are forecast targets, at a
nominal six-minute interval. The reported split uses 551 training sequences
from 2011-2017 and 2019-2021, and 88 test sequences from 2018, without a
validation subset in these 100-epoch runs. See [data format](data/README.md).

```bash
uv run ddnet-revision audit-data
```

## Train And Evaluate

```bash
uv run ddnet-revision train --model ddnet --run-name seed0 --seed 0
RUN=artifacts/ddnet_incidence_attention_large_e100/seed0
uv run ddnet-revision predict --model ddnet --run-name seed0 --seed 0 --checkpoint "$RUN/last.pt"
uv run ddnet-revision evaluate --model ddnet --run-name seed0 --seed 0 --predictions "$RUN/predictions.npy"
```

To resume an interrupted run, repeat the training command with
`--resume "$RUN/last.pt"`. Each run saves its configuration, data hashes,
environment, epoch logs, and an atomic `last.pt`. Existing artifacts are
protected against accidental overwrite. Only load trusted checkpoints.

DD-Net trains for 100 epochs with batch size 4, AdamW (lr 1e-4, weight decay
0.01), cosine decay to 1e-6 without warmup, maximum DropPath 0.1, and MSE+HFFL
(outer coefficient 1; internal HFFL weight 0.001). Selection uses the last epoch,
not the best test score. Earlier architecture exploration did consult 2018
results, so this is not an untouched confirmation set.

## Baselines And Reproduction

Main aliases: `ddnet`, `wast`, `convlstm`, `trajgru`, `predrnn`, `predrnnv2`,
`mim`, `motionrnn`, `swinlstm`, and `optical_flow`.
Sources, pinned versions, adaptations and actual model-specific training
recipes are documented in [baseline implementations](docs/BASELINES.md).
They are not all unchanged official implementations or author-recommended
radar presets. All models share the data and evaluation pipeline.

PySTEPS uses Lucas-Kanade motion and deterministic semi-Lagrangian
extrapolation, with no training or checkpoint:

```bash
uv run ddnet-revision predict --model optical_flow --run-name deterministic
uv run ddnet-revision evaluate --model optical_flow --run-name deterministic --predictions artifacts/optical_flow_pysteps/deterministic/predictions.npy
```

Preview the main comparison without executing models:

```bash
uv run python tools/run_suite.py --suite main
```

After preparing data and external sources, add `--execute` to run sequentially.
The main suite uses seeds 0/1/2. Only final configurations are included, not
historical searches.

```bash
uv run python tools/aggregate_metrics.py --suite main --output artifacts/main-summary.json
uv run python tools/export_lead_time.py --suite main --output artifacts/lead-time.csv
```

Evaluation reports MSE/MAE/PSNR/SSIM and CSI/POD/FAR at 10/20/30/40 dBZ,
including each lead time from 6 to 120 minutes. Categorical counts are pooled;
overall CSI is not the mean of lead-time CSI. Multi-seed summaries use sample
standard deviation.

## Efficiency

```bash
uv run ddnet-revision profile --model ddnet --checkpoint "$RUN/last.pt" --batch-size 1 --warmup 10 --iterations 100 --output artifacts/ddnet-profile.json
```

Use trained checkpoints on the same idle GPU with the same precision settings.
Latency is per full 20-frame forecast. PySTEPS uses CPU profiling without a
checkpoint. Captured FLOPs may omit custom operators and are not complete
theoretical costs.

## Checks

```bash
uv run python tools/verify_release.py
uv run pytest -q
```

For an optional synthetic GPU check, run `uv run python tools/gpu_qa.py`.
A fresh clean Linux/CUDA end-to-end acceptance run of this packaged release
is still pending; CPU/static tests do not replace GPU acceptance.
Without the reference data and checkpoints, downloading this code alone does
not independently verify the paper's numerical results.

## License And Attribution

Project contributions use [Apache-2.0](LICENSE). See [NOTICE](NOTICE),
[third-party notices](THIRD_PARTY_NOTICES.md) and [licensing](LICENSING.md)
for third-party terms. Cite each baseline's original paper and implementation.
For PySTEPS cite Pulkkinen et al. (2019),
[doi:10.5194/gmd-12-4185-2019](https://doi.org/10.5194/gmd-12-4185-2019).
