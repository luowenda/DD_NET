# Controlled Ablations

The final comparisons use the same 100-epoch engine, data and evaluation as
DD-Net, with seed 0. Compare with the seed-0 reference, not its three-seed mean.
Configurations are retained for reproduction; result tables and historical
candidate searches are not bundled.

## Component Comparisons

| Alias | Encoder | Core |
|---|---|---|
| `wast` | WaST convolutional stem | WaST |
| `coarse_hge_only` | 5-only attention-v1 hypergraph | WaST |
| `coarse_dcn_only` | WaST convolutional stem | DCN Core |
| `ddnet` | 5-only attention-v1 hypergraph | DCN Core |

## Fine-Grained Comparisons

| Alias | Change from DD-Net |
|---|---|
| `ablation_uniform` | Replace attention with strict-uniform two-stage aggregation |
| `ablation_small` | Use the 3x3x3 rather than 5x5x5 neighborhood |
| `ablation_dual` | Activate both neighborhoods with concatenation/linear fusion |
| `ablation_no_mask` | Set the modulation-mask gate to zero |
| `ablation_no_temporal` | Set the temporal-mixing residual gate to zero |
| `ablation_no_wavelet` | Replace backbone wavelets with learned down/up sampling |
| `ablation_mse` | Remove HFFL from the training objective |

Scale and mask/time gates retain inactive parameters. Zero-gated variants
therefore test functional removal, not parameter or latency savings.
Replacement WaST blocks and learned wavelet substitutes are constructed in a
forked RNG context to preserve common-weight initialization and training RNG.
Uniform versus attention is a module-level ablation with different parameter
counts and constructor RNG consumption, not a fully paired initialization
experiment. The no-wavelet variant retains HFFL's internal Haar operation.
The final model is single-scale; the dual-scale comparison is not a claim of
positive dual-scale benefit. These single-seed comparisons are not significance
tests and need not improve every metric.

## Run

```bash
uv run python tools/run_suite.py --suite coarse
uv run python tools/run_suite.py --suite fine
```

These commands preview plans. Add `--execute` after preparing the environment
and data. Individual configurations use the same interface:

```bash
uv run ddnet-revision train --model ablation_no_mask --run-name seed0 --seed 0
```

Settings are checked against `MODEL_REGISTRY.json` to prevent accidental
changes to the recorded comparisons.
