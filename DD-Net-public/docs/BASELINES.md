# Baseline Sources And Actual Training Recipes

These are the implementations used for the reviewed E100 comparison. Source
provenance and training-configuration provenance are separate. An upstream model
does not imply that its authors recommended this radar dataset's full recipe.
`MODEL_REGISTRY.json` and `configs/models/` record every effective setting.

## Fixed Implementations

| Public alias | Implementation used | Upstream reference and local integration |
|---|---|---|
| `convlstm` | OpenSTL benchmark implementation | OpenSTL `eecf8a3078f0a178dbc7b28723da20f94ce36985`, `openstl/models/convlstm_model.py` and `openstl/modules/convlstm_modules.py`; radar sequence/loss adapter |
| `predrnn` | OpenSTL benchmark implementation | Same OpenSTL commit, `openstl/models/predrnn_model.py`, `openstl/modules/predrnn_modules.py`; relative imports and radar adapter |
| `predrnnv2` | OpenSTL benchmark implementation | Same OpenSTL commit, `openstl/models/predrnnv2_model.py`, `openstl/modules/predrnnv2_modules.py`; relative imports and radar adapter |
| `mim` | OpenSTL PyTorch implementation | Same OpenSTL commit, `openstl/models/mim_model.py`, `openstl/modules/mim_modules.py`; authors' [MIM release](https://github.com/Yunbo426/MIM/tree/a61eb6a1be204cf0a7767ccbb61e2367a8b42401) uses TensorFlow 1; radar/state integration differs from a verbatim author release |
| `wast` | OpenSTL WaST implementation | Same OpenSTL commit, `openstl/models/wast_model.py`, `openstl/modules/wast_modules.py`; 20-frame radar adapter; OpenSTL grouping and shared LayerScale behavior retained |
| `trajgru` | Hzzone PyTorch port, not an author-official PyTorch release | [Runtime port](https://github.com/Hzzone/Precipitation-Nowcasting/tree/1e38d21aec77522dcbce9c1ecdd2dab766112788), checked against [official HKO-7 MXNet reference](https://github.com/sxjscience/HKO-7/tree/57b987bd893bd6910996494c247359ace371d333); 64x64 encoder/forecaster integration, device-safe states, dynamic output horizon and explicit `align_corners=True` |
| `motionrnn` | THUML author-maintained PyTorch release | [MotionRNN](https://github.com/thuml/MotionRNN/tree/002415f4e7384f14fd20501b76febe8c5caaca7e), `core/models/MotionRNN_PredRNN.py`, `core/layers/MotionGRU.py`, `core/layers/SpatioTemporalLSTMCell_Motion_Highway.py`; relative imports and runtime batch-sized states |
| `swinlstm` | Author-maintained SwinLSTM-D core | [SongTang-x/SwinLSTM](https://github.com/SongTang-x/SwinLSTM/tree/4425bcfefbebfac85c9fc6c6659361b8154d5ca3), `SwinLSTM_D.py`; canonical sequence wrapper and radar adapter |
| `optical_flow` | Installed PySTEPS 1.21.5 | [PySTEPS](https://github.com/pySTEPS/pysteps/tree/19e5f0df9eb65c1b0a5e03e2b0ec4e9223a945a7), Lucas-Kanade motion from the last three observed frames and deterministic semi-Lagrangian extrapolation for 20 frames |

OpenSTL's pinned source is available at
https://github.com/chengtan9907/OpenSTL/tree/eecf8a3078f0a178dbc7b28723da20f94ce36985.
Do not describe its implementations collectively as unchanged original-author code.
The currently selected ConvLSTM is **not** the ndrplz diagnostic implementation.
Historical alternate implementations are not included.

## Effective Recipes

All neural experiments use 100 epochs, batch size 4, training `drop_last=true`,
the same 551 training sequences and 88 test sequences, and the final checkpoint.
The shared engine uses the same physical-dBZ input and evaluator. Optimizers,
losses, sampling, clipping, and schedules remain model-specific and are not
claimed to be equally optimal for all architectures.

| Alias | Optimizer / configured LR / weight decay | Schedule | Objective / sampling / gradient clip |
|---|---|---|---|
| `ddnet`, `wast` | AdamW / 1e-4 / 0.01 | Epoch cosine to 1e-6, no warmup | MSE + HFFL (internal 0.001); no teacher forcing; no clipping |
| `convlstm` | AdamW / 1e-4 / 0.01 | Epoch cosine to 1e-6 | Future-20-frame MSE, no HFFL; frame-wise scheduled sampling; no clipping |
| `predrnn`, `predrnnv2`, `mim` | Adam / 5e-4 / 0 | OneCycle, maximum LR 5e-4 | Native sequence objective; frame-wise sampling; no clipping |
| `motionrnn` | Adam / 3e-4 / 0 | Fixed LR | Native sequence MSE; frame-wise sampling; no clipping |
| `trajgru` | Adam / 1e-4 / 0, betas (0.5,0.999) | Fixed LR | Plain MSE; no scheduled sampling; gradient norm clip 50 |
| `swinlstm` | Adam / 1e-4 / 0 | Fixed LR | Native next-frame sequence MSE; no scheduled sampling; no clipping |
| `optical_flow` | No optimizer or learned weights | Not applicable | PySTEPS deterministic extrapolation; no model training |

For OneCycle, the configured LR is its maximum; the scheduler establishes the
actual starting LR. Adam is constructed with betas (0.9,0.999), eps 1e-8 unless
explicitly overridden. OneCycle uses the PyTorch defaults: `pct_start=0.3`,
cosine annealing, `div_factor=25`, `final_div_factor=10000`, two phases and
`cycle_momentum=true`. For Adam this also cycles beta1 between 0.95 and 0.85;
it is not a fixed-beta1 recipe. See the
[PyTorch 2.12 OneCycleLR reference](https://docs.pytorch.org/docs/2.12/generated/torch.optim.lr_scheduler.OneCycleLR.html).
AdamW in the final DD-Net/WaST/ConvLSTM cosine recipes explicitly fixes betas
(0.9,0.999) and eps 1e-8 and does not use OneCycle momentum cycling.
Sampling masks are drawn per sample and frame, not independently per pixel.

### Which Settings Came From The Authors Or OpenSTL?

- **ConvLSTM:** architecture and sampling design reference OpenSTL Moving-MNIST;
  future-only MSE and DD-Net-matched AdamW/cosine E100 budget are project choices,
  not an author-recommended precipitation preset.
- **PredRNN, PredRNNv2, MIM:** reference OpenSTL `configs/mmnist/PredRNN.py`,
  `PredRNNv2.py`, and `MIM.py`. The radar grid, 20-to-20 horizon, batch/epoch
  budget and sampling timeline are adaptations. PredRNNv2 uses reverse-sampling
  transitions at 6,850 and 13,700 updates and exponential scale 2,740.
- **MotionRNN:** references the author's
  `human_script/MotionRNN_PredRNN_human_train.sh`, retaining its Adam/LR starting
  point; radar layout, horizon, E100 budget and sampling timeline are adaptations.
- **TrajGRU:** network widths/links and optimizer settings reference the official
  HKO-7 configuration recorded in the TOML. The PyTorch port, 64x64 network
  interface, 20-to-20 horizon, plain MSE and E100 protocol are radar adaptations.
- **SwinLSTM-D:** core and architecture defaults reference the author release;
  dataset, horizon, E100 protocol and sequence wrapper are radar adaptations.
- **WaST:** the backbone is OpenSTL. The selected AdamW/cosine/HFFL E100 recipe
  was reconstructed from the project's historical training code; it is not
  described as an unchanged original-author recommended configuration.
- **DD-Net:** the final architecture and recipe are project decisions.

Model-native losses may include predictions of observed context transitions;
DD-Net/WaST and the selected ConvLSTM objective use the forecast targets as
documented. Shared evaluation does not mean identical training objectives.

## Preprocessing, Selection And Evaluation

All wrappers consume the same preprocessed 64x64 arrays, split the first 20 and
last 20 frames, and normalize dBZ by 70. Input tensors are N,T,1,H,W. The
canonical preprocessing product is identified by its array SHA256; this release
does not pretend that a complete raw-radar-to-array construction pipeline or
an authoritative array-index timestamp map has been recovered.

Prediction uses eval mode and only observed inputs. Recurrent wrappers supply
zero future placeholders and inference masks; they do not consume future truth.
The shared evaluator uses `>=` thresholds at 10/20/30/40 dBZ, pooled categorical
counts, sample-frame continuous aggregation, and prediction clipping to 0-70 dBZ.
No best-test checkpoint selection occurs in these training runs. Earlier
architecture exploration did consult the test-year results and must not be
described as an untouched independent final confirmation.

PySTEPS records any persistence fallback in `pysteps_status.json`. The suite
aggregator refuses to label a run with nonzero fallbacks as pure PySTEPS output.
Report that status, rather than hiding failures in an optical-flow score.

## Citation

For PySTEPS cite Pulkkinen et al. (2019), *Pysteps: an open-source Python library
for probabilistic precipitation nowcasting (v1.0)*, GMD 12, 4185-4219,
https://doi.org/10.5194/gmd-12-4185-2019. Also cite each baseline's original paper
for the algorithm, and the actual implementation source above for the code.
