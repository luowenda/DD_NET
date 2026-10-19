# Third-Party Notices

## OpenSTL / WaST Backbone

- Repository: https://github.com/chengtan9907/OpenSTL
- Pinned commit: `eecf8a3078f0a178dbc7b28723da20f94ce36985`
- Reference files: `openstl/models/wast_model.py` and
  `openstl/modules/wast_modules.py`.
- License text: [Apache License 2.0](licenses/OPENSTL-APACHE-2.0.txt).

The DD-Net implementation is a modified derivative of the WaST backbone;
it is not an unchanged official WaST implementation. The final supported
configuration preserves the pinned OpenSTL wavelet grouping and shared
LayerScale behavior, replaces the input convolution with a sample-isolated
local hypergraph encoder using incidence attention-v1, and uses the DD-Net
DCN Core in the translator blocks. The training adapter normalizes physical
dBZ, and the separate HFFL implementation is device-safe.

The code-only release preserves the final experiment's numerical class
definitions. Its separate entry point selects only the final configuration.
OpenSTL-derived ConvLSTM, PredRNN, PredRNNv2 and MIM integrations are also
included at the same pinned commit under Apache-2.0. Local changes include
package imports, radar adapters and the MIM internal-state reset needed to keep
independent forwards separate. These are benchmark implementations, not a
blanket claim of unchanged original-author implementations.

## SwinLSTM-D

The author-maintained core comes from SongTang-x/SwinLSTM at commit
`4425bcfefbebfac85c9fc6c6659361b8154d5ca3`, under the MIT License.
The core is retained as `src/ddnet_revision/models/vendor/swinlstm/official.py`;
the radar sequence wrapper is separate. The original MIT notice is retained in
`licenses/SWINLSTM-MIT.txt`.

## MotionRNN And TrajGRU

No model source from the two repositories with unresolved redistribution
permissions is bundled. Fixed upstream references, raw/prepared hashes,
explicit preparation tools and the project-owned integration are documented in
`docs/EXTERNAL_BASELINES.md`. Acknowledgment of those terms does not grant rights
to external source. The Apache-2.0 project license does not relicense it.

## Installed Dependencies

PyTorch, torchvision, PyTorch Geometric, timm, PyWavelets, einops, NumPy,
SciPy, scikit-image, and other packages are installed from the pinned dependency
graph, not vendored here. Their own licenses apply to their distributions.
PySTEPS 1.21.5 is an installed BSD-3-Clause dependency, not a vendored library.
Its optical-flow/extrapolation adapter is included; the library itself is not
modified. Its BSD license text is retained in `licenses/PYSTEPS-BSD-3-CLAUSE.txt`.

The project's own contributions are licensed under Apache-2.0. That project
license and citations do not replace third-party notices or license terms.
