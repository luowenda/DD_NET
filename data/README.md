# Radar Data Contract

No dataset or download authorization is included in this code-only repository.
Provide arrays that you have permission to use. No model weights are included.

The default configuration expects NumPy arrays in **T,N,H,W** order:

| File | Shape | Physical units |
|---|---|---|
| train_data_new.npy | 40,551,64,64 | dBZ in [0,70] |
| test_data_new.npy | 40,88,64,64 | dBZ in [0,70] |

The first 20 frames are observed context; the next 20 are forecast targets.
The frame interval is 6 minutes. Arrays must be numeric, finite, and readable
with `numpy.load(..., allow_pickle=False)`. Do not pre-normalize to [0,1];
the adapter divides dBZ by 70 and converts model outputs back to dBZ.

The reference training years are 2011-2017 and 2019-2021. The test year is 2018.
Data paths are resolved relative to the repository, not the calling directory.
Only paths/device/workers may be changed within the frozen reproduction
protocol. Another dataset needs a separately declared configuration/registry,
not silent changes to the reported E100 comparison. Year-wise splitting must
be established from data provenance, not inferred from array shapes.

Optional byte-identity checksums for the reference arrays:

```text
train: ece842a05fcca781fa2136ad0f23de3c9367dba568b0bf13d7f1c69e34ce7121
test:  648ed414dbb80609c2613de2cfc3b9fd0bfefcba9e994010f191f512728bea81
```

`audit-data` checks shape, values, and finiteness. Its `--verify-reference` flag
also checks the above hashes. Neither mode can establish chronological
independence from array values alone. Data files are excluded from Git.

`split_summary.json` records the split counts and years. Detailed timestamps
and internal provenance audits are not distributed. This public summary is not
an independently verifiable sequence-level non-overlap proof. There is no
validation subset in the reported 100-epoch runs.
