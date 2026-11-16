# Third-party notices

EDA360-specific code is provided under Apache-2.0; see [LICENSE](LICENSE).
The following third-party portions retain their upstream licenses.

## ZipDepth

`model/architecture.py` adapts the ZipDepth architecture for panoramic depth
inference, including panoramic padding and cross-scale feature fusion.

- Source: https://github.com/fabiotosi92/ZipDepth
- License: MIT
- Copyright (c) 2026 Fabio Tosi
- Full license: [model/zipdepth_LICENSE.txt](model/zipdepth_LICENSE.txt)

## CBAM and PanoFormer

`model/c4_attention.py` contains attention blocks adapted from CBAM and PanoFormer
for panoramic feature grids. They are used by `model/last_fusion.py`.

The upstream references, copyright notices, and complete MIT licenses are
included in [model/c4_attention_LICENSES.md](model/c4_attention_LICENSES.md).
