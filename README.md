# EG360: Single-Panorama Depth Inference

Standalone inference code and pretrained checkpoint for:

**[EG360: Efficient and Generalizable 360° Depth Estimation via Adaptive Panorama-Perspective Distillation and Spherical Cross-Level Feature Fusion](https://arxiv.org/abs/2611.16793)**

Laiyan Ding, Chongxi Meng, Rui Huang

[arXiv](https://arxiv.org/abs/2611.16793) |
[HTML](https://arxiv.org/html/2611.16793v1) |
[PDF](https://arxiv.org/pdf/2611.16793)

A standalone inference package for estimating depth from a single 360-degree
equirectangular RGB panorama. The model implementation, inference script, and
pretrained checkpoint are included in this folder.

## Files

```text
infer.py                 Single-image inference and visualization
model/                   Model implementation and third-party license notices
checkpoints/eda360.pth    Pretrained checkpoint
requirements.txt         Python dependencies
```

## Installation

Use Python 3.9 or newer. From this folder, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU inference, use a CUDA-enabled PyTorch build compatible with your system.
All model source files are included; no additional source repository or
pretrained backbone download is required.

## Run inference

Provide the path to an RGB panorama:

```bash
python infer.py --input path/to/panorama.jpg --output_dir outputs
```

The script uses CUDA when available and falls back to CPU otherwise. To select
a device explicitly:

```bash
python infer.py --input path/to/panorama.jpg --output_dir outputs --device cpu
python infer.py --input path/to/panorama.jpg --output_dir outputs --device cuda
```

CUDA inference uses FP16 mixed precision by default. Add `--fp32` to use FP32.
The default checkpoint is `checkpoints/eda360.pth`, resolved relative to
`infer.py`. Use `--model_path path/to/eda360.pth` to select another compatible
checkpoint. Run `python infer.py --help` to list the available options.

## Input and outputs

Input should be a full equirectangular panorama, normally with a 2:1 width-to-height
ratio. The image is decoded as RGB, resized to 512 x 1024 with cubic interpolation,
scaled to [0, 1], and normalized with ImageNet mean and standard deviation.

For an input named `panorama.jpg`, the script saves:

| File | Content |
| --- | --- |
| `panorama_depth.npy` | Float32 depth in metres, shape 512 x 1024 |
| `panorama_disparity.npy` | Float32 predicted disparity, shape 512 x 1024 |
| `panorama_depth.png` | Depth visualization using the magma color map |
| `panorama_depth_kitti.png` | Depth visualization using the KITTI color map |
| `panorama_preview.jpg` | Resized RGB above the magma depth visualization |

Depth is computed in float32 as `1 / max(predicted_disparity, 1e-4)`.
All predictions are saved at 512 x 1024 resolution.

The visualization range defaults to the predicted depth's 95th percentile.
Use `--vis_max_depth 10`, for example, to set the visualization maximum to
10 metres. This option affects the colors, not the saved numerical predictions.

## Citation

If you use this code or model in your research, please cite:

```bibtex
@article{ding2026eg360,
  title={{EG360}: Efficient and Generalizable $360^\circ$ Depth Estimation via Adaptive Panorama-Perspective Distillation and Spherical Cross-Level Feature Fusion},
  author={Ding, Laiyan and Meng, Chongxi and Huang, Rui},
  journal={arXiv preprint arXiv:2611.16793},
  year={2026},
  url={https://arxiv.org/abs/2611.16793}
}
```

## License

Project-specific code is provided under Apache-2.0; see [LICENSE](LICENSE).
Third-party code retains its respective licenses and copyright notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
