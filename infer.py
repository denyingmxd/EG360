#!/usr/bin/env python3
"""Predict depth from one equirectangular panorama using the pretrained EDA360 model."""

import argparse
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

from model import load_model


ROOT = Path(__file__).resolve().parent


def preprocess_image(image_path, height=512, width=1024):
    """Decode RGB, resize with cubic interpolation, and apply ImageNet normalization."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Input image not found: {path}')
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Could not decode the input image: {path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    # Convert uint8 RGB to normalized NCHW float32 input.
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous().float().div(255)
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized = tensor.sub(mean).div(std).unsqueeze(0)
    return rgb, normalized


@torch.no_grad()
def predict_image(model, image_path):
    """Return resized RGB and float32 disparity/depth at the model resolution."""
    rgb, normalized = preprocess_image(image_path, model.height, model.width)
    device = next(model.parameters()).device
    disparity = model(normalized.to(device))['pred_disp'].detach().float().cpu()
    if not torch.isfinite(disparity).all():
        raise ValueError('Model predicted non-finite disparity')
    # Convert predicted disparity to depth in metres without scale alignment.
    depth = disparity.clamp_min(1e-4).reciprocal()
    return rgb, disparity[0, 0].numpy(), depth[0, 0].numpy()


def kitti_colormap(disparity, maxval=1.0):
    """Apply the KITTI color map and return uint8 BGR values."""
    if maxval < 0:
        maxval = np.max(disparity)
    colormap = np.asarray([[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],
                           [0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1,0]])
    weights = np.asarray([8.771929824561404,5.405405405405405,8.771929824561404,
                          5.747126436781609,8.771929824561404,5.405405405405405,
                          8.771929824561404,0])
    cumsum = np.asarray([0,0.114,0.299,0.413,0.587,0.701,0.8859999999999999,0.9999999999999999])
    colored = np.zeros([disparity.shape[0], disparity.shape[1], 3])
    values = np.expand_dims(np.minimum(np.maximum(disparity/maxval, 0.), 1.), -1)
    bins = np.repeat(np.repeat(np.expand_dims(np.expand_dims(cumsum, axis=0), axis=0),
                               disparity.shape[1], axis=1), disparity.shape[0], axis=0)
    diffs = np.where(np.repeat(values, 8, axis=-1) - bins > 0,
                     -1000, np.repeat(values, 8, axis=-1) - bins)
    index = np.argmax(diffs, axis=-1) - 1
    weight = 1 - (values[:, :, 0] - cumsum[index]) * weights[index]
    for channel in range(3):
        colored[:, :, 2-channel] = (weight * colormap[index][:, :, channel]
                                     + (1-weight) * colormap[index+1][:, :, channel])
    return (colored * np.expand_dims(disparity > 0, -1) * 255).astype(np.uint8)


def save_prediction(output_dir, stem, rgb, disparity, depth, vis_max_depth=None):
    """Save numerical arrays and magma/KITTI previews without modifying predictions."""
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    scale = (float(vis_max_depth) if vis_max_depth is not None
             else float(np.percentile(depth, 95)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Visualization maximum depth must be finite and positive')
    normalized = np.clip(depth / scale, 0, 1)
    magma = matplotlib.colormaps['magma'](normalized)
    magma_bgr = (magma[..., :3][..., ::-1] * 255).astype(np.uint8)
    kitti_bgr = kitti_colormap(normalized)

    paths = [output / (stem + suffix) for suffix in [
        '_depth.npy', '_disparity.npy', '_depth.png', '_depth_kitti.png', '_preview.jpg']]
    np.save(paths[0], depth, allow_pickle=False)
    np.save(paths[1], disparity, allow_pickle=False)
    for path, image in [(paths[2], magma_bgr), (paths[3], kitti_bgr),
                        (paths[4], np.concatenate([rgb[..., ::-1], magma_bgr], axis=0))]:
        if not cv2.imwrite(str(path), image):
            raise OSError(f'Could not write output image: {path}')
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='one equirectangular panorama')
    parser.add_argument('--model_path', type=Path, default=ROOT / 'checkpoints/eda360.pth')
    parser.add_argument('--output_dir', type=Path, default=Path('outputs'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--fp32', action='store_true', help='disable CUDA mixed precision')
    parser.add_argument('--vis_max_depth', type=float, default=None,
                        help='visualization range in metres; default: predicted depth 95th percentile')
    args = parser.parse_args(argv)
    if args.vis_max_depth is not None and (
            not math.isfinite(args.vis_max_depth) or args.vis_max_depth <= 0):
        parser.error('--vis_max_depth must be finite and positive')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available; use --device cpu')
    if device.type == 'cpu':
        torch.set_num_threads(4)
    cv2.setNumThreads(0)
    try:
        model = load_model(args.model_path, device=device,
                           mixed_precision=device.type == 'cuda' and not args.fp32)
        rgb, disparity, depth = predict_image(model, args.input)
        paths = save_prediction(args.output_dir, args.input.stem, rgb, disparity, depth,
                                args.vis_max_depth)
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f'EDA360: {model.height}x{model.width}, {device}, depth range '
          f'{float(depth.min()):.4f} to {float(depth.max()):.4f} m')
    for path in paths:
        print(path.resolve())


if __name__ == '__main__':
    main()
