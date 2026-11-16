"""EDA360 inference model and pretrained checkpoint loader."""

import math
from pathlib import Path

import torch
from torch import nn

from .architecture import create_model


class EDA360(nn.Module):
    """Map ImageNet-normalized panoramic RGB tensors to predicted disparity."""

    def __init__(self, height, width, args, mixed_precision=False):
        super().__init__()
        self.height = height
        self.width = width
        self.mixed_precision = mixed_precision
        self.output_eps = float(getattr(args, 'output_eps', 1e-3))
        self.zipdepth_division = float(getattr(args, 'zipdepth_division', 1.0))
        if not math.isfinite(self.output_eps) or self.output_eps < 0:
            raise ValueError('Checkpoint output_eps must be finite and nonnegative')
        if not math.isfinite(self.zipdepth_division) or self.zipdepth_division <= 0:
            raise ValueError('Checkpoint zipdepth_division must be finite and positive')

        # Validate the architecture settings stored in the released checkpoint.
        expected = {
            'zipdepth_variant': 'base', 'zipdepth_output_stride': 32,
            'last_fusion_type': 'cross_attn', 'last_fusion_round': 1,
            'last_fusion_shared': False, 'last_fusion_random_round': False,
            'last_fusion_ffn': 'depthwise', 'last_fusion_ffn_padding': 'pano',
            'last_fusion_sphere_distance': True, 'last_fusion_c4_first': False,
            'in_stage4': True, 'zipdepth_npu': False,
            'zipdepth_pad_type': None, 'zipdepth_global_pad_type': None,
            'zipdepth_global_pad_size': 0, 'zipdepth_replk_experiment': None,
        }
        for name, value in expected.items():
            actual = getattr(args, name, value)
            if actual != value:
                raise ValueError(f'Checkpoint setting {name}={actual!r} does not match the final model')
        if (height, width) != (512, 1024):
            raise ValueError('The released checkpoint uses a 512 x 1024 input')

        self.model = create_model(
            variant=args.zipdepth_variant, upsample_unfold=True,
            output_stride=args.zipdepth_output_stride,
            last_fusion_type=args.last_fusion_type,
            last_fusion_round=args.last_fusion_round,
            last_fusion_shared=args.last_fusion_shared,
            last_fusion_random_round=args.last_fusion_random_round,
            last_fusion_ffn=args.last_fusion_ffn,
            last_fusion_ffn_padding=args.last_fusion_ffn_padding,
            last_fusion_sphere_distance=args.last_fusion_sphere_distance,
            last_fusion_c4_first=args.last_fusion_c4_first,
            in_stage4=args.in_stage4,
        )

    def forward(self, inputs):
        x = inputs['normalized_rgb'] if isinstance(inputs, dict) else inputs
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, self.height, self.width):
            raise ValueError(f'Expected normalized RGB of shape B x 3 x {self.height} x {self.width}')
        # Apply mixed precision to the network before disparity scaling and offset.
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            pred = self.model(x)
        if self.zipdepth_division != 1.0:
            pred = pred / self.zipdepth_division
        return {'pred_disp': pred + self.output_eps}


def load_model(checkpoint_path, device='cpu', mixed_precision=False):
    """Load model weights and architecture settings from an EDA360 checkpoint."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Model checkpoint not found: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('net') != 'ZipDepth' or 'args' not in checkpoint:
        raise ValueError('Expected the released EDA360 checkpoint with its original model metadata')
    model = EDA360(checkpoint['height'], checkpoint['width'], checkpoint['args'], mixed_precision)
    state = {key: value for key, value in checkpoint.items() if isinstance(value, torch.Tensor)}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
