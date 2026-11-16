"""C4 attention controls adapted from the official CBAM and PanoFormer code.

Sources and MIT notices: c4_attention_LICENSES.md. No upstream model or optional
visualization/CUDA extension dependencies are imported. Geometry follows the
released PanoFormer sampler, including its align_corners=False convention.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CBAMBlock(nn.Module):
    """Native channel-then-spatial multiplicative CBAM, without a residual/FFN.

    Match upstream defaults: avg+max pooled shared MLP (reduction 16), then
    max/mean channel pooling, a zero-padded 7x7 conv and BatchNorm (momentum .01).
    This gate intentionally does NOT start as the identity.
    """

    def __init__(self, dim):
        super().__init__()
        self.channel_mlp = nn.Sequential(nn.Linear(dim, max(1, dim // 16)), nn.ReLU(),
                                         nn.Linear(max(1, dim // 16), dim))
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.spatial_bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01)

    def reset_residual_parameters(self):
        # Native CBAM has no additive residual output to zero-initialize.
        pass

    def forward(self, x, grid_size):
        batch, length, channels = x.shape
        if length != math.prod(grid_size):
            raise ValueError('CBAM grid does not match its tokens')
        feat = x.transpose(1, 2).reshape(batch, channels, *grid_size)
        avg = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(feat, 1).flatten(1)
        channel_gate = (self.channel_mlp(avg) + self.channel_mlp(maximum)).sigmoid()
        feat = feat * channel_gate[:, :, None, None]
        spatial = torch.cat((feat.max(dim=1, keepdim=True).values,
                             feat.mean(dim=1, keepdim=True)), dim=1)
        feat = feat * self.spatial_bn(self.spatial_conv(spatial)).sigmoid()
        return feat.flatten(2).transpose(1, 2)


def panoformer_reference_grid(height, width):
    """Official 3x3 tangent-plane reference points, returned as [1,H,W,9,(y,x)]."""
    if height < 2 or width != 2 * height:
        raise ValueError('PanoFormer requires a 2:1 ERP feature grid with height >= 2')
    axis = np.arange(-1, 2, dtype=np.float64)
    dx, dy = 2 * np.pi / width, np.pi / height
    kx, ky = np.meshgrid(np.tan(axis * dx), np.tan(axis * dy) / np.cos(axis * dx))
    rho = np.sqrt(kx ** 2 + ky ** 2)
    rho[1, 1] = 1e-8
    nu = np.arctan(rho)
    lat = ((np.arange(height) / height) - .5) * np.pi
    lon = ((np.arange(width) / width) - .5) * (2 * np.pi)
    lat = lat[:, None, None]
    # Clip only floating-point roundoff at the arcsin endpoints.
    latitude = np.arcsin(np.clip(np.cos(nu) * np.sin(lat) +
                                ky * np.sin(nu) * np.cos(lat) / rho, -1, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        longitude = np.arctan(kx * np.sin(nu) /
                             (rho * np.cos(lat) * np.cos(nu) - ky * np.sin(lat) * np.sin(nu)))
    latitude = np.broadcast_to(latitude[:, None], (height, width, 3, 3))
    longitude = longitude[:, None] + lon[None, :, None, None]
    y = (latitude / np.pi + .5) * height
    x = ((longitude / (2 * np.pi) + .5) * width) % width
    grid = np.stack((y, x), axis=-1).reshape(1, height, width, 9, 2).astype(np.float32)
    if not np.isfinite(grid).all():
        raise ValueError('Non-finite PanoFormer reference grid')
    return torch.from_numpy(grid)


class PanoSelfAttention(nn.Module):
    """Single-scale official PSA: learned offsets/weights on nine sphere samples.

    q_proj predicts offsets and softmax weights; k_proj produces sampled values.
    This is not dot-product attention. The 256-channel projection uses four heads
    of 64, while the residual stream and LeFF retain the input channel width.
    """

    NUM_HEADS = 4
    NUM_POINTS = 9
    ATTENTION_DIM = 256

    def __init__(self, dim, context_dim=None):
        super().__init__()
        self.dim = dim
        self.attention_dim = self.ATTENTION_DIM
        self.head_dim = self.attention_dim // self.NUM_HEADS
        self.q_proj = nn.Linear(dim, self.attention_dim)
        self.k_proj = nn.Linear(dim if context_dim is None else context_dim, self.attention_dim)
        self.offset_proj = nn.Linear(self.attention_dim, 2 * self.NUM_HEADS * self.NUM_POINTS)
        self.A_proj = nn.Linear(self.attention_dim, self.NUM_HEADS * self.NUM_POINTS)
        self.wm_proj = nn.Linear(self.attention_dim, dim)
        self._reference_cache = None

    def _apply(self, fn):
        self._reference_cache = None
        return super()._apply(fn)

    def reference_grid(self, grid_size, device, dtype):
        signature = (tuple(grid_size), device, dtype)
        if self._reference_cache is None or self._reference_cache[0] != signature:
            # A grid first built under inference_mode must also work in backward.
            with torch.inference_mode(False), torch.no_grad():
                grid = panoformer_reference_grid(*grid_size).to(device=device, dtype=dtype)
            self._reference_cache = (signature, grid)
        return self._reference_cache[1]

    def reset_residual_parameters(self):
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)
        nn.init.zeros_(self.A_proj.weight)
        nn.init.constant_(self.A_proj.bias, 1 / self.NUM_POINTS)
        nn.init.zeros_(self.wm_proj.weight)
        nn.init.zeros_(self.wm_proj.bias)

    def forward(self, x, grid_size, context=None, context_grid_size=None, sample_bias=None):
        """Optionally sample C4 values at C3 query locations for panoformerv2.

        Reference points and offsets stay in query-grid pixels. Their normalized
        coordinates also address a lower-resolution ERP context. sample_bias, if
        supplied, maps normalized (x, y) coordinates to per-sample logit biases.
        The original self-attention path and sampling convention are unchanged.
        """
        batch, length, channels = x.shape
        height, width = grid_size
        if length != height * width:
            raise ValueError('PanoFormer grid does not match its tokens')
        if context is None:
            source, source_grid = x, grid_size
        else:
            if context_grid_size is None:
                raise ValueError('PanoFormer cross-attention requires a context grid')
            source, source_grid = context, context_grid_size
            if source.shape[0] != batch or source.shape[1] != math.prod(source_grid):
                raise ValueError('Context grid or batch does not match its tokens')
            if source_grid[0] < 2 or source_grid[1] != 2 * source_grid[0]:
                raise ValueError('PanoFormer requires a 2:1 ERP context grid with height >= 2')
        heads, points = self.NUM_HEADS, self.NUM_POINTS
        query = self.q_proj(x)
        offsets = self.offset_proj(query).reshape(batch, height, width, heads, points, 2)
        logits = self.A_proj(query).reshape(batch, length, heads, points)
        weights = logits.softmax(-1)
        values = self.k_proj(source).reshape(batch, *source_grid, heads, self.head_dim)
        values = values.permute(0, 3, 4, 1, 2).reshape(batch * heads, self.head_dim, *source_grid)
        # grid_sample on CUDA under AMP is FP32; keep its coordinates FP32 too.
        sample_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            ref = self.reference_grid(grid_size, x.device, sample_dtype)
            sample = ref[:, :, :, None] + offsets.to(sample_dtype)
            gx = 2 * sample[..., 1] / (width - 1) - 1
            gy = 2 * sample[..., 0] / (height - 1) - 1
            grid = torch.stack((gx, gy), dim=-1)
            if sample_bias is not None:
                bias = sample_bias(grid, grid_size).reshape(batch, length, heads, points)
                weights = (logits.to(sample_dtype) + bias).softmax(-1)
            grid = grid.permute(0, 3, 1, 2, 4, 5)
            grid = grid.reshape(batch * heads, height, width * points, 2)
            sampled = F.grid_sample(values.to(sample_dtype), grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=False)
            sampled = sampled.reshape(batch, heads, self.head_dim, length, points)
            weights = weights.permute(0, 2, 1, 3).to(sample_dtype)
            mixed = (sampled * weights[:, :, None]).sum(-1)
            mixed = mixed.permute(0, 3, 1, 2).reshape(batch, length, self.attention_dim)
        return self.wm_proj(mixed.to(query.dtype))


class PanoFormerBlock(nn.Module):
    """Pre-LN PSA + LeFF (expansion=1, two GELUs, circular W / zero H pad).

    Adaptation: residual output projections start at zero, as in our trans
    control. No stochastic depth, shifted windows, cross-scale input or D.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.dattn = PanoSelfAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, groups=dim)
        self.linear2 = nn.Linear(dim, dim)
        self.reset_residual_parameters()

    def reset_residual_parameters(self):
        self.dattn.reset_residual_parameters()
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x, grid_size):
        height, width = grid_size
        x = x + self.dattn(self.norm1(x), grid_size)
        hidden = F.gelu(self.linear1(self.norm2(x)))
        hidden = hidden.transpose(1, 2).reshape(x.shape[0], -1, height, width)
        hidden = F.pad(F.pad(hidden, (1, 1, 0, 0), mode='circular'), (0, 0, 1, 1))
        hidden = F.gelu(self.dwconv(hidden)).flatten(2).transpose(1, 2)
        return x + self.linear2(hidden)
