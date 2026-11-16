"""Aligned spatial mixers on the final ERP encoder features."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from .c4_attention import CBAMBlock, PanoFormerBlock, PanoSelfAttention

from .fusion_config import (CROSS_FUSION_TYPES, C3_FUSION_TYPES,
                                  resolve_last_fusion_type, resolve_last_fusion_ffn,
                                  validate_last_fusion_config)


class FullAttentionBlock(nn.Module):
    """Pre-LN full attention + FFN with zero-initialized residual outputs.

    Q/K/V use 256 total channels in four heads by default. The FFN is
    dim -> 384 -> dim. Every query sees every spatial key.
    Uses PyTorch's native SDPA with its default automatic backend selection.
    """

    ATTENTION_DIM = 256
    NUM_HEADS = 4
    FFN_HIDDEN = 384

    def __init__(self, dim, attention_dim=None, hidden=None):
        super().__init__()
        self.attention_dim = self.ATTENTION_DIM if attention_dim is None else attention_dim
        hidden = self.FFN_HIDDEN if hidden is None else hidden
        if (isinstance(self.attention_dim, bool) or not isinstance(self.attention_dim, int)
                or self.attention_dim < 1 or self.attention_dim % self.NUM_HEADS):
            raise ValueError('attention_dim must be a positive multiple of NUM_HEADS')
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError('hidden must be a positive integer')
        self.norm1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, self.attention_dim)
        self.kv = nn.Linear(dim, 2 * self.attention_dim)
        self.out_proj = nn.Linear(self.attention_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self._inference_cache = None
        self.reset_residual_parameters()

    def train(self, mode=True):
        self._inference_cache = None
        return super().train(mode)

    def _apply(self, fn):
        # Cached tensors are derived weights, not parameters or checkpoint state.
        self._inference_cache = None
        return super()._apply(fn)

    def _inference_weights(self):
        """Pack Q/K/V and retain autocast weights between inference calls.

        The wrapper opens a fresh autocast context for every image. Its temporary
        casts otherwise repeat every time. Keep the same rounded weights here;
        parameter versions invalidate them after normal in-place updates or loads.
        Autograd forwards always use the original parameterized layers below.
        """
        layers = (self.q, self.kv, self.out_proj, self.fc1, self.fc2)
        dtype = (torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled()
                 else self.q.weight.dtype)
        parameters = tuple(p for layer in layers for p in (layer.weight, layer.bias))
        signature = (dtype, tuple((id(p), p._version, p.device, p.dtype) for p in parameters))
        if self._inference_cache is None or self._inference_cache[0] != signature:
            weights = [(torch.cat((self.q.weight, self.kv.weight)).to(dtype=dtype),
                        torch.cat((self.q.bias, self.kv.bias)).to(dtype=dtype))]
            weights += [(layer.weight.to(dtype=dtype), layer.bias.to(dtype=dtype))
                        for layer in layers[2:]]
            self._inference_cache = (signature, weights)
        return self._inference_cache[1]

    def reset_residual_parameters(self):
        for layer in (self.out_proj, self.fc2):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _attention(self, q, k, v, weights):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    def forward(self, x):
        batch, length, _ = x.shape
        z = self.norm1(x)
        head_dim = self.attention_dim // self.NUM_HEADS
        weights = (self._inference_weights()
                   if not self.training and not torch.is_grad_enabled() else None)
        if weights is None:
            q = self.q(z).reshape(batch, length, self.NUM_HEADS, head_dim).transpose(1, 2)
            kv = self.kv(z).reshape(batch, length, 2, self.NUM_HEADS,
                                   head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            q, k, v = F.linear(z, *weights[0]).reshape(
                batch, length, 3, self.NUM_HEADS, head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        attended = self._attention(q, k, v, weights)
        attended = attended.transpose(1, 2).reshape(batch, length, self.attention_dim)
        if weights is None:
            x = x + self.out_proj(attended)
            return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        x = x + F.linear(attended, *weights[1])
        return x + F.linear(F.gelu(F.linear(self.norm2(x), *weights[2])), *weights[3])


class RegisterAttentionBlock(FullAttentionBlock):
    """The full-attention block with only its attention core replaced.

    Eight learned tokens first aggregate spatial values, then spatial queries
    read those values. As in ZipDepth's original global-token attention, the
    register keys are projected from the learned tokens, not the updated values.
    Spatial Q/K/V, Pre-LN, output projection, residuals and FFN are inherited
    unchanged. Both attention stages use native Torch SDPA with 64-wide heads.
    """

    def __init__(self, dim, attention_dim=None, hidden=None, num_tokens=8):
        super().__init__(dim, attention_dim=attention_dim, hidden=hidden)
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens < 1:
            raise ValueError('num_tokens must be a positive integer')
        self.num_tokens = num_tokens
        self.tokens = nn.Parameter(torch.empty(1, num_tokens, dim))
        nn.init.trunc_normal_(self.tokens, std=0.02)
        self.q_tokens = nn.Linear(dim, self.attention_dim, bias=False)
        self.k_proj_tokens = nn.Linear(dim, self.attention_dim, bias=False)
        self._token_inference_cache = None

    def train(self, mode=True):
        self._token_inference_cache = None
        return super().train(mode)

    def _apply(self, fn):
        self._token_inference_cache = None
        return super()._apply(fn)

    def _inference_weights(self):
        weights = super()._inference_weights()
        dtype = weights[0][0].dtype
        parameters = (self.tokens, self.q_tokens.weight, self.k_proj_tokens.weight)
        signature = (dtype, tuple((id(p), p._version, p.device, p.dtype) for p in parameters))
        if self._token_inference_cache is None or self._token_inference_cache[0] != signature:
            packed = torch.cat((self.q_tokens.weight, self.k_proj_tokens.weight)).to(dtype=dtype)
            q_tokens, k_tokens = F.linear(self.tokens.to(dtype=dtype), packed).chunk(2, dim=-1)
            shape = (1, self.num_tokens, self.NUM_HEADS, self.attention_dim // self.NUM_HEADS)
            q_tokens = q_tokens.reshape(shape).transpose(1, 2)
            k_tokens = k_tokens.reshape(shape).transpose(1, 2)
            # These projections depend only on parameters, never the input image.
            # Retain their head layout too; all three source parameters are in
            # the signature, so optimizer updates/checkpoint loads invalidate it.
            self._token_inference_cache = (signature, (q_tokens, k_tokens))
        return weights + [self._token_inference_cache[1]]

    def _register_queries_keys(self, batch, weights):
        if weights is None:
            q_tokens = self.q_tokens(self.tokens)
            k_tokens = self.k_proj_tokens(self.tokens)
            shape = (1, self.num_tokens, self.NUM_HEADS, self.attention_dim // self.NUM_HEADS)
            q_tokens = q_tokens.reshape(shape).transpose(1, 2)
            k_tokens = k_tokens.reshape(shape).transpose(1, 2)
        else:
            q_tokens, k_tokens = weights[4]
        if batch == 1:
            return q_tokens, k_tokens
        return q_tokens.expand(batch, -1, -1, -1), k_tokens.expand(batch, -1, -1, -1)

    def _attention(self, q, k, v, weights):
        q_tokens, k_tokens = self._register_queries_keys(q.shape[0], weights)
        values = F.scaled_dot_product_attention(q_tokens, k, v, dropout_p=0.0)
        return F.scaled_dot_product_attention(q, k_tokens, values, dropout_p=0.0)


def make_spatial_conv(hidden, mode, padding='pano'):
    if mode == 'pointwise':
        return None
    return nn.Conv2d(hidden, hidden, 3, groups=hidden if mode == 'depthwise' else 1,
                     padding=1 if padding == 'zero' else 0, bias=True)


def reset_spatial_conv(conv):
    if conv is not None:
        nn.init.dirac_(conv.weight, groups=conv.groups)
        nn.init.zeros_(conv.bias)


def pad_erp(x):
    """Wrap longitude and continue over each pole with a half-turn shift."""
    shift = x.shape[-1] // 2
    top = torch.roll(x[:, :, :1], shift, dims=-1)
    bottom = torch.roll(x[:, :, -1:], shift, dims=-1)
    padded = torch.cat((top, x, bottom), dim=-2)
    return torch.cat((padded[..., -1:], padded, padded[..., :1]), dim=-1)


def spatial_ffn(block, x, grid_size):
    hidden = block.fc1(block.norm2(x))
    if block.spatial_conv is not None:
        hidden = hidden.transpose(1, 2).reshape(x.shape[0], -1, *grid_size)
        hidden = block.spatial_conv(pad_erp(hidden) if block.ffn_padding == 'pano' else hidden).flatten(2).transpose(1, 2)
    return x + block.fc2(F.gelu(hidden))


class SpatialAttentionBlock(FullAttentionBlock):
    """Self/cross attention with optional spherical distance D and local FFN.

    D subtracts softplus(scale_h) * (1 - ray_q dot ray_k) from each head's
    attention logits. ERP rays use pixel centers; no key-area term or gate.
    """

    def __init__(self, dim, ffn='pointwise', sphere_distance=False, context_dim=None,
                 ffn_padding='pano'):
        super().__init__(dim)
        self.ffn = ffn
        self.ffn_padding = ffn_padding
        self.local = ffn != 'pointwise'
        self.sphere_distance = sphere_distance
        self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None
        if context_dim is not None:
            self.kv = nn.Linear(context_dim, 2 * self.attention_dim)
        self.spatial_conv = make_spatial_conv(self.fc1.out_features, ffn, ffn_padding)
        if sphere_distance:
            initial_scales = torch.tensor([0.25, 1.0, 2.0, 4.0])
            self.sphere_log_scale = nn.Parameter(initial_scales.expm1().log())
        self._geometry_cache = None
        self.reset_residual_parameters()

    def reset_residual_parameters(self):
        super().reset_residual_parameters()
        reset_spatial_conv(getattr(self, 'spatial_conv', None))

    def _apply(self, fn):
        self._geometry_cache = None
        return super()._apply(fn)

    @staticmethod
    def directions(grid_size, device, dtype):
        height, width = grid_size
        lat = (0.5 - (torch.arange(height, device=device, dtype=dtype) + 0.5) / height) * math.pi
        lon = ((torch.arange(width, device=device, dtype=dtype) + 0.5) / width * 2 - 1) * math.pi
        lat, lon = torch.meshgrid(lat, lon, indexing='ij')
        return torch.stack((lat.cos()*lon.cos(), lat.sin(), lat.cos()*lon.sin()), -1).reshape(-1,3)

    def distance_bias(self, grid_size, context_grid_size, device, dtype):
        if not self.sphere_distance:
            return None
        gdtype = torch.float64 if dtype == torch.float64 else torch.float32
        signature = (tuple(grid_size), tuple(context_grid_size), device, gdtype)
        if self._geometry_cache is None or self._geometry_cache[0] != signature:
            with torch.inference_mode(False), torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                q = self.directions(grid_size, device, gdtype)
                k = self.directions(context_grid_size, device, gdtype)
                distance = (1 - q @ k.T).clamp(0,2)
            self._geometry_cache = (signature, distance)
        # Cache only static geometry. Rebuild the learned bias for every forward.
        scale = F.softplus(self.sphere_log_scale).to(gdtype)
        return (-scale[:,None,None] * self._geometry_cache[1]).to(dtype)[None]

    pad_erp = staticmethod(pad_erp)

    def forward(self, x, grid_size, context=None, context_grid_size=None):
        batch, length, _ = x.shape
        if length != math.prod(grid_size):
            raise ValueError('Query grid does not match its tokens')
        z = self.norm1(x)
        if self.norm_context is None:
            source = z
            context_grid_size = grid_size
        else:
            if context is None or context_grid_size is None:
                raise ValueError('Cross-attention requires C4 context and its grid')
            if context.shape[0] != batch or context.shape[1] != math.prod(context_grid_size):
                raise ValueError('Context grid or batch does not match its tokens')
            source = self.norm_context(context)
        head_dim = self.attention_dim // self.NUM_HEADS
        q = self.q(z).reshape(batch, length, self.NUM_HEADS, head_dim).transpose(1, 2)
        k, v = self.kv(source).reshape(batch, source.shape[1], 2, self.NUM_HEADS,
                                      head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.distance_bias(grid_size, context_grid_size, q.device, q.dtype)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, length, self.attention_dim)
        residual = self.out_proj(attended)
        return spatial_ffn(self, x + residual, grid_size)


class PanoFormerV2Block(nn.Module):
    """E08/full with only the attention branch replaced by cross-scale PSA.

    C3 predicts the offsets/weights; normalized C4 supplies sampled values.
    Keep E08's Pre-LN, distance scales, residuals, FFN384 and ERP padding.
    In particular this does not use the original PanoFormerBlock's LeFF.
    """

    def __init__(self, dim, context_dim, ffn='depthwise', sphere_distance=False,
                 ffn_padding='pano'):
        super().__init__()
        self.ffn = ffn
        self.ffn_padding = ffn_padding
        self.local = ffn != 'pointwise'
        self.sphere_distance = sphere_distance
        self.norm1 = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)
        self.dattn = PanoSelfAttention(dim, context_dim=context_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, FullAttentionBlock.FFN_HIDDEN)
        self.spatial_conv = make_spatial_conv(self.fc1.out_features, ffn, ffn_padding)
        self.fc2 = nn.Linear(self.fc1.out_features, dim)
        if sphere_distance:
            initial_scales = torch.tensor([0.25, 1.0, 2.0, 4.0])
            self.sphere_log_scale = nn.Parameter(initial_scales.expm1().log())
        self.reset_residual_parameters()

    def reset_residual_parameters(self):
        self.dattn.reset_residual_parameters()
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        reset_spatial_conv(self.spatial_conv)

    def sample_distance_bias(self, grid, grid_size):
        # With align_corners=False, normalized grid coordinates map to ERP
        # pixel-center angles lon=pi*x, lat=-pi*y/2 at ANY context resolution.
        # Evaluate the same E08 D=-softplus(scale)*(1-ray_q.dot(ray_k)) at the
        # learned sample positions; do not drop D when replacing dense attention.
        lon, lat = grid[..., 0] * math.pi, grid[..., 1] * (-math.pi / 2)
        keys = torch.stack((lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()), -1)
        queries = SpatialAttentionBlock.directions(grid_size, grid.device, grid.dtype)
        queries = queries.reshape(1, *grid_size, 1, 1, 3)
        distance = (1 - (queries * keys).sum(-1)).clamp(0, 2)
        scale = F.softplus(self.sphere_log_scale).to(grid.dtype)
        return -scale[None, None, None, :, None] * distance

    def forward(self, x, grid_size, context=None, context_grid_size=None):
        if context is None or context_grid_size is None:
            raise ValueError('PanoFormerV2 cross-attention requires C4 context and its grid')
        residual = self.dattn(self.norm1(x), grid_size,
                              self.norm_context(context), context_grid_size,
                              self.sample_distance_bias if self.sphere_distance else None)
        return spatial_ffn(self, x + residual, grid_size)


class LocalFFNBlock(nn.Module):
    """FFN-only control with the identical FFN used by SpatialAttentionBlock."""

    def __init__(self, dim, hidden=384, ffn='depthwise', ffn_padding='pano'):
        super().__init__()
        self.ffn = ffn
        self.ffn_padding = ffn_padding
        self.local = ffn != 'pointwise'
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.spatial_conv = make_spatial_conv(hidden, ffn, ffn_padding)
        self.fc2 = nn.Linear(hidden, dim)
        self.reset_residual_parameters()

    def reset_residual_parameters(self):
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        reset_spatial_conv(self.spatial_conv)

    def forward(self, x, grid_size, context=None, context_grid_size=None):
        if x.shape[1] != math.prod(grid_size) or context is not None:
            raise ValueError('Local FFN requires its own spatial grid without context')
        return spatial_ffn(self, x, grid_size)


class CBAMV2Block(LocalFFNBlock):
    """Native C3 CBAM inside E08's Pre-LN/residual + spatial FFN scaffold.

    CBAM gates a single C3 feature map, so there is no C4 input or distance D.
    Its native reduction16 / zero-padded spatial7x7+BN remain unchanged; ERP
    padding belongs to the inherited FFN. A zero-initialized output projection
    preserves E08's identity initialization for the attention residual as well.
    """

    def __init__(self, dim, ffn='depthwise', ffn_padding='pano'):
        super().__init__(dim, hidden=FullAttentionBlock.FFN_HIDDEN,
                         ffn=ffn, ffn_padding=ffn_padding)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CBAMBlock(dim)
        self.out_proj = nn.Linear(dim, dim)
        self.reset_residual_parameters()

    def reset_residual_parameters(self):
        super().reset_residual_parameters()
        # LocalFFNBlock also calls this before the attention branch is created.
        if hasattr(self, 'out_proj'):
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, grid_size, context=None, context_grid_size=None):
        if context is not None or context_grid_size is not None:
            raise ValueError('CBAMV2 requires only C3 features, without C4 context')
        residual = self.out_proj(self.attn(self.norm1(x), grid_size))
        return super().forward(x + residual, grid_size)


class LastFusion(nn.Module):
    """Repeated spatial mixing on C3 or C4 at the selected encoder position.

    By default each round owns a block. Shared rounds reuse one entire block,
    including normalization, FFN and any learned register tokens.
    Optional training samples one positive prefix for the whole batch; eval
    always uses the configured maximum unless explicitly overridden.
    """

    def __init__(self, dim, fusion_type, rounds=1, shared=False, random_round=False,
                 context_dim=None, ffn=None, sphere_distance=False, c4_first=False, ffn_padding='pano'):
        super().__init__()
        validate_last_fusion_config(fusion_type, rounds, shared, False, random_round,
                                   ffn, sphere_distance, c4_first, ffn_padding)
        kind, _ = resolve_last_fusion_type(fusion_type)
        self.fusion_type = kind
        self.rounds = rounds
        self.shared = shared
        self.random_round = random_round
        self.ffn = resolve_last_fusion_ffn(kind, ffn)
        self.ffn_padding = ffn_padding
        self.sphere_distance = sphere_distance
        self.requires_context = kind in CROSS_FUSION_TYPES
        self.updates_c3 = kind in C3_FUSION_TYPES
        if self.requires_context and context_dim is None:
            raise ValueError('{} requires context_dim'.format(kind))
        self._batch_rounds = None
        self.last_rounds = rounds
        # The C4 prefix executes once, before any of the C3 cross rounds, and
        # its updated C4 also continues through the original neck and decoder.
        self.c4_prefix = FullAttentionBlock(context_dim) if c4_first else None
        def make_block():
            if kind == 'cbam':
                return CBAMBlock(dim)
            if kind == 'cbamv2':
                return CBAMV2Block(dim, ffn=self.ffn, ffn_padding=ffn_padding)
            if kind == 'panoformer':
                return PanoFormerBlock(dim)
            if kind == 'panoformerv2':
                return PanoFormerV2Block(dim, context_dim, ffn=self.ffn,
                                         sphere_distance=sphere_distance, ffn_padding=ffn_padding)
            if kind in ('trans', 'full_attn_c3') and self.ffn == 'pointwise' and not sphere_distance:
                return FullAttentionBlock(dim)
            if kind == 'reg_attn':
                return RegisterAttentionBlock(dim)
            if kind in ('local_ffn', 'ffn_c3'):
                return LocalFFNBlock(dim, ffn=self.ffn, ffn_padding=ffn_padding)
            return SpatialAttentionBlock(dim, ffn=self.ffn, sphere_distance=sphere_distance,
                                         ffn_padding=ffn_padding,
                                         context_dim=context_dim if self.requires_context else None)
        self.blocks = nn.ModuleList(make_block() for _ in range(1 if shared else rounds))

    def reset_residual_parameters(self):
        if self.c4_prefix is not None:
            self.c4_prefix.reset_residual_parameters()
        for block in self.blocks:
            block.reset_residual_parameters()

    def train(self, mode=True):
        self.end_batch()
        return super().train(mode)

    def _sample_rounds(self, device):
        if not (self.training and self.random_round and self.rounds > 1):
            return self.rounds
        rounds = int(torch.randint(1, self.rounds + 1, (), device='cpu').item())
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # One depth for the global batch, including ranks with different RNG states.
            value = torch.tensor(rounds, dtype=torch.long, device=device)
            torch.distributed.broadcast(value, src=0)
            rounds = int(value.item())
        return rounds

    def begin_batch(self, device):
        """Sample before DataParallel replication; reuse for all batch forwards."""
        self._batch_rounds = self._sample_rounds(device)
        self.last_rounds = self._batch_rounds
        return self._batch_rounds

    def end_batch(self):
        self._batch_rounds = None

    def forward(self, x, context=None, return_context=False):
        rounds = (self._batch_rounds if self.training and self._batch_rounds is not None
                  else self._sample_rounds(x.device))
        self.last_rounds = rounds
        if self.c4_prefix is not None and rounds > 0:
            if context is None:
                raise ValueError('C4 prefix requires context')
            cb, cc, ch, cw = context.shape
            context = self.c4_prefix(context.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(cb, cc, ch, cw)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        context_tokens = None
        context_grid_size = None
        if self.requires_context:
            if context is None:
                raise ValueError('Cross-attention requires C4 context')
            context_tokens = context.flatten(2).transpose(1, 2)
            context_grid_size = context.shape[-2:]
        for index in range(rounds):
            block = self.blocks[0 if self.shared else index]
            if isinstance(block, (CBAMBlock, PanoFormerBlock)):
                tokens = block(tokens, (height, width))
            elif isinstance(block, (SpatialAttentionBlock, PanoFormerV2Block, LocalFFNBlock)):
                tokens = block(tokens, (height, width), context_tokens, context_grid_size)
            else:
                tokens = block(tokens)
        output = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return (output, context) if return_context else output
