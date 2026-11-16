"""Lightweight last-fusion configuration, also usable before model imports."""

import argparse


FUSION_TYPES = ('trans', 'cbam', 'cbamv2', 'panoformer', 'panoformerv2', 'full_attn_c3', 'reg_attn', 'full_attn_local', 'cross_attn',
                'cross_attn_local', 'local_ffn', 'ffn_c3')
CROSS_FUSION_TYPES = ('cross_attn', 'cross_attn_local', 'panoformerv2')
C3_FUSION_TYPES = CROSS_FUSION_TYPES + ('ffn_c3', 'full_attn_c3', 'cbamv2')
FFN_TYPES = ('pointwise', 'depthwise', 'conv')
FFN_PADDINGS = ('zero', 'pano')


def resolve_last_fusion_type(value):
    """Resolve the configured attention mixer."""
    if value in FUSION_TYPES:
        return value, None
    raise ValueError('last_fusion_type must be one of {}'.format(', '.join(FUSION_TYPES)))


def last_fusion_type_arg(value):
    """Validate CLI values while retaining their explicit spelling in args."""
    try:
        resolve_last_fusion_type(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def _boolean_arg(value, name):
    """Parse explicit booleans without treating the string 'false' as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
    raise argparse.ArgumentTypeError('{} must be true or false'.format(name))


def last_fusion_shared_arg(value):
    return _boolean_arg(value, 'last_fusion_shared')


def last_fusion_random_round_arg(value):
    return _boolean_arg(value, 'last_fusion_random_round')


def last_fusion_sphere_distance_arg(value):
    return _boolean_arg(value, 'last_fusion_sphere_distance')


def last_fusion_c4_first_arg(value):
    return _boolean_arg(value, 'last_fusion_c4_first')


def resolve_last_fusion_ffn(fusion_type, ffn=None):
    if fusion_type in ('cbam', 'panoformer'):
        if ffn is not None:
            raise ValueError('{} has a fixed native block; omit last_fusion_ffn'.format(fusion_type))
        return 'none' if fusion_type == 'cbam' else 'leff'
    if ffn is None:
        ffn = ('depthwise' if fusion_type in ('full_attn_local', 'cross_attn_local', 'local_ffn', 'panoformerv2', 'cbamv2')
               else 'pointwise')
    if ffn not in FFN_TYPES:
        raise ValueError('last_fusion_ffn must be one of {}'.format(', '.join(FFN_TYPES)))
    return ffn


def in_stage4_arg(value):
    return _boolean_arg(value, 'in_stage4')


def validate_last_fusion_placement(fusion_type, in_stage4):
    """Validate the optional Stage4 placement flag."""
    if not isinstance(in_stage4, bool):
        raise ValueError('in_stage4 must be a boolean')


def validate_last_fusion_config(fusion_type, rounds=1, shared=False, in_stage4=False,
                               random_round=False, ffn=None, sphere_distance=False, c4_first=False, ffn_padding='pano'):
    validate_last_fusion_placement(fusion_type, in_stage4)
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError('last_fusion_round must be a positive integer')
    for name, value in (('last_fusion_shared', shared), ('last_fusion_random_round', random_round),
                        ('last_fusion_sphere_distance', sphere_distance), ('last_fusion_c4_first', c4_first)):
        if not isinstance(value, bool):
            raise ValueError('{} must be a boolean'.format(name))
    if fusion_type is not None:
        resolve_last_fusion_type(fusion_type)
    mode = resolve_last_fusion_ffn(fusion_type, ffn)
    if ffn_padding not in FFN_PADDINGS:
        raise ValueError('last_fusion_ffn_padding must be zero or pano')
    if fusion_type is None and (mode != 'pointwise' or sphere_distance or c4_first):
        raise ValueError('Fusion options require last_fusion_type')
    if fusion_type == 'reg_attn' and (mode != 'pointwise' or sphere_distance or c4_first):
        raise ValueError('reg_attn retains its original pointwise FFN without distance bias or C4 prefix')
    if fusion_type in ('cbam', 'panoformer') and (sphere_distance or c4_first or ffn_padding != 'pano'):
        raise ValueError('{} uses its native padding and no D/C4 prefix; omit geometry/FFN overrides'.format(fusion_type))
    if fusion_type == 'cbamv2' and (sphere_distance or c4_first):
        raise ValueError('cbamv2 gates C3 without C4 context or pairwise distance logits; D/C4 prefix must be false')
    if sphere_distance and fusion_type in ('ffn_c3', 'local_ffn'):
        raise ValueError('D requires an attention branch')
    if c4_first and fusion_type not in CROSS_FUSION_TYPES:
        raise ValueError('last_fusion_c4_first requires cross attention')
