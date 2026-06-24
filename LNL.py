"""
Author: Omid Nejati
Email: omid_nejaty@alumni.iust.ac.ir
LNL : Introducing locality mechanism into Transformer in Transformer (TNT)
"""
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp
from timm.models.registry import register_model
from models.localvit import LocalityFeedForward
from models.tnt import Attention, TNT
import math
import torch.hub as hub


# URL weight ImageNet cua TNT goc (chi co ban Small la cong khai)
_TNT_URLS = {
    'LNL_Ti': '',
    'LNL_S': 'https://github.com/contrastive/pytorch-image-models/releases/download/TNT/tnt_s_patch16_224.pth.tar',
}


def _load_tnt_pretrained(model, url):
    """Nap weight TNT ImageNet vao LNL (partial, SHAPE-SAFE).
    Bo qua: head ImageNet, key khong ton tai, va key trung ten nhung LECH SHAPE
    (load_state_dict strict=False van raise RuntimeError neu lech shape -> phai loc tay).
    - conv (locality FFN) la lop moi -> khong co trong checkpoint
    - outer-MLP cua TNT goc -> LNL khong dung
    """
    sd = hub.load_state_dict_from_url(url, map_location='cpu', progress=True)
    sd = sd.get('state_dict', sd)
    model_sd = model.state_dict()
    take, skip_shape = {}, 0
    for k, v in sd.items():
        if k.startswith('head.'):
            continue
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                take[k] = v
            else:
                skip_shape += 1
    msg = model.load_state_dict(take, strict=False)
    total = len(model_sd)
    print(f'[LNL] pretrained: loaded {len(take)}/{total} keys | '
          f'{len(msg.missing_keys)} new/not-loaded | {skip_shape} skipped(shape mismatch)')
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'pixel_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'tnt_t_conv_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_s_conv_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_conv_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class LayerScale(nn.Module):
    """CaiT LayerScale: he so scale hoc duoc tren moi nhanh residual.
    init=1.0 -> khoi dau dung hanh vi pretrained (an toan khi fine-tune);
    init nho (vd 1e-1) -> on dinh hon khi train from-scratch."""
    def __init__(self, dim, init_value=1.0):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    """ TNT Block (+ LayerScale tuy chon, + locality-act tuy chon) """

    def __init__(self, dim, in_dim, num_pixel, num_heads=12, in_num_head=4, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 conv_act='hs+se', ls_init=None):
        super().__init__()
        # Inner transformer
        self.norm_in = norm_layer(in_dim)
        self.attn_in = Attention(
            in_dim, in_dim, num_heads=in_num_head, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm_mlp_in = norm_layer(in_dim)
        self.mlp_in = Mlp(in_features=in_dim, hidden_features=int(in_dim * 4),
                          out_features=in_dim, act_layer=act_layer, drop=drop)

        self.norm1_proj = norm_layer(in_dim)
        self.proj = nn.Linear(in_dim * num_pixel, dim, bias=True)
        # Outer transformer
        self.norm_out = norm_layer(dim)
        self.attn_out = Attention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # conv_act: 'hs+se' (goc) | 'hs+eca' (nhe hon, thuong tot hon)
        self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, act=conv_act, reduction=dim)

        # LayerScale tren 3 nhanh residual (attn_in, mlp_in, attn_out). None -> tat (Identity).
        if ls_init is not None:
            self.ls_attn_in = LayerScale(in_dim, ls_init)
            self.ls_mlp_in = LayerScale(in_dim, ls_init)
            self.ls_attn_out = LayerScale(dim, ls_init)
        else:
            self.ls_attn_in = self.ls_mlp_in = self.ls_attn_out = nn.Identity()


    def forward(self, pixel_embed, patch_embed):
        # inner
        x, _ = self.attn_in(self.norm_in(pixel_embed))
        pixel_embed = pixel_embed + self.drop_path(self.ls_attn_in(x))
        pixel_embed = pixel_embed + self.drop_path(self.ls_mlp_in(self.mlp_in(self.norm_mlp_in(pixel_embed))))

        # outer
        B, N, C = patch_embed.size()
        Nsqrt = int(math.sqrt(N))
        patch_embed[:, 1:] = patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))
        x, weights = self.attn_out(self.norm_out(patch_embed))
        patch_embed = patch_embed + self.drop_path(self.ls_attn_out(x))

        cls_token, patch_embed = torch.split(patch_embed, [1, N - 1], dim=1)                 # (B, 1, dim), (B, 196, dim)
        patch_embed = patch_embed.transpose(1, 2).view(B, C, Nsqrt, Nsqrt)   # (B, dim, 14, 14)
        patch_embed = self.conv(patch_embed).flatten(2).transpose(1, 2)                                 # (B, 196, dim)
        patch_embed = torch.cat([cls_token, patch_embed], dim=1)

        return pixel_embed, patch_embed, weights


class LocalViT_TNT(TNT):
    """ Transformer in Transformer - https://arxiv.org/abs/2103.00112
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, in_dim=48, depth=12,
                 num_heads=12, in_num_head=4, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, first_stride=4,
                 conv_act='hs+se', ls_init=None, gap_fusion=False, normalize_input=False):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, in_dim, depth,
                 num_heads, in_num_head, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                 drop_path_rate, norm_layer, first_stride)
        new_patch_size = self.pixel_embed.new_patch_size
        num_pixel = new_patch_size ** 2

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(Block(
                dim=embed_dim, in_dim=in_dim, num_pixel=num_pixel, num_heads=num_heads, in_num_head=in_num_head,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, conv_act=conv_act, ls_init=ls_init))
        self.blocks = nn.ModuleList(blocks)

        # --- cai tien thuat toan (mac dinh TAT de backward-compatible) ---
        self.gap_fusion = gap_fusion              # CLS + global-average-pool cua patch tokens
        self.normalize_input = normalize_input    # tu chuan hoa anh tho [0,1] (plug-and-play voi transform khong-normalize)
        self.register_buffer('in_mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('in_std',  torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        self.apply(self._init_weights)

    def forward_features(self, x):
        attn_weights = []
        B = x.shape[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)
        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(
            pixel_embed.reshape(B, self.num_patches, -1))))
        patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)
        for blk in self.blocks:
            pixel_embed, patch_embed, weights = blk(pixel_embed, patch_embed)
            attn_weights.append(weights)
        patch_embed = self.norm(patch_embed)
        cls = patch_embed[:, 0]
        if self.gap_fusion:
            cls = cls + patch_embed[:, 1:].mean(dim=1)   # CLS + GAP fusion
        return cls, attn_weights

    def forward(self, x, vis=False):
        if self.normalize_input:
            x = (x - self.in_mean) / self.in_std
        x, attn_weights = self.forward_features(x)
        x = self.head(x)
        if vis:
            return x, attn_weights
        return x


@register_model
def LNL_Ti(pretrained=False, **kwargs):
    model = LocalViT_TNT(patch_size=16, embed_dim=192, in_dim=12, depth=12, num_heads=3, in_num_head=3,
                         qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_t_conv_patch16_224']
    if pretrained:
        if _TNT_URLS['LNL_Ti']:
            _load_tnt_pretrained(model, _TNT_URLS['LNL_Ti'])
        else:
            print('[LNL] khong co pretrained TNT-tiny cong khai -> train from scratch')
    return model


@register_model
def LNL_S(pretrained=False, **kwargs):
    model = LocalViT_TNT(patch_size=16, embed_dim=384, in_dim=24, depth=12, num_heads=6, in_num_head=4,
                         qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_s_conv_patch16_224']
    if pretrained:
        _load_tnt_pretrained(model, _TNT_URLS['LNL_S'])
    return model
