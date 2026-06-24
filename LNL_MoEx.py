"""
Author: Omid Nejati
Email: omid_nejaty@alumni.iust.ac.ir

Introducing locality mechanism into Transformer in Transformer (TNT)

"""
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp
from timm.models.registry import register_model
from models.localvit import LocalityFeedForward
from models.tnt_moex import Attention, TNT, moex
import math


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
    """CaiT LayerScale: he so scale hoc duoc tren moi nhanh residual."""
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

        self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, act=conv_act, reduction=dim)

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
        self.gap_fusion = gap_fusion
        self.normalize_input = normalize_input
        self.register_buffer('in_mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('in_std',  torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        self.apply(self._init_weights)

    def forward_features(self, x, swap_index, moex_norm, moex_epsilon,
                         moex_layer, moex_positive_only):
        attn_weights = []
        B = x.shape[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)
        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(
            pixel_embed.reshape(B, self.num_patches, -1))))
        patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)

        # moex injection o stem (giu nguyen co che goc)
        if swap_index is not None and moex_layer == 'stem':
            patch_embed = moex(patch_embed, swap_index, moex_norm, moex_epsilon, moex_positive_only)

        for blk in self.blocks:
            pixel_embed, patch_embed, weights = blk(pixel_embed, patch_embed)
            attn_weights.append(weights)
        patch_embed = self.norm(patch_embed)
        cls = patch_embed[:, 0]
        if self.gap_fusion:
            cls = cls + patch_embed[:, 1:].mean(dim=1)   # CLS + GAP fusion
        return cls, attn_weights

    def forward(self, x, swap_index=None, moex_norm='pono', moex_epsilon=1e-5,
                moex_layer='stem', moex_positive_only=False, vis=False):
        if self.normalize_input:
            x = (x - self.in_mean) / self.in_std
        x, attn_weights = self.forward_features(x, swap_index, moex_norm, moex_epsilon,
                                                moex_layer, moex_positive_only)
        x = self.head(x)
        if vis:
            return x, attn_weights
        return x


@register_model
def LNL_MoEx_Ti(pretrained=False, **kwargs):
    model = LocalViT_TNT(patch_size=16, embed_dim=192, in_dim=12, depth=12, num_heads=3, in_num_head=3,
                         qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_t_conv_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def LNL_MoEx_S(pretrained=False, **kwargs):
    model = LocalViT_TNT(patch_size=16, embed_dim=384, in_dim=24, depth=12, num_heads=6, in_num_head=4,
                         qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_s_conv_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
