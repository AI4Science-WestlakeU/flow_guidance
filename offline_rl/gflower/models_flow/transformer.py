import math
import torch
from torch import nn

# Transformer model used for (B, C, T) time series input, adapted from DiT 
# (remove patch embedding, add positional encoding)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List

import torch
from torch import nn as nn
import torch.nn.functional as F

from itertools import repeat
import collections.abc

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

#################################################################################
#                    Transformer Blocks from DiT                                #
#################################################################################

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v: (B, num_heads, N, C // num_heads)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, dtype=torch.float32):
    grid = torch.arange(grid_size, dtype=dtype)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid) # (T, D)
    return pos_embed



#################################################################################
#                   Core Transformer Layers from DiT                            #
#################################################################################

class Layer(nn.Module):
    """
    An attention layer with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU() # removed approximate="tanh" for old torch version
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimeSeriesEmbedder(nn.Module):
    """ 1D Sequence to Embedding
    """
    def __init__(
            self,
            seq_len=224,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
            proj=True, 
            conv_k=1, 
    ):
        super().__init__()
        self.seq_len = seq_len
        self.flatten = flatten
        # compared to DiT, the "patch size" here is 1
        if proj == 'conv':
            self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=conv_k, stride=1, bias=bias, padding='same')
        elif proj == 'linear':
            self.proj_module = nn.Linear(in_chans, embed_dim, bias=bias)
            self.proj = lambda x: self.proj_module(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise ValueError(f"Unknown proj type: {proj}")
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def initialize_weights(self):
        if isinstance(self.proj, nn.Conv1d):
            w = self.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.proj.bias, 0)
        else:
            nn.init.xavier_uniform_(self.proj_module.weight)
            nn.init.constant_(self.proj_module.bias, 0)
        
    def forward(self, x):
        B, C, T = x.shape
        assert T == self.seq_len, f"Input seq len ({T}) doesn't match model ({self.seq_len})."
        x = self.proj(x)
        
        if self.flatten:
            x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        return x

class Transformer(nn.Module):
    """
    Denoising model in the diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        seq_len=2, 
        in_channels=4, 
        out_channels=None, 
        hidden_size=1152,
        depth=28, 
        num_heads=16,
        mlp_ratio=4.0,
        x_emb_proj='conv', 
        x_emb_proj_conv_k=1, 
        channel_as_token=False,
    ):
        """
        Args:
            seq_len: Number of timesteps in the input sequence. N in (B, N, C)
            in_channels: Number of input channels. C in (B, N, C)
            hidden_size: Hidden dimension of the transformer.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of hidden dimension in MLP to hidden dimension in attention layer.
            x_emb_proj: Type of input embedding. 'conv' for Conv1d, 'identity' for Identity.
        """
        super().__init__()
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels if not channel_as_token else seq_len
        self.seq_len = seq_len if not channel_as_token else in_channels
        self.num_heads = num_heads
        self.channel_as_token = channel_as_token
        self.maybe_channel_as_token_transpose = lambda x: x.transpose(1, 2) if self.channel_as_token else x
        # t is denoising timestep in DDPM, (B, )
        self.t_embedder = TimestepEmbedder(hidden_size) # denoising timestep
        # x is time series input, (B, T, C)
        self.x_embedder = TimeSeriesEmbedder(
            self.seq_len, self.in_channels, hidden_size, bias=True, proj=x_emb_proj, conv_k=x_emb_proj_conv_k
        )
        # Will use fixed sin-cos position (physical time) embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            Layer(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        self.x_embedder.initialize_weights()
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, x_self_cond=None, residual=None):
        """
        Forward pass of Transformer.
        x: (B, C, T) tensor of spatial inputs (images or latent representations of images)
            C is channel, T is number of timesteps.
        t: (B,) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        """
        assert x_self_cond is None, "x_self_cond not supported."
        assert residual is None, "Physical law residual not supported."
        x = self.maybe_channel_as_token_transpose(x)
        x = self.x_embedder(x) + self.pos_embed  # (B, C, T) -> (B, T, D), where D is hidden_size
        t = self.t_embedder(t)                   # (B,) -> (B, D)
        c = t                                    # no label condition, only t as condition
        for block in self.blocks:
            x = block(x, c)                      # (B, T, D) -> (B, T, D)
        x = self.final_layer(x, c)               # (B, T, D) -> (B, T, C)
        
        x = self.maybe_channel_as_token_transpose(x)               
        return x.permute(0, 2, 1)                # (B, T, C) -> (B, C, T)

#################################################################################
#                Wrapper for Flow Matching and Diffuser                         #
#################################################################################


class TransformerFlow(nn.Module):
    """
    Transformer model as Flow matching backbone. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transformer = Transformer(*args, **kwargs)

    def forward(self, x, t, *args, **kwargs):
        b, n, c = x.shape
        x = x.transpose(1, 2) # (B, T, C) -> (B, C, T)
        while t.dim() > 1: # remove tailing dimensions
            t = t[..., 0]
        if t.dim() == 0:
            t = t.repeat(b)
        # Now, t (B,); x (B, C, T)
        x = self.transformer(x, t) # (B, C, T) -> (B, C, T)
        x = x.transpose(1, 2) # (B, C, T) -> (B, T, C)
        return x

