# --------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type, Tuple


# ----------------------- Basic modules -----------------------
class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 mlp_dim: int,
                 act: Type[nn.Module] = nn.GELU,
                 dropout: float = 0.0,
                 ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, mlp_dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, embedding_dim)
        self.drop2 = nn.Dropout(dropout)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class ViTBlock(nn.Module):
    def __init__(
            self,
            dim                :int,
            qkv_bias           :bool  = False,
            num_heads          :int   = 8,
            num_patches        :int   = None,
            mlp_ratio          :float = 4.0,
            prefix_causal_mask :bool  = False,
            dropout            :float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, qkv_bias, num_heads, num_patches, prefix_causal_mask, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLPBlock(dim, int(dim * mlp_ratio), nn.GELU, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))

        return x

class Attention(nn.Module):
    def __init__(self,
                 dim                :int,
                 qkv_bias           :bool  = False,
                 num_heads          :int   = 8,
                 num_patches        :int   = None,
                 prefix_causal_mask :bool  = False,
                 dropout            :float = 0.):
        super().__init__()
        # --------------- Basic parameters ---------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.prefix_causal_mask = prefix_causal_mask

        # --------------- Network parameters ---------------
        self.qkv_proj = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.register_buffer(
            "attn_mask",
            torch.ones(1, num_patches, num_patches, dtype=torch.bool).tril(diagonal=0),
        )


    def forward(self, x, prefix_mask=None):
        B, N, C = x.shape
        # ----------------- Prefix mask -----------------
        if self.prefix_causal_mask:
            assert prefix_mask is not None, "A mask is required for the PrefixLM Causal Attention."
            prefix_mask = prefix_mask.unsqueeze(1).expand(-1, N, -1).bool()
            attn_mask = self.attn_mask.clone().expand(B, -1, -1)
            attn_mask = torch.logical_or(attn_mask, prefix_mask)
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, N)
        else:
            attn_mask = None

        # ----------------- Input proj -----------------
        qkv = (
            self.qkv_proj(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        # ----------------- Multi-head Attn -----------------
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, C)

        # ----------------- Output -----------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class PatchEmbed(nn.Module):
    def __init__(self,
                 in_chans    : int = 3,
                 embed_dim   : int = 768,
                 kernel_size : int = 16,
                 padding     : int = 0,
                 stride      : int = 16,
                 ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class AttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        in_dim      : int,
        out_dim     : int,
        num_heads   : int = 12,
        qkv_bias    : bool = False,
        num_queries : int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = in_dim // num_heads
        self.scale = head_dim**-0.5

        self.k = nn.Linear(in_dim, in_dim, bias=qkv_bias)
        self.v = nn.Linear(in_dim, in_dim, bias=qkv_bias)

        self.cls_token = nn.Parameter(torch.randn(1, num_queries, in_dim) * 0.02)
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(in_dim, affine=False, eps=1e-6)

        self.num_queries = num_queries

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape

        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1)

        out = self.linear(x_cls)

        return out, x_cls


# ---------------------- Model functions ----------------------
def window_partition(x: torch.Tensor,
                     window_size: int,
                     ) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows, (Hp, Wp)

def window_unpartition(windows: torch.Tensor,
                       window_size: int,
                       pad_hw: Tuple[int, int],
                       hw: Tuple[int, int],
                       ) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
        
    return x

def get_rel_pos(q_size: int,
                k_size: int,
                rel_pos: torch.Tensor,
                )-> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(attn : torch.Tensor,
                           q    : torch.Tensor,
                           rel_pos_h : torch.Tensor,
                           rel_pos_w : torch.Tensor,
                           q_size    : Tuple[int, int],
                           k_size    : Tuple[int, int],
                           ) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn
