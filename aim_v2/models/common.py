import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ----------------------- Common modules -----------------------
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def drop_path(self, x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
            Input:
                x: torch.Tensor -> [bs, seq_len, c]
        """
        # y = x / sqrt(E[x^2] + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGluFFN(nn.Module):
    def __init__(self,
                 d_model: int,
                 mlp_dim: int,
                 ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp_dim)
        self.fc2 = nn.Linear(d_model, mlp_dim)
        self.fc3 = nn.Linear(mlp_dim, d_model)
        self.norm = RMSNorm(mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x)) * self.fc2(x)
        x = self.norm(x)

        x = self.fc3(x)
        
        return x

class MlpDecoder(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 mlp_ratio: float,
                 num_blocks: int,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = max(round(in_dim * mlp_ratio), 2048)
        self.pixel_decoder = nn.Sequential(*[
            SwiGluFFN(in_dim, self.hidden_dim)
            for _ in range(num_blocks)])
        self.pixel_predictor = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.pixel_decoder(x)
        x = self.pixel_predictor(x)

        return x

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis


# ----------------------- ViT modules -----------------------
class ViTPatchEmbed(nn.Module):
    def __init__(self,
                 in_chans    : int = 3,
                 embed_dim   : int = 768,
                 kernel_size : int = 16,
                 padding     : int = 0,
                 stride      : int = 16,
                 ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x  # [bs, seq, c]

class AimV2ViTAttention(nn.Module):
    def __init__(self,
                 dim       :int,
                 qkv_bias  :bool  = False,
                 num_heads :int   = 8,
                 attn_drop :float = 0.,
                 proj_drop :float = 0.,
                 is_causal: bool = False,
                 ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        # --------------- Basic parameters ---------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_causal = is_causal
        self.scale = self.head_dim ** -0.5

        # --------------- Network parameters ---------------
        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,
                x: torch.Tensor,
                prefix_mask: torch.Tensor = None,
                ):
        bs, seq_len, _ = x.shape
        # ----------------- Prefix mask -----------------
        if self.is_causal:
            # attention mask: [bs, seq_len, seq_len,]
            attn_mask = torch.ones(1, seq_len, seq_len, dtype=torch.bool).tril(diagonal=0)
            attn_mask = attn_mask.expand(bs, -1, -1)
            attn_mask = attn_mask.to(x.device)

            # prefix attention mask
            if prefix_mask is not None:
                prefix_mask = prefix_mask.unsqueeze(1).expand(-1, seq_len, -1).bool() # [bs, c] -> [bs, seq_len, 1]
                attn_mask = torch.logical_or(attn_mask, prefix_mask)

            # [bs, 1, seq_len, seq_len]
            attn_mask = attn_mask.unsqueeze(1)
        else:
            attn_mask = None

        # ----------------- Input proj -----------------
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        ## [bs, seq_len, c] -> [bs, seq_len, nh, dh], c = nh x dh
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, seq_len, self.num_heads, self.head_dim)
        v = v.view(bs, seq_len, self.num_heads, self.head_dim)

        # [bs, seq_len, nh, dh] -> [bs, nh, seq_len, dh]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ----------------- Multi-head Attn -----------------
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = self.attn_drop(x)

        # ----------------- Output -----------------
        x = x.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class AimV2ViTBlock(nn.Module):
    """ Transformer module for Vision Transformer """
    def __init__(self,
                 dim       :int,
                 num_heads :int,
                 mlp_ratio :float = 4.0,
                 qkv_bias  :bool  = False,
                 proj_drop :float = 0.,
                 attn_drop :float = 0.,
                 drop_path :float = 0.,
                 is_causal :bool = False,
                 ) -> None:
        super().__init__()
        # -------------- Model parameters --------------
        self.norm1 = RMSNorm(dim, eps=1e-5)
        self.attn  = AimV2ViTAttention(
            dim       = dim,
            qkv_bias  = qkv_bias,
            num_heads = num_heads,
            attn_drop = attn_drop,
            proj_drop = proj_drop,
            is_causal = is_causal,
            )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = RMSNorm(dim, eps=1e-5)
        self.mlp   = SwiGluFFN(
            d_model = dim,
            mlp_dim = int(dim * mlp_ratio),
            )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,
                x: torch.Tensor,
                prefix_mask: torch.Tensor = None,
                ) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x), prefix_mask))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


# ----------------------- LM modules -----------------------
class LMAttention(nn.Module):
    def __init__(self,
                 dim       :int,
                 qkv_bias  :bool  = False,
                 num_heads :int   = 8,
                 attn_drop :float = 0.,
                 proj_drop :float = 0.,
                 is_causal: bool = False,
                 ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        # --------------- Basic parameters ---------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_causal = is_causal
        self.scale = self.head_dim ** -0.5

        # --------------- Network parameters ---------------
        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        
        # Reshape for broadcast
        ndim = xq_.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (xq_.shape[1], xq_.shape[-1])

        shape = [d if i == 1 or i == ndim - 1 else 1
                for i, d in enumerate(xq_.shape)]
        
        freqs_cis = freqs_cis.view(*shape)

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self,
                x: torch.Tensor,
                freqs_cis: torch.Tensor,
                attention_mask: torch.Tensor = None,
                ):
        bs, seq_len, _ = x.shape
        # ----------------- Prefix mask -----------------
        if self.is_causal:
            q_seq_len = kv_seq_len = seq_len
            # attention mask: [bs, q_seq_len, kv_seq_len,]
            causal_mask = torch.ones(1, q_seq_len, kv_seq_len, dtype=torch.bool).tril(diagonal=0)
            causal_mask = causal_mask.expand(bs, -1, -1)
            causal_mask = causal_mask.to(x.device)

            # [bs, 1, seq_len, seq_len]
            causal_mask = causal_mask.unsqueeze(1)
        else:
            causal_mask = None

        # ----------------- Input proj -----------------
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        ## [bs, seq_len, c] -> [bs, seq_len, nh, dh], c = nh x dh
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, seq_len, self.num_heads, self.head_dim)
        v = v.view(bs, seq_len, self.num_heads, self.head_dim)

        # Add RoPE
        q, k = self.apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # [bs, seq_len, nh, dh] -> [bs, nh, seq_len, dh]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ----------------- Multi-head Attn -----------------
        if attention_mask is not None:
            if causal_mask is not None:
                attn_mask = torch.logical_and(causal_mask, attention_mask.bool()[:, None, None, :].repeat(1, 1, seq_len, 1))
            else:
                attn_mask = attention_mask.bool()[:, None, None, :].repeat(1, 1, seq_len, 1)
        else:
            attn_mask = causal_mask

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = self.attn_drop(x)

        # ----------------- Output -----------------
        x = x.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class LMBlock(nn.Module):
    """ Transformer module for VLM Transformer """
    def __init__(self,
                 dim       :int,
                 num_heads :int,
                 mlp_ratio :float = 4.0,
                 qkv_bias  :bool  = False,
                 proj_drop :float = 0.,
                 attn_drop :float = 0.,
                 drop_path :float = 0.,
                 is_causal :bool = False,
                 ) -> None:
        super().__init__()
        # -------------- Model parameters --------------
        self.norm1 = RMSNorm(dim, eps=1e-5)
        self.attn  = LMAttention(dim       = dim,
                                  qkv_bias  = qkv_bias,
                                  num_heads = num_heads,
                                  attn_drop = attn_drop,
                                  proj_drop = proj_drop,
                                  is_causal = is_causal,
                                  )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = RMSNorm(dim, eps=1e-5)
        self.mlp   = SwiGluFFN(d_model = dim,
                               mlp_dim = int(dim * mlp_ratio),
                               )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,
                x: torch.Tensor,
                freqs_cis: torch.Tensor,
                attention_mask: torch.Tensor = None,
                ) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x), freqs_cis, attention_mask))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


# ----------------------- Classifier modules -----------------------
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
        self.bn = nn.BatchNorm1d(in_dim, affine=False, eps=1e-5)

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
