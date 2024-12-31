import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------- Basic modules -----------------------
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # y = x / sqrt(E[x^2] + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
def scaled_dot_product_attention(query, key, value, attn_mask=None):
    """
    :param query: Query 向量 (batch_size, n_heads, seq_len, d_k)
    :param key: Key 向量 (batch_size, n_heads, seq_len, d_k)
    :param value: Value 向量 (batch_size, n_heads, seq_len, d_v)
    :param attn_mask: 注意力掩码 (batch_size, n_heads, seq_len, seq_len)
    :return: 输出向量 (batch_size, n_heads, seq_len, d_v)
    """
    scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, n_heads, seq_len, seq_len)
    
    dk = torch.tensor(key.size(-1), dtype=torch.float32)  # d_k
    scores = scores / torch.sqrt(dk)  # 缩放点积
    
    if attn_mask is not None:
        attn_mask_ = attn_mask[:, :, :scores.shape[-2], :scores.shape[-1]]
        scores = scores.masked_fill(attn_mask_ == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)  # (batch_size, n_heads, seq_len, seq_len)
    
    output = torch.matmul(attn_weights, value)  # (batch_size, n_heads, seq_len, d_v)
    
    return output

class FeedForward(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 mlp_dim: int,
                 dropout: float = 0.0,
                 ) -> None:
        super().__init__()
        self.fc1   = nn.Linear(embedding_dim, mlp_dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(mlp_dim, embedding_dim)
        self.drop2 = nn.Dropout(dropout)
        self.act   = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim       :int,
                 qkv_bias  :bool  = False,
                 qk_norm   :bool  = False,
                 num_heads :int   = 8,
                 attn_drop :float = 0.,
                 proj_drop :float = 0.,
                 prefix_causal_mask :bool  = False,
                 ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        # --------------- Basic parameters ---------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.prefix_causal_mask = prefix_causal_mask

        # --------------- Network parameters ---------------
        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,
                x: torch.Tensor,
                prefix_mask: torch.Tensor = None,
                num_patches: int = None,
                ):
        bs, seq_len, _ = x.shape
        # ----------------- Prefix mask -----------------
        if self.prefix_causal_mask:
            assert num_patches is not None, "A mask is required for the PrefixLM Causal Attention."
            # attention mask: [bs, seq_len, seq_len,]
            attn_mask = torch.ones(1, num_patches, num_patches, dtype=torch.bool).tril(diagonal=0)
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

        ## [B, N, C] -> [B, N, H, C_h] -> [B, H, N, C_h]
        q = q.view(bs, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(bs, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(bs, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        # ----------------- Multi-head Attn -----------------
        x = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # ----------------- Output -----------------
        x = x.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

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

    
# ----------------------- Model modules -----------------------
class ViTBlock(nn.Module):
    def __init__(self,
                 dim       :int,
                 num_heads :int,
                 mlp_ratio :float = 4.0,
                 qkv_bias  :bool  = False,
                 qk_norm  :bool  = False,
                 proj_drop :float = 0.,
                 attn_drop :float = 0.,
                 drop_path :float = 0.,
                 prefix_causal_mask :bool = False,
                 ) -> None:
        super().__init__()
        # -------------- Model parameters --------------
        self.norm1 = RMSNorm(dim)
        self.attn  = Attention(dim       = dim,
                               qkv_bias  = qkv_bias,
                               qk_norm   = qk_norm,
                               num_heads = num_heads,
                               attn_drop = attn_drop,
                               proj_drop = proj_drop,
                               prefix_causal_mask = prefix_causal_mask,
                               )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = RMSNorm(dim)
        self.mlp   = FeedForward(embedding_dim=dim,
                                 mlp_dim=int(dim * mlp_ratio),
                                 dropout=proj_drop,
                                 )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,
                x: torch.Tensor,
                prefix_mask: torch.Tensor = None,
                num_patches: int = None,
                ) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x), prefix_mask, num_patches))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

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


# ----------------------- classifier modules -----------------------
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
