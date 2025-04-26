# --------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------
import math
import torch
import torch.nn as nn

try:
    from .common import TransformerBlock, PatchEmbed, RMSNorm, precompute_freqs_cis
except:
    from  common import TransformerBlock, PatchEmbed, RMSNorm, precompute_freqs_cis


# ---------------------- Vision transformer ----------------------
class ImageEncoderViT(nn.Module):
    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embed_dim: int,
                 depth: int,
                 max_seq_len: int = 2048,
                 rope_theta: int = 50000,
                 attn_drop_rate: float = 0.,
                 proj_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 ) -> None:
        super().__init__()
        # ----------- Basic parameters -----------
        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_heads  = embed_dim // 64
        self.embed_dim  = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        # ----------- Model parameters -----------
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size, stride=patch_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            TransformerBlock(dim = embed_dim,
                             num_heads = embed_dim // 64,
                             mlp_ratio = 4.0,
                             qkv_bias  = True,
                             qk_norm   = False,
                             drop_path = dpr[i],
                             attn_drop = attn_drop_rate,
                             proj_drop = proj_drop_rate,
                             )
            for i in range(depth)])
        self.norm = RMSNorm(embed_dim, eps=1e-6)

        self._init_weights()

        # RoPE
        self.freqs_cis = precompute_freqs_cis(
            dim = embed_dim // self.num_heads,
            end = max_seq_len * 2,
            theta = rope_theta,
        )

    def _init_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        for m in self.modules():           
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # patch embed
        x = self.patch_embed(x)

        # patchify: [bs, c, h, w] -> [bs, c, seq_len] -> [bs, seq_len, c], seq_len = hw
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        num_patches = x.shape[1]

        # get RoPE
        freqs_cis = self.freqs_cis[:num_patches].to(x.device)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x, freqs_cis)

        # final norm
        x = self.norm(x)

        return x


# ------------------------ Model Functions ------------------------
def build_vit(model_name="vit_t", img_size=224, patch_size=16, img_dim=3, window_size=0):
    if model_name == "vit_t":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               embed_dim=192,
                               depth=12,
                               drop_path_rate=0.1,
                               )
    if model_name == "vit_s":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               embed_dim=384,
                               depth=12,
                               drop_path_rate=0.1,
                               )
    if model_name == "vit_b":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               embed_dim=768,
                               depth=12,
                               drop_path_rate=0.1,
                               )
    if model_name == "vit_l":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               embed_dim=1024,
                               depth=24,
                               drop_path_rate=0.1,
                               )
    if model_name == "vit_h":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               embed_dim=1280,
                               depth=32,
                               drop_path_rate=0.1,
                               )
    

if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    img_size = 224
    patch_size = 16
    x = torch.randn(2, 3, img_size, img_size)

    # Build model
    model = build_vit(model_name="vit_t", img_size=img_size, patch_size=patch_size, img_dim=3)

    # Inference
    outputs = model(x)

    # Compute FLOPs & Params
    print('============ Params & FLOPs ============')
    model.eval()
    x = torch.randn(1, 3, img_size, img_size)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
