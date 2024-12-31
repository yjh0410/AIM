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
    from .common import PatchEmbed, ViTBlock
except:
    from  common import PatchEmbed, ViTBlock


# ---------------------- Vision transformer ----------------------
class ImageEncoderViT(nn.Module):
    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 patch_embed_dim: int,
                 depth: int,
                 num_heads: int,
                 mlp_ratio: float,
                 attn_drop_rate: float = 0.,
                 proj_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 ) -> None:
        super().__init__()
        # ----------- Basic parameters -----------
        self.img_size = img_size
        self.patch_size = patch_size
        self.image_embedding_size = img_size // ((patch_size if patch_size > 0 else 1))
        self.patch_embed_dim = patch_embed_dim
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2
        # ----------- Model parameters -----------
        self.patch_embed = PatchEmbed(in_chans, patch_embed_dim, patch_size, stride=patch_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            ViTBlock(patch_embed_dim,
                     num_heads,
                     mlp_ratio,
                     qkv_bias = True,
                     drop_path=dpr[i],
                     attn_drop=attn_drop_rate,
                     proj_drop=proj_drop_rate
                     )
            for i in range(depth)])
        self.norm = nn.LayerNorm(patch_embed_dim)

        self._init_weights()

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

    def get_posembed(self, embed_dim, grid_shape, temperature=10000):
        scale = 2 * math.pi
        grid_h, grid_w = grid_shape
        num_pos_feats = embed_dim // 2
        # get grid
        y_embed, x_embed = torch.meshgrid([torch.arange(grid_h, dtype=torch.float32),
                                           torch.arange(grid_w, dtype=torch.float32)])
        # normalize grid coords
        y_embed = y_embed / (grid_h + 1e-6) * scale
        x_embed = x_embed / (grid_w + 1e-6) * scale
    
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        pos_x = torch.div(x_embed[..., None], dim_t)
        pos_y = torch.div(y_embed[..., None], dim_t)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

        # [H, W, C] -> [N, C]
        pos_embed = torch.cat((pos_y, pos_x), dim=-1).view(-1, embed_dim)

        return pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embed
        B = x.shape[0]
        x = self.patch_embed(x)
        bs, c, h, w = x.shape

        # [bs, c, h, w] -> [bs, c, hw] -> [bs, hw, c]
        x = x.flatten(2).permute(0, 2, 1).contiguous()

        # Calculate pos embed
        pos_embed = self.get_posembed(embed_dim=c, grid_shape=[h, w])
        pos_embed = pos_embed.to(x.device)

        # Add pos embed
        x = x + pos_embed

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x


# ------------------------ Model Functions ------------------------
def build_vit(model_name="vit_t", img_size=224, patch_size=16, img_dim=3):
    if model_name == "vit_t":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               patch_embed_dim=192,
                               depth=12,
                               num_heads=3,
                               mlp_ratio=4.0,
                               drop_path_rate=0.1,
                               )
    if model_name == "vit_s":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               patch_embed_dim=384,
                               depth=12,
                               num_heads=6,
                               mlp_ratio=4.0,
                               drop_path_rate=0.1,
                               )
    if model_name == "vit_b":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               patch_embed_dim=768,
                               depth=12,
                               num_heads=12,
                               mlp_ratio=4.0,
                               drop_path_rate=0.1,
                               )
    if model_name == "vit_l":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               patch_embed_dim=1024,
                               depth=24,
                               num_heads=16,
                               mlp_ratio=4.0,
                               drop_path_rate=0.1,
                               )
    if model_name == "vit_h":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               patch_embed_dim=1280,
                               depth=32,
                               num_heads=16,
                               mlp_ratio=4.0,
                               drop_path_rate=0.1,
                               )
    

if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, c, h, w = 2, 3, 224, 224
    x = torch.randn(bs, c, h, w)
    patch_size = 16

    # Build model
    model = build_vit(patch_size=patch_size)

    # Inference
    outputs = model(x)

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
