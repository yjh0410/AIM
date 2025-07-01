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
    from .common import AimV2ViTBlock, ViTPatchEmbed, RMSNorm
except:
    from  common import AimV2ViTBlock, ViTPatchEmbed, RMSNorm


# --------------- Vision transformer ---------------
class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embed_dim: int,
                 num_blocks: int,
                 is_causal: bool = False,
                 attn_drop_rate: float = 0.,
                 proj_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 ) -> None:
        super().__init__()
        # ----------- Basic parameters -----------
        self.img_dim    = in_chans
        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_heads  = embed_dim // 64
        self.embed_dim  = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # ----------- Model parameters -----------
        self.patch_embed = ViTPatchEmbed(in_chans, embed_dim, patch_size, stride=patch_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            AimV2ViTBlock(dim = embed_dim,
                          num_heads = embed_dim // 64,
                          mlp_ratio = 4.0,
                          qkv_bias  = True,
                          drop_path = dpr[i],
                          attn_drop = attn_drop_rate,
                          proj_drop = proj_drop_rate,
                          is_causal = is_causal,
                          )
            for i in range(num_blocks)])
        
        self.norm = RMSNorm(embed_dim, eps=1e-5)
        self.pos_embed = nn.Parameter(torch.rand(1, self.num_patches, embed_dim))
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self,
                x: torch.Tensor,
                prefix_mask: torch.Tensor = None,
                ) -> torch.Tensor:
        # patch embed: [bs, c, h, w] -> [bs, seq_len, c]
        tokens = self.patch_embed(x)
        num_tokens = tokens.shape[1]

        # add pos_embed
        tokens += self.pos_embed[:, :num_tokens]

        # apply Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, prefix_mask)

        # final norm
        tokens = self.norm(tokens)
        
        return tokens


# --------------- Model Functions ---------------
def build_vision_transformer(model_cfg,):
    return VisionTransformer(
        img_size   = model_cfg.vit_img_size,
        patch_size = model_cfg.vit_patch_size,
        in_chans   = model_cfg.vit_img_dim,
        embed_dim  = model_cfg.vit_embed_dim,
        num_blocks = model_cfg.vit_num_blocks,
        is_causal  = model_cfg.vit_is_causal,
        drop_path_rate = model_cfg.vit_drop_path_rate,
        )
    

if __name__ == '__main__':
    import torch
    from thop import profile

    # model config
    class ModelConfig:
        vit_img_dim: int = 3
        vit_embed_dim: int = 192
        vit_patch_size: int = 16
        vit_img_size: int = 256
        vit_drop_path_rate: float = 0.0
        vit_num_blocks: int = 12
        vit_is_causal: bool = False

    cfg = ModelConfig()

    x = torch.randn(2, 3, cfg.vit_img_size, cfg.vit_img_size)

    # Build model
    model = build_vision_transformer(cfg)

    # Inference
    outputs = model(x)

    # Compute FLOPs & Params
    print('============ Params & FLOPs ============')
    model.eval()
    x = torch.randn(1, 3, cfg.vit_img_size, cfg.vit_img_size)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
