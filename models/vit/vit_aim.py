import math
import torch
import torch.nn as nn

try:
    from .common import ViTBlock, PatchEmbed, MLPBlock
except:
    from  common import ViTBlock, PatchEmbed, MLPBlock


# ------------------------ Basic Modules ------------------------
class AimEncoder(nn.Module):
    def __init__(self,
                 img_size      :int   = 224,
                 patch_size    :int   = 16,
                 in_dim        :int   = 3,
                 patch_embed_dim :int   = 768,
                 num_layers    :int   = 12,
                 num_heads     :int   = 12,
                 qkv_bias      :bool  = True,
                 mlp_ratio     :float = 4.0,
                 dropout       :float = 0.1,
                 prefix_causal_mask: bool = False,
                 ):
        super().__init__()
        # -------- basic parameters --------
        self.img_size = img_size
        self.in_dim = in_dim
        self.patch_embed_dim = patch_embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.prefix_causal_mask = prefix_causal_mask
        # -------- model parameters --------
        self.patch_embed = PatchEmbed(in_dim, patch_embed_dim, patch_size, stride=patch_size)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches, patch_embed_dim), requires_grad=False)
        self.norm_layer  = nn.LayerNorm(patch_embed_dim)
        self.blocks      = nn.ModuleList([ViTBlock(patch_embed_dim, qkv_bias, num_heads, self.num_patches, mlp_ratio, prefix_causal_mask, dropout)
                                          for _ in range(num_layers)])

        self._init_weights()

    def _init_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = self.get_posembed(self.pos_embed.shape[-1], int(self.num_patches**.5))
        self.pos_embed.data.copy_(pos_embed)

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

    def get_posembed(self, embed_dim, grid_size, temperature=10000):
        scale = 2 * torch.pi
        grid_h, grid_w = grid_size, grid_size
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

        return pos_embed.unsqueeze(0)

    def forward(self, x, mask=None):
        # Patch embed
        x = self.patch_embed(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()

        # Add pos embed
        x = x + self.pos_embed

        # Apply Transformer blocks
        for block in self.blocks     :
            x = block(x, mask)
        x = self.norm_layer(x)

        return x

class AimDecoder(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 mlp_ratio: float,
                 num_blocks: int,
                 dropout:float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = round(in_dim * mlp_ratio)
        self.pixel_decoder = nn.Sequential(*[MLPBlock(in_dim, self.hidden_dim, nn.GELU, dropout) for _ in range(num_blocks)])
        self.pixel_predictor = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.pixel_decoder(x)
        x = self.pixel_predictor(x)

        return x


# ------------------------ MAE Vision Transformer ------------------------
class ViTforAutoRegression(nn.Module):
    def __init__(self,
                 encoder :AimEncoder,
                 decoder :AimDecoder,
                 norm_pix_loss :bool = False):
        super().__init__()
        self.aim_encoder = encoder
        self.aim_decoder = decoder
        self.norm_pix_loss = norm_pix_loss

    def patchify(self, imgs, patch_size):
        """
        imgs: (B, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x, patch_size):
        """
        x: (B, N, patch_size**2 *3)
        imgs: (B, 3, H, W)
        """
        p = patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        return imgs

    def compute_loss(self, x, output):
        # Patchify the image
        target = self.patchify(x, self.aim_encoder.patch_size)
        bs, seq_length = x.shape[:2]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # Shift one position to the left
        pred = output["x_pred"]
        target = target[:, 1:, :]

        # Compute L1 loss
        loss = (pred[:, :-1, :] - target) ** 2
        loss = loss.mean(dim=-1).sum() / (seq_length - 1) / bs
        
        return loss

    def forward(self, x, prefix_mask=None):
        """
        Inputs:
            x: (torch.Tensor) -> [B, C, H, W]. Input image.
        """
        # ---------- infer & loss ----------
        imgs = x
        x = self.aim_encoder(x, prefix_mask)
        x = self.aim_decoder(x)
        output = {
            'x_pred': x,
        }

        if self.training:
            loss = self.compute_loss(imgs, output)
            output["loss"] = loss

        return output


# ------------------------ Model Functions ------------------------
def build_vit_aim(model_name="vit_t", img_size=224, patch_size=16, img_dim=3, norm_pix_loss=False):
    # ---------------- MAE Encoder ----------------
    if model_name == "vit_t":
        encoder = AimEncoder(img_size=img_size,
                             patch_size=patch_size,
                             in_dim=img_dim,
                             patch_embed_dim=192,
                             num_layers=12,
                             num_heads=3,
                             qkv_bias=False,
                             mlp_ratio=4.0,
                             dropout=0.1,
                             prefix_causal_mask=True)
    if model_name == "vit_s":
        encoder = AimEncoder(img_size=img_size,
                             patch_size=patch_size,
                             in_dim=img_dim,
                             patch_embed_dim=384,
                             num_layers=12,
                             num_heads=6,
                             qkv_bias=False,
                             mlp_ratio=4.0,
                             dropout=0.1,
                             prefix_causal_mask=True)
    if model_name == "vit_b":
        encoder = AimEncoder(img_size=img_size,
                             patch_size=patch_size,
                             in_dim=img_dim,
                             patch_embed_dim=768,
                             num_layers=12,
                             num_heads=12,
                             qkv_bias=False,
                             mlp_ratio=4.0,
                             dropout=0.1,
                             prefix_causal_mask=True)
    if model_name == "vit_l":
        encoder = AimEncoder(img_size=img_size,
                             patch_size=patch_size,
                             in_dim=img_dim,
                             patch_embed_dim=1024,
                             num_layers=24,
                             num_heads=16,
                             qkv_bias=False,
                             mlp_ratio=4.0,
                             dropout=0.1,
                             prefix_causal_mask=True)
    if model_name == "vit_h":
        encoder = AimEncoder(img_size=img_size,
                             patch_size=patch_size,
                             in_dim=img_dim,
                             patch_embed_dim=1280,
                             num_layers=32,
                             num_heads=16,
                             qkv_bias=False,
                             mlp_ratio=4.0,
                             dropout=0.1,
                             prefix_causal_mask=True)
    
    # ---------------- MAE Decoder ----------------
    decoder = AimDecoder(in_dim=encoder.patch_embed_dim,
                         out_dim=patch_size**2 * img_dim,
                         mlp_ratio=4.0,
                         num_blocks=8,
                         dropout=0.1,)
    
    return ViTforAutoRegression(encoder, decoder, norm_pix_loss)


if __name__ == '__main__':
    import torch
    import random
    from thop import profile

    print('===============  AIM pipeline  ===============')
    # parameters
    is_train = True
    batch_size = 4
    img_size = 224
    patch_size = 16
    num_patches = (img_size // patch_size) ** 2

    # generate input data
    images = []
    prefix_masks = []
    for i in range(batch_size):
        x = torch.randn(3, img_size, img_size).float()
        prefix_length = random.randint(1, num_patches-1)
        prefix_mask = torch.zeros(num_patches).bool()
        prefix_mask[:prefix_length] = True
        images.append(x)
        prefix_masks.append(prefix_mask)

    images = torch.stack(images)
    prefix_masks = torch.stack(prefix_masks)

    # Build model
    model = build_vit_aim(patch_size=patch_size)
    model.train()

    # inference
    outputs = model(images, prefix_masks)
    if "loss" in outputs and outputs["loss"]:
        print("Loss: ", outputs["loss"].item())

    # compute FLOPs & Params
    print('==============================')
    x = images[:1]
    prefix_mask = prefix_masks[:1]
    flops, params = profile(model, inputs=(x, prefix_mask), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
