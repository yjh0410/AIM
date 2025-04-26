import math
import torch
import torch.nn as nn

try:
    from .common import ViTBlock, PatchEmbed, FeedForward
except:
    from  common import ViTBlock, PatchEmbed, FeedForward


# ------------------------ Basic Modules ------------------------
class AimEncoder(nn.Module):
    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 patch_embed_dim: int,
                 depth: int,
                 num_heads: int,
                 mlp_ratio: float,
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
        self.patch_embed = PatchEmbed(in_chans, patch_embed_dim, patch_size, 0, patch_size)
        self.blocks = nn.ModuleList([
            ViTBlock(patch_embed_dim,
                     num_heads,
                     mlp_ratio,
                     qkv_bias=True,
                     qk_norm=False,
                     prefix_causal_mask=True,
                     )
            for _ in range(depth)])
        self.norm = nn.LayerNorm(patch_embed_dim, eps=1e-6)
        
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

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def get_posembed(self, embed_dim, grid_shape, temperature=10000,):
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

    def forward(self, x: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
        # patch embed
        x = self.patch_embed(x)
        bs, c, h, w = x.shape

        # [bs, c, h, w] -> [bs, c, seq_len] -> [bs, seq_len, c], seq_len = hw
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        num_patches = x.shape[1]

        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = self.get_posembed(self.patch_embed_dim, [h, w])
        pos_embed = pos_embed.to(x.device).unsqueeze(0)

        # add pos embed w/o cls token
        x = x + pos_embed

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x, prefix_mask, num_patches)
        x = self.norm(x)
        
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
        self.hidden_dim = max(round(in_dim * mlp_ratio), 2048)
        self.pixel_decoder = nn.Sequential(*[FeedForward(in_dim, self.hidden_dim, dropout) for _ in range(num_blocks)])
        self.pixel_predictor = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.pixel_decoder(x)
        x = self.pixel_predictor(x)

        return x


# ------------------------ AutoRegression for Vision Transformer ------------------------
class AIMv1(nn.Module):
    def __init__(self,
                 encoder :AimEncoder,
                 decoder :AimDecoder,
                 norm_pix_loss :bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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

    def compute_loss(self, x, output, prefix_mask):
        """
            Input:
                x: torch.Tensor -> [bs, 3, h, w]
                output['x_pred']: torch.Tensor -> [bs, seq_len, p*p*3]
                prefix_mask: torch.Tensor -> [bs, seq_len]
            Output:
            loss: torch.Tensor
        """
        # Patchify the image: [bs, 3, h, w] -> [bs, seq_len, c]
        target = self.patchify(x, self.encoder.patch_size)
        bs, seq_length = target.shape[:2]

        # normalize patch pixel values
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # Shift one position to the left
        pred = output["x_pred"]
        
        # Compute L1 loss
        loss = (pred[:, :-1, :] - target[:, 1:, :]) ** 2

        # Keep the losses on the non-prefix tokens
        non_prefix_mask = (1.0 - prefix_mask.float())[:, 1:]
        loss = loss * non_prefix_mask.unsqueeze(2)

        # Normalize loss
        norm_factor = torch.sum(non_prefix_mask)
        loss = loss.mean(dim=-1).sum() / norm_factor
        
        return loss

    def forward(self, x: torch.Tensor, prefix_mask: torch.Tensor = None):
        imgs = x
        x = self.encoder(x, prefix_mask)
        x = self.decoder(x)
        output = {
            'x_pred': x,
        }

        if self.training:
            loss = self.compute_loss(imgs, output, prefix_mask)
            output["loss"] = loss

        return output


def build_aimv1(model_name: str = "aimv1_t",
                  img_size: int = 224,
                  patch_size: int = 16,
                  img_dim: int = 3,
                  norm_pix_loss=False,
                  ):
    # ---------------- AIM Encoder (ViT) ----------------
    if model_name == "aimv1_t":
        encoder = AimEncoder(img_size        = img_size,
                             patch_size      = patch_size,
                             in_chans        = img_dim,
                             patch_embed_dim = 192,
                             depth           = 12,
                             num_heads       = 3,
                             mlp_ratio       = 4.0,
                             )
    if model_name == "aimv1_s":
        encoder = AimEncoder(img_size        = img_size,
                             patch_size      = patch_size,
                             in_chans        = img_dim,
                             patch_embed_dim = 384,
                             depth           = 12,
                             num_heads       = 6,
                             mlp_ratio       = 4.0,
                             )
    if model_name == "aimv1_b":
        encoder = AimEncoder(img_size        = img_size,
                             patch_size      = patch_size,
                             in_chans        = img_dim,
                             patch_embed_dim = 768,
                             depth           = 12,
                             num_heads       = 12,
                             mlp_ratio       = 4.0,
                             )
    if model_name == "aimv1_l":
        encoder = AimEncoder(img_size        = img_size,
                             patch_size      = patch_size,
                             in_chans        = img_dim,
                             patch_embed_dim = 1024,
                             depth           = 24,
                             num_heads       = 16,
                             mlp_ratio       = 4.0,
                             )
    if model_name == "aimv1_h":
        encoder = AimEncoder(img_size        = img_size,
                             patch_size      = patch_size,
                             in_chans        = img_dim,
                             patch_embed_dim = 1280,
                             depth           = 32,
                             num_heads       = 16,
                             mlp_ratio       = 4.0,
                             )
    
    # ---------------- AIM Decoder (MLP) ----------------
    decoder = AimDecoder(in_dim     = encoder.patch_embed_dim,
                         out_dim    = patch_size**2 * img_dim,
                         mlp_ratio  = 4.0,
                         num_blocks = 12,
                         dropout    = 0.0,
                         )
    
    return AIMv1(encoder, decoder, norm_pix_loss)


if __name__ == '__main__':
    import random
    import torch
    from thop import profile

    print('===============  AIM pipeline  ===============')
    # parameters
    is_train = True
    batch_size = 2
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
    model = build_aimv1(model_name="aimv1_s", patch_size = patch_size, norm_pix_loss=True)
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