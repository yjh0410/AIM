import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .common import TransformerBlock, MlpDecoder, PatchEmbed, RMSNorm, precompute_freqs_cis
except:
    from  common import TransformerBlock, MlpDecoder, PatchEmbed, RMSNorm, precompute_freqs_cis


# ------------------------ Basic Modules ------------------------
class AimV2VisionEncoder(nn.Module):
    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embed_dim: int,
                 depth: int,
                 max_seq_len: int = 2048,
                 rope_theta: int = 50000,
                 ) -> None:
        super().__init__()
        # ----------- Basic parameters -----------
        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_heads  = embed_dim // 64
        self.embed_dim  = embed_dim

        # ----------- Model parameters -----------
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size, 0, patch_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim = embed_dim,
                             num_heads = embed_dim // 64,
                             mlp_ratio = 4.0,
                             qkv_bias  = True,
                             qk_norm   = False,
                             causal_mask = True,
                             )
            for _ in range(depth)])
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

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
        # patch embed
        x = self.patch_embed(x)

        # patchify: [bs, c, h, w] -> [bs, c, seq_len] -> [bs, seq_len, c], seq_len = hw
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        num_patches = x.shape[1]

        # get RoPE
        freqs_cis = self.freqs_cis[:num_patches].to(x.device)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x, freqs_cis, prefix_mask)

        # final norm
        x = self.norm(x)
        
        return x

class AimV2Decoder(nn.Module):
    def __init__(self,
                 img_dim: int = 3,
                 patch_size: int = 16,
                 img_embed_dim: int = 512,
                 txt_embed_dim: int = 512,
                 depth: int = 24,
                 max_seq_len: int = 2048,
                 vocab_size: int = 1024,
                 rope_theta: int = 50000,
                 ) -> None:
        super().__init__()
        # ----------- Basic parameters -----------
        self.img_embed_dim  = img_embed_dim
        self.txt_embed_dim  = txt_embed_dim
        self.depth = depth

        self.max_seq_len = max_seq_len
        self.vocab_size  = vocab_size
        self.rope_theta  = rope_theta

        self.tok_emb = nn.Embedding(vocab_size, txt_embed_dim)
        self.img_projector = nn.Linear(img_embed_dim, txt_embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim       = txt_embed_dim,
                             num_heads = txt_embed_dim // 64,
                             mlp_ratio = 4.0,
                             qkv_bias  = True,
                             qk_norm   = False,
                             causal_mask = True,
                             ) for _ in range(depth)])
        self.norm = RMSNorm(txt_embed_dim)

        self.img_head = MlpDecoder(
            in_dim     = txt_embed_dim,
            out_dim    = patch_size ** 2 * img_dim,
            mlp_ratio  = 4.0,
            num_blocks = 12,
            )
        self.txt_head  = MlpDecoder(
            in_dim     = txt_embed_dim,
            out_dim    = vocab_size,
            mlp_ratio  = 4.0,
            num_blocks = 6,
            )

        self._init_weights()

        # RoPE
        num_heads = txt_embed_dim // 64
        self.freqs_cis = precompute_freqs_cis(
            txt_embed_dim // num_heads,
            max_seq_len * 2,
            rope_theta,
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, img_embed, token_ids):
        # Mapping visual tokens to text space
        img_tokens = self.img_projector(img_embed)    # [bs, vseq_len, c]
        img_tokens_len = img_tokens.shape[1]

        # text embeddings
        txt_tokens = self.tok_emb(token_ids)  # [bs, tseq_len, c]

        # Concatenate image-text embeddings
        x = torch.cat([img_tokens, txt_tokens], dim=1)   # [bs, vseq_len + tseq_len, c]
        mseq_len = x.shape[1]

        # get RoPE
        freqs_cis = self.freqs_cis[:mseq_len].to(x.device)

        # Apply transformer block
        for block in self.blocks:
            x = block(x, freqs_cis)

        # final norm
        x = self.norm(x)

        # output
        vlogits = self.img_head(x[:, :img_tokens_len, :])
        tlogits = self.txt_head(x[:, img_tokens_len:, :])

        return vlogits, tlogits


# ------------------------ MAE Vision Transformer ------------------------
class AIMv2(nn.Module):
    def __init__(self,
                 encoder: AimV2VisionEncoder,
                 decoder: AimV2Decoder,
                 norm_pix_loss :bool = False,
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.norm_pix_loss = norm_pix_loss

    def forward(self,
                imgs: torch.Tensor,
                text_token_ids: torch.Tensor,
                prefix_mask: torch.Tensor = None,
                token_mask: torch.Tensor = None,
                compute_loss: bool = False,
                ):
        # vision encoder: output shape -> [bs, v_seq_len, c]
        img_embed = self.encoder(imgs, prefix_mask)

        # multi-modal decoder output shape -> [bs, v_seq_len + t_seq_len, c]
        vlogits, tlogits = self.decoder(img_embed, text_token_ids)

        output = {}
        if self.training or compute_loss:
            output = self.compute_loss(
                imgs, vlogits, prefix_mask, text_token_ids, tlogits, token_mask)

        return output

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

    def compute_loss(self,
                     # image
                     imgs, img_pred, prefix_mask,
                     # text
                     txt, txt_pred, token_mask,
                     ):
        loss_vis = self.compute_vis_loss(imgs, img_pred, prefix_mask)
        loss_txt = self.compute_txt_loss(txt, txt_pred, token_mask)

        λ = 1.0
        loss_dict = {}
        loss_dict["loss"] = loss_vis + λ * loss_txt
        loss_dict["loss_vis"] = loss_vis
        loss_dict["loss_txt"] = loss_txt

        return loss_dict

    def compute_vis_loss(self, imgs, pred, prefix_mask):
        """
            Input:
                imgs: torch.Tensor -> [bs, 3, h, w]
                pred: torch.Tensor -> [bs, seq_len, p*p*3]
                prefix_mask: torch.Tensor -> [bs, seq_len]
            Output:
                loss: torch.Tensor
        """
        # Patchify the image: [bs, 3, h, w] -> [bs, seq_len, c]
        target = self.patchify(imgs, self.encoder.patch_size)

        # normalize patch pixel values
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        # Compute L1 loss
        loss = (pred[:, :-1, :] - target[:, 1:, :]) ** 2

        # Keep the losses on the non-prefix tokens and normalize the loss
        non_prefix_mask = (1.0 - prefix_mask.float())[:, 1:]
        loss = loss * non_prefix_mask.unsqueeze(2)

        # Normalize loss
        norm_factor = torch.sum(non_prefix_mask)
        loss = loss.mean(dim=-1).sum() / norm_factor
        
        return loss

    def compute_txt_loss(self, texts, pred, token_masks):
        """
            Input:
                label: torch.Tensor -> [bs, seq_len,]
                pred: torch.Tensor -> [bs, seq_len, vocab_size]
                token_masks: torch.Tensor -> [bs, seq_len,]
            Output:
                loss: torch.Tensor
        """
        # cross-entropy
        pred_flatten = pred[:, :-1, :].flatten(0, 1)
        texts_flatten = texts[:, 1:].flatten()
        loss = F.cross_entropy(pred_flatten, texts_flatten, reduction="none")
        
        # keep the loss of non-padding tokens and normalize the loss.
        masks_flatten = token_masks[:, 1:].flatten()
        loss = (loss * masks_flatten).sum() / masks_flatten.sum()

        return loss
    

def build_aimv2(model_name: str = "aimv2_t",
                img_dim: int = 3,
                img_size: int = 224,
                patch_size: int = 16,
                norm_pix_loss=False,
                ):
    # ---------------- AIMv2 Encoder ----------------
    if model_name == "aimv2_t":
        encoder = AimV2VisionEncoder(
            img_size = img_size,
            patch_size = patch_size,
            in_chans = img_dim,
            embed_dim = 192,
            depth = 12,
            max_seq_len = 1024,
            rope_theta = 50000,
            )
        decoder = AimV2Decoder(
            img_dim = img_dim,
            patch_size = patch_size,
            img_embed_dim = encoder.embed_dim,
            txt_embed_dim = 192,
            depth = 6,
            max_seq_len = 1024,
            vocab_size = 130000,
            rope_theta = 50000,
        )

    if model_name == "aimv2_s":
        encoder = AimV2VisionEncoder(
            img_size = img_size,
            patch_size = patch_size,
            in_chans = img_dim,
            embed_dim = 384,
            depth = 12,
            max_seq_len = 1024,
            rope_theta = 50000,
            )
        decoder = AimV2Decoder(
            img_dim = img_dim,
            patch_size = patch_size,
            img_embed_dim = encoder.embed_dim,
            txt_embed_dim = 384,
            depth = 6,
            max_seq_len = 1024,
            vocab_size = 130000,
            rope_theta = 50000,
        )

    if model_name == "aimv2_b":
        encoder = AimV2VisionEncoder(
            img_size = img_size,
            patch_size = patch_size,
            in_chans = img_dim,
            embed_dim = 768,
            depth = 12,
            max_seq_len = 1024,
            rope_theta = 50000,
            )
        decoder = AimV2Decoder(
            img_dim = img_dim,
            patch_size = patch_size,
            img_embed_dim = encoder.embed_dim,
            txt_embed_dim = 768,
            depth = 6,
            num_heads = 12,
            max_seq_len = 1024,
            vocab_size = 130000,
            rope_theta = 50000,
        )

    if model_name == "aimv2_l":
        encoder = AimV2VisionEncoder(
            img_size = img_size,
            patch_size = patch_size,
            in_chans = img_dim,
            embed_dim = 1024,
            depth = 24,
            max_seq_len = 1024,
            rope_theta = 50000,
            )
        decoder = AimV2Decoder(
            img_dim = img_dim,
            patch_size = patch_size,
            img_embed_dim = encoder.embed_dim,
            txt_embed_dim = 1024,
            depth = 12,
            num_heads = 16,
            max_seq_len = 1024,
            vocab_size = 130000,
            rope_theta = 50000,
        )
        
    if model_name == "aimv2_h":
        encoder = AimV2VisionEncoder(
            img_size = img_size,
            patch_size = patch_size,
            in_chans = img_dim,
            embed_dim = 1280,
            depth = 32,
            max_seq_len = 1024,
            rope_theta = 50000,
            )
        decoder = AimV2Decoder(
            img_dim = img_dim,
            patch_size = patch_size,
            img_embed_dim = encoder.embed_dim,
            txt_embed_dim = 1280,
            depth = 16,
            num_heads = 16,
            max_seq_len = 1024,
            vocab_size = 130000,
            rope_theta = 50000,
        )
    
    return AIMv2(encoder, decoder, norm_pix_loss)


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
    tseq_len = 8

    # generate input data
    images = []
    prefix_masks = []
    token_masks = []
    for i in range(batch_size):
        x = torch.randn(3, img_size, img_size).float()
        prefix_length = random.randint(1, num_patches-1)
        prefix_mask = torch.zeros(num_patches).bool()
        prefix_mask[:prefix_length] = True

        token_mask = torch.zeros(tseq_len).bool()
        token_len = 4
        token_mask[:token_len] = True

        images.append(x)
        prefix_masks.append(prefix_mask)
        token_masks.append(token_mask)

    images = torch.stack(images)
    text_token_ids = torch.randint(low=0, high=400, size=[batch_size, tseq_len],)
    prefix_masks = torch.stack(prefix_masks)
    token_masks = torch.stack(token_masks)

    # Build model
    model = build_aimv2(model_name="aimv2_t", patch_size = patch_size, norm_pix_loss=True)
    model.train()

    # inference
    outputs = model(images, text_token_ids, prefix_masks, token_masks)
    for k in outputs:
        print(k, outputs[k].item())

    # compute FLOPs & Params
    print('==============================')
    x1 = images[:1]
    x2 = text_token_ids[:1]
    prefix_mask = prefix_masks[:1]
    flops, params = profile(model, inputs=(x1, x2, prefix_mask), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))