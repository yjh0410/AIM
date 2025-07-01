import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .vision_transformer import build_vision_transformer
    from .language_model import build_language_model
    from .common import MlpDecoder
except:
    from  vision_transformer import build_vision_transformer
    from  language_model import build_language_model
    from  common import MlpDecoder


# ------------------------ MAE Vision Transformer ------------------------
class AIMv2(nn.Module):
    def __init__(self, model_cfg,):
        super().__init__()
        self.norm_pix_loss = model_cfg.vit_norm_pix_loss

        self.vis_encoder = build_vision_transformer(model_cfg)
        self.lm_decoder = build_language_model(model_cfg)

        self.projector = nn.Linear(model_cfg.vit_embed_dim, model_cfg.lm_embed_dim)

        self.vhead = MlpDecoder(
            in_dim     = model_cfg.vit_embed_dim,
            out_dim    = model_cfg.vit_patch_size ** 2 * model_cfg.vit_img_dim,
            mlp_ratio  = 4.0,
            num_blocks = 12,
            )
        self.thead = MlpDecoder(
            in_dim     = model_cfg.lm_embed_dim,
            out_dim    = model_cfg.lm_vocab_size,
            mlp_ratio  = 4.0,
            num_blocks = 6,
            )

    def forward(self,
                images: torch.Tensor,
                image_prefix_masks: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                compute_loss: bool = False,
                ):
        # vision encoder: output shape -> [bs, v_seq, c]
        img_tokens = self.vis_encoder(images, image_prefix_masks)
        img_tokens = self.projector(img_tokens)

        # get the token embeddings: [bs, t_seq, c]
        txt_tokens = self.lm_decoder.token_embeddings(input_ids)

        # concatenate vl tokens
        tokens = torch.cat([img_tokens, txt_tokens], dim=1)

        # lm decoder
        outputs = self.lm_decoder(
            input_embeddings = tokens,
            attention_mask = attention_mask,
            )

        # multi-modal decoder output shape -> [bs, v_seq_len + t_seq_len, c]
        last_hidden_stat = outputs["last_hidden_state"]
        vlogits = self.vhead(last_hidden_stat[:, :img_tokens.shape[1]])
        tlogits = self.thead(last_hidden_stat[:, img_tokens.shape[1]:])

        output = {}
        if self.training or compute_loss:
            output = self.compute_loss(
                input_imgs = images,
                pred_imgs = vlogits,
                img_prefix_mask = image_prefix_masks,
                input_ids = input_ids,
                pred_logits = tlogits,
                attention_mask = attention_mask,
                )

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
                     input_imgs,
                     pred_imgs,
                     img_prefix_mask,
                     # text
                     input_ids,
                     pred_logits,
                     attention_mask = None,
                     ):
        loss_vis = self.compute_vis_loss(input_imgs, pred_imgs, img_prefix_mask)
        loss_txt = self.compute_txt_loss(input_ids, pred_logits, attention_mask[:, pred_imgs.shape[1]:])

        λ = 1.0
        loss_dict = {}
        loss_dict["loss"] = loss_vis + λ * loss_txt
        loss_dict["loss_vis"] = loss_vis
        loss_dict["loss_txt"] = loss_txt

        return loss_dict

    def compute_vis_loss(self, input_imgs, pred_imgs, img_prefix_mask):
        """
            Input:
                input_imgs: torch.Tensor -> [bs, 3, h, w]
                pred_imgs: torch.Tensor -> [bs, seq_len, p*p*3]
                img_prefix_mask: torch.Tensor -> [bs, seq_len]
            Output:
                loss: torch.Tensor
        """
        # Patchify the image: [bs, 3, h, w] -> [bs, seq_len, c]
        target = self.patchify(input_imgs, self.vis_encoder.patch_size)

        # normalize patch pixel values
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        # Compute L1 loss
        loss = (pred_imgs[:, :-1, :] - target[:, 1:, :]) ** 2

        # Keep the losses on the non-prefix tokens and normalize the loss
        non_prefix_mask = (1.0 - img_prefix_mask.float())[:, 1:]
        loss = loss * non_prefix_mask.unsqueeze(2)

        # Normalize loss
        norm_factor = torch.sum(non_prefix_mask)
        loss = loss.mean(dim=-1).sum() / norm_factor
        
        return loss

    def compute_txt_loss(self, input_ids, pred_logits, attention_mask=None):
        """
            Input:
                label: torch.Tensor -> [bs, seq_len,]
                pred: torch.Tensor -> [bs, seq_len, vocab_size]
                token_masks: torch.Tensor -> [bs, seq_len,]
            Output:
                loss: torch.Tensor
        """
        # cross-entropy
        pred_logits = pred_logits[:, :-1, :].flatten(0, 1)
        input_ids = input_ids[:, 1:].flatten()
        loss = F.cross_entropy(pred_logits, input_ids, reduction="none")
        
        # keep the loss of non-padding tokens and normalize the loss.
        if attention_mask is not None:
            masks_flatten = attention_mask[:, 1:].flatten()
            loss = (loss * masks_flatten).sum() / masks_flatten.sum()
        else:
            loss = loss.mean()

        return loss


if __name__ == '__main__':
    import torch
    from thop import profile

    # model config
    class ModelConfig:
        vit_img_dim: int = 3
        vit_embed_dim: int = 192
        vit_patch_size: int = 16
        vit_img_size: int = 256   # num_vision_tokens = (img_size // patch_size) ** 2
        vit_drop_path_rate: float = 0.0
        vit_num_blocks: int = 12
        vit_is_causal: bool = True
        vit_norm_pix_loss: bool = True

        lm_embed_dim: int = 192
        lm_num_blocks: int = 12
        lm_max_length: int = 288
        lm_vocab_size: float = 130000
        lm_rope_theta: int = 10000

    cfg = ModelConfig()

    # randomly create images
    images = torch.randn(1, 3, cfg.vit_img_size, cfg.vit_img_size)
    num_vtokens = (cfg.vit_img_size // cfg.vit_patch_size) ** 2
    prefix_mask = torch.zeros([1, num_vtokens])
    prefix_mask[:, :4] = 1.0

    # randomly create token ids
    input_ids = torch.randint(low=0, high=400, size=[1, cfg.lm_max_length - num_vtokens],)

    # prepare an attention mask
    attention_mask = torch.zeros([1, cfg.lm_max_length])
    attention_mask[:, :264] = 1.0

    # create aimv2 model
    model = AIMv2(cfg)
    model.train()

    # inference
    outputs = model(
        images = images,
        image_prefix_masks = prefix_mask,
        input_ids = input_ids,
        attention_mask = attention_mask,
        compute_loss = True,
    )

    for k in outputs:
        print(k, outputs[k].item())

    # # compute FLOPs & Params
    # print('============ Params & FLOPs ============')
    # flops, params = profile(model, inputs=(images, prefix_mask, input_ids, attention_mask), verbose=False)
    # print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    # print('Params : {:.2f} M'.format(params / 1e6))