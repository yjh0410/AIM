import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .common import LMBlock, RMSNorm, precompute_freqs_cis
except:
    from  common import LMBlock, RMSNorm, precompute_freqs_cis


# ----------------- Basic Modules -----------------
class LanguageModel(nn.Module):
    def __init__(self,
                 embed_dim: int = 512,
                 num_blocks: int = 24,
                 max_length: int = 2048,
                 vocab_size: int = 1024,
                 rope_theta: int = 10000,
                 ) -> None:
        super().__init__()
        # ----------- Basic parameters -----------
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.rope_theta = rope_theta

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            LMBlock(dim = embed_dim,
                    num_heads = embed_dim // 64,
                    mlp_ratio = 4.0,
                    qkv_bias  = True,
                    is_causal = True,
                    ) for _ in range(num_blocks)])
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.apply(self._init_weights)

        # RoPE
        num_heads = embed_dim // 64
        self.freqs_cis = precompute_freqs_cis(
            embed_dim // num_heads,
            max_length * 2,
            rope_theta,
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,
                input_ids: torch.Tensor = None,
                input_embeddings: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                ):
        # text embeddings
        if input_embeddings is None:
            tokens = self.token_embeddings(input_ids)  # [bs, seq, c]
        else:
            tokens = input_embeddings  # [bs, seq, c]

        # get RoPE
        freqs_cis = self.freqs_cis[:tokens.shape[1]].to(tokens.device)

        # Apply transformer block
        for block in self.blocks:
            tokens = block(tokens, freqs_cis, attention_mask)

        # final norm
        tokens = self.norm(tokens)

        # output
        logits = self.lm_head(tokens)

        outputs = {
            "last_hidden_state": tokens,
            "logits": logits,
        }

        return outputs


# ----------------- Model Functions -----------------
def build_language_model(model_cfg,):
    return LanguageModel(
        embed_dim  = model_cfg.lm_embed_dim,
        num_blocks = model_cfg.lm_num_blocks,
        max_length = model_cfg.lm_max_length,
        vocab_size = model_cfg.lm_vocab_size,
        rope_theta = model_cfg.lm_rope_theta,
    )


if __name__ == '__main__':
    import torch
    from thop import profile

    # model config
    class ModelConfig:
        lm_embed_dim: int = 192
        lm_num_blocks: int = 12
        lm_max_length: int = 512
        lm_vocab_size: float = 130000
        lm_rope_theta: int = 10000

    cfg = ModelConfig()

    # randomly create token ids
    token_ids = torch.randint(low=0, high=400, size=[1, 8],)
    attention_mask = torch.zeros([1, 8])
    attention_mask[:, :4] = 1.0

    # create model
    model = build_language_model(cfg)

    # Inference
    outputs = model(token_ids, attention_mask=attention_mask)

    # Compute FLOPs & Params
    print('============ Params & FLOPs ============')
    model.eval()
    token_ids = torch.randint(low=0, high=400, size=[1, 8],)
    flops, params = profile(model, inputs=(token_ids, None, attention_mask), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
