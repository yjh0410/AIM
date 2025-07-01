def build_config(model_name):
    print('\n =============== Create config for model: {} =============== '.format(model_name))
    # ----------- AimV2 -----------
    if model_name == "aimv2_tiny":
        return AimV2TinyConfig()

    if model_name == "aimv2_small":
        return AimV2SmallConfig()

    if model_name == "aimv2_base":
        return AimV2BaseConfig()

    if model_name == "aimv2_large":
        return AimV2LargeConfig()

    if model_name == "aimv2_huge":
        return AimV2HugeConfig()

    # ----------- Vision Transformer -----------
    if model_name == "vit_tiny":
        return ViTTinyConfig()

    if model_name == "vit_small":
        return ViTSmallConfig()

    if model_name == "vit_base":
        return ViTBaseConfig()

    if model_name == "vit_large":
        return ViTLargeConfig()

    if model_name == "vit_huge":
        return ViTHugeConfig()


# ------------ AimV2 model config ------------
class AimV2TinyConfig:
    vit_img_dim: int = 3
    vit_img_size: int = 256   # num_vision_tokens = (img_size // patch_size) ** 2
    vit_patch_size: int = 16
    vit_embed_dim: int = 192
    vit_num_blocks: int = 12
    vit_is_causal: bool = True
    vit_drop_path_rate: float = 0.0
    vit_norm_pix_loss: bool = True

    lm_embed_dim: int = 192
    lm_num_blocks: int = 12
    lm_max_length: int = 288
    lm_vocab_size: float = 130000
    lm_rope_theta: int = 10000

class AimV2SmallConfig:
    vit_img_dim: int = 3
    vit_img_size: int = 256   # num_vision_tokens = (img_size // patch_size) ** 2
    vit_patch_size: int = 16
    vit_embed_dim: int = 384
    vit_num_blocks: int = 12
    vit_is_causal: bool = True
    vit_drop_path_rate: float = 0.0
    vit_norm_pix_loss: bool = True

    lm_embed_dim: int = 384
    lm_num_blocks: int = 12
    lm_max_length: int = 288
    lm_vocab_size: float = 130000
    lm_rope_theta: int = 10000

class AimV2BaseConfig:
    vit_img_dim: int = 3
    vit_img_size: int = 256   # num_vision_tokens = (img_size // patch_size) ** 2
    vit_patch_size: int = 16
    vit_embed_dim: int = 768
    vit_num_blocks: int = 12
    vit_is_causal: bool = True
    vit_drop_path_rate: float = 0.0
    vit_norm_pix_loss: bool = True

    lm_embed_dim: int = 768
    lm_num_blocks: int = 12
    lm_max_length: int = 288
    lm_vocab_size: float = 130000
    lm_rope_theta: int = 10000

class AimV2LargeConfig:
    vit_img_dim: int = 3
    vit_img_size: int = 256   # num_vision_tokens = (img_size // patch_size) ** 2
    vit_patch_size: int = 16
    vit_embed_dim: int = 1024
    vit_num_blocks: int = 24
    vit_is_causal: bool = True
    vit_drop_path_rate: float = 0.0
    vit_norm_pix_loss: bool = True

    lm_embed_dim: int = 1024
    lm_num_blocks: int = 24
    lm_max_length: int = 288
    lm_vocab_size: float = 130000
    lm_rope_theta: int = 10000

class AimV2HugeConfig:
    vit_img_dim: int = 3
    vit_img_size: int = 256   # num_vision_tokens = (img_size // patch_size) ** 2
    vit_patch_size: int = 16
    vit_embed_dim: int = 1280
    vit_num_blocks: int = 32
    vit_is_causal: bool = True
    vit_drop_path_rate: float = 0.0
    vit_norm_pix_loss: bool = True

    lm_embed_dim: int = 1280
    lm_num_blocks: int = 32
    lm_max_length: int = 288
    lm_vocab_size: float = 130000
    lm_rope_theta: int = 10000

# ------------ ViT model config ------------
class ViTTinyConfig:
    vit_img_dim: int = 3
    vit_img_size: int = 256   # num_vision_tokens = (img_size // patch_size) ** 2
    vit_patch_size: int = 16
    vit_embed_dim: int = 192
    vit_num_blocks: int = 12
    vit_is_causal: bool = False
    vit_drop_path_rate: float = 0.0

class ViTSmallConfig:
    vit_img_dim: int = 3
    vit_img_size: int = 256   # num_vision_tokens = (img_size // patch_size) ** 2
    vit_patch_size: int = 16
    vit_embed_dim: int = 384
    vit_num_blocks: int = 12
    vit_is_causal: bool = False
    vit_drop_path_rate: float = 0.0

class ViTBaseConfig:
    vit_img_dim: int = 3
    vit_img_size: int = 256   # num_vision_tokens = (img_size // patch_size) ** 2
    vit_patch_size: int = 16
    vit_embed_dim: int = 768
    vit_num_blocks: int = 12
    vit_is_causal: bool = False
    vit_drop_path_rate: float = 0.0

class ViTLargeConfig:
    vit_img_dim: int = 3
    vit_img_size: int = 256   # num_vision_tokens = (img_size // patch_size) ** 2
    vit_patch_size: int = 16
    vit_embed_dim: int = 1024
    vit_num_blocks: int = 24
    vit_is_causal: bool = False
    vit_drop_path_rate: float = 0.0

class ViTHugeConfig:
    vit_img_dim: int = 3
    vit_img_size: int = 256   # num_vision_tokens = (img_size // patch_size) ** 2
    vit_patch_size: int = 16
    vit_embed_dim: int = 1280
    vit_num_blocks: int = 32
    vit_is_causal: bool = False
    vit_drop_path_rate: float = 0.0
