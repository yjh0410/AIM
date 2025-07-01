import os
import torch

from .vision_transformer import build_vision_transformer
from .aimv2 import AIMv2
from .vit_cls import ViTForImageClassification


def build_model(args, model_cfg, model_type='default'):
    # ----------- Masked Image Modeling task -----------
    if model_type == 'aim':
        print(" - Build AIMv2 model ... ")
        return AIMv2(model_cfg)
   
    # ----------- Vision Transformer -----------
    model = build_vision_transformer(model_cfg)
    load_pretrained(model, args.pretrained)

    if model_type == 'cls':
        print(" - Build ViT based classifier ... ")
        model = ViTForImageClassification(model, num_classes=args.num_classes)
            
    return model

def load_pretrained(model, ckpt=None):
    if ckpt is not None:
        # check path
        if not os.path.exists(ckpt):
            print(" => Not found the pretrained model.")
            return model
        print(' => Loading the pretrained from: {}'.format(ckpt))
        checkpoint = torch.load(ckpt, map_location='cpu')
        
        # checkpoint state dict
        encoder_state_dict = checkpoint.pop("pretrained_encoder")
        for k in encoder_state_dict.keys():
            print(' - ', k)

        # load encoder weight into ViT's encoder
        model.load_state_dict(encoder_state_dict)
