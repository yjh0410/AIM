import os
import torch

from .vit import build_vit
from .aimv1 import build_aimv1
from .vit_cls import ViTForImageClassification


def build_model(args, model_type='default'):
    # ----------- Masked Image Modeling task -----------
    if model_type == 'aim':
        print(" - Build AIMv1 model ... ")
        return build_aimv1(args.model,
                             args.img_size,
                             args.patch_size,
                             args.img_dim,
                             args.norm_pix_loss,
                             )
   
    # ----------- Vision Transformer -----------
    model = build_vit(args.model, args.img_size, args.patch_size, args.img_dim)
    load_pretrained(model, args.pretrained)

    if model_type == 'cls':
        print(" - Build ViT based classifier ... ")
        model = ViTForImageClassification(model, num_classes=args.num_classes)
            
    return model

def load_pretrained(model, ckpt=None):
    if ckpt is not None:
        # check path
        if not os.path.exists(ckpt):
            print(" - No pretrained model.")
            return model
        print('- Loading pretrained from: {}'.format(ckpt))
        checkpoint = torch.load(ckpt, map_location='cpu')
        
        # checkpoint state dict
        encoder_state_dict = checkpoint.pop("pretrained_encoder")
        for k in encoder_state_dict.keys():
            print(' - ', k)

        # load encoder weight into ViT's encoder
        model.load_state_dict(encoder_state_dict)
