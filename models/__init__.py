import torch

from .aim.build import build_aim, build_aim_cls


def build_model(args, is_train=False):
    # --------------------------- AIM series ---------------------------
    if   args.model in ['aim_nano', 'aim_tiny', 'aim_base', 'aim_large', 'aim_huge']:
        model = build_aim(args, is_train)

    elif args.model in ['aim_cls_nano', 'aim_cls_tiny', 'aim_cls_base', 'aim_cls_large', 'aim_cls_huge']:
        model = build_aim_cls(args)


    if args.resume and args.resume.lower() != 'none':
        print('loading trained weight for <{}> from <{}>: '.format(args.model, args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)

    return model
