import torch.nn as nn

try:
    from .common import AttentionPoolingClassifier
    from .vit    import ImageEncoderViT
except:
    from  common import AttentionPoolingClassifier
    from  vit    import ImageEncoderViT


class ViTForImageClassification(nn.Module):
    def __init__(self,
                 image_encoder :ImageEncoderViT,
                 num_classes   :int   = 1000,
                 qkv_bias      :bool  = True,
                 ):
        super().__init__()
        # -------- Model parameters --------
        self.encoder    = image_encoder
        self.classifier = AttentionPoolingClassifier(
            image_encoder.patch_embed_dim, num_classes, image_encoder.num_heads, qkv_bias, num_queries=1)

    def forward(self, x):
        """
        Inputs:
            x: (torch.Tensor) -> [B, C, H, W]. Input image.
        """
        x = self.encoder(x)
        x, x_cls = self.classifier(x)

        return x


if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, c, h, w = 2, 3, 224, 224
    x = torch.randn(bs, c, h, w)
    patch_size = 16

    # Build model
    encoder = ImageEncoderViT(img_size=h,
                               patch_size=patch_size,
                               in_dim=c,
                               patch_embed_dim=192,
                               num_layers=12,
                               num_heads=3,
                               mlp_ratio=4.0,
                               qkv_bias=False,
                               dropout=0.1)
    model = ViTForImageClassification(encoder, num_classes=1000, qkv_bias=True)

    # Inference
    outputs = model(x)

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
