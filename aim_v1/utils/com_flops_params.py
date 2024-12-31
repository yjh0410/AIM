import torch
from thop import profile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def FLOPs_and_Params(model, img_size, patch_size, task_type="aim"):
    x = torch.randn(1, 3, img_size, img_size).to(device)
    num_patches = (img_size // patch_size) ** 2
    if task_type == "aim":
        prefix_mask = torch.zeros(1, num_patches).bool().to(device)
        prefix_mask[:, :3] = True
        model.eval()

        flops, params = profile(model, inputs=(x, prefix_mask))
    else:
        flops, params = profile(model, inputs=(x,))
        print('GFLOPs : ', flops / 1e9 * 2)
        print('Params : ', params / 1e6, ' M')

    model.train()


if __name__ == "__main__":
    pass
