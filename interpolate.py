import os
import sys
import torch

from model import Generator
from torchvision.utils import save_image

if __name__ == "__main__":
    assert len(sys.argv) == 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load latents
    latent1 = torch.load(sys.argv[1]).to(device)
    latent2 = torch.load(sys.argv[2]).to(device)

    # Target_directory
    target_dir = sys.argv[3]
    os.makedirs(target_dir, exist_ok=True)

    # Load model
    g = Generator(1024, 512, 8, pretrained=True).to(device).train()
    for param in g.parameters():
        param.requires_grad = False

    steps = 90

    for t in range(steps):
        w = latent1 + (t / steps) * (latent2 - latent1)
        img, _ = g([w], input_is_latent=True)

        save_image(img, os.path.join(target_dir, '{}.png').format(str(t + 1).zfill(3)),
                   normalize=True, range=(-1, 1))
