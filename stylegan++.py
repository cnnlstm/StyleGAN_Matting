import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator
from compare import compare

def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to('cpu')
        .numpy()
        .squeeze(0)
    )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='stylegan2-ffhq-config-f.pt')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--w_lr', type=float, default=0.01)
    parser.add_argument('--n_lr', type=float, default=5)
    parser.add_argument('--w_step', type=int, default=5000)
    parser.add_argument('--n_step', type=int, default=3000)
    parser.add_argument('--is_face', action='store_true')
    parser.add_argument('files', metavar='FILES', nargs='+')

    args = parser.parse_args()
    n_mean_latent = 10000

    resize = args.size

    transform = transforms.Compose(
        [
            transforms.Resize((resize,resize)),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
        
    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    
    percept = lpips.PerceptualLoss(
            model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
        )

    for file_idx, imgfile in enumerate(args.files):
        with torch.no_grad():
            if args.is_face:
                noise_sample = torch.randn(n_mean_latent, 512, device=device)
                latent_out = g_ema.style(noise_sample)
                latent_in = latent_out.mean(0).detach().clone().unsqueeze(0).unsqueeze(0).repeat(1, g_ema.n_latent, 1).cuda()
            else:
                # uniform initialize
                latent_in = torch.FloatTensor(g_ema.n_latent, 512).unsqueeze(0).uniform_(-1, 1).cuda()
                
        noises = g_ema.make_noise()
        optimizer_w = optim.Adam([latent_in], lr=args.w_lr)
        optimizer_n = optim.Adam(noises, lr=args.n_lr)

        img = transform(Image.open(imgfile).convert('RGB')).unsqueeze(0).cuda()
        img_list = [img.detach().cpu()]
        
        w_pbar = tqdm(range(args.w_step))
        n_pbar = tqdm(range(args.n_step))
        for i in w_pbar:
            # fix n
            latent_in.requires_grad = True
            for noise in noises:
                noise.requires_grad = False
    
            img_gen, _ = g_ema([latent_in], input_is_latent=True, noise=noises)
            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256
                p_img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                p_img_gen = p_img_gen.mean([3, 5])
                p_img = img.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                p_img = p_img.mean([3, 5])


            p_loss = percept(p_img_gen, p_img).sum()
            mse_loss = F.mse_loss(img_gen, img)

            loss = 0.01 * p_loss + 1 * mse_loss

            optimizer_w.zero_grad()
            loss.backward()
            optimizer_w.step()
            w_pbar.set_description(
                (
                    f'perceptual: {p_loss.item():.4f}; mse: {mse_loss.item():.4f}'
                )
            )
        img_list.append(img_gen.detach().cpu())
        for i in n_pbar:
            # fix w
            latent_in.requires_grad = False
            for noise in noises:
                noise.requires_grad = True
    
            img_gen, _ = g_ema([latent_in], input_is_latent=True, noise=noises)

            mse_loss = F.mse_loss(img_gen, img)
            loss = 0.1 * mse_loss

            optimizer_n.zero_grad()
            loss.backward()
            optimizer_n.step()
            n_pbar.set_description(
                (
                    f'mse: {mse_loss.item():.4f}'
                )
            )
        img_name = os.path.basename(imgfile).split(".")[0]
        arr = {"latent_in": latent_in, "noises":noises}
        torch.save(arr, "dataset/testing/param/" + img_name +".pt")
            

