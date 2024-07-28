from __future__ import annotations

import math
import os
import random
import shutil
import sys
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
from tqdm import tqdm

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model."):]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--edit", required=True, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--process", type=str, default='crop')
    args = parser.parse_args()

    root = '/root/paddlejob/workspace/env_run/datasets/afs3/yanglingfeng/Add-P2P-dev'
    if not os.path.exists(args.ckpt):
        print('Downloading Models...')
        os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
        shutil.copyfile(f'{root}/{args.ckpt}', args.ckpt)
        print('Finished!')

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    os.makedirs(args.output, exist_ok=True)
    shutil.copyfile(args.input, f"{args.output}/{os.path.basename(args.input)}")
    for i in tqdm(range(args.repeat), ncols=100):
        seed = random.randint(0, 100000) if args.seed is None else args.seed
        input_image = Image.open(args.input).convert("RGB")

        ori_width, ori_height = input_image.size
        factor = args.resolution / max(ori_width, ori_height)
        factor = math.ceil(min(ori_width, ori_height) * factor / 64) * 64 / min(ori_width, ori_height)
        width = int((ori_width * factor) // 64) * 64
        height = int((ori_height * factor) // 64) * 64

        if args.process == 'crop':
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
        elif args.process == 'resize':
            input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
        elif args.process == 'pad':
            real_ratio = min(width / ori_width, height / ori_height)
            scale_width = int(ori_width * real_ratio)
            scale_height = int(ori_height * real_ratio)
            bottom = height - scale_height
            right = width - scale_width
            input_image = input_image.resize((scale_width, scale_height), Image.Resampling.LANCZOS)
            input_image = add_margin(input_image, 0, 0, right, bottom, (124, 116, 104))
        else:
            raise NotImeplementedError

        if args.edit == "":
            input_image.save(args.output)
            return

        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            cond = {}
            cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
            cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = model_wrap.get_sigmas(args.steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": args.cfg_text,
                "image_cfg_scale": args.cfg_image,
            }
            torch.manual_seed(seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args, disable=True)
            x = model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")

            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            if args.process == 'crop':
                pass
            elif args.process == 'resize':
                edited_image = edited_image.resize((ori_width, ori_height), Image.Resampling.LANCZOS)
            elif args.process == 'pad':
                edited_image = edited_image.crop((0, 0, scale_width, scale_height))
                edited_image = edited_image.resize((ori_width, ori_height), Image.Resampling.LANCZOS)
            else:
                raise NotImplementedError

        basename = args.output.split('/')[-1]
        edited_image.save(f"{args.output}/{basename}_{i}.jpg")


if __name__ == "__main__":
    main()
