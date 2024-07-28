import json
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
import torch.distributed as dist
import torch.multiprocessing as mp
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

def load_model(config_path, ckpt, device, vae_ckpt=None):
    device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt, vae_ckpt)
    model.eval().to(device)
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    return model_wrap, model_wrap_cfg

def random_sample(freq_file, no_freq):
    freq = freq_file["freq"]
    valid_label = freq_file["valid_name"]
    # print(valid_label)
    # print(freq)
    if no_freq:
        label = random.choices(valid_label, k=1)[0]
    else:
        label = random.choices(valid_label, weights=freq, k=1)[0]
    return label


def add_margin(pil_img, left, top, right, bottom, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def main_worker(gpu, args):
    print(args.output)
    os.makedirs(args.output, exist_ok=True)
    exits_dir = os.listdir(args.output)

    rank = args.nr * args.gpus + gpu
    print("rank: {}".format(rank))
    print("world_size: {}".format(args.world_size))
    if args.dist_type == 'tcp':
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=rank)
    else:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    print("gpu: {}".format(gpu))
    torch.cuda.set_device(gpu)
    print(f"ngpu: {rank}/{args.world_size}")

    config = OmegaConf.load(args.config)

    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    if args.is_lvis:
        with open("data/json/lvis_image_label_freq.json") as f:
            freq_file_ = json.load(f)
        freq_file = []
        for k in args.lvis_label_selection:
            print('load', k)
            freq_file += freq_file_[k]
    else:
        with open("data/json/coco_image_label_freq.json") as f:
            freq_file = json.load(f)

    file = sorted(os.listdir(args.input))
    
    sequence = np.arange(len(file))
    sequence = sequence[sequence % args.world_size == rank]
    for idx in tqdm(sequence, ncols=80):
        fn = file[idx]
        image_id_str = fn[0]
        seed = random.randint(0, 100000) if args.seed is None else args.seed

        input_dir = os.path.join(args.input, fn)

        try:
            input_image = Image.open(input_dir).convert("RGB")
        except:
            continue

        if args.multi:
            num = random.choice(['one', 'two', 'three', 'four', 'five'])
            edit_template = args.edit_template.replace(' a ', f" {num} ")
        else:
            edit_template = args.edit_template
            
        image_id = str(int(image_id_str))  # 000000001355 -> 1335
        if args.is_lvis:
            label = random.choices(freq_file, k=1)[0]['name']
            label = label.replace('_', ' ')
        else:
            if image_id in freq_file:
                if args.no_superlabel:
                    label = random_sample(freq_file["all"], args.no_freq)
                else:
                    label = random_sample(freq_file[image_id], args.no_freq)
                # print(edit, fn)
            else:
                label = random_sample(freq_file["all"], args.no_freq)
                # input_image.save(os.path.join(args.output, fn))
                # continue
        edit = edit_template.replace("<obj>", label)

        print(edit, label)
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


        g_list = [np.array(input_image)]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)

        seed = random.randint(0, 100000) if args.seed is None else args.seed
        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            cond = {}
            cond["c_crossattn"] = [model.get_learned_conditioning([edit])]
            cond["c_concat"] = [model.encode_first_stage(input_image.clone()).mode()]

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
            edited_image = x.type(torch.uint8).cpu().numpy()
            g_list.append(edited_image)

        edited_image = Image.fromarray(edited_image)

        if args.process == 'crop':
            pass
        elif args.process == 'resize':
            edited_image = edited_image.resize((ori_width, ori_height), Image.Resampling.LANCZOS)
        elif args.process == 'pad':
            edited_image = edited_image.crop((0, 0, scale_width, scale_height))
            edited_image = edited_image.resize((ori_width, ori_height), Image.Resampling.LANCZOS)
        else:
            raise NotImplementedError

        os.makedirs(os.path.join(args.output, image_id_str), exist_ok=True)
        edited_image.save(os.path.join(args.output, image_id_str, f'{label}.jpg'))

def main():
    parser = ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--addr', default='127.0.0.1', type=str)
    parser.add_argument('--port', default=8888, type=int)
    parser.add_argument('--dist_type', default='tcp', type=str, choices=['tcp', 'env'])
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--edit_template", type=str, default="add a <obj>")
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--process", type=str, default='crop')
    parser.add_argument("--multi", action='store_true', default=False)
    parser.add_argument("--is_lvis", action='store_true', default=False)
    parser.add_argument("--lvis_label_selection", nargs="+", type=str, default=['r'])
    parser.add_argument("--no_superlabel", action='store_true', default=False)
    parser.add_argument("--no_freq", action='store_true', default=False)

    args = parser.parse_args()

    root = '../pretrained/'
    if not os.path.exists(args.ckpt):
        print('Downloading Models...')
        os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
        shutil.copyfile(f'{root}/{args.ckpt}', args.ckpt)
        print('Finished!')

    args.gpus = torch.cuda.device_count()
    print(args.gpus)
    args.world_size = args.gpus * args.nodes
    # multiprocess
    if args.dist_type == 'tcp':
        args.dist_url = f'tcp://{args.addr}:{args.port}'
    else:
        os.environ['MASTER_ADDR'] = args.addr
        os.environ['MASTER_PORT'] = str(args.port)
    if args.gpus == 1:
        main_worker(gpu=0, args=args)
    else:
        torch.multiprocessing.set_start_method('spawn')
        mp.spawn(main_worker, nprocs=args.gpus, args=(args,))


if __name__ == "__main__":
    main()
