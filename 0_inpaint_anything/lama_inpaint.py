import os
from pickle import LIST
import sys
import numpy as np
import torch
from tqdm import tqdm
import yaml
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo

from utils import load_img_to_array, save_array_to_img
import math


@torch.no_grad()
def inpaint_img_with_lama(
        img: np.ndarray,
        masks: list or tuple or np.ndarray,
        config_p: str,
        ckpt_p: str,
        mod=8,
        device="cuda",
):
    ##### start config models #####
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    # device = torch.device(predict_config.device)
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)
    out_key = predict_config.out_key
    ##### end config models #####

    ##### start infer models #####
    if isinstance(masks, np.ndarray):
        masks = [masks]
    # print("Total {} masks".format(len(masks)))
    total_cur_res = []
    img = torch.from_numpy(img).float().div(255.)
    for mask in masks:
        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255
        mask = torch.from_numpy(mask).float()

        batch = {}
        batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
        batch['mask'] = mask[None, None]
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
        batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1

        batch = model(batch)
        cur_res = batch[out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        # return cur_res
        total_cur_res.append(cur_res)
    ##### end infer models #####
    return total_cur_res


def build_lama_model(        
        config_p: str,
        ckpt_p: str,
        device="cuda",
        DDP=False
):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(train_config, checkpoint_path, strict=False)
    model.to(device)
    if not DDP:
        model.freeze()
    # model.eval()
    out_key = predict_config.out_key
    return model, out_key


@torch.no_grad()
def inpaint_img_with_builded_lama(
        model,
        out_key,
        img: np.ndarray,
        masks: np.ndarray,
        config_p=None,
        mod=8,
        device="cuda"
):
    if isinstance(masks, np.ndarray):
        masks = [masks]

    img = torch.from_numpy(img).float().div(255.)
    total_cur_res = []
    for mask in masks:
        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255
        mask = torch.from_numpy(mask).float()

        batch = {}
        batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
        batch['mask'] = mask[None, None]
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
        batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1

        batch = model(batch)
        cur_res = batch[out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        total_cur_res.append(cur_res)
    return total_cur_res

@torch.no_grad()
def inpaint_batch_img_with_builded_lama(
        model,
        out_key,
        batch: dict,
        split_sizes: list,
        ori_img_sizes: list,
        config_p=None,
        mod=8,
        device="cuda",
        max_batch_size=32,
):
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1
    
    #### model inference ####
    if batch['image'].shape[0] > max_batch_size:
        part_size = math.ceil(batch['image'].shape[0] / max_batch_size)
        batch_list = []
        for i in range(part_size):
            start_idx = i * max_batch_size
            end_idx = min((i + 1) * max_batch_size, batch['image'].shape[0])
            batch_list.append(model(dict(image=batch['image'][start_idx:end_idx], mask=batch['mask'][start_idx:end_idx])))
        outputs = torch.cat([batch[out_key] for batch in batch_list])
        del batch_list
    else:
        batch = model(batch)
        outputs = batch[out_key]
        del batch

    #### postprocess ####
    outputs = torch.split(outputs, split_sizes)

    img_inpainted_lists = []
    for output, o_size in zip(outputs, ori_img_sizes):
        if len(output) == 0:
            img_inpainted_lists.append(output)
        else:
            orig_height, orig_width = o_size
            output = output.permute(0, 2, 3, 1)
            output = output[:, :orig_height, :orig_width, :]
            output = output.detach().cpu().numpy()
            output = np.clip(output * 255, 0, 255).astype('uint8')
            img_inpainted_lists.append(output)

    return img_inpainted_lists

def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--input_mask_glob", type=str, required=True,
        help="Glob to input masks",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )


if __name__ == "__main__":
    """Example usage:
    python lama_inpaint.py \
        --input_img FA_demo/FA1_dog.png \
        --input_mask_glob "results/FA1_dog/mask*.png" \
        --output_dir results \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_stem = Path(args.input_img).stem
    mask_ps = sorted(glob.glob(args.input_mask_glob))
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_img_to_array(args.input_img)
    for mask_p in mask_ps:
        mask = load_img_to_array(mask_p)
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted = inpaint_img_with_lama(
            img, mask, args.lama_config, args.lama_ckpt, device=device)
        save_array_to_img(img_inpainted, img_inpainted_p)