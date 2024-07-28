from symbol import annassign
import os
from tkinter import PhotoImage
import torch
import sys
import tqdm
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from traitlets import default

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama, inpaint_img_with_builded_lama, build_lama_model
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

from mmcv import Config, DictAction
from mmdet.datasets import build_dataset
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.lvis import LVISV1Dataset

from tqdm import tqdm
import time
import pycocotools

def setup_args(parser):
    parser.add_argument('--config', help='train config file path', 
                        default='./configs/datasets/coco_instance_infer.py',
                        )
    parser.add_argument(
        "--input_img", type=str, required=False,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str, required=False,
        default="key_in", choices=["click", "key_in", "box_in"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=False,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=False,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--box_coords", type=float, nargs='+', required=False,
        help="The coordinate of the box prompt, [coord_X0 coord_Y0 coord_X1 coord_Y1].",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
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
    ### load existing dataset
    parser.add_argument(
        '--data_root', help='dataset root',
    )
    
def build_dataset(info):
    if info.type == "CocoDataset":
        datasets = CocoDataset(
            ann_file=info.ann_file,
            pipeline=info.pipeline,
            img_prefix=info.img_prefix,
        )
    elif info.type == "LVISV1Dataset":
        datasets = LVISV1Dataset(
            ann_file=info.ann_file,
            pipeline=info.pipeline,
            img_prefix=info.img_prefix,
        )
    
    return datasets

if __name__ == "__main__":
    """Example usage:
    python remove_anything_with_GTbox.py \
        --config ./configs/datasets/coco_instance_infer.py \
        --point_labels 1 \
        --dilate_kernel_size 10 \
        --output_dir ./results/coco/ \
        --sam_model_type vit_h \
        --sam_ckpt ../pretrained/sam_vit_h_4b8939.pth \
        --lama_config ./lama/configs/prediction/default.yaml \
        --lama_ckpt ../pretrained/big-lama \
    """

    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg = Config.fromfile(args.config)
    
    #### build dataset
    if args.data_root is not None:
        cfg.data_root = args.data_root
    datasets = build_dataset(cfg.data.train)
    
    #### load model
    lama_model, lama_out_key = build_lama_model(
        args.lama_config,
        args.lama_ckpt,
        device=device,
    )

    wrong_img_ids = []
    if cfg.data.train.type == "CocoDataset":
        print("Total {} images.".format(len(datasets.data_infos)))
        for im_idx, info in tqdm(enumerate(datasets.data_infos)):
            img_name = info['file_name']
            img_stem = Path(img_name).stem
            out_dir = Path(args.output_dir) / img_stem
            if os.path.exists(out_dir) and os.listdir(out_dir):
                continue

            img_id = info['id']
            img_height, img_width = info['height'], info['width']
            if not getattr(pycocotools, '__version__', '0') >= '12.0.2':
                anns_ids = datasets.coco.getAnnIds(imgIds=[img_id])
                anns = datasets.coco.loadAnns(anns_ids)
            else:
                anns_ids = datasets.coco.get_ann_ids(img_ids=[img_id])
                anns = datasets.coco.load_anns(anns_ids)
            anno = datasets._parse_ann_info(info, anns)

            bboxes = anno['bboxes']
            masks = anno['masks']
            labels = anno['labels']
            label_names = [datasets.CLASSES[l] for l in labels]
            img_path = os.path.join(cfg.data.train.img_prefix, img_name)
            img = load_img_to_array(img_path) # H, W, C

            masks = np.array([datasets.poly2mask(mask, img_height, img_width) for mask in masks])
            assert len(masks) == len(bboxes)

            masks = masks.astype(np.uint8) * 255
            # dilate mask to avoid unmasked edge effect
            if args.dilate_kernel_size is not None:
                masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

            out_dir.mkdir(parents=True, exist_ok=True)
            
            img_inpainted_lists = inpaint_img_with_builded_lama(
                    lama_model, lama_out_key, img, masks, mod=8, device=device
            )
            for idx, (img_inpainted, l_name) in enumerate(zip(img_inpainted_lists, label_names)):
                mask_p = out_dir / f"mask_{idx}_{l_name}.png"
                img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
                save_array_to_img(img_inpainted, img_inpainted_p)
                

    if cfg.data.train.type == "LVISV1Dataset":
        print("Total {} images.".format(len(datasets.data_infos)))
        instance_number = 0
        for im_idx, info in tqdm(enumerate(datasets.data_infos)):
            img_name = info['filename']
            img_stem = Path(img_name).stem
            out_dir = Path(args.output_dir) / img_stem
            if os.path.exists(out_dir) and os.listdir(out_dir):
                continue

            img_id = info['id']
            img_height, img_width = info['height'], info['width']
            anns_ids = datasets.coco.get_ann_ids(img_ids=[img_id])
            instance_number += len(anns_ids)
            
            anns = datasets.coco.load_anns(anns_ids)
            anno = datasets._parse_ann_info(info, anns)

            bboxes = anno['bboxes']
            masks = anno['masks']
            labels = anno['labels']
            label_names = [datasets.CLASSES[l] for l in labels]
            img_path = os.path.join(cfg.data.train.img_prefix, img_name)
            img = load_img_to_array(img_path) # H, W, C
            
            masks = np.array([datasets.poly2mask(mask, img_height, img_width) for mask in masks])
            assert len(masks) == len(bboxes)

            masks = masks.astype(np.uint8) * 255
            # dilate mask to avoid unmasked edge effect
            if args.dilate_kernel_size is not None:
                masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

            # visualize the segmentation results
            out_dir.mkdir(parents=True, exist_ok=True)

            img_inpainted_lists = inpaint_img_with_builded_lama(
                    lama_model, lama_out_key, img, masks, mod=8, device=device
            )
            for idx, (img_inpainted, l_name) in enumerate(zip(img_inpainted_lists, label_names)):
                mask_p = out_dir / f"mask_{idx}_{l_name}.png"
                img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
                save_array_to_img(img_inpainted, img_inpainted_p)

        print(f"Total instance number: {instance_number}")
