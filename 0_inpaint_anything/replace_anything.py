import cv2
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List
from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import replace_img_with_sd
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict, annotate, load_image_fast


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in", "box_in", "grounding"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--box_coords", type=float, nargs='+', required=False,
        help="The coordinate of the box prompt, [coord_X0 coord_Y0 coord_X1 coord_Y1].",
    )
    parser.add_argument(
        "--text_prompt", type=str, required=True,
        help="Text prompt",
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
        "--seed", type=int,
        help="Specify seed for reproducibility.",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic algorithms for reproducibility.",
    )
    parser.add_argument(
        "--model_config", type=str, 
        help="if coords_type is gronding, load the model config file.",
        default="grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    )
    parser.add_argument(
        "--model_path", type=str, 
        help="if coords_type is gronding, load the model path.",
        default="/root/paddlejob/workspace/env_run/output/zhangxinyu14/baidu/personal-code/pretrained/groundingdino_swinb_cogcoor.pth",
    )

if __name__ == "__main__":
    """Example usage:
    python replace_anything.py \
        --input_img ./example/replace-anything/dog.png \
        --coords_type key_in \
        --point_coords 750 500 \
        --point_labels 1 \
        --text_prompt "sit on the swing" \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.coords_type == "click":
        latest_coords = get_clicked_point(args.input_img)
    elif args.coords_type == "key_in":
        latest_coords = args.point_coords
    elif args.coords_type == "box_in":
        latest_coords = args.box_coords
        latest_coords = [latest_coords]
    elif args.coords_type == "grounding":
        latest_coords = None
        model = load_model(args.model_config, args.model_path)

    import pdb; pdb.set_trace()
    if args.coords_type in ["click", "key_in", "box_in"]:
        img = load_img_to_array(args.input_img)
    elif args.coords_type == "grounding":
        image_source, img_transformed = load_image_fast(args.input_img)
        img = np.array(image_source)
        TEXT_PROMPT = "jacket ."
        BOX_TRESHOLD = 0.3 # default: 0.35
        TEXT_TRESHOLD = 0.25
        boxes, logits, phrases = predict(
            model=model,
            image=img_transformed,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        latest_coords = boxes

    masks, _, _ = predict_masks_with_sam(
        img,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
        point_coords = [latest_coords] if args.coords_type in ["click", "key_in"] else None,
        point_labels=args.point_labels,
        box = latest_coords if args.coords_type in ["box_in", "grounding"] else None,
    )
    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    ### this is for visualization
    annotated_frame = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(out_dir / f"grounding.png", annotated_frame)
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [latest_coords], args.point_labels,
                    size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    import pdb; pdb.set_trace()
    # fill the masked image
    for idx, mask in enumerate(masks):
        if args.seed is not None:
            torch.manual_seed(args.seed)
        mask_p = out_dir / f"mask_{idx}.png"
        img_replaced_p = out_dir / f"replaced_with_{Path(mask_p).name}"
        img_replaced = replace_img_with_sd(
            img, mask, args.text_prompt, device=device)
        save_array_to_img(img_replaced, img_replaced_p)
