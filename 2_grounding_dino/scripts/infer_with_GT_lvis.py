import json
from groundingdino.util.inference import load_model, load_image, predict, annotate, load_image_fast
import cv2
import os, sys
from tqdm import tqdm
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.lvis import LVISV1Dataset
from torchvision.ops import box_convert, box_iou, nms
import torch
import math
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.bbs import BoundingBox
import copy
import pycocotools
import pickle
from show import draw_box, draw_mask, draw_text
import argparse

def resizeImageAndBoundingBoxes(bboxes, inputW, inputH, targetImgW, targetImgH):
    #### Method 1: directly resize the image and bounding boxes
    bboxes[:, 0] = bboxes[:, 0] * targetImgW / inputW
    bboxes[:, 1] = bboxes[:, 1] * targetImgH / inputH
    bboxes[:, 2] = bboxes[:, 2] * targetImgW / inputW
    bboxes[:, 3] = bboxes[:, 3] * targetImgH / inputH
    return bboxes

def get_sd_crop_size(GT_boxes, img_w, img_h, target_resolution):
    factor = target_resolution / max(img_w, img_h)
    factor = math.ceil(min(img_w, img_h) * factor / 64) * 64 / min(img_w, img_h)
    gen_img_w = int((img_w * factor) // 64) * 64
    gen_img_h = int((img_h * factor) // 64) * 64
    # calculate the aspect ratio of the live_size
    live_size_ratio = img_w / img_h
    # calculate the aspect ratio of the output image
    gen_output_ratio = gen_img_w / gen_img_h
    # figure out if the sides or top/bottom will be cropped off
    if live_size_ratio == gen_output_ratio:
        # live_size is already the needed ratio
        crop_width = img_w
        crop_height = img_h
    elif live_size_ratio >= gen_output_ratio:
        # live_size is wider than what's needed, crop the sides
        crop_width = gen_output_ratio * img_h
        crop_height = img_h
    else:
        # live_size is taller than what's needed, crop the top and bottom
        crop_width = img_w
        crop_height = img_w / gen_output_ratio
        
    # make the crop
    crop_left = (img_w - crop_width) * 0.5
    crop_top = (img_h - crop_height) * 0.5
    GT_boxes[:, 0] = GT_boxes[:, 0] - crop_left
    GT_boxes[:, 1] = GT_boxes[:, 1] - crop_top
    GT_boxes[:, 2] = GT_boxes[:, 2] - crop_left
    GT_boxes[:, 3] = GT_boxes[:, 3] - crop_top
    crop = (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
    return crop, GT_boxes

def get_resize_box(GT_boxes, in_w, in_h, out_w, out_h):
    outNewBBoxes = resizeImageAndBoundingBoxes(GT_boxes, int(in_w), int(in_h), out_w, out_h)
    if isinstance(outNewBBoxes, np.ndarray):
        return outNewBBoxes
    resize_GT_boxes = []
    for box in outNewBBoxes:
        resize_GT_boxes.append([box.x1, box.y1, box.x2, box.y2])
    resize_GT_boxes = np.array(resize_GT_boxes)
    return resize_GT_boxes

def get_project_box(GT_boxes, img_path, img_w, img_h):
    crop, GT_boxes = get_sd_crop_size(GT_boxes, img_w, img_h, RESOLUTION)
    GT_boxes = get_resize_box(GT_boxes, crop[2]-crop[0], crop[3]-crop[1], img_w, img_h)
    return GT_boxes

def setup_args(parser):
    parser.add_argument('--dataset_name', help='dataset name for infer grounding dino', 
                        default='coco_gen_multi',
                        )
    parser.add_argument(
        '--data_root', help='dataset root', default='/root/paddlejob/workspace/datasets/COCO/',
        )
    parser.add_argument(
        '--save_path', help='dataset root', default=None,
        )
    parser.add_argument(
        '--gen_data_root', help='generated dataset root', default='/root/paddlejob/workspace/datasets/',
        )
    parser.add_argument(
        '--model_config', help='config for pretrained model', default='groundingdino/config/GroundingDINO_SwinB_cfg.py',
        )
    parser.add_argument(
        '--model_path', help='path for pretrained model', default='../../pretrained/groundingdino_swinb_cogcoor.pth',
        )
    parser.add_argument(
            "--image_resolution", type=int, 
            help="image resolution for inference", default=512,
        )
    parser.add_argument(
            "--box_threshold", type=float, 
            help="box_threshold for detection", default=0.3,
        )
    parser.add_argument(
            "--text_threshold", type=float, 
            help="text_threshold for detection", default=0.25,
        )
    parser.add_argument(
            "--start_index", type=int, 
            help="start index for detection", default=0,
        )
    parser.add_argument(
            "--end_index", type=int, 
            help="end index for detection", default=-1,
        )
    parser.add_argument("--save_annotations", action='store_true', help="whether save annotations or not")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NAME = args.dataset_name
    data_root = args.data_root
    gen_data_root = os.path.join(args.gen_data_root, NAME)
    save_path = gen_data_root.replace("{}".format(NAME), "{}_visualization_all".format(NAME)) if args.save_path is None else args.save_path
    os.makedirs(save_path, exist_ok=True)
    model_config = args.model_config
    model_path = args.model_path
    RESOLUTION = args.image_resolution
    BOX_THRESHOLD = args.box_threshold # default: 0.35
    TEXT_THRESHOLD = args.text_threshold
    START = args.start_index
    END = args.end_index
    SAVE_ANNOTATIONS = args.save_annotations

    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    ]
    
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1333, 800),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    datasets = LVISV1Dataset(
        ann_file=data_root + 'annotations/lvis_v1_train.json',
        pipeline=train_pipeline,
        img_prefix=data_root + 'train2017/',
    )
    
    model = load_model(model_config, model_path)
    
    ##### get generated image names #####
    gen_img_lists = os.listdir(os.path.join(gen_data_root, 'train2017/'))
    print("total {} images in gen_img_lists".format(len(gen_img_lists)))
    
    new_jsons = {}
    new_jsons['info'] = datasets.coco.dataset['info']
    new_jsons['licenses'] = datasets.coco.dataset['licenses']
    new_jsons['images'] = []
    new_jsons['annotations'] = []
    new_jsons['categories'] = datasets.coco.dataset['categories']

    ##### get new annotations #####
    start = START
    end = len(datasets.data_infos) if END == -1 else END
    
    MAX_IMG_ID = max(datasets.img_ids)
    ADDED_IMG_ID = MAX_IMG_ID + 100 # if new added images, need to add a new id

    TOTAL_CLASSES = list(datasets.CLASSES)
    for im_idx, info in tqdm(enumerate(datasets.data_infos[start:end])):
        if 'file_name' in info:
            img_name = info['file_name']
        else:
            img_name = info['filename']
        img_name = os.path.basename(img_name)
        img_id = info['id']

        anns_ids = datasets.coco.get_ann_ids(img_ids=[img_id]) # 该image id下的annotation id
        anns = datasets.coco.load_anns(anns_ids) # 得到annotation id对应下的annotation # datasets.coco.anns[anns_ids[ann_idx]]
        anno = datasets._parse_ann_info(info, anns)
        if img_name[:-4] not in gen_img_lists:
            continue
        ##### step 1: get GT boxes and labels #####
        GT_boxes = copy.deepcopy(anno['bboxes']) ### x1, y1, x2, y2
        img_h, img_w = info['height'], info['width']
        GT_logits = torch.ones(GT_boxes.shape[0])
        labels = anno['labels']
        GT_label_names = [datasets.CLASSES[l] for l in labels]
        
        ##### step2: get boxes projected to the size of SD's output and labels #####
        ##### step 1.1. get GT boxes #####
        GT_boxes = get_project_box(GT_boxes, os.path.join(data_root + 'train2017/', img_name), img_w, img_h)        
        
        # clip box to min 0 and max img size
        GT_boxes[:, 0:4:2] = np.clip(GT_boxes[:, 0:4:2], 0.0, img_w)
        GT_boxes[:, 1:4:2] = np.clip(GT_boxes[:, 1:4:2], 0.0, img_h)
        GT_boxes = box_convert(boxes=torch.tensor(GT_boxes), in_fmt="xyxy", out_fmt="cxcywh") ### change to cx, cy, w, h
        GT_boxes = GT_boxes / torch.Tensor([img_w, img_h, img_w, img_h])
        
        ##### step 2: generate boxes for added obj #####
        gen_img_paths = os.listdir(os.path.join(gen_data_root, 'train2017/', img_name[:-4]))
        
        for gen_idx, gen_img_path in enumerate(gen_img_paths):
            if SAVE_ANNOTATIONS:
                if os.path.exists(os.path.join(save_path, 'annotations', img_name[:-4], gen_img_path.replace('.jpg', '.pickle'))):
                    continue
            ori_phrase = gen_img_path.split('.jpg')[0]
            TEXT_PROMPT = gen_img_path.split('.jpg')[0]
            TEXT_PROMPT = TEXT_PROMPT + ' .'

            image_source, image = load_image_fast(os.path.join(gen_data_root, 'train2017/', img_name[:-4], gen_img_path))
            new_img_w, new_img_h = image_source.size[:2] # Note that this is the Image format, therefore the size is (w, h)
            
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )

            # ### change box coordinates to original image size
            new_boxes = torch.cat((GT_boxes, boxes), dim=0)
            new_logits = torch.cat((GT_logits, logits), dim=0)
            new_phrases = GT_label_names + phrases
            
            print(img_name, TEXT_PROMPT, boxes, logits, phrases)
            
            if len(boxes) != 0:
                keep = nms(box_convert(boxes=new_boxes * torch.Tensor([new_img_w, new_img_h, new_img_w, new_img_h]), in_fmt="cxcywh", out_fmt="xyxy"), new_logits, iou_threshold=0.5)
                new_boxes = new_boxes[keep]
                new_logits = new_logits[keep]
                new_phrases = [new_phrases[k] for k in keep]
            
            category_ids = []
            for idx_, phr in enumerate(new_phrases):
                try:
                    phr_index = TOTAL_CLASSES.index(phr)
                except:
                    phr_index = TOTAL_CLASSES.index(ori_phrase.replace(' ', '_').replace(')_', ') '))
                    new_phrases[idx_] = ori_phrase.replace(' ', '_').replace(')_', ') ')
                category_ids.append([k for k, v in datasets.cat2label.items() if v == phr_index][0])

            if 'file_name' in info: info['file_name'] = os.path.join(img_name[:-4], gen_img_path)
            if 'filename' in info: info['filename'] = os.path.join(img_name[:-4], gen_img_path)
            info['height'], info['width'] = new_img_h, new_img_w
            new_jsons['images'].append(info)
            
            save_boxes = box_convert(boxes=new_boxes * torch.Tensor([new_img_w, new_img_h, new_img_w, new_img_h]), in_fmt="cxcywh", out_fmt="xywh")
            
            if gen_idx == 0:
                new_id = copy.deepcopy(img_id)
            else:
                new_id = copy.deepcopy(ADDED_IMG_ID)
                ADDED_IMG_ID += 1
            
            save_info = {
                'image_id': new_id,
                'category_ids': category_ids,
                'save_boxes': save_boxes,
                'new_logits': new_logits,
                "new_phrases": new_phrases,
            }
                        
            if SAVE_ANNOTATIONS:
                os.makedirs(os.path.join(save_path, 'annotations', img_name[:-4]), exist_ok=True)
                with open(os.path.join(save_path, 'annotations', img_name[:-4], gen_img_path.replace('.jpg', '.pickle')), 'wb') as handle:
                    pickle.dump(save_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

            category_ids, save_boxes, new_logits = save_info['category_ids'], save_info['save_boxes'], save_info['new_logits']
            for cate_id, box, logit_ in zip(category_ids, save_boxes, new_logits):
                new_ann_id = len(new_jsons['annotations'])
                new_ann_info = {
                    'area': (box[2] * box[3]).item(),
                    'iscrowd': 0,
                    'image_id': new_id,
                    'bbox': box.tolist(),
                    'logit': logit_.item(),
                    'category_id': cate_id, 
                    'id': new_ann_id,
                }
                new_jsons['annotations'].append(new_ann_info)  

    with open(os.path.join(save_path, 'add_sd_generated_image_instances_train2017_{}_{}_wScore.json'.format(start, end)), 'w') as f:
        f.write(json.dumps(new_jsons))
    print("Total increase {} images".format(ADDED_IMG_ID - 100 - MAX_IMG_ID))
    print("Finish!")
