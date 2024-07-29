# Add-SD: Rational Generation without Manual Reference

This is the *official* repository with PyTorch implementation of [Add-SD: Rational Generation without Manual Reference](https://arxiv.org/pdf/).

☀️ If you find this work useful for your research, please kindly star our repo and cite our paper! ☀️

## Catalogue

- [1. Introduction](#1)
- [2. Method](#2)
- [3. Overall Pipeline](#3)
- [4. Main Results](#7)
- [5. References](#8)
- [6. Citation](#9)

<a name='1'></a>

## 1. Introduction

We propose Add-SD, a novel visual generation method for instruction-based object addition, demonstrating significant advancements in seamlessly integrating objects into realistic scenes using only textual instructions.


<a name='2'></a>

## 2. Mehtod
Add-SD consists of three essential stages to complete the object addition task: 
1. Creating image pairs by removing objects, 
2. Fine-tuning Add-SD, 
3. Generating synthetic data for downstream tasks.
   

<p align="center">
  <img src="assets/main_architecture.svg" alt="main_architecture.svg" style="width: 100%;"/>
  <br>
  <span style="display: block; text-align: center; font-size: 14px; color: #555;">Main architecture of Add-SD</span>
</p>



<a name='3'></a>

## 3. Overall Pipeline

### Step 0: Creating image pairs by removing objects

1) Follow the instructions in [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything) repository to install the necessary dependencies.

2) Download pretrained models, including [sam_vit_h_4b8939](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and [big-lama](https://drive.google.com/drive/folders/1pxea8PQ83Y9pBkCv2adk68BFskFlldZb?usp=drive_link), into the ```pretrained``` directory

3) Navigate to the 0_inpaint_anything directory and run the script to process COCO and LVIS data:

```shell
cd 0_inpaint_anything
sh script/remove_anything_with_GTbox.sh ### containing both COCO and LVIS
```

### Step 1: Fine-tuning Add-SD

1) Follow the installation instructions in the [instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix) repository.

2) Download pretrained model, [v1-5-pruned-emaonly.ckpt](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt), in the ```pretrained``` directory.

3) Download the required [JSON](https://drive.google.com/drive/folders/1AfwU843MJ9X1yI8VVGXgt89sN3w39OaA?usp=sharing) files and organize them as follows:

```
1_AddSD/data/
  ├── json/
   ├── seeds_coco_multi_vanilla.json
   ├── seeds_coco_multi_vanilla.json
   ├── seeds_lvis_vanilla.json
   ├── seeds_lvis_multi_vanilla.json
   ├── seeds_vg_vanilla.json
   └── seeds_vgcut_vanilla.json
```

4) (Optional) If you want to make your own datasets, conduct the following steps:

```shell
cd 1_AddSD
python utils/gen_train_data_annos.py
```

5) Train Add-SD

```shell
cd 1_AddSD
python run_train.sh
```

Make sure place the datasets, such as COCO, LVIS, VG, VGCUT, RefCOCO, RefCOCO+, and RefCOCOg, in the data directory with the following structure:

```
1_AddSD/data/
  ├── coco/
     ├── train2017/
     ├── val2017/
     ├── train2017_remove_image/  ## coco single object remove datasets
     ├── train2017_remove_image_multiobj/  ## coco multiple objects remove datasets
     ├── lvis_remove_image/  ## lvis single object remove datasets
     ├── lvis_remove_image_multiobj/  ## lvis multiple objects remove datasets
     └── annotations/
     	├── instances_train2017.json
     	└── instances_val2017.json
  ├── lvis/
     ├── lvis_v1_train.json
     └── lvis_v1_val.json
  ├── refcoco/
     ├── refcoco/
     	└── instances.json
     ├── refcoco+/
     	└── instances.json
     ├── refcocog/
     	└── instances.json
  ├── refcoco_remove/
  ├── vg/
     ├── images/
     ├── metas/
     ├── caption_vg_all.json
     └── caption_vg_train.json
  ├── vg_remove/
  ├── vgcut/
     ├── refer_train.json
     ├── refer_val.json
     ├── refer_input_train.json
     └── refer_input_val.json
  └── vgcut_remove/
```

6) Generating synthetic data

Download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1Hrmr5GnCVxXnxcLiVXRMXgTBQHcl6gSs?usp=sharing).

Run the dataset generation script:

```shell
cd 1_AddSD
sh utils/gen_datasets.sh
```

Here are examples of generation on COCO and LVIS datasets.

**COCO object generation**

```shell
python edit_cli_datasets.py --config configs/generate.yaml \
-n $NNODES -nr $NODE_RANK --addr $ADDR --port $PORT --input $INPUT --output $OUTPUT --ckpt $MODEL --seed $SEED \
```

- By default, use super-label-based sampling strategy to restrict the category of the added object. If do not use it, please add ```--no_superlabel``` parameter.

- By default, generate single object. If want to generate multiple objects, please add ```--multi``` parameter.

**LVIS object generation**

```shell
python edit_cli_datasets.py --config configs/generate.yaml -n $NNODES -nr $NODE_RANK --addr $ADDR --port $PORT --input $INPUT --output $OUTPUT --ckpt $MODEL --seed $SEED \
--is_lvis --lvis_label_selection r
```

- Need to add ```--is_lvis``` parameter to generate on LVIS dataset.

- By default, add object with rare classes. If want to use common or frequent classes, please change ```--lvis_label_selection f c r``` parameter, where f, c, r represents frequent, common, rare class, respectively.

### Step 2: Postprocessing synthetic data to localize the added objects from Add-SD

1) Follow the installation instructions in the [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) repository.

2) Download pretrained model, [groundingdino_swinb_cogcoor.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth), in the ```pretrained``` directory.

2) Navigate to the ```2_grounding_dino``` directory and run the inference script:

```shell
cd 2_grounding_dino
sh run_infer_with_GT_for_AddSD.sh
```

### Step 3: Train detectors with original data and synthetic data from Add-SD

1) Follow the installation instructions in the [XPaste](https://github.com/yoctta/XPaste) repository.

2) Navigate to the ```3_XPaste``` directory and run the inference script:

```shell
cd 3_XPaste
sh train.sh
```


<a name='4'></a>

## 4. Main Results

<p align="center">
  <img src="assets/visualization.svg" alt="visualization.svg" style="width: 100%;"/>
  <br>
  <span style="display: block; text-align: center; font-size: 14px; color: #555;">Visualization on image editing</span>
</p>

<p align="center">
  <img src="assets/visualization_instruction.svg" alt="visualization_instruction.svg" style="width: 100%;"/>
  <br>
  <span style="display: block; text-align: center; font-size: 14px; color: #555;">Visualization under different instructions</span>
</p>


<a name='5'></a>

## 5. References
Our project is conducted based on the following public paper with code:

- [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix)
- [XPaste](https://github.com/yoctta/XPaste)



<a name='6'></a>

## 6. Citation

If you find this code useful in your research, please kindly consider citing our paper:

```bibtex
    @article{yang2024add,
        title={Add-SD: Rational Generation without Manual Reference},
        author={Yang, Lingfeng and Zhang, Xinyu and Li, Xiang and Chen, Jinwen and Yao, Kun and Zhang, Gang and Liu, Lingqiao and Wang, Jingdong and Yang, Jian},
        journal={arXiv preprint arXiv},
        year={2024}
    }
```

