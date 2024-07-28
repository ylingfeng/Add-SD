#!/bin/bash

##### train COCO #####
bash launch.sh --config-file configs/centernet2/AddSD/COCO_Base-C2_L_R5021k_640b64_4x_coco_gen_multi_ratio41_bs_mask.yaml

##### train LVIS #####
bash launch.sh --config-file configs/centernet2/AddSD/LVIS_Base-C2_L_R5021k_640b64_4x_lvis_gen_single_ratio41_bs_mask.yaml