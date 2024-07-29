export WANDB_DISABLED='true'

# ##### Train COCO single object remove dataset #####
python main.py --name default --base configs/train_rm_coco.yaml --train --gpus 0,1,2,3,4,5,6,7
# ##### Train LVIS single object remove dataset #####
# python main.py --name default --base configs/train_rm_lvis.yaml --train --gpus 0,1,2,3,4,5,6,7
# ##### Train COCO multiple objects remove dataset #####
# python main.py --name default --base configs/train_rm_coco_multi.yaml --train --gpus 0,1,2,3,4,5,6,7
# ##### Train LVIS multiple objects remove dataset #####
# python main.py --name default --base configs/train_rm_lvis_multi.yaml --train --gpus 0,1,2,3,4,5,6,7
##### Train COCO+LVIS+VG+VGCUT+REFCOCO remove dataset #####
# python main.py --name default --base configs/train_rm_coco_lvis_vg_vgcut_refcoco.yaml --train --gpus 0,1,2,3,4,5,6,7
