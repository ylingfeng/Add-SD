SEED=0

######### COCO ##########
CUDA_VISIBLE_DEVICES=0 python \
    scripts/infer_with_GT_coco.py \
    --dataset_name addsd_gen_coco_multi_coco-lvis-vg-vgcut_seed${SEED}_superlabel \
    --start_index 0 \
    --end_index -1 \
    # --save_annotations \

######### LVIS ##########
CUDA_VISIBLE_DEVICES=0 python \
    scripts/infer_with_GT_lvis.py \
    --dataset_name addsd_gen_lvis_single_coco-lvis-vg-vgcut_seed${SEED}_r \
    --start_index 0 \
    --end_index -1 \
    # --save_annotations \
    # --save_path temp/lvis \
