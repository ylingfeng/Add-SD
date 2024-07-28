############## COCO dataset for single object remove ##############

python remove_anything_with_GTbox.py \
    --config ./configs/datasets/coco_instance_infer.py \
    --point_labels 1 \
    --dilate_kernel_size 10 \
    --output_dir ./results/coco/ \
    --sam_model_type vit_h \
    --sam_ckpt ./pretrained/sam_vit_h_4b8939.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ../pretrained/big-lama \


############## LVIS dataset for single object remove ##############

python remove_anything_with_GTbox.py \
    --config ./configs/datasets/lvis_instance_infer.py \
    --point_labels 1 \
    --dilate_kernel_size 10 \
    --output_dir ./results/lvis/ \
    --sam_model_type vit_h \
    --sam_ckpt ../pretrained/sam_vit_h_4b8939.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ../pretrained/big-lama \

############## COCO dataset for multiple objects remove ##############

python remove_anything_with_GTbox_multiobj.py \
    --config ./configs/datasets/coco_instance_infer.py \
    --point_labels 1 \
    --dilate_kernel_size 10 \
    --output_dir ./results/coco_multi/ \
    --sam_model_type vit_h \
    --sam_ckpt ./pretrained/sam_vit_h_4b8939.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ../pretrained/big-lama \

############## LVIS dataset for multiple objects remove ##############

python remove_anything_with_GTbox_multiobj.py \
    --config ./configs/datasets/lvis_instance_infer.py \
    --point_labels 1 \
    --dilate_kernel_size 10 \
    --output_dir ./results/lvis_multi/ \
    --sam_model_type vit_h \
    --sam_ckpt ../pretrained/sam_vit_h_4b8939.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ../pretrained/big-lama \