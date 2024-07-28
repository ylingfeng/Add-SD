python replace_anything.py \
    --input_img ./example/shuigang/sleep_baidu000002__00001872.jpg \
    --coords_type grounding \
    --point_coords 750 500 \
    --point_labels 1 \
    --text_prompt "sit on the swing" \
    --output_dir ./results/shuigang/ \
    --sam_model_type "vit_h" \
    --sam_ckpt /root/paddlejob/workspace/datasets/afs3/zhangxinyu14/pretrained/sam_vit_h_4b8939.pth