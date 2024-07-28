MODEL=addsd_coco_lvis_vg_vgcut/last.ckpt
INPUT=data/coco/train2017
SEED=0
NNODES=1
NODE_RANK=0
ADDR=10.127.0.0
PORT=8890

##### COCO single object generation #####
OUTPUT=./logs/addsd_gen_coco_single_coco-lvis-vg-vgcut_seed${SEED}_superlabel/
python edit_cli_datasets.py --config configs/generate.yaml \
-n $NNODES -nr $NODE_RANK --addr $ADDR --port $PORT --input $INPUT --output $OUTPUT --ckpt $MODEL --seed $SEED \

##### COCO multiple objects generation #####
OUTPUT=./logs/addsd_gen_coco_multi_coco-lvis-vg-vgcut_seed${SEED}_superlabel/
python edit_cli_datasets.py --config configs/generate.yaml \
-n $NNODES -nr $NODE_RANK --addr $ADDR --port $PORT --input $INPUT --output $OUTPUT --ckpt $MODEL --seed $SEED --multi \

##### LVIS single object generation with only rare classes #####
OUTPUT=./logs/addsd_gen_lvis_single_coco-lvis-vg-vgcut_seed${SEED}_r/
python edit_cli_datasets.py --config configs/generate.yaml -n $NNODES -nr $NODE_RANK --addr $ADDR --port $PORT --input $INPUT --output $OUTPUT --ckpt $MODEL --seed $SEED \
--is_lvis --lvis_label_selection r

##### LVIS single object generation with frequent, common, rare classes #####
OUTPUT=./logs/addsd_gen_lvis_single_coco-lvis-vg-vgcut_seed${SEED}_fcr/
python edit_cli_datasets.py --config configs/generate.yaml -n $NNODES -nr $NODE_RANK --addr $ADDR --port $PORT --input $INPUT --output $OUTPUT --ckpt $MODEL --seed $SEED \
--is_lvis --lvis_label_selection f c r

