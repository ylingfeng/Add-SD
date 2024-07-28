# tmux
outnet apt-get update
outnet apt-get install tmux -y

# abosolute path
export afs3=/root/paddlejob/workspace/env_run/datasets/afs3
export afs3_repo=/root/paddlejob/workspace/env_run/datasets/afs3/yanglingfeng/Add-P2P-dev
export repo=/root/paddlejob/workspace/env_run/output/yanglingfeng/Add-P2P-dev
export datasets_root=/root/paddlejob/workspace/datasets

# checkpoints and data
rsync -lrvu ${afs3_repo}/checkpoints ${repo}
ln -s ${datasets_root} ${repo}/data
rsync -lrvu --exclude 'json/not_using' ${afs3_repo}/data/json ${repo}/data

# coco datasets
# wget -c http://10.127.10.20:8011/datasets/COCO.tar -P ${datasets_root}
cp ${afs3}/zhangxinyu14/data/public/COCO.tar ${datasets_root}
tar -xvf ${datasets_root}/COCO.tar -C ${datasets_root}
ln -s ${datasets_root}/COCO ${repo}/data/coco

# coco remove datasets
wget -c http://10.127.27.25:8901/datasets/COCO/train2017_remove_image.tar.gz ${datasets_root}/COCO/
tar -xzvf ${datasets_root}/COCO/train2017_remove_image.tar.gz -C ${datasets_root}/COCO/

# copy checkpoints
cp ${afs3_repo}/logs/train_rm_default/checkpoints/trainstep_checkpoints/epoch=000048-step=000005000.ckpt ${repo}/logs/train_rm_default/checkpoints/trainstep_checkpoints/
cp -rf ${repo}/logs/train_rm_default/checkpoints/trainstep_checkpoints ${afs3_repo}/logs/train_rm_default/checkpoints/trainstep_checkpoints

