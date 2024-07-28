import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.register_coco import register_coco_instances
from .lvis_v1 import get_lvis_instances_meta, custom_register_lvis_instances
from detectron2.data import DatasetCatalog

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "addsd_gen_coco_multi_coco-lvis-vg-vgcut_seed0_superlabel": (
        "addsd_gen_coco_multi_coco-lvis-vg-vgcut_seed0_superlabel/train2017",
        "addsd_gen_coco_multi_coco-lvis-vg-vgcut_seed0_superlabel_visualization_all/add_sd_generated_image_instances_train2017_0_117266.json"
    ),
    # "coco_gen_coco_lvis_2017_train": (
    #     "coco_gen_coco_lvis/train2017",
    #     "coco_gen_coco_lvis_visualization_all/add_sd_generated_image_instances_train2017_0_117266.json"
    # ),
    # "coco_gen_coco_lvis_pad_2017_train": (
    #     "coco_gen_coco_lvis_pad/train2017",
    #     "coco_gen_coco_lvis_pad_visualization_all/add_sd_generated_image_instances_train2017_0_117266.json"
    # ),
    # "coco_gen_multi_2017_train": (
    #     "coco_gen_multi/train2017",
    #     "coco_gen_multi_visualization_all/add_sd_generated_image_instances_train2017_0_117266.json"
    # ),
    # "coco_gen_multi_avail_2017_train": (
    #     "coco_gen_multi/train2017",
    #     "coco_gen_multi_visualization_all/add_sd_generated_image_instances_train2017_0_88313_avail.json"
    # ),
    # "coco_gen_parking_meter_train2017": (
    #     "coco_gen_parking_meter/train2017",
    #     "coco_gen_parking_meter_visualization_all/add_sd_generated_image_instances_train2017_0_12880.json"
    # ),
    # "coco_gen_multi_seed1_2017_train": (
    #     "coco_gen_multi_seed1/train2017",
    #     "coco_gen_multi_seed1_visualization_all/add_sd_generated_image_instances_train2017_0_117266.json"
    # ),
    # "coco_gen_multi_filter_2017_train": (
    #     "coco_gen_multi/train2017",
    #     "coco_gen_multi_visualization_all/add_sd_generated_image_instances_train2017_0_117266_filter.json"
    # ),
    # "coco_gen_multi_seed1_filter_2017_train": (
    #     "coco_gen_multi_seed1/train2017",
    #     "coco_gen_multi_seed1_visualization_all/add_sd_generated_image_instances_train2017_0_117266_filter.json"
    # ),
    # "coco_gen_multi_filter_badimage_2017_train": (
    #     "coco_gen_multi/train2017",
    #     "coco_gen_multi_visualization_all/add_sd_generated_image_instances_train2017_0_117266_filter_badimage.json"
    # ),
    # "coco_gen_multi_seed1_filter_badimage_2017_train": (
    #     "coco_gen_multi_seed1/train2017",
    #     "coco_gen_multi_seed1_visualization_all/add_sd_generated_image_instances_train2017_0_117266_filter_badimage.json"
    # ),
    # "coco_gen_multi_filter_low_aesthetic_score_image_2017_train": (
    #     "coco_gen_multi/train2017",
    #     "coco_gen_multi_visualization_all/add_sd_generated_image_instances_train2017_0_117266_filter_low_aesthetic_score_image.json"
    # ),
    # "coco_gen_multi_filter_small_large_instances_2017_train": (
    #     "coco_gen_multi/train2017",
    #     "coco_gen_multi_visualization_all/add_sd_generated_image_instances_train2017_0_117266_filter_small_large_instances.json"
    # ),
    # "coco_gen_multi_filter_small_large_instances_onlyGen_2017_train": (
    #     "coco_gen_multi/train2017",
    #     "coco_gen_multi_visualization_all/add_sd_generated_image_instances_train2017_0_117266_filter_small_large_instances_onlyGen.json"
    # ),
    # # no superlabel no freq
    # "coco_gen_multi_no_superlabel_2017_train": (
    #     "coco_gen_multi_no_superlabel/train2017",
    #     "coco_gen_multi_no_superlabel_visualization_all/add_sd_generated_image_instances_train2017_0_117266_wScore.json"
    # ),
    # "coco_gen_multi_no_freq_2017_train": (
    #     "coco_gen_multi_no_freq/train2017",
    #     "coco_gen_multi_no_freq_visualization_all/add_sd_generated_image_instances_train2017_0_117266_wScore.json"
    # ),
    # "coco_gen_multi_no_superlabel_no_freq_2017_train": (
    #     "coco_gen_multi_no_superlabel_no_freq/train2017",
    #     "coco_gen_multi_no_superlabel_no_freq_visualization_all/add_sd_generated_image_instances_train2017_0_117266_wScore.json"
    # ),
    # "coco_gen_single_coco_lvis_ref_vg_vgcut_new_seed0_superlabel_2017_train": (
    #     "coco_gen_single_coco_lvis_ref_vg_vgcut_new_seed0_superlabel/train2017",
    #     "coco_gen_single_coco_lvis_ref_vg_vgcut_new_seed0_superlabel_visualization_all/add_sd_generated_image_instances_train2017_0_117266_wScore.json"
    # ),
    # "coco_gen_multi_coco_lvis_ref_vg_vgcut_new_seed0_superlabel_2017_train": (
    #     "coco_gen_multi_coco_lvis_ref_vg_vgcut_new_seed0_superlabel/train2017",
    #     "coco_gen_multi_coco_lvis_ref_vg_vgcut_new_seed0_superlabel_visualization_all/add_sd_generated_image_instances_train2017_0_117266_wScore.json"
    # ),
}


root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))

for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
    for key, (image_root, json_file) in splits_per_dataset.items():
        # Assume pre-defined datasets live in `./datasets`.
        if not key in DatasetCatalog.list():
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "addsd_gen_lvis_single_coco-lvis-vg-vgcut_seed0_r": (
            "addsd_gen_lvis_single_coco-lvis-vg-vgcut_seed0_r/train2017",
            "addsd_gen_lvis_single_coco-lvis-vg-vgcut_seed0_r_visualization_all/add_sd_generated_image_instances_train2017_0_99388_wScore.json"
        ),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            if not key in DatasetCatalog.list():
                custom_register_lvis_instances(
                    key,
                    get_lvis_instances_meta(dataset_name),
                    os.path.join(root, json_file) if "://" not in json_file else json_file,
                    os.path.join(root, image_root),
                )


register_all_lvis(root)
