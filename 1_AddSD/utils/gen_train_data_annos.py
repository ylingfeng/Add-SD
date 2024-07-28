import json
import os
import random
from collections import defaultdict
import sys
import copy
sys.path.append("./")
sys.path.append("./Pattern/")
sys.path.append("./Pattern/pattern/")

import torch
from tqdm import tqdm

from prompt_candidates import prompt_edit_add, prompt_edit_add_with_description


def save_annos(dataset, seeds_dict, root, strategy):
    seeds_list = []
    for k, v in seeds_dict.items():
        if len(v) > 0:
            seeds_list.append([k, v])
        else:
            print(k)

    with open(os.path.join(root, "json", f"seeds_{dataset}_{strategy}.json"), "w") as f:
        json.dump(seeds_list, f)

    print(f"Num images for {dataset}: {len(seeds_dict)}")
    print("-------------------------------------------------------")
    return


def get_gemini():
    import google.generativeai as genai
    gemini_key = genai.configure(api_key="AIzaSyBd_8K0jtSer3-9VEc5MLj2t7m1xFqfEUk")

    generation_config = {
        'temperature': 0,
        'top_p': 1,
        'top_k': 1,
        'max_output_tokens': 2048,
    }

    safety_settingfs = [
        {
            'category': 'HARM_CATEGORY_HARASSMENT',
            'threshold': 'BLOCK_NONE'
        },
        {
            'category': 'HARM_CATEGORY_HATE_SPEECH',
            'threshold': 'BLOCK_NONE'
        },
        {
            'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
            'threshold': 'BLOCK_NONE'
        },
        {
            'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
            'threshold': 'BLOCK_NONE'
        },
    ]

    gemini_model = genai.GenerativeModel(model_name='gemini-pro',
                                         generation_config=generation_config,
                                         safety_settings=safety_settingfs)

    return gemini_model


def gen_annos_ip2p(root="data/clip-filtered-dataset"):
    with open(os.path.join(root, "seeds.json"), "r") as f:
        seeds_list = json.load(f)

    seeds_dict = {}
    for name, seed in tqdm(seeds_list, ncols=100):
        seeds_dict[name] = seed

    new_seeds_list = []
    for fn in tqdm(os.listdir(f"{root}/subset"), ncols=100):
        new_seeds_list.append([fn, seeds_dict[fn]])

        assert "metadata.jsonl" in os.listdir(f"{root}/subset/{fn}")
        assert "prompt.json" in os.listdir(f"{root}/subset/{fn}")
        for sub_fn in seeds_dict[fn]:
            assert f"{sub_fn}_0.jpg" in os.listdir(f"{root}/subset/{fn}")
            assert f"{sub_fn}_1.jpg" in os.listdir(f"{root}/subset/{fn}")

    with open(os.path.join(root, "seeds_subset.json"), "w") as f:
        json.dump(new_seeds_list, f)


def gen_annos_coco_by_instance(root="data", strategy="vanilla"):
    seeds_dict = {}
    with open(f'{root}/json/coco_remove/instances_train2017_with_remove_image.json', 'r') as f:
        annotation = json.load(f)
    print("coco images num", len(annotation['images']))

    if strategy == 'gemini':
        with open(f'{root}/json/llava/llava_v1_5_mix665k.json', 'r') as f:
            anno_llava = json.load(f)
        print("llava coco cap num", len(anno_llava))
        anno_llava_id2anno = {}
        for line in tqdm(anno_llava, ncols=100):
            # {
            #     'id': '000000033471', 'image': 'coco/train2017/000000033471.jpg',
            #     'conversations': [
            #         {'from': 'human', 'value': '<image>\nWhat are the colors of the bus in the image?'},
            #         {'from': 'gpt', 'value': 'The bus in the image is white and red.'},
            #         {'from': 'human', 'value': 'What feature can be seen on the back of the bus?'},
            #         {'from': 'gpt', 'value': 'The back of the bus features an advertisement.'},
            #     ]
            # }
            image_id = line['id']

            if image_id not in anno_llava_id2anno:
                anno_llava_id2anno[image_id] = [line]
            else:
                anno_llava_id2anno[image_id].append(line)
        # print(anno_llava_id2anno["000000000025"])
        gemini_model = get_gemini()
    else:
        pass

    for anno in tqdm(annotation['annotations'], ncols=100):
        # anno = {
        #     'segmentation': [
        #         [239.97, 260.24, 222.04, 270.49, 199.84, 253.41, 213.5, 227.79, 259.62, 200.46, 274.13,
        #          202.17, 277.55, 210.71, 249.37, 253.41, 237.41, 264.51, 242.54, 261.95, 228.87, 271.34]
        #     ],
        #     'area': 2765.1486500000005, 'iscrowd': 0, 'image_id': 558840, 'bbox': [199.84, 200.46, 77.71, 70.88], 'category_id': 58, 'id': 156,
        #     'remove_image': 'train2017_remove_image/000000558840/inpainted_with_mask_0_hot dog.png'
        # }
        if "remove_image" not in anno:
            continue
        # 'train2017_remove_image/000000558840/inpainted_with_mask_0_hot dog.png'
        remove_image = anno['remove_image']
        image_id = remove_image.split('/')[1]
        remove_fn = remove_image.split('/')[-1]
        label = remove_fn.split('.png')[0].split('_')[-1]
        # src_dir = f"{root}/coco/{remove_image}"
        # tgt_dir = f"{root}/coco/train2017/{image_id}.jpg"

        mask_id = int(remove_fn.split('.png')[0].split('_')[-2])
        # print(label, mask_id, remove_fn)
        assert remove_fn in os.listdir(os.path.join(root, "coco/train2017_remove_image", image_id))

        # find matched image_id in llava captions
        if strategy == "vanilla":
            edit = f"add a {label}"
        elif strategy == "template":
            template = random.choice(prompt_edit_add)
            edit = template.replace("<obj>", label)
        elif strategy == "gemini":
            if image_id not in anno_llava_id2anno:
                # print(f"no image id {image_id} in llava")
                template = random.choice(prompt_edit_add)
                edit = template.replace("<obj>", label)
            else:
                llava_anno = anno_llava_id2anno[image_id]
                context = ''
                for i in range(len(llava_anno)):
                    conversations = llava_anno[i]["conversations"]
                    for idx_c, conversation in enumerate(conversations):
                        if idx_c % 2 == 0:
                            context += 'Q: ' + conversation['value'] + '\n'
                        else:
                            context += 'A: ' + conversation['value'] + '\n'
                            if '<image>\n' in context:
                                context = context.replace('<image>\n', '')

                input_txt = prompt_edit_add_with_description.replace("<obj>", label).replace("<cap>", context)
                # print(input_txt)
                response = gemini_model.generate_content(input_txt)
                edit = response.text
        else:
            raise NotImplementedError

        res = {"image_0_dir": f"coco/{remove_image}",
               "image_1_dir": f"coco/train2017/{image_id}.jpg",
               "edit": edit}
        if image_id in seeds_dict:
            seeds_dict[image_id].append(res)
        else:
            seeds_dict[image_id] = [res]

    for fn in tqdm(os.listdir(os.path.join(root, "coco/train2017_remove_image")), ncols=100):
        assert len(os.listdir(os.path.join(root, "coco/train2017_remove_image", fn))) == len(seeds_dict[fn])

    save_annos('coco', seeds_dict, root, strategy)
    return seeds_dict


def gen_annos_coco(root="data", strategy="vanilla", check_by_instance=False, check_by_images=False):
    seeds_dict = {}
    for image_id in tqdm(os.listdir(os.path.join(root, "coco/train2017_remove_image")), ncols=100):
        seeds_dict[image_id] = []

        for fn in os.listdir(os.path.join(root, "coco/train2017_remove_image", image_id)):
            label = fn.split('.png')[0].split('_')[-1]

            if strategy == "vanilla":
                edit = f"add a {label}"
            elif strategy == "template":
                template = random.choice(prompt_edit_add)
                edit = template.replace("<obj>", label)
            else:
                raise NotImplementedError
            # print(edit)

            res = {"image_0_dir": os.path.join("coco/train2017_remove_image", image_id, fn),
                   "image_1_dir": f"coco/train2017/{image_id}.jpg",
                   "edit": edit}
            seeds_dict[image_id].append(res)

    save_annos('coco', seeds_dict, root, strategy)
    return seeds_dict


def gen_annos_vgcut(root="data", strategy="vanilla"):
    seeds_dict = {}
    for image_id in tqdm(os.listdir(os.path.join(root, "vgcut_remove")), ncols=100):
        seeds_dict[image_id] = []

        for fn in os.listdir(os.path.join(root, "vgcut_remove", image_id)):
            if fn == "anno.png" or fn == "ori.png":
                continue
            try:
                assert "inpainted" in fn
            except:
                print(os.path.join(root, "vgcut_remove", image_id, fn))
            label = fn.split('.png')[0].split('_')[-1]

            if strategy == "vanilla":
                edit = f"add a {label}"
            elif strategy == "template":
                template = random.choice(prompt_edit_add)
                edit = template.replace("<obj>", label)
            else:
                raise NotImplementedError
            res = {"image_0_dir": os.path.join("vgcut_remove", image_id, fn),
                   "image_1_dir": os.path.join("vgcut_remove", image_id, "ori.png"),
                   "edit": edit}
            seeds_dict[image_id].append(res)

    save_annos('vgcut', seeds_dict, root, strategy)
    return seeds_dict


def gen_annos_vg(root="data", strategy="vanilla"):
    seeds_dict = {}
    for image_id in tqdm(os.listdir(os.path.join(root, "vg_remove")), ncols=100):
        seeds_dict[image_id] = []

        for fn in os.listdir(os.path.join(root, "vg_remove", image_id)):
            if fn == "anno.png" or fn == "ori.png" or fn == "selected.json":
                continue
            try:
                assert "inpainted" in fn
            except:
                print(os.path.join(root, "vg_remove", image_id, fn))
            label = fn.split('.png')[0].split('_')[-1]

            if strategy == "vanilla":
                edit = f"add a {label}"
            elif strategy == "template":
                template = random.choice(prompt_edit_add)
                edit = template.replace("<obj>", label)
            else:
                raise NotImplementedError
            res = {"image_0_dir": os.path.join("vg_remove", image_id, fn),
                   "image_1_dir": os.path.join("vg_remove", image_id, "ori.png"),
                   "edit": edit}
            seeds_dict[image_id].append(res)

    save_annos('vg', seeds_dict, root, strategy)
    return seeds_dict


def gen_annos_lvis_by_images(root="data", strategy="vanilla"):
    seeds_dict = {}
    with open(f"{root}/json/lvis_remove/addsd_lvis_v1_train_with_remove_image_convs.json", 'r') as f:
        file = json.load(f)

    for line in tqdm(file, ncols=100):
        fn = line['image'].split('/')[-1]
        image_id = fn.split('.')[0]
        remove_image = line["remove_image"]
        image_1_dir = f"coco/train2017/{fn}"

        for rm_img in remove_image:
            image_0_dir = rm_img  # coco/lvis_remove_image/000000475546/inpainted_with_mask_8_sink.png
            label = rm_img.split('/')[-1].split('inpainted_with_mask_')[-1].split('.png')[0].split('_')[1:]
            label = ' '.join(label)
            if strategy == "vanilla":
                edit = f"add a {label}"
            elif strategy == "template":
                template = random.choice(prompt_edit_add)
                edit = template.replace("<obj>", label)
            else:
                raise NotImplementedError

            res = {"image_0_dir": image_0_dir,
                   "image_1_dir": image_1_dir,
                   "edit": edit}
            # print(res)
            if image_id in seeds_dict:
                seeds_dict[image_id].append(res)
            else:
                seeds_dict[image_id] = [res]

    save_annos('lvis', seeds_dict, root, strategy)
    return seeds_dict


def gen_annos_lvis(root="data", strategy="vanilla", check_by_instance=False, check_by_images=False):
    if check_by_instance:
        with open(f"{root}/json/lvis_remove/lvis_v1_train_with_remove_image.json", 'r') as f:
            anno_instance = json.load(f)["annotations"]
        anno_instance_ = defaultdict(list)
        for anno in anno_instance:
            if "remove_image" not in anno:
                continue
            image_id = anno['remove_image'].split('/')[1]
            remove_fn = anno['remove_image']
            anno_instance_[image_id].append(remove_fn)
        anno_instance = anno_instance_
    if check_by_images:
        with open(f"{root}/json/lvis_remove/addsd_lvis_v1_train_with_remove_image_convs.json", 'r') as f:
            anno_images = json.load(f)
        anno_images = {anno["image"].split('/')[-1].split('.jpg')[0]: anno["remove_image"] for anno in anno_images}

    seeds_dict = {}
    for image_id in tqdm(os.listdir(os.path.join(root, "coco/lvis_remove_image")), ncols=100):
        seeds_dict[image_id] = []

        for fn in os.listdir(os.path.join(root, "coco/lvis_remove_image", image_id)):
            if check_by_instance:
                assert os.path.join("lvis_remove_image", image_id, fn) in anno_instance[image_id]
                anno_instance[image_id] = list(filter(lambda x: x != os.path.join(
                    "lvis_remove_image", image_id, fn), anno_instance[image_id]))
            if check_by_images:
                assert os.path.join("coco/lvis_remove_image", image_id, fn) in anno_images[image_id]
                anno_images[image_id] = list(filter(lambda x: x != os.path.join(
                    "coco/lvis_remove_image", image_id, fn), anno_images[image_id]))

            label = fn.split('inpainted_with_mask_')[-1].split('.png')[0].split('_')[1:]
            label = ' '.join(label)
            if strategy == "vanilla":
                edit = f"add a {label}"
            elif strategy == "template":
                template = random.choice(prompt_edit_add)
                edit = template.replace("<obj>", label)
            else:
                raise NotImplementedError
            # print(edit)

            res = {"image_0_dir": os.path.join("coco/lvis_remove_image", image_id, fn),
                   "image_1_dir": f"coco/train2017/{image_id}.jpg",
                   "edit": edit}
            seeds_dict[image_id].append(res)

    save_annos('lvis', seeds_dict, root, strategy)

    if check_by_instance:
        for k, v in anno_instance.items():
            if len(v):
                print(k, v)
    if check_by_images:
        for k, v in anno_images.items():
            if len(v):
                print(k, v)

    return seeds_dict


def gen_annos_coco_multi(root="data", strategy="vanilla"):
    import nltk
    import num2words
    nltk.set_proxy("http://172.19.56.199:3128")
    nltk.download('omw-1.4')
    nltk.download('popular')
    from pattern.en import pluralize

    seeds_dict = {}
    for image_id in tqdm(os.listdir(os.path.join(root, "coco/train2017_remove_image_multiobj")), ncols=100):
        seeds_dict[image_id] = []

        for fn in os.listdir(os.path.join(root, "coco/train2017_remove_image_multiobj", image_id)):

            label = fn.split('_')[4]
            num_ins = fn.split('_')[5]
            assert 'N' in num_ins
            num_ins = int(num_ins.split('N')[1])

            if strategy == "vanilla":
                if num_ins > 1:
                    if random.random() > 0.5:
                        label = pluralize(label)
                    else:
                        pass
                    if random.random() > 0.5:
                        edit = f"add {num2words.num2words(num_ins)} {label}"
                    else:
                        edit = f"add {num_ins} {label}"
                else:
                    assert num_ins == 1
                    if random.random() > 0.5:
                        edit = f"add a {label}"
                    else:
                        edit = f"add one {label}"
            else:
                raise NotImplementedError
            # print(edit)

            res = {"image_0_dir": os.path.join("coco/train2017_remove_image_multiobj", image_id, fn),
                   "image_1_dir": f"coco/train2017/{image_id}.jpg",
                   "edit": edit}
            seeds_dict[image_id].append(res)

    seeds_dict_single = gen_annos_coco(root, strategy)
    for k, v in seeds_dict_single.items():
        if k in seeds_dict:
            seeds_dict[k] += v
        else:
            seeds_dict[k] = v

    save_annos('coco_multi', seeds_dict, root, strategy)
    return seeds_dict


def gen_annos_lvis_multi(root="data", strategy="vanilla"):
    import nltk
    import num2words
    nltk.set_proxy("http://172.19.56.199:3128")
    nltk.download('omw-1.4')
    nltk.download('popular')
    from pattern.en import pluralize

    seeds_dict = {}
    for image_id in tqdm(os.listdir(os.path.join(root, "coco/lvis_remove_image_multiobj")), ncols=100):
        seeds_dict[image_id] = []

        for fn in os.listdir(os.path.join(root, "coco/lvis_remove_image_multiobj", image_id)):

            label = fn.split('_')[4:-2]
            label = ' '.join(label)

            num_ins = fn.split('_')[-2]
            assert 'N' in num_ins
            num_ins = int(num_ins.split('N')[1])

            if strategy == "vanilla":
                if num_ins > 1:
                    if random.random() > 0.5:
                        label = pluralize(label)
                    else:
                        pass
                    if random.random() > 0.5:
                        edit = f"add {num2words.num2words(num_ins)} {label}"
                    else:
                        edit = f"add {num_ins} {label}"
                else:
                    assert num_ins == 1
                    if random.random() > 0.5:
                        edit = f"add a {label}"
                    else:
                        edit = f"add one {label}"
            else:
                raise NotImplementedError
            # print(edit)

            res = {"image_0_dir": os.path.join("coco/lvis_remove_image_multiobj", image_id, fn),
                   "image_1_dir": f"coco/train2017/{image_id}.jpg",
                   "edit": edit}
            seeds_dict[image_id].append(res)

    seeds_dict_single = gen_annos_lvis(root, strategy)
    for k, v in seeds_dict_single.items():
        if k in seeds_dict:
            seeds_dict[k] += v
        else:
            seeds_dict[k] = v

    save_annos('lvis_multi', seeds_dict, root, strategy)
    return seeds_dict

if __name__ == "__main__":
    gen_annos_ip2p()
    gen_annos_coco(root="data", strategy="vanilla")
    # gen_annos_coco_multi(root="data", strategy="vanilla")
    # gen_annos_lvis(root="data", strategy="vanilla")
    # gen_annos_lvis_multi(root="data", strategy="vanilla")
    # gen_annos_vg(root="data", strategy="vanilla")
    # gen_annos_vgcut(root="data", strategy="vanilla")
