import glob
import os
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
import json

from torchvision.datasets.vision import VisionDataset
from .builder import DATASETS
import xml.etree.ElementTree as ET

@DATASETS.register_module()
class Flickr8kParser(HTMLParser):
    """Parser for extracting captions from the Flickr8k dataset web page."""

    def __init__(self, root: str) -> None:
        super().__init__()

        self.root = root

        # Data structure to store captions
        self.annotations: Dict[str, List[str]] = {}

        # State variables
        self.in_table = False
        self.current_tag: Optional[str] = None
        self.current_img: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.current_tag = tag

        if tag == "table":
            self.in_table = True

    def handle_endtag(self, tag: str) -> None:
        self.current_tag = None

        if tag == "table":
            self.in_table = False

    def handle_data(self, data: str) -> None:
        if self.in_table:
            if data == "Image Not Found":
                self.current_img = None
            elif self.current_tag == "a":
                img_id = data.split("/")[-2]
                img_id = os.path.join(self.root, img_id + "_*.jpg")
                img_id = glob.glob(img_id)[0]
                self.current_img = img_id
                self.annotations[img_id] = []
            elif self.current_tag == "li" and self.current_img:
                img_id = self.current_img
                self.annotations[img_id].append(data.strip())

@DATASETS.register_module()
class Flickr8k(VisionDataset):
    """`Flickr8k Entities <http://hockenmaier.cs.illinois.edu/8k-pictures.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)

        # Read annotations and store in a dict
        parser = Flickr8kParser(self.root)
        with open(self.ann_file) as fh:
            parser.feed(fh.read())
        self.annotations = parser.annotations

        self.ids = list(sorted(self.annotations.keys()))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]

        # Image
        img = Image.open(img_id).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)

def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      fn - full file path to the sentence file to parse
    
    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this 
                                    phrase belongs to

    """
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence' : ' '.join(words), 'phrases' : []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index' : index,
                                             'phrase' : phrase,
                                             'phrase_id' : p_id,
                                             'phrase_type' : p_type})

        annotations.append(sentence_data)

    return annotations

def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      fn - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the 
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall('size')[0]
    anno_info = {'boxes' : {}, 'scene' : [], 'nobox' : []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                if box_id not in anno_info['boxes']:
                    anno_info['boxes'][box_id] = []
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info

@DATASETS.register_module()
class Flickr30k(VisionDataset):
    """`Flickr30k Entities <https://bryanplummer.com/Flickr30kEntities/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)

        # Read annotations and store in a dict
        self.annotations = defaultdict(dict)
        with open(self.ann_file, 'r') as fh:
            dataset = json.load(fh)
        for line in dataset:
            img_id = line['image_id']
            image = line['image']
            if img_id in self.annotations:
                continue
            
            image_name = os.path.basename(image).split('.jpg')[0]
            sentences = get_sentence_data(os.path.join(root, 'Sentences', image_name + '.txt'))
            anno_info = get_annotations(os.path.join(root, 'Annotations', image_name + '.xml'))
            
            self.annotations[img_id] = {
                'image': os.path.basename(image),
                'img_id': img_id,
                'sentences': sentences,
                'annotations': anno_info,
            }

        self.ids = list(sorted(self.annotations.keys()))
        self.data_infos = self.annotations

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]

        # Image
        filename = os.path.join(self.root, img_id)
        img = Image.open(filename).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)