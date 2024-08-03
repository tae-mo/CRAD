from __future__ import division

import json
import logging

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.image_reader import build_image_reader
from datasets.transforms import RandomColorJitter

logger = logging.getLogger("global_logger")

cls_ids = {
    'bottle':0,
    'cable':1,
    'capsule':2,
    'carpet':3,
    'grid':4,
    'hazelnut':5,
    'leather':6,
    'metal_nut':7,
    'pill':8,
    'screw':9,
    'tile':10,
    'toothbrush':11,
    'transistor':12,
    'wood':13,
    'zipper':14,
}


def _init_fn(worker_id):
    np.random.seed(12)

def build_custom_dataloader(cfg, training, distributed=True):

    image_reader = build_image_reader(cfg.image_reader)

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
    if training:
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"])

    colorjitter_fn = None
    if cfg.get("colorjitter", None) and training:
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

    logger.info("building CustomDataset from: {}".format(cfg["meta_file"]))

    dataset = CustomDataset(
        image_reader,
        cfg["meta_file"],
        training,
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
        worker_init_fn=_init_fn
    )

    return data_loader


class CustomDataset(BaseDataset):
    def __init__(
        self,
        image_reader,
        meta_file,
        training,
        transform_fn,
        normalize_fn,
        colorjitter_fn=None,
    ):
        self.image_reader = image_reader
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn

        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]

        # read image
        filename = meta["filename"]
        label = meta["label"]
        image = self.image_reader(meta["filename"])
        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )
        
        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
            input['clsid'] = cls_ids[meta["clsname"]]
        else:
            input["clsname"] = filename.split("/")[-4]
            input['clsid'] = cls_ids[filename.split("/")[-4]]

        image = Image.fromarray(image, "RGB")

        # read / generate mask
        if meta.get("maskname", None):
            maskname = meta['maskname']
            mask = self.image_reader(meta["maskname"], is_mask=True)
        else:
            maskname = None
            if label == 0:  # good
                mask = np.zeros((image.height, image.width)).astype(np.uint8)
            elif label == 1:  # defective
                mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        mask = Image.fromarray(mask, "L")

        if self.transform_fn:
            image, mask = self.transform_fn(image, mask)
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        if self.normalize_fn:
            image = self.normalize_fn(image)
        input.update({"image": image, "mask": mask, "maskname": str(maskname)})
        return input
