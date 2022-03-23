import os
from os.path import join
from glob import glob
import ujson as json
from typing import List, Tuple, Union, Optional, Dict
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode as IM
from pycocotools.mask import decode
from datasets.base_dataset import BaseDataset


class DUTSDataset(BaseDataset):
    def __init__(
            self,
            dir_dataset: str,
            img_size: int = 224,
            use_pseudo_masks: bool = True,
            pseudo_masks_fp: Optional[str] = None,
            scale_range=(0.5, 2.0),
            use_copy_paste: bool = False,
            rank: Optional[int] = None,
            world_size: Optional[int] = None
    ):
        """At this moment, only test set is considered."""
        super(DUTSDataset, self).__init__()
        self.p_test_imgs = sorted(glob(join(dir_dataset, "DUTS-TE-Image", f"*.jpg")))
        self.p_test_gts = sorted(glob(join(dir_dataset, "DUTS-TE-Mask", f"*.png")))

        if not use_pseudo_masks and pseudo_masks_fp is None:
            self.pseudo_masks = None
            self.p_train_imgs = sorted(glob(join(dir_dataset, "DUTS-TR-Image", f"*.jpg")))
            self.p_train_gts = sorted(glob(join(dir_dataset, "DUTS-TR-Mask", f"*.png")))
            assert len(self.p_train_imgs) == len(self.p_train_gts), f"{len(self.p_train_imgs)} != {len(self.p_train_gts)}"
            assert len(self.p_test_imgs) == len(self.p_test_gts), f"{len(self.p_test_imgs)} != {len(self.p_test_gts)}"

        else:
            self.pseudo_masks = json.load(open(pseudo_masks_fp, 'r'))
            self.p_train_imgs = sorted(list(self.pseudo_masks.keys()))
            self.p_train_imgs = [join(dir_dataset, "DUTS-TR-Image", p_img) for p_img in self.p_train_imgs]
            self.p_train_gts = None

        # seg variables for data augmentation
        self.dir_dataset = dir_dataset
        self.img_size = img_size
        self.masks = None
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.name = "duts"
        self.use_pseudo_masks = use_pseudo_masks
        self.scale_range = scale_range
        self.use_aug = True
        self.use_copy_paste = use_copy_paste

        print(
            '\n'
            "Dataset summary\n"
            f"Dataset name: {self.name}\n"
            f"# train images: {len(self.p_train_imgs)}\n"
            f"# test images: {len(self.p_test_imgs)}\n"
        )

    @staticmethod
    def collate_fn(batch_data: List[dict]):
        """
        :param batch_data: [{
            'x': torch.Tensor,
            'm': torch.Tensor,
            "objectness_scores": List[float],
            "p_img": str,
        }]
        :return:
        """
        imgs: List[torch.Tensor] = list()
        masks: List[List[torch.Tensor]] = list()  # [[mask 0, ..., mask N], ..., [mask 0', ..., mask N']]
        p_imgs: List[str] = list()

        filenames: List[str] = list()

        for data in batch_data:
            m: torch.Tensor = data['m']  # h x w
            n, h, w = m.shape

            if n == 1 and (m == 0).sum() == (h * w):
                # in case there is no pseudo-masks for this image after filtering out
                continue

            imgs.append(data['x'])
            masks.append(data['m'])
            p_imgs.append(data["p_img"])
            filenames.append(data["filename"])

        return {
            "filename": filenames,
            'x': torch.stack(imgs, dim=0),
            'm': masks,
            "p_img": p_imgs,
        }

    def _get_pseudo_masks(self, filename: str) -> torch.Tensor:
        masks: np.ndarray = decode(self.pseudo_masks[filename])  # h x w (x n)
        if len(masks.shape) == 3:
            masks = masks.transpose((2, 0, 1))  # n x h x w
        elif len(masks.shape) == 2:
            masks = masks[None]  # 1 x h x w
        return torch.from_numpy(masks)

    def __getitem__(self, ind) -> dict:
        dict_data = dict()

        p_img = self.p_imgs[ind]
        image: Image.Image = Image.open(p_img).convert("RGB")

        filename = os.path.basename(p_img)
        dict_data["filename"] = filename

        if self.use_pseudo_masks and self.mode == "train":
            image = TF.resize(image, [self.img_size, self.img_size], IM.BILINEAR)
            masks: torch.Tensor = self._get_pseudo_masks(filename=filename)

        else:
            masks: Image.Image = torch.from_numpy(np.array(Image.open(self.p_gts[ind]).convert("L")))[None]  # 1 x h x w

        if self.mode == "train" and self.use_aug:
            image, masks = self._geometric_augmentations(
                image=image,
                mask=masks,
                random_scale_range=self.scale_range,
                random_crop_size=self.img_size,
                ignore_index=0,
                random_hflip_p=0.5
            )

            image: Image.Image = self._photometric_augmentations(image)

        image = TF.normalize(TF.to_tensor(image), self.mean, self.std)
        masks = np.asarray(masks, np.int64)
        if masks.max() > 1.0:
            masks = masks > 0

        dict_data.update({
            'x': image,
            'm': torch.tensor(masks, dtype=torch.long),
            "p_img": p_img
        })
        return dict_data
