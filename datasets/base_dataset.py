import os
from typing import Dict, Optional, Union, List, Tuple
import ujson as json
from time import time
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ColorJitter, RandomApply, RandomGrayscale
from torchvision.transforms.functional import InterpolationMode as IM
import torchvision.transforms.functional as TF
from tqdm import tqdm
# from datasets.mask_generator import MaskGenerator
from datasets.augmentations.geometric_transforms import random_crop, random_hflip, random_scale, resize
from datasets.augmentations.gaussian_blur import GaussianBlur


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.dir_dataset: str
        self.name: str
        self.ignore_index: int = -1
        self.img_size: Tuple[int, int]
        self.list_mask_info: List[dict] = list()
        self.mode: str = ''
        self.p_imgs: List[str]
        self.p_gts: List[str]

        self.mean: Tuple[float, float, float]
        self.std: Tuple[float, float, float]
        self.use_aug: bool
        self.use_pseudo_masks: bool = False

        self.rank: Optional[int] = None
        self.world_size: Optional[int] = None

    def denormalize(self, image_tensor: torch.Tensor, as_pil=True):
        if len(image_tensor.shape) == 3:
            image_tensor *= torch.tensor(self.std).view(3, 1, 1)
            image_tensor += torch.tensor(self.mean).view(3, 1, 1)
        elif len(image_tensor.shape) == 4:
            image_tensor *= torch.tensor(self.std).view(1, 3, 1, 1)
            image_tensor += torch.tensor(self.mean).view(1, 3, 1, 1)
        else:
            raise ValueError(f"image_tensor should have 3 or 4 dimensions, got {len(image_tensor.shape)}.")
        image_tensor *= 255

        image_tensor = image_tensor.cpu().numpy().squeeze()
        image_tensor = np.clip(image_tensor, 0, 255).astype(np.uint8)
        image_tensor = np.transpose(image_tensor, (1, 2, 0))

        if as_pil:
            return Image.fromarray(image_tensor)
        return image_tensor

    @staticmethod
    def _geometric_augmentations(
            image: Image.Image,
            random_scale_range: Optional[Tuple[float, float]] = None,
            random_crop_size: Optional[int] = None,
            random_hflip_p: Optional[float] = None,
            mask: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
            ignore_index: Optional[int] = None
    ):
        """Note. image and mask are assumed to be of base size, thus share a spatial shape."""
        if random_scale_range is not None:
            image, mask = random_scale(image=image, random_scale_range=random_scale_range, mask=mask)

        if random_crop_size is not None:
            crop_size = (random_crop_size, random_crop_size)

            fill = tuple(np.array(image).mean(axis=(0, 1)).astype(np.uint8).tolist())
            image, offset = random_crop(image=image, crop_size=crop_size, fill=fill)

            if mask is not None:
                assert ignore_index is not None
                mask = random_crop(image=mask, crop_size=crop_size, fill=ignore_index, offset=offset)[0]

        if random_hflip_p is not None:
            image, mask = random_hflip(image=image, p=random_hflip_p, mask=mask)
        return image, mask

    @staticmethod
    def _photometric_augmentations(
            image: Image.Image,
            random_color_jitter: Optional[Dict[str, float]] = None,
            random_grayscale_p: Optional[float] = 0.2,
            random_gaussian_blur: bool = True
    ):
        if random_color_jitter is None:  # note that "is None" rather than "is not None"
            color_jitter = ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            image = RandomApply([color_jitter], p=0.8)(image)

        if random_grayscale_p is not None:
            image = RandomGrayscale(random_grayscale_p)(image)

        if random_gaussian_blur:
            w, h = image.size
            image = GaussianBlur(kernel_size=int((0.1 * min(w, h) // 2 * 2) + 1))(image)
        return image

    def _preprocess_data(
            self,
            image: Image.Image,
            mask: Union[Image.Image, np.ndarray, torch.Tensor],
            ignore_index: Optional[int] = None,
            pasting_image: Optional[Image.Image] = None,
            pasting_mask: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Image.Image, Union[Image.Image, torch.Tensor]]:
        """Prepare data in a proper form for either training (data augmentation) or validation."""

        # resize to base size
        img_size = getattr(self, f"{self.mode}_image_size")
        image: Image.Image = resize(image, size=img_size, edge="shorter", interpolation="bilinear")

        if not isinstance(mask, torch.Tensor):
            mask: torch.Tensor = torch.tensor(mask)

        if self.mode == "train" and self.use_aug:
            assert ignore_index is not None
            image, mask = self._geometric_augmentations(
                image=image,
                mask=mask,
                random_scale_range=self.scale_range,
                random_crop_size=self.train_crop_size,
                ignore_index=ignore_index,
                random_hflip_p=0.5
            )

            image: Image.Image = self._photometric_augmentations(image)

        non_empty_masks = mask.sum(dim=(-2, -1)) > 0
        mask = mask[non_empty_masks]

        return image, mask

    def get_dataloader(self, with_tbar=False, **kwargs):
        try:
            world_size: Optional[int] = kwargs.pop("world_size")
            rank: int = kwargs.pop("rank")

            if world_size is None or rank == -1:
                sampler = None

            else:
                if "shuffle" in kwargs:
                    # shuffle is exclusive with sampler option when constructing a DataLoader
                    shuffle: bool = kwargs.pop("shuffle")
                else:
                    shuffle: bool = True
                sampler = torch.utils.data.distributed.DistributedSampler(
                    self, num_replicas=world_size, rank=rank, shuffle=shuffle
                )
        except KeyError:
            sampler = None

        if with_tbar:
            dataloader = DataLoader(self, sampler=sampler, **kwargs)
            iter_dataloader = iter(dataloader)
            return iter_dataloader, tqdm(range(len(dataloader)))

        else:
            return iter(DataLoader(self, sampler=sampler, **kwargs))

    def set_mode(self, mode: str) -> None:
        """
        Set dataset mode. Note that this changes the files that dataloader loads.
        """
        self.p_imgs, self.p_gts = getattr(self, f"p_{mode}_imgs"), getattr(self, f"p_{mode}_gts")
        self.mode = mode
        print(f"dataset mode: {mode}")

    def use_data_augmentation_(self, flag: bool) -> None:
        """
        Turn on/off data augmentation. Note that when mode == eval or test, data augmentation is not applied even if
        self.use_aug == True.
        """
        self.use_aug = flag
        print(f"Data augmentation is turned {'on' if flag else 'off'} for training.")

    # @torch.no_grad()
    # def generate_pseudo_masks(
    #         self,
    #         network: callable,
    #         k: int,
    #         scale_factor: int,
    #         arch: str,
    #         fp: Optional[str] = None
    # ) -> None:
    #     if os.path.exists(fp):
    #         print(fp)
    #         masks_info: dict = json.load(open(fp, 'r'))
    #         print(len(masks_info))
    #
    #         self.p_imgs: List[str] = [
    #             f"{self.dir_dataset}/{filename}" for filename in sorted(list(masks_info.keys()))
    #         ]
    #
    #         self.p_train_imgs: List[str] = [
    #             f"{self.dir_dataset}/train/{filename.split('_')[0]}/{filename}" for filename in sorted(list(masks_info.keys()))
    #         ]
    #         self.list_mask_info.append(masks_info)
    #         return
    #     st = time()
    #
    #     # generate new pseudo-masks based on the given k
    #     mask_generator = MaskGenerator(fp=fp)
    #     network.eval()
    #     self.list_mask_info.append(mask_generator(
    #         image_paths=self.p_train_imgs,
    #         network=network,
    #         k=k,
    #         image_size=self.train_image_size,
    #         cluster_type="spectral",
    #         arch=arch,
    #         scale_factor=scale_factor,
    #         batch_size=48 if self.train_image_size < 256 else 16,
    #     ))
    #     print(
    #         f"New pseudo masks with k={k} and img size={self.train_image_size} are generated. ({time() - st:.3f} sec.)"
    #     )
    #     self.p_imgs = list(self.list_mask_info[0].keys())

    def __len__(self) -> int:
        return len(self.p_imgs)

    def __getitem__(self, ind) -> dict:
        dict_data = dict()
        p_img = self.p_imgs[ind]
        image = Image.open(p_img).convert("RGB")
        dict_data["filename"] = os.path.basename(p_img)
        dict_data["p_img"] = p_img

        if self.use_pseudo_masks and self.mode not in ["val", "test"]:
            m: torch.Tensor = self.masks[ind]
            m = m * 255 if m.max() == 1 else m
            m: Image.Image = Image.fromarray(m.numpy()).convert("L")

            w, h = image.size
            crop_size = min(w, h)

            # temporary
            image, m = TF.center_crop(image, crop_size), TF.center_crop(m, crop_size)
            image, m = TF.resize(image, self.img_size, IM.BILINEAR), TF.resize(m, self.img_size, IM.NEAREST)

        else:
            m: Image.Image = Image.open(self.p_gts[ind]).convert("L")

        image = TF.normalize(TF.to_tensor(image), self.mean, self.std)
        m = np.asarray(m, np.int64)
        if m.max() > 1.0:
            m = m > 0

        dict_data.update({'x': image, 'm': torch.tensor(m, dtype=torch.long)})
        return dict_data

