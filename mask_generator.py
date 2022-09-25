from typing import Union, Optional, List, Dict, Tuple
from collections import defaultdict
from random import choice
from time import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pycocotools.mask import encode, decode
import clusterings
from datasets.custom_dataset import CustomDataset
from utils.misc import get_model, to_one_hot


class MaskGenerator:
    def __init__(
            self,
            cluster_sizes: Tuple[int, ...] = (2, 3, 4),
            cluster_type: str = "spectral",
            feature_types: List[str] = ["mocov2", "swav", "dino"],
            use_gpu: bool = True,  # whether to use a gpu for clustering
            device: torch.device = torch.device("cuda:0")
    ):
        assert cluster_type in ["k-means", "spectral"]
        self.cluster_sizes: Tuple[int, ...] = cluster_sizes
        self.feature_types: List[str] = feature_types
        self.device: torch.device = device

        if cluster_type == "k-means":
            self.clusterer: callable = clusterings.KMeansClustering(use_gpu=use_gpu)
        else:
            self.clusterer: callable = clusterings.SpectralClustering(use_gpu=use_gpu)

    @staticmethod
    def mask_to_bbox(mask: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
        """Given a binary mask, return a list of bounding box coordinates (ymin, ymax, xmin, xmax)."""
        mask_index_to_bbox = dict()
        if len(mask.shape) == 2:
            mask = mask[None]

        for mask_index, m in enumerate(mask):
            y_coords, x_coords = np.where(m)
            try:
                ymin, ymax, xmin, xmax = np.min(y_coords), np.max(y_coords), np.min(x_coords), np.max(x_coords)
            except ValueError:  # a mask which does not predict anything.
                continue
            mask_index_to_bbox[mask_index] = (ymin.item(), ymax.item(), xmin.item(), xmax.item())
        return mask_index_to_bbox

    @staticmethod
    def filter_masks(
            dt_masks: torch.Tensor,
            mask_index_to_bbox: dict,
            remove_long_masks: bool = True,
            remove_small_large_masks: bool = False
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        list_filtered_masks: list = list()
        new_index_to_prev_index: dict = dict()
        h, w = dt_masks.shape[-2:]
        new_index = 0

        for mask_index, bbox in mask_index_to_bbox.items():
            ymin, ymax, xmin, xmax = bbox
            if remove_long_masks:
                if ymin == 0 and ymax + 1 == h:
                    continue
                elif xmin == 0 and xmax + 1 == w:
                    continue

            if remove_small_large_masks:
                if dt_masks[mask_index].sum() < 0.05 * h * w:
                    continue
                elif (xmax - xmin) * (ymax - ymin) > 0.95 * h * w:
                    continue
            list_filtered_masks.append(dt_masks[mask_index])
            new_index_to_prev_index[new_index] = mask_index
            new_index += 1
        try:
            return torch.stack(list_filtered_masks, dim=0), new_index_to_prev_index
        except RuntimeError:  # rare case where all predictions are filtered.
            return dt_masks, {i: i for i in range(len(dt_masks))}

    def _visualise(
            self,
            image: torch.Tensor,
            best_mask: np.ndarray,
            result_dir: str,
            batch_index: int,
            mask: Optional[torch.Tensor] = None,
    ):
        mask: np.ndarray = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        h, w = image.shape[-2:]

        ratio = h / w
        plt.figure(figsize=(1, ratio))
        plt.imshow(self.dataset.denormalize(image))
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f"{result_dir}/{batch_index:04d}_input_image.png")
        plt.close()

        if mask is not None:
            plt.figure(figsize=(1, ratio))
            plt.imshow(mask, interpolation="none")
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(f"{result_dir}/{batch_index:04d}_gt_mask.png")
            plt.close()

        for mask_name, mask in zip(["best"], [best_mask]):
            plt.figure(figsize=(1, ratio))
            plt.imshow(mask, interpolation="none")
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(f"{result_dir}/{batch_index:04d}_{mask_name}.png")
            plt.close()

    @staticmethod
    def pad_input_image(image: torch.Tensor, total_stride: int):
        assert len(image.shape) == 4
        h_image, w_image = image.shape[-2:]
        pad_w = (total_stride - w_image % total_stride) % total_stride
        pad_h = (total_stride - h_image % total_stride) % total_stride
        image: torch.Tensor = TF.pad(image, [0, 0, pad_w, pad_h])

        h_padded_image, w_padded_image = h_image + pad_h, w_image + pad_w
        h_feat, w_feat = h_padded_image // total_stride, w_padded_image // total_stride
        return image, h_feat, w_feat

    def extract_candidate_masks(self, p_images: List[str]) -> Dict[str, np.ndarray]:
        filename_to_candidate_masks: Dict[str, list] = defaultdict(list)
        for feature_type in self.feature_types:
            print(f"========== Generating candidate masks with {feature_type} ==========")
            # load a model
            network = get_model(
                arch={"mocov2": "resnet50", "swav": "resnet50", "dino": "vit_small"}[feature_type],
                training_method=feature_type,
                patch_size=16  # for dino
            ).to(self.device)
            network.eval()

            dataloader = DataLoader(dataset=CustomDataset(image_paths=p_images), batch_size=1, shuffle=False)
            for dict_data in tqdm(dataloader):
                batch_imgs: torch.Tensor = dict_data["img"].to(self.device)  # b (=1) x 3 x H x W
                h_image, w_image = batch_imgs.shape[-2:]

                filename: str = dict_data["filename"][0]
                # extract features froom a given model
                if feature_type in ["mocov2", "swav"]:
                    # dilated resnet50
                    total_stride: int = 8
                    batch_imgs, h_feat, w_feat = self.pad_input_image(batch_imgs, total_stride=total_stride)
                    features: torch.Tensor = network(batch_imgs)[-1]
                else:
                    # ViT-S/16
                    total_stride: int = 16
                    batch_imgs, h_feat, w_feat = self.pad_input_image(batch_imgs, total_stride=total_stride)
                    try:
                        batch_tokens = network(batch_imgs, layer="layer12")
                    except AttributeError:
                        batch_tokens = network.encoder(batch_imgs, layer="layer12")

                    batch_patch_tokens = batch_tokens[:, 1:, :]  # b (=1) x (h_feat * w_feat) x n_dims

                    # batch_patch_tokens
                    batch_patch_tokens = batch_patch_tokens.view(batch_imgs.shape[0], h_feat, w_feat, -1)
                    features = batch_patch_tokens.permute(0, 3, 1, 2)

                # upsample by 2 before clustering
                features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=True)

                # iterate over a list of given cluster sizes
                for k in self.cluster_sizes:
                    # clustering
                    # batch_clusters: b (=1) x h x w
                    batch_clusters: np.ndarray = self.clusterer(features, k)

                    # batch_one_hot_masks: b (=1) x k x h x w -> k x h x w
                    batch_one_hot_masks: np.ndarray = to_one_hot(batch_clusters[0])

                    batch_one_hot_masks: torch.Tensor = F.interpolate(
                        batch_one_hot_masks[None],
                        scale_factor=(total_stride // 2, total_stride // 2),
                        mode="nearest"
                    )[0]
                    
                    batch_one_hot_masks: torch.Tensor = batch_one_hot_masks[..., :h_image, :w_image]
                    one_hot_masks: np.ndarray = batch_one_hot_masks.numpy().astype(np.uint8)
                    filename_to_candidate_masks[filename].append(one_hot_masks)

        # concatenate a list of masks for each image into a numpy array
        for filename, candidate_mask in filename_to_candidate_masks.items():
            filename_to_candidate_masks[filename]: np.ndaray = np.concatenate(candidate_mask, axis=0)
        return filename_to_candidate_masks

    def vote_mask(
            self,
            batch_pred_masks,
            remove_long_masks: bool = True,
            remove_small_large_masks: bool = False
    ):
        mask_index_to_bbox = self.mask_to_bbox(batch_pred_masks.squeeze(dim=0).cpu().numpy())
        masks, new_index_to_prev_index = self.filter_masks(
            dt_masks=batch_pred_masks.squeeze(dim=0),
            mask_index_to_bbox=mask_index_to_bbox,
            remove_long_masks=remove_long_masks,
            remove_small_large_masks=remove_small_large_masks
        )  # n x h x w
        masks_t = masks.permute(1, 2, 0)
        try:
            intersection = torch.logical_and(masks[..., None], masks_t[None]).sum(dim=(1, 2))  # n x n
            union = torch.logical_or(masks[..., None], masks_t[None]).sum(dim=(1, 2))  # n x n
        except RuntimeError:
            masks = masks.cpu()
            masks_t = masks_t.cpu()
            intersection = torch.logical_and(masks[..., None], masks_t[None]).sum(dim=(1, 2))  # n x n
            union = torch.logical_or(masks[..., None], masks_t[None]).sum(dim=(1, 2))  # n x n

        iou_table = intersection / (union + 1e-7)  # n x n
        ious = iou_table.sum(dim=1)  # n
        sorted_index = torch.argsort(ious, descending=True)
        best_mask_index = sorted_index[0].cpu().item()
        best_mask = masks[best_mask_index]
        return best_mask, best_mask_index, new_index_to_prev_index

    @torch.no_grad()
    def __call__(
            self,
            p_images: List[str],
            remove_long_masks: bool = True,
            remove_small_large_masks: bool = False,
    ) -> Dict[str, dict]:
        filename_to_encoded_salient_mask: Dict[str, np.ndarray] = dict()
        filename_to_candidate_masks: Dict[str, np.ndarray] = self.extract_candidate_masks(p_images=p_images)
        for filename, candidate_masks in filename_to_candidate_masks.items():
            candidate_masks: torch.Tensor = torch.tensor(candidate_masks).to(self.device)[None]  # 1 x k x h x w

            # salient_mask: torch.Tensor, h x w
            salient_mask, _, _ = self.vote_mask(
                candidate_masks,
                remove_long_masks=remove_long_masks,
                remove_small_large_masks=remove_small_large_masks
            )

            filename_to_encoded_salient_mask[filename] = encode(np.asfortranarray(salient_mask.cpu().numpy()))
        return filename_to_encoded_salient_mask


if __name__ == '__main__':
    import os
    from argparse import ArgumentParser
    import ujson as json
    from utils.misc import set_seeds

    parser = ArgumentParser()
    parser.add_argument("--seed", '-s', type=int, default=0)
    parser.add_argument("--p_images", type=str, nargs='+', default=[
        "/users/gyungin/datasets/DUTS/DUTS-TR-Image/ILSVRC2012_test_00000004.jpg"
    ])
    parser.add_argument("--fp", type=str, default="your_pseudo_masks.json")
    parser.add_argument("--cluster_type", '-ct', type=str, default="spectral", choices=["k-means", "spectral"])
    parser.add_argument("--cluster_sizes", '-cs', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument("--feature_types", '-ft', type=str, nargs='+', default=["mocov2", "swav", "dino"])
    parser.add_argument("--patch_size", '-ps', type=int, default=16)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu_id", type=int, default=3)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    seed: int = args.seed
    set_seeds(seed)

    # instantiate a mask generator
    mask_generator: callable = MaskGenerator()

    # set image paths
    p_images: List[str] = args.p_images

    # generate masks
    filename_to_encoded_salient_mask: Dict[str, dict] = mask_generator(p_images=p_images)

    # save the resulting file if necessary
    json.dump(filename_to_encoded_salient_mask, open(args.fp, 'w'), reject_bytes=False)

    # decode
    for filename, encoded_salient_mask in filename_to_encoded_salient_mask.items():
        salient_mask: np.ndarray = decode(encoded_salient_mask)
        break
