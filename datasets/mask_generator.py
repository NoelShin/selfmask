from typing import Optional, Union, List, Tuple
from copy import deepcopy
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm
import clusterings
from utils.misc import to_one_hot
from pycocotools.mask import encode
from torchvision.transforms.functional import resize


class MaskGenerator:
    def __init__(self, fp: Optional[str] = None):
        if fp is not None and os.path.exists(fp):
            self.mask_info: dict = json.load(open(fp, "r"))
        else:
            self.mask_info: dict = dict()
        self.fp = fp

    @torch.no_grad()
    def __call__(
            self,
            image_paths: List[str],
            network: callable,
            k: int,
            image_size: Optional[Union[int, Tuple[int, int]]] = None,
            arch: str = "resnet50",
            cluster_type: str = "spectral",
            scale_factor: int = 2,
            batch_size: Optional[int] = 1,
            num_workers: int = 8,
            pin_memory: bool = True,
            **dataloader_kwargs
    ):
        """
        Generate pseudo-masks from training images and update masks accordingly.
        This will first check whether there exists the file path fp containing the previously generated masks.

        If the file exists:
            it will skip the generation process.
        else:
            it will generate new masks with the given k and save the info for the new masks at fp.

        The resulting file will be a dictionary: { "image_filename": Run-length encoding (RLE) }

        If n_images is given, pick random n_images images to make pseudo-masks of. This is used to subsample ImageNet1k
        images due to its large number of images. No-op for other datasets such as CUB2011, Flowers102, etc. Set 0 to
        use all ImageNet1k training images.
        """
        assert len(image_paths) > 0
        if cluster_type == "spectral":
            clusterer = clusterings.SpectralClustering(use_gpu=True)
        elif cluster_type == "kmeans":
            clusterer = clusterings.KMeansClustering(use_gpu=True)
        else:
            raise ValueError(cluster_type)
        dataset = MaskDataset(image_paths, image_size=image_size)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **dataloader_kwargs
        )
        n_total_iters: int = len(dataloader)
        for ind_batch, dict_data in enumerate(tqdm(dataloader), start=1):
            batch_filenames: List[str] = dict_data["filename"]

            batch_imgs: torch.Tensor = dict_data["img"].to("cuda:0")
            h_image, w_image = batch_imgs.shape[-2:]

            ps = 8 if "resnet50" in arch else 16
            pad_w, pad_h = (ps - w_image % ps) % ps, (ps - h_image % ps) % ps
            batch_padded_imgs: torch.Tensor = TF.pad(batch_imgs, [0, 0, pad_w, pad_h])

            h_padded_image, w_padded_image = h_image + pad_h, w_image + pad_w

            if arch == "resnet50":
                features: torch.Tensor = network(batch_padded_imgs)[-1]

            else:
                try:
                    batch_tokens = network(batch_padded_imgs, layer="layer12")
                except AttributeError:
                    batch_tokens = network.encoder(batch_padded_imgs, layer="layer12")

                batch_cls_token, batch_patch_tokens = batch_tokens[:, 0, :], batch_tokens[:, 1:, :]

                # spatial_size, n_embs = int(sqrt(batch_patch_tokens.shape[1])), batch_patch_tokens.shape[-1]
                h_feature, w_feature = h_padded_image // ps, w_padded_image // ps
                batch_patch_tokens = batch_patch_tokens.view(batch_padded_imgs.shape[0], h_feature, w_feature, -1)
                features = batch_patch_tokens.permute(0, 3, 1, 2)

            # upsampling the output features from the encoder
            features: torch.Tensor = clusterer.upsample(features.detach(), scale_factor=scale_factor)

            # clustering
            batch_clusters: np.ndarray = clusterer(features, k)  # b x h x w
            batch_one_hot_masks: torch.Tensor = to_one_hot(batch_clusters)  # b x k x h x w

            batch_one_hot_masks: torch.Tensor = F.interpolate(
                batch_one_hot_masks,
                scale_factor=(ps // scale_factor, ps // scale_factor),  # note that this part is different from validation
                mode="nearest"
            ).numpy().astype(np.uint8)

            batch_one_hot_masks = batch_one_hot_masks[..., :h_image, :w_image]
            assert batch_one_hot_masks.shape[-2:] == batch_imgs.shape[-2:], \
                f"{batch_one_hot_masks.shape[-2:]} != {batch_imgs.shape[-2:]}"

            batch_one_hot_masks = batch_one_hot_masks.transpose((0, 2, 3, 1))  # b x h x w x k

            for i, (filename, one_hot_masks) in enumerate(zip(batch_filenames, batch_one_hot_masks)):
                rles: dict = encode(np.asfortranarray(one_hot_masks))
                self.mask_info.update({filename: rles})

            if (ind_batch % (n_total_iters // 20) == 0 or ind_batch == n_total_iters) and self.fp is not None:
                # store intermediate results every 5% of the loop.
                json.dump(self.mask_info, open(self.fp, "w"), reject_bytes=False)

            if ind_batch == 1:
                json.dump(self.mask_info, open(self.fp, "w"), reject_bytes=False)

        json.dump(self.mask_info, open(self.fp, "w"), reject_bytes=False)
        return self.mask_info


class MaskDataset(Dataset):
    def __init__(
            self,
            image_paths: List[str],
            image_size: Optional[int] = None,
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        assert len(image_paths) > 0, f"No image paths are given: {len(image_paths)}."
        self.image_paths: List[str] = image_paths
        self.image_size = image_size
        # self.image_size: Tuple[int, int] = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.mean: Tuple[float, float, float] = mean
        self.std: Tuple[float, float, float] = std

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path: str = self.image_paths[index]
        img: Image.Image = Image.open(image_path).convert("RGB")
        if self.image_size is not None:
            img = resize(img, size=self.image_size, interpolation=Image.BILINEAR)
            # img = img.resize(self.image_size, resample=Image.BILINEAR)
        img = TF.normalize(TF.to_tensor(img), mean=self.mean, std=self.std)
        return {"img": img, "filename": image_path.split('/')[-1]}


def decode_masks(masks_info: dict) -> torch.Tensor:
    """
    :param masks_info: {"mask_kN": np.ndarray}
    :return: (N x height x width) one-hot masks
    """
    masks_info = deepcopy(masks_info)
    h, w = masks_info.pop("height"), masks_info.pop("width")
    n_masks: int = len(masks_info)
    masks = torch.zeros((n_masks, h, w), torch.bool)
    for num_mask in range(n_masks):
        x_coords: List[int] = masks_info[f"mask_{num_mask}"]["x_coords"]
        y_coords: List[int] = masks_info[f"mask_{num_mask}"]["y_coords"]
        masks[num_mask, x_coords, y_coords] = True
    return masks


if __name__ == '__main__':
    import os
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    from argparse import Namespace
    from glob import glob
    import ujson as json
    import yaml
    from utils.misc import get_model, set_seeds
    import torch

    parser = ArgumentParser()
    parser.add_argument("--k", type=int)
    parser.add_argument("--arch", '-a', type=str, default="vit_small", choices=["maskformer", "vit_small", "resnet50"])
    parser.add_argument("--training_method", '-tm', type=str, default="dino", choices=["dino", "swav", "mocov2"])
    parser.add_argument("--patch_size", '-ps', type=int, default=16)
    # parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--dataset_name", type=str, default="imagenet1k", choices=["duts", "imagenet1k"])
    parser.add_argument("--dir_dataset", type=str, default="/users/gyungin/datasets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=str, default='3')
    parser.add_argument("--batch_size", '-bs', type=int, default=1)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    k: int = args.k
    # image_size: int = args.image_size
    seed: int = args.seed
    dataset_name: str = args.dataset_name

    # prepare a model
    set_seeds(seed=seed)
    torch.backends.cudnn.benchmark = True

    if args.arch == "maskformer":
        config: str = "/users/gyungin/sos/configs/imagenet1k-k2-224.yaml"
        args = yaml.safe_load(open(f"{config}", 'r'))
        args: Namespace = Namespace(**args)
        args.dataset_name = dataset_name  #

    network = get_model(
        arch=args.arch, training_method=args.training_method, patch_size=args.patch_size, configs=args
    ).to("cuda:0")
    network.eval()

    if args.dataset_name == "imagenet1k":
        dir_dataset: str = f"{args.dir_dataset}/ImageNet2012"

        # load another file that contains filename information.
        masks_info: dict = json.load(
            open(f"/scratch/shared/beegfs/gyungin/datasets/ImageNet2012/masks_k2_p10_is224_s{seed}.json", "r")
        )

        list_filenames: List[str] = sorted(list(masks_info.keys()))
        list_fp: List[str] = sorted([
            f"{dir_dataset}/train/{filename.split('_')[0]}/{filename}" for filename in list_filenames
        ])
        image_size = 256 # shorter edge of the image
        # mask_generator = MaskGenerator(
        #     fp=f"{args.dataset_name}_{args.arch}_{args.training_method}_masks_k{k}_p10_is{image_size}_s{seed}.json"
        # )

    else:
        dir_dataset: str = f"{args.dir_dataset}/DUTS"
        list_fp: List[str] = sorted(glob(
            f"{dir_dataset}/DUTS-TR-Image/*"
        ))
        image_size = None

    if args.training_method == "dino":
        fp = f"{args.dataset_name}_{args.arch}_{args.training_method}_p{args.patch_size}_masks_k{k}_s{seed}.json"
    else:
        fp = f"{args.dataset_name}_{args.arch}_{args.training_method}_masks_k{k}_s{seed}.json"

    mask_generator = MaskGenerator(fp=fp)
    print('\n', fp)
    print(f"Total {len(list_fp)} images will be used for generating pseudo masks (k={k}).")

    mask_generator(
        image_paths=list_fp,
        network=network,
        k=k,
        image_size=image_size,
        arch=args.arch,
        cluster_type="spectral",
        scale_factor=2,
        batch_size=args.batch_size,
    )