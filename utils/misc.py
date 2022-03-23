from pathlib import Path
import random
from typing import Union, Optional, Dict, List, Set, Tuple
from argparse import Namespace
import numpy as np
import torch
from natsort import natsorted


def to_one_hot(mask: Union[np.ndarray, torch.Tensor], k: Optional[int] = None):
    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask)

    if k is None:
        k = len(torch.unique(mask))

    if len(mask.shape) == 2:
        h, w = mask.shape
        m_flat = mask.view(h * w)
        m_one_hot = torch.zeros((h * w, k), dtype=torch.long, device=mask.device)
        m_flat = m_flat.to(torch.long).unsqueeze(dim=-1)
        m_one_hot.scatter_(dim=1, index=m_flat, src=torch.ones_like(m_flat))
        m_one_hot = m_one_hot.view(h, w, k).permute(2, 0, 1)

    elif len(mask.shape) == 3:
        b, h, w = mask.shape
        m_flat = mask.view(b, h * w)
        m_one_hot = torch.zeros((b, h * w, k), dtype=torch.long, device=mask.device)
        m_flat = m_flat.to(torch.long).unsqueeze(dim=-1)
        m_one_hot.scatter_(dim=2, index=m_flat, src=torch.ones_like(m_flat))
        m_one_hot = m_one_hot.view(b, h, w, k).permute(0, 3, 1, 2)

    else:
        raise ValueError(len(mask.shape))

    return m_one_hot.to(torch.float32)


def set_seeds(seed: int):
    [func(seed) for func in (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed)]


def get_dataset(
        dir_dataset: str,
        dataset_name: str,
        mode: str,
        train_img_size: Optional[int] = None,
        eval_img_size: Optional[int] = None,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        use_pseudo_masks: bool = False,
        k: int = 2,
        use_copy_paste: bool = False,
        repeat_image: bool = False,
        n_percent: int = 100,
        n_copy_pastes: Optional[int] = None,
        pseudo_masks_fp: Optional[str] = None
) -> torch.utils.data.Dataset:
    if dataset_name == "cub2011":
        assert mode == "test", "cub2011 dataset is only for test."
        kwargs = {
            "dataset_class": "CUB2011Dataset",
            "dir_dataset": f"{dir_dataset}/CUB_200_2011",
            "dataset_mode": mode,
            "img_size": eval_img_size,
            "use_pseudo_masks": use_pseudo_masks
        }

    elif dataset_name == "flowers102":
        assert mode == "test", "flowers102 dataset is only for test."
        kwargs = {
            "dataset_class": "Flowers102Dataset",
            "dir_dataset": f"{dir_dataset}/Flowers-102",
            "dataset_mode": mode,
            "img_size": eval_img_size,
            "use_pseudo_masks": use_pseudo_masks
        }

    elif dataset_name == "ecssd":
        assert mode in "test", "ecssd dataset is only for test."
        kwargs = {
            "dataset_class": "ECSSDDataset",
            "dir_dataset": f"{dir_dataset}/ECSSD",
            "img_size": eval_img_size,
            "dataset_mode": mode
        }

    elif dataset_name == "duts":
        assert mode in ["train", "test"]
        kwargs = {
            "dataset_class": "DUTSDataset",
            "dir_dataset": f"{dir_dataset}/DUTS",
            "img_size": train_img_size,
            "scale_range": scale_range,
            "dataset_mode": mode,
            "use_pseudo_masks": use_pseudo_masks,
            "pseudo_masks_fp": pseudo_masks_fp,
            "use_copy_paste": use_copy_paste
        }

    elif dataset_name == "dut_omron":
        assert mode == "test", "dut_omron dataset is only for test."
        kwargs = {
            "dataset_class": "DUTOMRONDataset",
            "dir_dataset": f"{dir_dataset}/DUT-OMRON",
            "img_size": eval_img_size,
            "dataset_mode": mode
        }

    elif dataset_name == "hku_is":
        assert mode == "test", "hku_is dataset is only for test."
        kwargs = {
            "dataset_class": "HKUISDataset",
            "dataset_mode": "test",
            "dir_dataset": f"{dir_dataset}/HKU-IS",
            "img_size": eval_img_size
        }

    elif dataset_name == "sod":
        assert mode == "test", "sod dataset is only for test."
        kwargs = {
            "dataset_class": "SODDataset",
            "dataset_mode": "test",
            "dir_dataset": f"{dir_dataset}/SOD",
            "img_size": eval_img_size
        }

    elif dataset_name == "imagenet1k":
        assert mode == "train", "sod dataset is only for train."
        kwargs = {
            "dataset_class": "ImageNet1KDataset",
            "dir_dataset": f"{dir_dataset}/ImageNet2012",
            "train_image_size": train_img_size,
            "train_crop_size": train_img_size,
            "use_copy_paste": use_copy_paste,
            "scale_range": scale_range,
            "dataset_mode": mode,
            "repeat_image": repeat_image,
            'k': k,
            "n_percent": n_percent,
            "pseudo_masks_fp": pseudo_masks_fp,
            "n_copy_pastes": n_copy_pastes
        }

    else:
        raise ValueError(f"Invalid dataset_name {dataset_name}.")
    import datasets
    dataset_class, dataset_mode = kwargs.pop("dataset_class"), kwargs.pop("dataset_mode")
    dataset = datasets.__dict__[dataset_class]
    dataset = dataset(**kwargs)
    dataset.set_mode(dataset_mode)
    return dataset


def get_lr_scheduler(optimizer, n_epochs: int, n_iters_per_epoch=-1, mode="poly", **kwargs):
    if mode == "poly":
        from utils.lr_scheduler import Poly
        lr_scheduler = Poly(optimizer, n_epochs, n_iters_per_epoch, **kwargs)
    else:
        raise ValueError(f"Unsupported lr scheduler type: {mode} (currently [poly] supported)")
    return lr_scheduler


def get_model(
        arch: str,
        patch_size: Optional[int] = None,
        training_method: Optional[str] = None,
        configs: Optional[Namespace] = None,
        **kwargs
):
    if arch == "maskformer":
        assert configs is not None
        from networks.maskformer.maskformer import MaskFormer
        model = MaskFormer(
            n_queries=configs.n_queries,
            n_decoder_layers=configs.n_decoder_layers,
            learnable_pixel_decoder=configs.learnable_pixel_decoder,
            lateral_connection=configs.lateral_connection,
            return_intermediate=configs.loss_every_decoder_layer,
            scale_factor=configs.scale_factor,
            abs_2d_pe_init=configs.abs_2d_pe_init,
            use_binary_classifier=configs.use_binary_classifier,
            arch=configs.arch,
            training_method=configs.training_method,
            patch_size=configs.patch_size
        )

        for n, p in model.encoder.named_parameters():
            p.requires_grad_(True)

    elif "vit" in arch:
        import networks.vision_transformer as vits
        import networks.timm_deit as timm_deit
        if training_method == "dino":
            arch = arch.replace("vit", "deit") if arch.find("small") != -1 else arch
            model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
            load_model(model, arch, patch_size)

        elif training_method == "deit":
            assert patch_size == 16
            model = timm_deit.deit_small_distilled_patch16_224(True)

        elif training_method == "supervised":
            assert patch_size == 16
            # model = timm_deit.deit_small_patch16_224(True)

            state_dict: dict = torch.load(
                "/users/gyungin/selfmask/networks/pretrained/deit_small_patch16_224-cd65a155.pth"
            )["model"]
            for k in list(state_dict.keys()):
                if k in ["head.weight", "head.bias"]:  # classifier head, which is not used in our network
                    state_dict.pop(k)

            model = get_model(arch="vit_small", patch_size=16, training_method="dino")
            model.load_state_dict(state_dict=state_dict, strict=True)

        else:
            raise NotImplementedError
        print(f"{arch}_p{patch_size}_{training_method} is built.")

    elif arch == "resnet50":
        from networks.resnet import ResNet50
        assert training_method in ["mocov2", "swav", "supervised"]
        model = ResNet50(training_method)

    else:
        raise ValueError(f"{arch} is not supported arch. Choose from [maskformer, resnet50, vit, dino]")
    return model


def load_model(model, arch: str, patch_size: int) -> None:
    url = None
    if arch == "deit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    elif arch == "deit_small" and patch_size == 8:
        # model used for visualizations in our paper
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    elif arch == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    elif arch == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    if url is not None:
        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("There is no reference weights available for this model => We use random weights.")


def get_image_paths(pattern, dir_base='.',  ext="png"):
    candidates = [p for p in Path(dir_base).rglob(f"*.{ext}")]
    assert len(candidates) > 0, f"No candidate files are found in subdirectories of {dir_base}"
    p_imgs = list()
    for p in candidates:
        if pattern.match(str(p.resolve())) is not None:
            p_imgs.append(str(p.resolve()))
    p_imgs = natsorted(p_imgs)

    if len(p_imgs) == 0:
        print(f"\nFound no files matching {pattern}\n")
        print("Candidate files:")
        for p in candidates:
            print(str(p.resolve()))
        raise FileNotFoundError

    print(f"Found {len(p_imgs)} matching the pattern.")
    return p_imgs


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


def filter_masks(
        dt_masks: torch.Tensor,
        remove_long_masks: bool = True,
        remove_small_large_masks: bool = False
) -> torch.Tensor:
    list_filtered_masks = list()
    new_index_to_prev_index: dict = dict()
    h, w = dt_masks.shape[-2:]
    new_index = 0

    mask_index_to_bbox: dict = mask_to_bbox(dt_masks.cpu().numpy())

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