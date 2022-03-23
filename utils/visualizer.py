import os
from typing import Optional, Union, List, Tuple, Dict
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


class Visualizer:
    @staticmethod
    def plot(
            coord_to_img: Dict[Tuple[int, int], Dict[str, Union[str, np.ndarray]]],
            fp: str
    ) -> None:
        """
        :param coord_to_img: {
            (int, int): {
                "image": np.ndarray,
                "xlabel": str,
                "axis_color": str
            }
        }
        :param fp:
        """
        nrows, ncols = np.max(list(coord_to_img.keys()), axis=0) + 1
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows), squeeze=False)

        for coord, img_info in coord_to_img.items():
            img: np.ndarray = img_info["image"]
            xlabel: str = img_info.get("xlabel", '')
            axis_color: str = img_info.get("axis_color", 'k')
            ax[coord].imshow(img) if len(img.shape) == 3 else ax[coord].imshow(img, cmap="gray")

            ax[coord].set_xlabel(xlabel)
            ax[coord].set_xticks([])
            ax[coord].set_yticks([])
            [ax[coord].spines[d].set_color(axis_color) for d in ("top", "bottom", "left", "right")]
            if axis_color not in ['k', "black"]:
                [ax[coord].spines[d].set_linewidth(3) for d in ("top", "bottom", "left", "right")]

        plt.tight_layout(pad=0.5)
        plt.savefig(fp)
        plt.close()
        return

    def visualize(
            self,
            input_image: np.ndarray,
            gt_masks: torch.Tensor,
            best_pred_masks: torch.Tensor,
            best_pred_masks_index: List[int],
            fp: str,
            all_pred_masks: Optional[torch.Tensor] = None,
            max_ncols: int = 10
    ) -> None:
        """
        :param input_image: (3 x H x W)
        :param gt_masks: (M x H x W)
        :param best_pred_masks: (M x H x W)
        :param best_pred_masks_index (M)
        :param fp:
        :param all_pred_masks (N_queries x H x W):
        :param max_ncols: (1)
        :return:
        """

        assert len(gt_masks) >= len(best_pred_masks), f"{len(gt_masks)} !>= {len(best_pred_masks)}"
        assert len(gt_masks) >= len(best_pred_masks_index), f"{len(gt_masks)} !>= {len(best_pred_masks_index)}"

        dir_vis: str = os.path.dirname(fp)
        os.makedirs(dir_vis, exist_ok=True)

        shared_name, ext = os.path.splitext(fp.split('/')[-1])

        # input_img: np.ndarray = self.dataset.denormalize(input_image.cpu(), as_pil=False)
        self.plot({(0, 0): {"image": input_image}}, fp=f"{dir_vis}/{shared_name}_input{ext}")

        coord_to_img: dict = dict()
        for col, (gt_mask, pred_mask, index) in enumerate(zip(gt_masks, best_pred_masks, best_pred_masks_index)):
            coord_to_img.update({
                (0, col): {"image": gt_mask.cpu().numpy(), "xlabel": f"gt mask {col + 1}"},
                (1, col): {"image": pred_mask.cpu().numpy(), "xlabel": f"query {index}"}
            })
        self.plot(coord_to_img, fp=f"{dir_vis}/{shared_name}_gt_comp{ext}")

        if all_pred_masks is not None:
            coord_to_img: dict = dict()
            for cnt, pred_mask in enumerate(all_pred_masks):
                coord_to_img.update({
                    (cnt // max_ncols, cnt % max_ncols): {
                        "image": pred_mask.cpu().numpy(),
                        "axis_color": "red" if cnt in best_pred_masks_index else 'k'
                    }
                })
            self.plot(coord_to_img, fp=f"{dir_vis}/{shared_name}_all_pred{ext}")

def denormalize(
        image_tensor: torch.Tensor,
        as_pil: bool = True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
):
    image_tensor = image_tensor.cpu()
    if len(image_tensor.shape) == 3:
        image_tensor *= torch.tensor().view(3, 1, 1)
        image_tensor += torch.tensor(mean).view(3, 1, 1)
    elif len(image_tensor.shape) == 4:
        image_tensor *= torch.tensor(std).view(1, 3, 1, 1)
        image_tensor += torch.tensor(mean).view(1, 3, 1, 1)
    else:
        raise ValueError(f"image_tensor should have 3 or 4 dimensions, got {len(image_tensor.shape)}.")
    image_tensor *= 255

    image_array: np.ndarray = image_tensor.numpy().squeeze()
    image_array: np.ndarray = np.clip(image_array, 0, 255).astype(np.uint8)

    if as_pil:
        if len(image_array.shape) == 3:
            image_array = np.transpose(image_array, (1, 2, 0))
            return Image.fromarray(image_array)

        else:
            image_array = np.transpose(image_array, (0, 2, 3, 1))
            list_pil_images: list = list()
            for arr in image_array:
                list_pil_images.append(Image.fromarray(arr))
            return list_pil_images
    else:
        return image_tensor
