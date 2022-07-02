import os
from typing import Union, Optional, List, Dict
from math import sqrt
import numpy as np
import torch
import torch.nn.functional as F
from metrics.f_measure import FMeasure
from metrics.mae import compute_mae
from metrics.iou import compute_iou
from metrics.pixel_acc import compute_pixel_accuracy
from metrics.s_measure import SMeasure
from metrics.average_meter import AverageMeter
from utils.misc import get_dataset, get_model, set_seeds, to_one_hot, filter_masks
from base_structure import BaseStructure


class Evaluator(BaseStructure):
    def __init__(
            self,
            network: callable,
            arch: str = "vit_small",
            dir_dataset: str = "/scratch/shared/beegfs/gyungin/datasets",
            visualizer: Optional[callable] = None,
            debug: bool = False
    ):
        super(Evaluator, self).__init__(model=network, visualizer=visualizer)
        assert os.path.exists(dir_dataset), f"{dir_dataset} does not exist."
        self.arch: str = arch
        self.debug: bool = debug
        self.dir_dataset: str = dir_dataset
        self.network: callable = network
        self.visualizer: callable = visualizer

    def _init_meters(self) -> None:
        # meters for internally selected masks (IS)
        self.f_score = AverageMeter()
        self.f_max = AverageMeter()
        self.f_mean = AverageMeter()
        self.mae = AverageMeter()
        self.iou = AverageMeter()
        self.pixel_acc = AverageMeter()
        self.s_measure = AverageMeter()

        # meters for upper bound masks (UB)
        self.f_score_ub = AverageMeter()
        self.f_max_ub = AverageMeter()
        self.f_mean_ub = AverageMeter()
        self.mae_ub = AverageMeter()
        self.iou_ub = AverageMeter()
        self.pixel_acc_ub = AverageMeter()
        self.s_measure_ub = AverageMeter()

    def _update_meters(
            self,
            pred_mask: torch.Tensor,
            gt_mask: torch.Tensor,
            ub_mask: Optional[torch.Tensor] = None
    ) -> None:
        """
        :param pred_mask: (H x W)
        :param gt_mask: (H x W)
        """
        assert pred_mask.shape == gt_mask.shape, f"{pred_mask.shape} != {gt_mask.shape}"
        assert len(pred_mask.shape) == len(gt_mask.shape) == 2
        iou = compute_iou(pred_mask, gt_mask)
        f_measures = FMeasure()(pred_mask, gt_mask)
        mae = compute_mae(pred_mask, gt_mask)
        pixel_acc = compute_pixel_accuracy(pred_mask, gt_mask)

        self.iou.update(val=iou.numpy(), n=1)
        self.f_score.update(val=f_measures["f_measure"].numpy(), n=1)
        self.f_max.update(val=f_measures["f_max"].numpy(), n=1)
        self.f_mean.update(val=f_measures["f_mean"].numpy(), n=1)
        self.s_measure.update(val=SMeasure()(pred_mask=pred_mask, gt_mask=gt_mask.to(torch.float32)), n=1)
        self.mae.update(val=mae.numpy(), n=1)
        self.pixel_acc.update(val=pixel_acc.numpy(), n=1)

        if ub_mask is not None:
            assert ub_mask.shape == gt_mask.shape
            assert len(ub_mask.shape) == len(gt_mask.shape) == 2
            iou = compute_iou(ub_mask, gt_mask)
            f_measures = FMeasure()(ub_mask, gt_mask)
            mae = compute_mae(ub_mask, gt_mask)
            pixel_acc = compute_pixel_accuracy(ub_mask, gt_mask)

            self.iou_ub.update(val=iou.numpy(), n=1)
            self.f_score_ub.update(val=f_measures["f_measure"].numpy(), n=1)
            self.f_max_ub.update(val=f_measures["f_max"].numpy(), n=1)
            self.f_mean_ub.update(val=f_measures["f_mean"].numpy(), n=1)
            self.s_measure_ub.update(val=SMeasure()(pred_mask=ub_mask, gt_mask=gt_mask.to(torch.float32)), n=1)
            self.mae_ub.update(val=mae.numpy(), n=1)
            self.pixel_acc_ub.update(val=pixel_acc.numpy(), n=1)

    def _get_upper_bound_mask(
            self,
            pred_mask: torch.Tensor,
            gt_mask: torch.Tensor,
            quantity: str = "iou"
    ) -> torch.Tensor:
        """
        :param pred_mask: (n queries x H x W)
        :param gt_mask: (H x W) or (1 x h x w)
        :param quantity: a quantity that will be considered to pick the best mask. Default: "iou"
        :return: best_mask: (H x W)
        """
        if quantity == "iou":
            n_queries = pred_mask.shape[0]
            if len(gt_mask.shape) == 2:
                gt_mask = gt_mask[None, ...].repeat(n_queries, 1, 1)
            elif len(gt_mask.shape) == 3 and gt_mask.shape[0] == 1:
                gt_mask = gt_mask.repeat(n_queries, 1, 1)
            ious = compute_iou(pred_mask=pred_mask, gt_mask=gt_mask)
            index = torch.argmax(ious)

        elif quantity in ["f_measure", "f_max"]:
            list_f_value: list = list()
            for _pred_mask in pred_mask:
                list_f_value.append(FMeasure()(pred_mask=_pred_mask, gt_mask=gt_mask)[quantity])
            index = torch.argmax(torch.tensor(list_f_value))

        elif quantity == "mae":
            n_queries = pred_mask.shape[0]
            maes = compute_mae(pred_mask=pred_mask, gt_mask=gt_mask[None, ...].repeat(n_queries, 1, 1))
            index = torch.argmin(maes)
        else:
            raise ValueError
        return index

    def _get_salient_mask(
            self,
            pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param pred_mask: (n queries x H x W)
        :param gt_mask: (H x W)
        :param quantity: a quantity that will be considered to pick the best mask. Default: "iou"
        :return: best_mask: (H x W)
        """
        masks, _ = filter_masks(pred_mask)
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
        return best_mask_index

    @torch.no_grad()
    def __call__(
            self,
            dataset_name: str,
            dir_ckpt: str,
            img_size: Optional[int] = None,
            scale_factor: int = 2,
            batch_size: int = 1,
            device=torch.device("cuda:0"),
            cost_type: str = "iou"
    ) -> dict:
        self._init_meters()
        dataset = get_dataset(
            dir_dataset=self.dir_dataset,
            dataset_name=dataset_name,
            mode="test",
            eval_img_size=img_size,
        )

        iter_dataloader, tbar = dataset.get_dataloader(
            shuffle=False,
            batch_size=batch_size,
            num_workers=4,
            with_tbar=True
        )

        for ind_batch in tbar:
            dict_data: dict = next(iter_dataloader)
            batch_gt_masks: torch.Tensor = dict_data['m'].to(device)
            h, w = batch_gt_masks.shape[-2:]
            dict_outputs: dict = self._forward(dict_data, device=device)

            batch_pred_masks: torch.Tensor = dict_outputs["mask_pred"]  # [0, 1]
            batch_objectness: torch.Tensor = dict_outputs.get("objectness", None)  # [0, 1]

            if len(batch_pred_masks.shape) == 5:
                # b x n_layers x n_queries x h x w -> b x n_queries x h x w
                batch_pred_masks = batch_pred_masks[:, -1, ...]  # extract the output from the last decoder layer

                if batch_objectness is not None:
                    # b x n_layers x n_queries x 1 -> b x n_queries x 1
                    batch_objectness = batch_objectness[:, -1, ...]

            # resize prediction to original resolution
            # note: upsampling by 4 and cutting the padded region allows for a better result
            batch_pred_masks = F.interpolate(
                batch_pred_masks, scale_factor=4, mode="bilinear", align_corners=False
            )[..., :h, :w]

            batch_best_gt_to_query: List[Dict[int, int]] = list()
            # iterate over batch dimension
            for batch_index, (pred_masks, gt_mask) in enumerate(zip(batch_pred_masks, batch_gt_masks)):
                upper_bound_index = self._get_upper_bound_mask(pred_mask=pred_masks > 0.5, gt_mask=gt_mask)

                # n_queries x 1 -> n_queries
                objectness: torch.Tensor = batch_objectness[batch_index].squeeze(dim=-1)
                ranks = torch.argsort(objectness, descending=True)  # n_queries
                pred_mask = pred_masks[ranks[0]]

                self._update_meters(
                    pred_mask=pred_mask.to(device=device),
                    gt_mask=gt_mask.squeeze().to(device),
                    ub_mask=pred_masks[upper_bound_index] if self.model.use_binary_classifier else None
                )
                batch_best_gt_to_query.append({batch_index: upper_bound_index})

            tbar.set_description(
                f"{dataset_name} (IS, UB) | "
                f"F-measure: ({self.f_score.avg:.3f}, {self.f_score_ub.avg:.3f}) | "
                f"F-max: ({self.f_max.avg:.3f}, {self.f_max_ub.avg:.3f}) | "
                f"MAE: ({self.mae.avg:.3f}, {self.mae_ub.avg:.3f}) | "
                f"S-measure: ({self.s_measure.avg:.3f}, {self.s_measure_ub.avg:.3f}) | "
                f"IoU: ({self.iou.avg:.3f}, {self.iou_ub.avg:.3f}) | "
                f"pixel acc.: ({self.pixel_acc.avg:.3f}, {self.pixel_acc_ub.avg:.3f})"
            )

            if ind_batch % 250 == 0:
                os.makedirs(f"{dir_ckpt}", exist_ok=True)

                batch_gt_masks = dict_data["m"]

                for i in range(batch_size):
                    gt_mask = batch_gt_masks[i]
                    if not self.model.use_binary_classifier:
                        objectness = batch_objectness[i, -1]  # n_queries x n_classes (=2)

                        # pred_masks: n_queries x h x w -> n_classes x h x w -> 1 x h x w
                        pred_masks = torch.einsum("qhw,qc->chw", pred_masks, objectness)
                        pred_masks = torch.argmax(pred_masks, dim=0, keepdim=True).cpu()
                        best_mask_to_query = {1: 0}

                    else:
                        pred_masks = (batch_pred_masks[i, ...] > 0.5).detach().cpu()  # n_queries x h x w
                        best_mask_to_query = batch_best_gt_to_query[i]

                    if gt_mask.sum() == 0:
                        continue
                    self._visualize(
                        img=dict_data['x'][i],
                        mask_pred=pred_masks,
                        gt_mask=gt_mask,
                        best_mask_to_query=best_mask_to_query,
                        dataset=dataset,
                        fp=f"{dir_ckpt}/{ind_batch:05d}_{i}.png",
                        max_ncols=int(sqrt(pred_masks.shape[0])),
                        objectness=batch_objectness[i].squeeze(dim=-1)
                    )

            if self.debug:
                break

        with open(f"{dir_ckpt}/metrics_{dataset_name}.txt", 'w') as f:
            f.write("iou,pixel_acc,f_score,f_max,f_mean,mae,s_measure,miou_ub,pixel_acc_ub,f_score_ub,f_max_ub,f_mean_ub,mae_ub,s_measure_ub\n")
            f.write(
                f"{self.iou.avg},"
                f"{self.pixel_acc.avg},"
                f"{self.f_score.avg},"
                f"{self.f_max.avg},"
                f"{self.f_mean.avg},"
                f"{self.mae.avg},"
                f"{self.s_measure.avg},"
                f"{self.iou_ub.avg},"
                f"{self.pixel_acc_ub.avg},"
                f"{self.f_score_ub.avg},"
                f"{self.f_max_ub.avg},"
                f"{self.f_mean_ub.avg},"
                f"{self.mae_ub.avg},"
                f"{self.s_measure_ub.avg}"
            )
            f.close()
        return {
            "iou": self.iou.avg,
            "pixel_accuarcy": self.pixel_acc.avg,
            "f_score": self.f_score.avg,
            "f_max": self.f_max.avg,
            "f_mean": self.f_mean.avg,
            "mae": self.mae.avg,
            "s_measure": self.s_measure.avg,
            "iou_ub": self.iou_ub.avg,
            "pixel_accuarcy_ub": self.pixel_acc_ub.avg,
            "f_score_ub": self.f_score_ub.avg,
            "f_max_ub": self.f_max_ub.avg,
            "f_mean_ub": self.f_mean_ub.avg,
            "mae_ub": self.mae_ub.avg,
            "s_measure_ub": self.s_measure_ub.avg
        }


if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace
    import yaml
    from utils.visualizer import Visualizer
    from main import define_experim_name
    parser = ArgumentParser("SelfMask evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/noel/projects/selfmask/configs/duts-dino-k234-nq20-224-swav-mocov2-dino-p16-sr10100.yaml"
    )

    parser.add_argument(
        "--p_state_dict",
        type=str,
        default="/Users/noel/Desktop/selfmask_nq20_model.pt",
    )

    parser.add_argument(
        "--dataset_name", '-dn', type=str, default="duts",
        choices=["dut_omron", "duts", "ecssd"]
    )

    # independent variables
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--dir_root", type=str, default="..")
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--suffix", type=str, default='')
    args: Namespace = parser.parse_args()
    base_args = yaml.safe_load(open(f"{args.config}", 'r'))
    base_args.pop("dataset_name")
    args: dict = vars(args)
    args.update(base_args)
    args: Namespace = Namespace(**args)
    args.experim_name = define_experim_name(args)
    dir_ckpt = f"{os.path.dirname(args.p_state_dict)}"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")

    # set seed
    set_seeds(args.seed)

    state_dict = torch.load(args.p_state_dict, map_location=device)
    model = get_model(arch="maskformer", configs=args).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Pre-trained weights are loaded from {args.p_state_dict}.")

    evaluator = Evaluator(
        network=model,
        dir_dataset=args.dir_dataset,
        visualizer=Visualizer()
    )

    evaluator(
        dataset_name=args.dataset_name,
        dir_ckpt=dir_ckpt,
        scale_factor=args.scale_factor,
        batch_size=1
    )
