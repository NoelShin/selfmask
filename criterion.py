from typing import Dict, List, Tuple, Union, Optional
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import torch.nn.functional as F
from metrics.average_meter import AverageMeter


class Loss:
    def __init__(
            self,
            weight_dice_loss: float = 1.0,
            weight_focal_loss: float = 20.0,
            weight_bce_loss: float = 0.0,
    ):
        """
        :param weight_dice_loss: a weight for dice loss following the MaskFormer implementation. Default: 1.0
        :param weight_focal_loss: a weight for focal loss following the MaskFormer implementation. Default: 20.0
        :param weight_bce_loss: a weight for binary cross-entropy loss. Default: 0.0
        """
        self.weight_bce_loss: float = weight_bce_loss
        self.weight_dice_loss: float = weight_dice_loss
        self.weight_focal_loss: float = weight_focal_loss
        self.weight_cls_loss: float = 0.05
        self.weight_rank_loss: float = 1.0

        self.meter_loss = AverageMeter()
        self.meter_bce_loss = AverageMeter()
        self.meter_dice_loss = AverageMeter()
        self.meter_total_dice_loss = AverageMeter()
        self.meter_focal_loss = AverageMeter()
        self.meter_iou = AverageMeter()
        self.meter_mask_loss = AverageMeter()
        self.meter_classification_loss = AverageMeter()
        self.meter_ranking_loss = AverageMeter()

    def _update_meters(self, **kwargs) -> None:
        for meter_name, list_values in kwargs.items():
            assert isinstance(list_values, list), f"{type(list_values)} is not a list."
            getattr(self, meter_name).update(val=np.mean(list_values), n=len(list_values))

    def reset_metrics(self) -> None:
        self.meter_loss.reset()
        self.meter_bce_loss.reset()
        self.meter_dice_loss.reset()
        self.meter_focal_loss.reset()
        self.meter_iou.reset()
        self.meter_mask_loss.reset()
        self.meter_classification_loss.reset()
        self.meter_ranking_loss.reset()

    # adapted from the MaskFormer official code.
    # https://github.com/facebookresearch/MaskFormer/blob/2ae8543c892c5d57352fc10d55103a7254a03dad/mask_former/modeling/matcher.py#L66
    @staticmethod
    def _dice_loss(
            pred_masks: torch.Tensor,
            one_hot_gt_masks: torch.Tensor
    ):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            pred_masks:
                    N_queries x (H * W)
                    A float tensor of arbitrary shape.
                    The predictions for each example.
            one_hot_gt_masks:
                    M_gt_masks_per_image x (H * W)
                    A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        numerator = 2 * torch.einsum("nc,mc->nm", pred_masks, one_hot_gt_masks)

        denominator = pred_masks.sum(-1)[:, None] + one_hot_gt_masks.sum(-1)[None, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

    # adapted from the MaskFormer official code.
    # https://github.com/facebookresearch/MaskFormer/blob/2ae8543c892c5d57352fc10d55103a7254a03dad/mask_former/modeling/matcher.py#L66
    @staticmethod
    def _focal_loss(
            pred_masks,
            one_hot_targets,
            alpha: float = 0.25,
            gamma: float = 2
    ):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            pred_masks: N_queries x (H * W) A float tensor of arbitrary shape.
                    The predictions for each example.
            one_hot_targets: M x (H * W) A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        hw = pred_masks.shape[1]

        focal_pos = ((1 - pred_masks) ** gamma) * F.binary_cross_entropy(
            pred_masks, torch.ones_like(pred_masks), reduction="none"
        )
        focal_neg = (pred_masks ** gamma) * F.binary_cross_entropy(
            pred_masks, torch.zeros_like(pred_masks), reduction="none"
        )
        if alpha >= 0:
            focal_pos = focal_pos * alpha
            focal_neg = focal_neg * (1 - alpha)

        loss = torch.einsum("nc,mc->nm", focal_pos, one_hot_targets) + torch.einsum("nc,mc->nm", focal_neg, (1 - one_hot_targets))
        return loss / hw

    @staticmethod
    def _binary_cross_entropy_loss(pred_masks: torch.Tensor, one_hot_targets: torch.Tensor):
        """
        :param pred_masks: N_queries x (h * w)
        :param one_hot_targets: M x (h * w)
        :return:
        """
        n_queries, n_one_hot_masks = len(pred_masks), len(one_hot_targets)

        pred_masks = pred_masks.unsqueeze(dim=1).repeat(1, n_one_hot_masks, 1)  # N_queries x M x (hw)
        one_hot_targets = one_hot_targets.unsqueeze(dim=0).repeat(n_queries, 1, 1)  # N_queries x M x (hw)

        loss = F.binary_cross_entropy(pred_masks, one_hot_targets, reduction="none").mean(dim=-1)  # N_queries x M
        return loss

    def _forward(
            self,
            batch_pred_masks: torch.Tensor,
            batch_one_hot_gt_mask: torch.Tensor,
            batch_objectness: Optional[torch.Tensor] = None,
            use_classification_loss: bool = False
    ):
        """
        Note: compute all the possible losses between the predictions and labels and pick the least losses as many as
        batch_size

        :param batch_pred_masks: b x N x H x W or b x n_layers x n_queries x H x W
        :param batch_one_hot_gt_mask: b x M x H x W, where M varies with an image.
        """
        assert 0 <= batch_pred_masks.min() <= 1
        assert 0 <= batch_pred_masks.max() <= 1

        # iterate over the batch axis
        batch_gt_to_query: List[dict] = list()
        dice_loss: torch.Tensor = torch.tensor(0., device=batch_pred_masks.device)
        ranking_loss: torch.Tensor = torch.tensor(0., device=batch_pred_masks.device)
        classification_loss: torch.Tensor = torch.tensor(0., device=batch_pred_masks.device)

        batch_losses: List[float] = list()
        batch_dice_losses: List[float] = list()
        batch_ranking_losses: List[float] = list()
        batch_classification_losses: List[float] = list()

        list_ious: List[float] = list()

        for num_batch, (pred_masks, one_hot_gt_masks) in enumerate(zip(batch_pred_masks, batch_one_hot_gt_mask)):
            # one_hot_gt_masks: n_objects x h x w
            one_hot_gt_masks: torch.Tensor = one_hot_gt_masks.to(batch_pred_masks.device, torch.float32)
            if one_hot_gt_masks.sum() == 0:
                # in case where there is no object in ground-truth masks of an image.
                batch_gt_to_query.append(None)
                continue

            if len(one_hot_gt_masks.shape) == 2:
                one_hot_gt_masks = one_hot_gt_masks[None]  # add a mask dimension
            h, w = one_hot_gt_masks.shape[-2:]

            # resize predicted masks to the shape of ground-truth masks
            # pred_masks: (n_layers x) n_queries x h' x w' -> (n_layers x) n_queries x h x w
            if len(pred_masks.shape) == 3:
                pred_masks: torch.Tensor = pred_masks[None, ...]

            pred_masks: torch.Tensor = F.interpolate(
                pred_masks, size=one_hot_gt_masks.shape[-2:], mode="bilinear", align_corners=False
            )

            # pred_masks: (n_layers x) n_queries x h x w -> (n_layers x) n_queries x hw
            pred_masks = pred_masks.flatten(start_dim=-2)

            # gt_masks: n_objects x h x w -> n_objects x hw
            if use_classification_loss:
                from utils.misc import to_one_hot
                # one_hot_gt_masks: 1 x h x w -> 2 x hw
                one_hot_gt_masks = to_one_hot(one_hot_gt_masks).flatten(start_dim=-2).squeeze(dim=0)
            else:
                one_hot_gt_masks = one_hot_gt_masks.flatten(start_dim=-2)

            # iterate over the transformer decoder layers
            batch_dice_loss = torch.tensor(0., device=batch_pred_masks.device)
            batch_ranking_loss = torch.tensor(0., device=batch_pred_masks.device)
            batch_classification_loss = torch.tensor(0., device=batch_pred_masks.device)
            for num_layer, pred_masks_per_layer in enumerate(pred_masks):  # pred_masks_per_layer: n_queries x hw
                if batch_objectness is not None:
                    if use_classification_loss:
                        _dice_loss: torch.Tensor = self._dice_loss(
                            pred_masks=pred_masks_per_layer, one_hot_gt_masks=one_hot_gt_masks
                        ).permute(1, 0)

                        # n_queries x 2 -> 2 x n_queries
                        objectness_per_layer = batch_objectness[num_batch, num_layer, ...].t()
                        cost = _dice_loss - objectness_per_layer

                        gt_indices, query_indices = linear_sum_assignment(cost.detach().cpu().numpy(), maximize=False)
                        gt_to_query: dict = dict()
                        for gt_index, query_index in zip(gt_indices, query_indices):
                            gt_to_query.update({gt_index: query_index})
                            batch_dice_loss += _dice_loss[gt_index, query_index]
                            batch_classification_loss += - torch.log(objectness_per_layer[gt_index, query_index] + 1e-7)

                    else:
                        # _dice_loss: n x k -> k x n (k = 1)
                        _dice_loss: torch.Tensor = self._dice_loss(
                            pred_masks=pred_masks_per_layer, one_hot_gt_masks=one_hot_gt_masks
                        ).permute(1, 0)

                        batch_dice_loss += _dice_loss.sum()
                        gt_to_query: dict = {0: torch.argmin(_dice_loss.squeeze(dim=0)).item()}

                        objectness_per_layer = batch_objectness[num_batch, num_layer, ...].squeeze(dim=-1)  # n_queries #x 1
                        # query_scores = objectness_per_layer[query_indices]  # n_objects

                        # hinge loss without margin, but aligned.
                        # binary_masks_per_layer = pred_masks_per_layer > 0.5  # n_q x hw
                        # matched_binary_mask = binary_masks_per_layer[query_indices.squeeze()][None]  # 1 x hw
                        # matched_binary_mask = matched_binary_mask.repeat(len(binary_masks_per_layer), 1)  # 1 x hw
                        #
                        # intersection = torch.logical_and(matched_binary_mask, binary_masks_per_layer).sum(dim=-1)
                        # union = torch.logical_or(matched_binary_mask, binary_masks_per_layer).sum(dim=-1)
                        # ious: torch.Tensor = (intersection / (union + 1e-7))

                        sorted_indices = torch.argsort(_dice_loss.squeeze(dim=0), descending=True)
                        objectness_scores = objectness_per_layer[sorted_indices][:, None]  # n_q -> n_q x 1
                        objectness_scores_t = objectness_scores.t()  # 1 x n_q

                        upper_triangular_matrix = torch.triu(objectness_scores - objectness_scores_t, diagonal=1)
                        batch_ranking_loss += upper_triangular_matrix[upper_triangular_matrix < 0].abs().sum()

            dice_loss += batch_dice_loss
            ranking_loss += batch_ranking_loss
            classification_loss += batch_classification_loss

            pred_masks = pred_masks[-1]  # n_layers x n_queries x hw -> n_queries x hw

            # assert len(gt_to_query) <= len(one_hot_gt_masks), f"{len(gt_to_query)} != {len(one_hot_gt_masks)}"

            batch_gt_to_query.append(gt_to_query)

            batch_dice_losses.append(batch_dice_loss.detach().cpu().item())
            batch_ranking_losses.append(batch_ranking_loss.detach().cpu().item())
            batch_classification_losses.append(batch_classification_loss.detach().cpu().item())

            if use_classification_loss:
                batch_losses.append(batch_dice_loss.detach().cpu().item() + batch_classification_loss.detach().cpu().item())
            else:
                batch_losses.append(batch_dice_loss.detach().cpu().item() + batch_ranking_loss.detach().cpu().item())

            for gt_index, query_index in gt_to_query.items():
                gt_mask, dt_mask = one_hot_gt_masks[gt_index], pred_masks[query_index] > 0.5
                intersection = torch.logical_and(gt_mask, dt_mask.detach()).sum().cpu().item()
                union = torch.logical_or(gt_mask, dt_mask.detach()).sum().cpu().item()
                list_ious.append(intersection / (union + 1e-7))

        if use_classification_loss:
            loss = dice_loss + classification_loss
        else:
            loss = dice_loss + self.weight_rank_loss * ranking_loss

        loss = loss / len(batch_one_hot_gt_mask)

        self._update_meters(
            meter_loss=batch_losses,
            meter_dice_loss=batch_dice_losses,
            # meter_total_dice_loss=batch_total_dice_losses,
            meter_ranking_loss=batch_ranking_losses,
            meter_classification_loss=batch_classification_losses,
            meter_iou=list_ious
        )

        return {
            "loss": loss,
            "batch_best_gt_to_query": batch_gt_to_query,
            "avg_loss": self.meter_loss.avg,
            "avg_dice_loss": self.meter_dice_loss.avg,
            # "avg_total_dice_loss": self.meter_total_dice_loss.avg,
            "avg_ranking_loss": self.meter_ranking_loss.avg,
            "avg_classification_loss": self.meter_classification_loss.avg,
            "avg_iou": self.meter_iou.avg
        }

    def __call__(
            self,
            batch_pred_masks: torch.Tensor,
            batch_one_hot_gt_mask: Union[torch.Tensor, List[torch.Tensor]],
            batch_objectness: Optional[torch.Tensor] = None,
            use_classification_loss: bool = False
    ) -> dict:
        """
        :param batch_pred_masks: b x n_dims x h x w or b x n_layers x n_dims x h x w
        :param batch_one_hot_gt_mask: b x h x w
        :param batch_objectness: b x n_q x 1
        :return:
        """
        assert batch_pred_masks.shape[0] == len(batch_one_hot_gt_mask), \
            f"prediction ({len(batch_pred_masks)}) and gt masks ({len(batch_one_hot_gt_mask)}) should share the same batch size."

        return self._forward(
            batch_pred_masks=batch_pred_masks,
            batch_one_hot_gt_mask=batch_one_hot_gt_mask,
            batch_objectness=batch_objectness,
            use_classification_loss=use_classification_loss
        )
