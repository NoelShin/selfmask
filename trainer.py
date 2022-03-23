import os
from copy import deepcopy
from math import sqrt
from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn.functional as F
import wandb
from base_structure import BaseStructure


class Trainer(BaseStructure):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            model: callable,
            criterion: callable,
            optimizer,
            clusterer: callable,
            lr_scheduler=None,
            evaluator=None,
            seed: int = 0,
            arch: str = "vit_small",
            training_method: str = "dino",
            batch_size: int = 1,
            dir_ckpt: str = '',
            experim_name: str = '',
            k: Union[int, List[int]] = 3,
            n_percent: int = 100,
            scale_factor: int = 2,
            visualizer: Optional[callable] = None,
            benchmarks: Optional[Tuple[str, ...]] = None,
            eval_image_size: Optional[int] = None,
            debug: bool = False
    ):
        super(Trainer, self).__init__(model=model, visualizer=visualizer)
        self.arch: str = arch
        self.training_method: str = training_method
        self.batch_size: int = batch_size
        self.benchmarks = ["ecssd", "duts", "dut_omron"] if benchmarks is None else benchmarks
        self.clusterer: callable = clusterer
        self.criterion: callable = criterion
        self.dataset: torch.utils.data.Dataset = dataset
        self.dataset_name: str = dataset.name
        self.debug: bool = debug
        self.dir_ckpt: str = dir_ckpt
        self.evaluator: callable = evaluator
        self.experim_name: str = experim_name
        self.k: Union[int, List[int]] = k
        self.lr_scheduler = lr_scheduler
        self.n_percent: int = n_percent
        self.optimizer = optimizer
        self.scale_factor: int = scale_factor
        self.seed: int = seed
        self.eval_image_size: Optional[int] = eval_image_size

        self.iter_total = 0
        self.iter_vis = 1000

    # backward pass
    def _backward(self, dict_losses: Dict[str, torch.Tensor], clip_grad_norm: bool = False) -> float:
        loss: torch.Tensor = dict_losses["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.detach().item()

    def _train_epoch(self, num_epoch: int) -> None:
        self.model.train()
        self.dataset.set_mode("train")
        self.dataset.use_data_augmentation_(True)
        self.criterion.reset_metrics()

        iter_dataloader, tbar = self.dataset.get_dataloader(
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            with_tbar=True,
            num_workers=4,
            collate_fn=self.dataset.collate_fn if "collate_fn" in dir(self.dataset) else None
        )
        for ind_batch in tbar:
            dict_data = next(iter_dataloader)
            dict_outputs: dict = self._forward(dict_data)

            if self.dataset.name in ["imagenet1k", "duts"]:
                batch_one_hot_gt_mask: Union[List[torch.Tensor], torch.Tensor] = dict_data["m"]
            else:
                batch_one_hot_gt_mask: torch.Tensor = dict_data["m"][:, None]  # b x h x w -> b x 1 x h x w

            dict_losses: Dict[str, torch.Tensor] = self.criterion(
                batch_pred_masks=dict_outputs["mask_pred"],
                batch_one_hot_gt_mask=batch_one_hot_gt_mask,
                batch_objectness=dict_outputs.get("objectness", None),
                use_classification_loss=not self.model.use_binary_classifier
            )

            self._backward(dict_losses=dict_losses)

            tbar.set_description(
                f"Epoch {num_epoch}: {self.experim_name} | "
                f"avg loss: {dict_losses['avg_loss']:.3f} | "
                f"avg dice loss: {dict_losses['avg_dice_loss']:.3f} | "
                f"avg ranking loss: {dict_losses['avg_ranking_loss']:.3f} | "
                f"avg iou: {dict_losses['avg_iou']:.3f} | "
                f"lr: {self.lr_scheduler.get_lr()[0]:.5f}"
            )

            if self.iter_total % (len(iter_dataloader) // 10) == 0:
                batch_pred_masks: torch.Tensor = dict_outputs["mask_pred"].detach()
                batch_gt_masks: List[torch.Tensor] = deepcopy(dict_data["m"])
                for num_batch in range(len(batch_gt_masks)):
                    gt_masks: torch.Tensor = batch_gt_masks[num_batch]
                    pred_masks: torch.Tensor = batch_pred_masks[num_batch]
                    batch_objectness = dict_outputs.get("objectness", None)

                    if gt_masks.sum() == 0:
                        continue

                    if len(pred_masks.shape) == 4:
                        # in case where the prediction includes intermediate layers' outputs
                        pred_masks = pred_masks[-1, ...]  # n_queries x h x w

                    pred_masks = F.interpolate(
                        pred_masks[None, ...], size=gt_masks.shape[-2:], mode="bilinear", align_corners=False
                    )[0]

                    pred_masks = pred_masks.cpu() > 0.5
                    best_mask_to_query = dict_losses["batch_best_gt_to_query"][num_batch]

                    self._visualize(
                        img=deepcopy(dict_data['x'][num_batch]),
                        mask_pred=pred_masks,
                        gt_mask=gt_masks,
                        best_mask_to_query=best_mask_to_query,
                        dataset=self.dataset,
                        fp=f"{self.dir_ckpt}/{self.dataset_name}/{num_epoch:02d}/{ind_batch:05d}_{num_batch}.png",
                        max_ncols=int(sqrt(pred_masks.shape[0])),
                        objectness=batch_objectness[num_batch][-1].squeeze(dim=-1)
                    )

            self.iter_total += 1
            if self.debug:
                break

        wandb.log({
            "epoch": num_epoch,
            "avg_loss": dict_losses["avg_loss"],
            "avg_dice_loss": dict_losses["avg_dice_loss"],
            "avg_ranking_loss": dict_losses["avg_ranking_loss"],
            "avg_iou": dict_losses["avg_iou"],
        })

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "n_epochs": num_epoch,
            "n_iters": self.iter_total
        },
            f"{self.dir_ckpt}/latest_model.pt"
        )

    def _evaluate(
            self,
            num_epoch: int,
    ) -> None:
        self.model.eval()

        for dataset_name in self.benchmarks:
            dict_results: dict = self.evaluator(
                dataset_name=dataset_name,
                dir_ckpt=f"{self.dir_ckpt}/eval/{dataset_name}/{num_epoch:02d}",
                batch_size=1,  # batch size should be 1 due to varying image sizes
            )
            current_score = dict_results["iou"]

            for k in list(dict_results.keys()):
                new_k = k + f" ({dataset_name.upper()})"
                v = dict_results.pop(k)
                dict_results.update({new_k: v})

            dict_results.update({"epoch": num_epoch})
            wandb.log(dict_results)

            try:
                best_score: float = getattr(self, f"best_score_{dataset_name}")
            except AttributeError:
                best_score: float = 0.

            if current_score > best_score:
                setattr(self, f"best_score_{dataset_name}", current_score)
                torch.save({
                    "n_epochs": num_epoch,
                    "n_iters": self.iter_total,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict()
                }, f"{self.dir_ckpt}/eval/{dataset_name}/best_model.pt")
                print(
                    f"\nBest score for {dataset_name} dataset has changed from {best_score:.3f} to {current_score:.3f} "
                    f"(Epoch: {num_epoch}, n iters: {self.iter_total})\n"
                )

    def __call__(self, n_epochs: int, device: torch.device = torch.device("cuda:0")) -> None:
        os.makedirs(f"{self.dir_ckpt}/{self.dataset_name}", exist_ok=True)
        for num_epoch in range(1, n_epochs + 1):
            self._train_epoch(num_epoch=num_epoch)
            self._evaluate(num_epoch=num_epoch)
