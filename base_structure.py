import os.path
from typing import Dict, List, Optional, Iterable, Union
import torch
import matplotlib.pyplot as plt


class BaseStructure:
    def __init__(
            self,
            model: callable,
            visualizer: Optional[callable] = None,
            device: torch.device = torch.device("cuda:0")
    ):
        self.device = device
        self.model = model
        self.visualizer = visualizer

    def _forward(
            self,
            dict_data: dict,
            encoder_only: bool = False,
            device: torch.device = torch.device("cuda:0"),
            skip_decoder: bool = False
    ) -> Dict[str, torch.Tensor]:
        return self.model(dict_data['x'].to(device), encoder_only=encoder_only, skip_decoder=skip_decoder)

    def _extract_selected_predictions(
            self,
            mask_pred: torch.Tensor,
            best_mask_to_query: Union[Dict[int, int], List[Dict[int, int]]]
    ) -> torch.Tensor:
        """
        :param mask_pred: (b) x (n_layers) x n_queries x h x w
        :param best_mask_to_query: {gt_index: query_index} or [{gt_index: query_index}]
        :return:
        """
        batch_best_pred_per_mask: list = list()

        if len(mask_pred.shape) == 4:  # b x n_queries x h x w
            for batch_index, mtq in enumerate(best_mask_to_query):
                best_pred_per_mask: list = list()
                for m, q in sorted(mtq.items()):
                    best_pred_per_mask.append(mask_pred[batch_index, q, ...])  # k x h x w
                batch_best_pred_per_mask.append(torch.stack(best_pred_per_mask, dim=0))
            return torch.stack(batch_best_pred_per_mask, dim=0)  # b x k x h x w

        elif len(mask_pred.shape) == 3:  # n_queries x h x w
            best_pred_per_mask: list = list()
            for m, q in sorted(best_mask_to_query.items()):
                best_pred_per_mask.append(mask_pred[q, ...])
            return torch.stack(best_pred_per_mask, dim=0)  # k x h x w

        else:
            raise ValueError(mask_pred.shape)

    def _visualize(
            self,
            img: torch.Tensor,
            mask_pred: torch.Tensor,
            gt_mask: torch.Tensor,
            best_mask_to_query: Dict[int, int],
            dataset: Iterable,
            fp: str,
            max_ncols: int = 10,
            objectness: Optional[torch.Tensor] = None
    ) -> None:
        assert gt_mask.shape[0] >= len(best_mask_to_query), f"{gt_mask.shape[0]} !>= {len(best_mask_to_query)}"
        batch_best_pred_per_mask: torch.Tensor = self._extract_selected_predictions(
            mask_pred=mask_pred, best_mask_to_query=best_mask_to_query
        )
        if len(gt_mask.shape) == 2:
            gt_mask = gt_mask.unsqueeze(0)

        self.visualizer.visualize(
            input_image=dataset.denormalize(img.cpu(), as_pil=False),
            gt_masks=gt_mask,
            best_pred_masks=batch_best_pred_per_mask.detach(),
            best_pred_masks_index=list(best_mask_to_query.values()),
            fp=fp,
            all_pred_masks=mask_pred.detach().cpu(),
            max_ncols=max_ncols
        )

        if objectness is not None:
            ranks = torch.argsort(objectness, descending=True)
            n_queries = len(objectness)
            nrows, ncols = n_queries // 5, 5

            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(ncols, nrows))
            cnt = 0
            for num_row in range(nrows):
                for num_col in range(ncols):
                    ax[num_row, num_col].imshow(mask_pred[ranks[cnt]].sigmoid() > 0.5)
                    ax[num_row, num_col].set_xticks([])
                    ax[num_row, num_col].set_yticks([])
                    ax[num_row, num_col].set_xlabel(f"{objectness[ranks[cnt]]:.3f}")
                    cnt += 1
            plt.tight_layout(pad=0.5)
            base_fp = os.path.splitext(fp)[0]
            plt.savefig(f"{base_fp}_object.png")
            plt.close()
