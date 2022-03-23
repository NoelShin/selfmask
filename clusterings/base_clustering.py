from typing import Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class BaseClustering:
    def __init__(self, max_iter: int = 300, n_init: int = 10, use_gpu: bool = True):
        """Cluster features
        Args:
            max_iter: maximum number of clustering iteration
            n_init: the number of different clustering to be done
            use_gpu: if True, use a gpu-accelerated spectral clustering instead of SpectralClustering method from
            sklearn package. No-op when performing K-means as its default is gpu-accelerated.
        """
        self.max_iter = max_iter
        self.n_init = n_init
        self.use_gpu = use_gpu

    @staticmethod
    def upsample(x: Union[np.ndarray, torch.Tensor], scale_factor: int = None, target_size: Tuple[int, int] = None):
        """
        Args:
            x: np.ndarray or torch.Tensor to be upsampled
            scale_factor: a factor that x will be upsampled by
            target_size: If given, upsample to the given size. No-op when x is torch.Tensor.
        """
        assert type(x) in (np.ndarray, torch.Tensor), f"Invalid type {type(x)}"

        if isinstance(x, torch.Tensor):
            assert scale_factor is not None, "when input is torch.Tensor, scale_factor should be given."
            assert len(x.shape) in [3, 4], f"Input needs to be 3D or 4D, got {len(x.shape)}D."
            if len(x.shape) == 3:
                x = x.unsqueeze(dim=0)  # add a batch dimension
                x = F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=True).squeeze(dim=0)
            else:
                x = F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=True)

        else:
            assert target_size is not None, "when input is np.ndarray, target_size should be given."
            if len(x.shape) == 2:
                # add batch & channel dimension
                x = torch.tensor(x).unsqueeze(dim=0).unsqueeze(dim=1)

            elif len(x.shape) == 3:
                # add channel dimension
                x = torch.tensor(x).unsqueeze(dim=1)

            try:
                x = F.interpolate(x, size=target_size, mode="nearest").squeeze(dim=1)
            except RuntimeError:
                x = TF.resize(x, size=target_size, interpolation=TF.InterpolationMode.NEAREST).squeeze(dim=1)
            x = x.numpy()
        return x

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, patch_tokens: torch.Tensor, k: int, **kwargs):
        try:
            clusters: np.ndarray = getattr(self, "forward")(patch_tokens, k, **kwargs)
        except RuntimeError:
            clusters: np.ndarray = getattr(self, "forward")(patch_tokens.cpu(), k, **kwargs)
        return clusters
