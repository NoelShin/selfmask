from typing import List, Optional, Tuple
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize, resize, to_tensor
from PIL import Image


class CustomDataset(Dataset):
    def __init__(
            self,
            image_paths: List[str],
            image_size: Optional[int] = None,
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        assert len(image_paths) > 0, f"No image paths are given: {len(image_paths)}."
        super(CustomDataset, self).__init__()
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
        img = normalize(to_tensor(img), mean=self.mean, std=self.std)
        return {"img": img, "filename": image_path.split('/')[-1]}