from os.path import join
from glob import glob
from typing import Optional
from datasets.base_dataset import BaseDataset


class ECSSDDataset(BaseDataset):
    def __init__(
            self,
            dir_dataset: str,
            img_size: int = 128,
            rank: Optional[int] = None,
            world_size: Optional[int] = None
    ):
        """At this moment, only test set is considered."""
        super(ECSSDDataset, self).__init__()
        self.p_test_imgs = sorted(glob(join(dir_dataset, "images", f"*.jpg")))
        self.p_test_gts = sorted(glob(join(dir_dataset, "ground_truth_mask", f"*.png")))
        assert len(self.p_test_imgs) == len(self.p_test_gts), f"{len(self.p_test_imgs)} != {len(self.p_test_gts)}"

        # seg variables for data augmentation
        self.img_size = (img_size, img_size)
        self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.name = "ecssd"

        print(
            '\n'
            "Dataset summary\n"
            f"Dataset name: {self.name}\n"
            f"# images: {len(self.p_test_imgs)}\n"
        )


if __name__ == '__main__':
    dataset = ECSSDDataset("/scratch/shared/beegfs/gyungin/datasets/ECSSD")
    dataset.set_mode("test")
    print(len(dataset))
    for dict_data in dataset:
        print(dict_data.keys())
        print(dict_data['m'].max())
        break