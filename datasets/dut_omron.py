from os.path import join
from glob import glob
from typing import Optional
from datasets.base_dataset import BaseDataset


class DUTOMRONDataset(BaseDataset):
    def __init__(
            self,
            dir_dataset: str,
            img_size: int = 128,
            rank: Optional[int] = None,
            world_size: Optional[int] = None
    ):
        """At this moment, only test set is considered."""
        super(DUTOMRONDataset, self).__init__()
        self.p_test_imgs = sorted(glob(join(dir_dataset, "DUT-OMRON-image", f"*.jpg")))
        self.p_test_gts = sorted(glob(join(dir_dataset, "pixelwiseGT-new-PNG", f"*.png")))
        assert len(self.p_test_imgs) == len(self.p_test_gts), f"{len(self.p_test_imgs)} != {len(self.p_test_gts)}"

        # seg variables for data augmentation
        self.img_size = (img_size, img_size)
        self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.name = "dut-omron"

        print(
            '\n'
            "Dataset summary\n"
            f"Dataset name: {self.name}\n"
            f"# images: {len(self.p_test_imgs)}\n"
        )


if __name__ == '__main__':
    dataset = DUTOMRONDataset("/scratch/shared/beegfs/gyungin/datasets/DUT-OMRON")
    dataset.set_mode("test")
    for dict_data in dataset:
        print(dict_data['m'].min(), dict_data['m'].max())
        break