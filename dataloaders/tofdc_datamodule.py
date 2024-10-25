from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule
from .tofdc import TOFDCDataset
from .registry import DATAMODULES


@DATAMODULES.register_module('tofdc')
class TOFDCDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self.tofdc_train = TOFDCDataset(self.cfg.dataset.root_dir, 'train', 7, 2.5)
            self.tofdc_val = TOFDCDataset(self.cfg.dataset.root_dir, 'test', 0, 0)

        if stage == 'test' or stage is None:
            self.tofdc_test = TOFDCDataset(self.cfg.dataset.root_dir, 'test', 0, 0)

    def train_dataloader(self):
        return DataLoader(
            self.tofdc_train,
            self.cfg.training.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg.training.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tofdc_val,
            1,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.tofdc_test,
            1,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
