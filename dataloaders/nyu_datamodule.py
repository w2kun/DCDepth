from easydict import EasyDict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataloader import preprocessing_transforms, DataLoadPreprocess
from .registry import DATAMODULES


@DATAMODULES.register_module('nyu')
class NYUDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # prepare args
        args = {
            'filenames_file_eval': 'data_splits/nyudepthv2_test_files_with_gt.txt',  # files for online eval
            'filenames_file': 'data_splits/nyudepthv2_train_files_with_gt.txt',  # files for training
            'dataset': 'nyu',
            'use_right': False,
            'data_path': self.cfg.dataset.data_path,
            'data_path_eval': self.cfg.dataset.data_path_eval,
            'gt_path': self.cfg.dataset.data_path,
            'gt_path_eval': self.cfg.dataset.data_path_eval,
            'do_kb_crop': self.cfg.evaluation.do_kb_crop,
            'input_height': self.cfg.dataset.input_height,
            'input_width': self.cfg.dataset.input_width,
            'do_random_rotate': True,
            'degree': 2.5,
            'max_translation_x': 32
        }
        self.args = EasyDict(args)

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self.nyu_train = DataLoadPreprocess(self.args, 'train',
                                                  transform=preprocessing_transforms('train'))
            self.nyu_val = DataLoadPreprocess(self.args, 'online_eval',
                                              transform=preprocessing_transforms('online_eval'))

        if stage == 'test' or stage is None:
            self.nyu_test = DataLoadPreprocess(self.args, 'online_eval',
                                               transform=preprocessing_transforms('online_eval'))

    def train_dataloader(self):
        return DataLoader(
            self.nyu_train,
            self.cfg.training.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg.training.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.nyu_val,
            1,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.nyu_test,
            1,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )