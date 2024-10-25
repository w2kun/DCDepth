from torch.utils.data import DataLoader

from easydict import EasyDict
from pytorch_lightning import LightningDataModule
from .kitti_official import preprocessing_transforms, DataLoadPreprocess
from .registry import DATAMODULES


@DATAMODULES.register_module('kitti_official')
class KITTIDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # prepare args
        args = {
            'filenames_file_test': 'data_splits/kitti_official_test.txt',
            'filenames_file_eval': 'data_splits/kitti_official_valid.txt',  # files for online eval
            'filenames_file': 'data_splits/kitti_depth_prediction_train_2.txt',  # files for training
            'dataset': 'kitti',
            'use_right': True,
            'data_path': self.cfg.dataset.data_path,
            'data_path_eval': self.cfg.dataset.data_path_eval,
            'data_path_test': self.cfg.dataset.data_path_test,
            'gt_path': self.cfg.dataset.gt_path,
            'do_kb_crop': self.cfg.evaluation.do_kb_crop,
            'input_height': self.cfg.dataset.input_height,
            'input_width': self.cfg.dataset.input_width,
            'do_random_rotate': True,
            'degree': 1.0,
            'max_translation_x': 8
        }
        self.args = EasyDict(args)

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self.kitti_train = DataLoadPreprocess(self.args, 'train',
                                                  transform=preprocessing_transforms('train'))
            self.kitti_val = DataLoadPreprocess(self.args, 'online_eval',
                                                transform=preprocessing_transforms('online_eval'))

        if stage == 'test' or stage is None:
            self.kitti_test = DataLoadPreprocess(self.args, 'test',
                                                 transform=preprocessing_transforms('test'))

    def train_dataloader(self):
        return DataLoader(
            self.kitti_train,
            self.cfg.training.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg.training.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.kitti_val,
            1,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.kitti_test,
            1,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
