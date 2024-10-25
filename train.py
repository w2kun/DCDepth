import os.path as osp
import warnings
from argparse import ArgumentParser

try:
    from mmcv import Config
except ImportError:
    from mmengine import Config

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from dataloaders import DATAMODULES
from models import MODELS


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'config_name',
        type=str,
        help='The name of configuration file.'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1
    )

    return parser.parse_args()


def main():
    warnings.filterwarnings(action='ignore')

    # configurations
    arg = parse_args()
    cfg = Config.fromfile(osp.join('configs/', f'{arg.config_name}.yaml'))

    # set random seed
    seed_everything(cfg.training.seed)

    # data module
    data = DATAMODULES.build(
        {
            'type': cfg.dataset.name,
            'cfg': cfg
        }
    )

    # model
    model = MODELS.build({
        'type': cfg.model.type,
        'cfg': cfg
    })
    print(f'Training with model {type(model).__name__}...')

    # get resume path
    resume_path = cfg.training.resume_from
    if resume_path is not None:
        print(f'The training is resumed from {resume_path}.')
    else:
        print(f'The training is from scratch.')

    # define checkpoint configurations
    work_dir = osp.join(cfg.training.work_dir, arg.config_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=work_dir,
        every_n_epochs=cfg.evaluation.every_n_epochs,
        monitor='val/rms',
        mode='min',
        save_weights_only=False,
        save_top_k=cfg.training.get('save_top_k', 3)
    )
    # define trainer
    trainer = Trainer(
        precision=cfg.training.precision,
        accelerator='gpu',
        default_root_dir=work_dir,
        devices=arg.gpus,
        max_epochs=cfg.training.max_epochs,
        check_val_every_n_epoch=cfg.evaluation.every_n_epochs,
        callbacks=[checkpoint_callback],
        strategy=DDPStrategy(find_unused_parameters=cfg.training.find_unused_parameters, static_graph=False),
        sync_batchnorm=(arg.gpus > 1),
        num_nodes=1,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
    )

    # training
    trainer.fit(model, data, ckpt_path=resume_path)

    # training done.
    print('The training is done.')


if __name__ == '__main__':
    main()
