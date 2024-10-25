import os
import os.path as osp
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.utils import MetricTool
from utils import compute_errors_pth, flip_lr, post_process_depth

try:
    from mmcv import Config
except ImportError:
    from mmengine import Config

from dataloaders import DATAMODULES
from models import MODELS
from tqdm import tqdm
from utils import inv_normalize


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'config_name',
        type=str,
        help='The name of configuration file.'
    )
    parser.add_argument(
        'resume_from',
        type=str,
        help='Where to load checkpoint.'
    )
    parser.add_argument(
        '--vis',
        action='store_true'
    )

    return parser.parse_args()


def to_metric_depth(output: torch.Tensor, output_space: str):
    if output_space == 'log':
        return torch.exp(output)
    elif output_space == 'metric':
        return output
    else:
        raise NotImplementedError


@torch.no_grad()
def test(post_process: bool):
    # warnings.filterwarnings(action='ignore')
    device = torch.device('cuda')

    # configurations
    arg = parse_args()
    cfg = Config.fromfile(osp.join('configs/', f'{arg.config_name}.yaml'))

    # data module
    data = DATAMODULES.build(
        {
            'type': cfg.dataset.name,
            'cfg': cfg
        }
    )
    data.setup('test')
    loader = data.test_dataloader()
    dataset = cfg.dataset.name

    # read list file
    list_file = {
        'nyu': 'nyudepthv2_test_files_with_gt.txt',
        'kitti_eigen': 'eigen_test_files_with_gt.txt',
        'tofdc': 'TOFDC/TOFDC_test.txt'
    }[dataset]
    with open(osp.join('data_splits', list_file), 'r') as f:
        lines = f.readlines()

    # model
    model = MODELS.build({
        'type': cfg.model.type,
        'cfg': cfg
    })
    print(f'Testing with model {type(model).__name__}...')
    checkpoint = torch.load(arg.resume_from, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint, strict=True)
    print(f'Checkpoint is successfully loaded from {arg.resume_from}.')
    model = model.model
    model.to(device)
    model.eval()

    # define checkpoint configurations
    work_dir = osp.join(cfg.training.work_dir, arg.config_name)

    # metric tool
    metric_tool = MetricTool(work_dir)

    # create vis folder
    if arg.vis:
        vis_dir = osp.join(work_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)

    # begin test
    for batch_idx, batch in enumerate(tqdm(loader)):
        # fetch data
        image = batch['image'].to(device)
        gt_depth = batch['depth'].to(device)
        has_valid_depth = batch['has_valid_depth']

        depths = model(image)
        depth = to_metric_depth(depths[-1], cfg.model.output_space)
        if post_process:
            image_flipped = flip_lr(image)
            depth_flipped = to_metric_depth(model(image_flipped)[-1], cfg.model.output_space)
            pred_depth = post_process_depth(depth, depth_flipped)
        else:
            pred_depth = depth

        pred_depth = pred_depth.squeeze()

        pred_depth[pred_depth < cfg.dataset.min_depth] = cfg.dataset.min_depth
        pred_depth[pred_depth > cfg.dataset.max_depth] = cfg.dataset.max_depth
        pred_depth[torch.isinf(pred_depth)] = cfg.dataset.max_depth
        pred_depth[torch.isnan(pred_depth)] = cfg.dataset.min_depth

        # compute metrics
        if has_valid_depth:
            gt_depth = gt_depth.squeeze()

            if cfg.evaluation.do_kb_crop:
                height, width = gt_depth.shape
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                pred_depth_uncropped = torch.zeros(height, width).type_as(pred_depth)
                pred_depth_uncropped[top_margin: top_margin + 352, left_margin: left_margin + 1216] = pred_depth
                pred_depth = pred_depth_uncropped

            valid_mask = torch.logical_and(gt_depth > cfg.dataset.min_depth, gt_depth < cfg.dataset.max_depth)

            if cfg.evaluation.garg_crop or cfg.evaluation.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = torch.zeros_like(valid_mask)

                if cfg.evaluation.garg_crop:
                    eval_mask[int(0.40810811 * gt_height): int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width): int(0.96405229 * gt_width)] = 1

                elif cfg.evaluation.eigen_crop:
                    if cfg.dataset.name == 'kitti':
                        eval_mask[int(0.3324324 * gt_height): int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width): int(0.96405229 * gt_width)] = 1
                    elif cfg.dataset.name == 'nyu':
                        eval_mask[45: 471, 41: 601] = 1
                    else:
                        raise NotImplementedError

                valid_mask = torch.logical_and(valid_mask, eval_mask)

            measures = compute_errors_pth(gt_depth[valid_mask], pred_depth[valid_mask])

            # log
            metric_tool.add(measures)

        # vis
        if arg.vis:
            if dataset == 'nyu':
                name = lines[batch_idx].split()[0][: -4]
                name = name.split('/')
                name = f'{name[0]}_{name[1]}_depth_dcdepth.png'
                pred_depth = pred_depth.cpu().numpy()
                gt_depth = gt_depth.cpu().numpy()
                plt.imsave(os.path.join(vis_dir, name), pred_depth, cmap='jet', vmin=np.min(gt_depth),
                           vmax=np.max(gt_depth))
            elif dataset == 'kitti_eigen':
                name = lines[batch_idx].split()[0][: -4]
                name = name.split('/')
                name = f'{name[1]}_{name[4]}_dcdepth'
                image = inv_normalize(image.squeeze()).permute(1, 2, 0).cpu().numpy()
                pred_depth = pred_depth.cpu().numpy()
                gt_depth = gt_depth.cpu().numpy()
                mask = gt_depth < 0.1
                gt_depth = np.log10(np.clip(gt_depth, a_min=0.1, a_max=100.))
                gt_depth[mask] = 0.
                plt.imsave(os.path.join(vis_dir, f'{name}_rgb.png'), image)
                plt.imsave(os.path.join(vis_dir, f'{name}_pred.png'), np.log10(pred_depth), cmap='magma')
                plt.imsave(os.path.join(vis_dir, f'{name}_gt.png'), gt_depth, cmap='magma')
            elif dataset == 'tofdc':
                name = f'{batch_idx:05}'
                image = inv_normalize(image.squeeze()).permute(1, 2, 0).cpu().numpy()
                pred_depth = pred_depth.cpu().numpy()
                gt_depth = gt_depth.cpu().numpy()
                plt.imsave(os.path.join(vis_dir, f'{name}_rgb.png'), image)
                plt.imsave(os.path.join(vis_dir, f'{name}_pred.png'), pred_depth, cmap='jet', vmin=np.min(gt_depth),
                           vmax=np.max(gt_depth))
                plt.imsave(os.path.join(vis_dir, f'{name}_gt.png'), gt_depth, cmap='jet')
            else:
                pass

    # summary
    metric_tool.summary()


if __name__ == '__main__':
    test(True)
