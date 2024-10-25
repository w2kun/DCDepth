import math
import os.path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricTool:
    def __init__(self, work_dir: str):
        self.metrics = []
        self.work_dir = work_dir

    def add(self, metrics: Dict[str, torch.Tensor]):
        self.metrics.append(
            {k: v.cpu().numpy() for k, v in metrics.items()}
        )

    def clear(self):
        self.metrics.clear()

    def summary(self, postfix: str = ''):
        df = pd.DataFrame(self.metrics, dtype=np.float32)

        avg = df.mean()
        print('-' * 32)
        print(f'Test samples:{len(self.metrics)}.')
        print(avg)
        print('-' * 32)

        # save to csv
        if postfix is None:
            postfix = ''

        if len(postfix) > 0:
            postfix = '_' + postfix

        # create work dir
        os.makedirs(self.work_dir, exist_ok=True)

        avg.to_csv(
            os.path.join(self.work_dir, f'result_avg{postfix}.csv'),
            header=False,
        )
        df.to_csv(
            os.path.join(self.work_dir, f'result{postfix}.csv')
        )


class FrequencySparseRegularity(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()

        self.patch_size = patch_size
        meshgrid = torch.meshgrid(
            torch.arange(self.patch_size),
            torch.arange(self.patch_size),
            indexing='ij'
        )
        weight = meshgrid[0] + meshgrid[1]
        weight = 1.2 ** weight - 1.0
        weight = weight / weight.sum()
        self.register_buffer('_weight', weight.flatten(), persistent=True)  # (P * P)

    def forward(self, x: torch.Tensor):
        """
        Compute regularity
        :param x: (..., P, P)
        :return:
        """
        assert x.shape[-2:] == (self.patch_size, self.patch_size)

        loss = (x.flatten(-2).abs() * self._weight).sum(-1)
        return loss.mean()


class SmoothRegularity(nn.Module):
    def __init__(self):
        super().__init__()

        self.huber = nn.HuberLoss('none', math.log(1.01))

    def forward(self, depth_log: torch.Tensor, image: torch.Tensor):
        """
        Compute smooth loss
        :param depth_log: (B, 1, H, W), depth in log space
        :param image: (B, 3, H, W)
        :return:
        """
        grad_depth_x = (depth_log[:, :, :, :-1] - depth_log[:, :, :, 1:]).abs()
        grad_depth_y = (depth_log[:, :, :-1, :] - depth_log[:, :, 1:, :]).abs()
        # # using huber loss
        # grad_depth_x = self.huber(grad_depth_x, torch.zeros_like(grad_depth_x))
        # grad_depth_y = self.huber(grad_depth_y, torch.zeros_like(grad_depth_y))

        image = image.mean(dim=1, keepdim=True)
        grad_img_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
        grad_img_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])

        grad_depth_x *= torch.exp(-grad_img_x)
        grad_depth_y *= torch.exp(-grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()


def shift_image(x: torch.Tensor, shift: int):
    B, C, H, W = x.shape
    if shift >= 0:
        return F.pad(x, (shift, 0, shift, 0), 'constant', 0.)[:, :, : H, : W]
    else:
        return F.pad(x, (0, -shift, 0, -shift), 'constant', 0.)[:, :, -H:, -W:]
