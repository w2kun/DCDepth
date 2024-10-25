import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.DCDepth import DCDepth
from pytorch_lightning import LightningModule
from torch.optim import AdamW

from utils import post_process_depth, flip_lr, compute_errors_pth, colormap, inv_normalize, colormap_magma
from .registry import MODELS
from .utils import SmoothRegularity


class SILogLossInstance(nn.Module):
    def __init__(self, variance_focus: float, patch_size: int = 8, min_valid_pixels: int = 4, square_root: bool = True):
        super().__init__()

        assert 0 <= variance_focus <= 1.
        self.variance_focus = variance_focus

        self.patch_size = patch_size
        self.min_valid_pixels = min_valid_pixels
        self.square_root = square_root
        self.register_buffer(
            '_weight',
            torch.ones(1, 1, self.patch_size, self.patch_size, dtype=torch.float32)
        )

    def forward(self, depth_log: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor, **kwargs):
        """
        Compute the silog loss
        :param depth_log: depth prediction in log space
        :param depth_gt: depth ground truth in metric
        :param mask: valid mask, binary
        :return:
        """
        mask = mask.float()
        assert depth_log.shape == mask.shape

        # filter mask
        if self.min_valid_pixels > 0:
            patch_mask = F.conv2d(mask, self._weight, stride=self.patch_size)
            patch_mask = (patch_mask >= self.min_valid_pixels).float()
            patch_mask = patch_mask.repeat_interleave(self.patch_size, dim=-1).repeat_interleave(self.patch_size, dim=-2)
            mask = mask * patch_mask

        B, _, H, W = depth_log.shape
        # convert gt to log space
        depth_gt = torch.log(depth_gt.clamp_min(1.0e-3))
        # flatten
        depth_log = depth_log.flatten(1)
        depth_gt = depth_gt.flatten(1)
        mask = mask.flatten(1)
        # compute difference
        diff = (depth_log - depth_gt) * mask
        # compute silog loss for each sample
        num = mask.sum(1)
        loss = diff.square().sum(1) / num - self.variance_focus * (diff.sum(1) / num).square()  # (B,)
        if self.square_root:
            loss = loss.sqrt()
        loss = 10. * loss
        # compute weight
        loss = loss.mean()

        return loss


@MODELS.register_module()
class DCTProg(LightningModule):
    """
    Bisection depth model
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.patch_size = 8
        self.max_depth = self.cfg.dataset.max_depth
        self.min_depth = self.cfg.dataset.min_depth

        # output space, (metric or log)
        self.output_space = self.cfg.model.output_space
        assert self.output_space in ['metric', 'log']

        # model
        self.model = DCDepth(
            self.cfg.model.encoder,
            self.cfg.model.pretrain,
            scale=(math.log(self.max_depth) if self.output_space == 'log' else self.max_depth),
            img_size=(self.cfg.dataset.input_height, self.cfg.dataset.input_width),
            ape=self.cfg.model.ape,
            drop_path_rate=self.cfg.model.drop_path_rate,
            drop_path_rate_crf=self.cfg.model.drop_path_rate_crf,
            seq_dropout_rate=self.cfg.model.seq_dropout_rate
        )

        # loss
        self.si_log = SILogLossInstance(self.cfg.loss.variance_focus, self.patch_size,
                                        self.cfg.loss.min_valid_pixels, self.cfg.loss.square_root)

        self.smooth_regularity = SmoothRegularity()

        self.beta = self.cfg.loss.beta
        if self.beta is not None:
            assert 0.5 <= self.beta <= 1.5

        self.total_steps = None

        print(f'Output Space={self.output_space}.')

    def output2metric(self, out: torch.Tensor):
        """
        Convert output in metric or log space to metric depth
        :param out:
        :return:
        """
        if self.output_space == 'log':
            return out.exp()
        elif self.output_space == 'metric':
            return out
        else:
            raise NotImplementedError

    def output2log(self, out: torch.Tensor):
        """
        Convert output in metric or log space to log space
        :param out:
        :return:
        """
        if self.output_space == 'log':
            return out
        elif self.output_space == 'metric':
            return out.clamp_min(1.0e-4).log()
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        self.total_steps = self.trainer.estimated_stepping_batches

        optimizer = AdamW(
            [
                {
                    'params': self.model.parameters_5x(),
                    'lr': self.cfg.optimization.max_lr,
                    'weight_decay': 0.
                },
                {
                    'params': self.model.parameters_1x(),
                    'lr': self.cfg.optimization.max_lr / self.cfg.optimization.lr_ratio,
                    'weight_decay': self.cfg.optimization.weight_decay
                },
            ]
        )
        # scheduler
        lrs = [group['lr'] for group in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, lrs, self.total_steps, div_factor=self.cfg.optimization.div_factor,
            final_div_factor=self.cfg.optimization.final_div_factor, pct_start=self.cfg.optimization.pct_start,
            anneal_strategy=self.cfg.optimization.anneal_strategy
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    @torch.no_grad()
    def log_images(self, image, image_aug, depth_preds, depth_gt):
        writer = self.logger.experiment

        depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e-3, depth_gt)
        global_step = self.global_step

        # visualize rgb
        writer.add_image(f'train/image', inv_normalize(image[0, :, :, :]), global_step)
        writer.add_image(f'train/image_aug', inv_normalize(image_aug[0, :, :, :]), global_step)

        # visualize depth
        n_pred = len(depth_preds)
        if self.cfg.dataset.name in ['nyu', 'tofdc']:
            writer.add_image(f'train/depth_gt', colormap(depth_gt[0, :, :, :]), global_step)

            for idx in range(n_pred):
                depth_pred = self.output2metric(depth_preds[idx].detach()[0])
                writer.add_image(f'train/depth_pred_{idx}', colormap(depth_pred), global_step)

        else:
            writer.add_image(f'train/depth_gt', colormap_magma(torch.log10(depth_gt[0, :, :, :])), global_step)

            for idx in range(n_pred):
                depth_pred = self.output2metric(depth_preds[idx].detach()[0])
                writer.add_image(f'train/depth_pred_{idx}', colormap_magma(torch.log10(depth_pred)), global_step)

    def get_exponential_weights(self, n: int):
        xs = np.arange(n)
        ys = self.beta ** xs
        ys = ys / ys.sum()
        weights = ys.tolist()
        return list(reversed(weights))

    def on_train_epoch_start(self) -> None:
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        # fetch data
        image = batch['image']
        image_aug = batch['image_aug']
        depth_gt = batch['depth']

        # get model prediction
        depths, freq_regs = self.model(image_aug)

        # compute loss
        mask = depth_gt >= self.min_depth

        total_loss = 0.
        weight_func = self.get_exponential_weights
        for idx, (depth_log, freq_reg, weight) in enumerate(zip(depths, freq_regs, weight_func(len(depths)))):
            # get depth in log space
            depth_log = self.output2log(depth_log)

            # compute si-log loss
            si_log = self.si_log(depth_log, depth_gt, mask)
            if idx > 3:
                smooth_reg = self.smooth_regularity(depth_log, image)
            else:
                smooth_reg = torch.zeros(1, dtype=si_log.dtype, device=si_log.device)

            self.log(f'loss/si_log_{idx}', si_log.item(), on_step=True, on_epoch=False)
            self.log(f'loss/freq_reg_{idx}', freq_reg.item(), on_step=True, on_epoch=False)
            self.log(f'loss/smooth_reg_{idx}', smooth_reg.item(), on_step=True, on_epoch=False)

            total_loss += weight * (si_log + self.cfg.loss.freq_reg_weight * freq_reg +
                                    self.cfg.loss.smooth_reg_weight * smooth_reg)

        # # log total loss
        # self.log(f'loss/total', total_loss.item(), on_step=True, on_epoch=False, prog_bar=True)

        # log images
        if self.global_step % self.cfg.training.log_freq == 0:
            self.log_images(image, image_aug, depths, depth_gt)

        # log lr
        optim = self.optimizers()
        for idx, group in enumerate(optim.optimizer.param_groups):
            self.log(f'learning_rate/group_{idx}', group['lr'])

        return total_loss

    def evaluate_depth(self, batch, batch_idx):
        post_process = True

        # fetch data
        image = batch['image']
        gt_depth = batch['depth']
        has_valid_depth = batch['has_valid_depth']

        if not has_valid_depth:
            # print('Has no valid depth.')
            return

        depths = self.model(image)
        depth = self.output2metric(depths[-1])
        if post_process:
            image_flipped = flip_lr(image)
            depth_flipped = self.output2metric(self.model(image_flipped)[-1])
            pred_depth = post_process_depth(depth, depth_flipped)
        else:
            pred_depth = depth

        pred_depth = pred_depth.squeeze()
        gt_depth = gt_depth.squeeze()

        if self.cfg.evaluation.do_kb_crop:
            assert self.cfg.dataset.name in ['kitti_eigen', 'kitti_official']
            height, width = pred_depth.shape
            top_margin = 352 - height
            left_margin = (1216 - width) // 2
            pred_depth_uncropped = torch.zeros(352, 1216).type_as(pred_depth)
            pred_depth_uncropped[top_margin:, left_margin: left_margin + width] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < self.min_depth] = self.min_depth
        pred_depth[pred_depth > self.max_depth] = self.max_depth
        pred_depth[torch.isinf(pred_depth)] = self.max_depth
        pred_depth[torch.isnan(pred_depth)] = self.min_depth

        valid_mask = torch.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth)

        if self.cfg.evaluation.garg_crop or self.cfg.evaluation.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = torch.zeros_like(valid_mask)

            if self.cfg.evaluation.garg_crop:
                eval_mask[int(0.40810811 * gt_height): int(0.99189189 * gt_height),
                int(0.03594771 * gt_width): int(0.96405229 * gt_width)] = 1

            elif self.cfg.evaluation.eigen_crop:
                if self.cfg.dataset.name == 'kitti':
                    eval_mask[int(0.3324324 * gt_height): int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width): int(0.96405229 * gt_width)] = 1
                elif self.cfg.dataset.name == 'nyu':
                    eval_mask[45: 471, 41: 601] = 1
                else:
                    raise NotImplementedError

            valid_mask = torch.logical_and(valid_mask, eval_mask)

        # compute metrics
        measures = compute_errors_pth(gt_depth[valid_mask], pred_depth[valid_mask])

        # log
        for metric_name, metric in measures.items():
            self.log(f'val/{metric_name}', metric, sync_dist=True)

        # log image
        if batch_idx < 8:
            writer = self.logger.experiment

            writer.add_image(f'val/image_{batch_idx}', inv_normalize(image[0]), global_step=self.global_step)

            # plot error map
            error_map = ((pred_depth - gt_depth).abs() * valid_mask).unsqueeze(0)
            writer.add_image(f'val/error_map_{batch_idx}', colormap(error_map), global_step=self.global_step)

            # pred_depth = remove_border(pred_depth, 4)
            if self.cfg.dataset.name in ['nyu', 'tofdc']:
                writer.add_image(f'val/pred_{batch_idx}', colormap(pred_depth.unsqueeze(0)),
                                 global_step=self.global_step)
                writer.add_image(f'val/gt_{batch_idx}', colormap(gt_depth.unsqueeze(0)),
                                 global_step=self.global_step)
            else:
                writer.add_image(f'val/pred_{batch_idx}', colormap_magma(pred_depth.unsqueeze(0).log10()),
                                 global_step=self.global_step)
                writer.add_image(f'val/gt_{batch_idx}', colormap_magma(gt_depth.clamp_min(1.0e-3).unsqueeze(0).log10()),
                                 global_step=self.global_step)

    def validation_step(self, batch, batch_idx):
        self.evaluate_depth(batch, batch_idx)
