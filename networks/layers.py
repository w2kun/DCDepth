import torch
import torch.nn as nn

from typing import List
from networks.util import DCT2, conv_norm_act, ChannelAttention, ds_conv


class Downsample(nn.Module):
    def __init__(self, factor: int, kernel: int, reduction: int, in_dim: int, out_dim: int, strategy: str):
        super().__init__()

        assert strategy in ['bilinear', 'pixel_unshuffle']

        self.factor = factor

        if strategy == 'bilinear':
            self.downsample = nn.Upsample(scale_factor=(1.0 / factor), mode='bilinear', align_corners=True)
            mid_dim = in_dim
        elif strategy == 'pixel_unshuffle':
            mid_dim = factor ** 2 * in_dim // reduction
            self.downsample = nn.Sequential(
                nn.PixelUnshuffle(factor),
                nn.Conv2d(in_dim * (factor ** 2), mid_dim, 1, groups=in_dim)
            )
        else:
            raise NotImplementedError

        self.conv = nn.Sequential(
            nn.Conv2d(mid_dim, out_dim, 1, bias=(kernel > 1)),
            nn.Conv2d(out_dim, out_dim, kernel, padding=(kernel - 1) // 2,
                      bias=False, groups=out_dim) if kernel > 1 else nn.Identity(),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x: torch.Tensor):
        """
        Downsample the input tensor
        :param x: (B, C, H, W)
        :return:
        """
        # downsample
        x = self.downsample(x)

        # conv
        x = self.conv(x)

        return x


class DctDownsample(nn.Module):
    def __init__(self, factor: int, kernel: int, reduction: int, in_dim: int, out_dim: int):
        super().__init__()

        self.factor = factor
        self.dct = DCT2(self.factor)

        self.n_freq = self.factor ** 2
        mid_dim = self.n_freq // reduction * in_dim
        self.conv1 = nn.Conv2d(self.n_freq * in_dim, mid_dim, 1, groups=in_dim)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_dim, out_dim, 1, bias=(kernel > 1)),
            nn.Conv2d(out_dim, out_dim, kernel, padding=(kernel - 1) // 2,
                      bias=False, groups=out_dim) if kernel > 1 else nn.Identity(),
            nn.BatchNorm2d(out_dim)
        )

    def patchify(self, x: torch.Tensor):
        """
        Patchify images to patches
        :param x: (B, C, H, W)
        :return:
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.factor, self.factor, W // self.factor, self.factor).transpose(3, 4).contiguous()
        return x

    def forward(self, x: torch.Tensor):
        """
        Downsample the input tensor
        :param x: (B, C, H, W)
        :return:
        """
        B, C, H, W = x.shape
        H_, W_ = H // self.factor, W // self.factor

        # patchify
        x = self.patchify(x).view(B, C, H_, W_, self.factor, self.factor)
        # to frequency domain
        x = self.dct.transform(x).flatten(-2)  # (B, C, H_, W_, P * P)
        # stack
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(B, C * self.factor * self.factor, H_, W_)
        # conv
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class PyramidFeatureFusionV2(nn.Module):
    def __init__(self, scales: List[int], in_dims: List[int], out_dim: int, downsample: str = 'dct'):
        super().__init__()

        self.downs = nn.ModuleList()
        cat_dims = 0
        for scale, in_dim in zip(scales, in_dims):
            if scale > 1:
                self.downs.append(
                    DctDownsample(scale, 5, 4, in_dim, out_dim)
                    if downsample == 'dct' else Downsample(scale, 5, 4, in_dim, out_dim, downsample)
                )
                cat_dims += out_dim
            else:
                self.downs.append(
                    nn.Identity()
                )
                cat_dims += in_dim
        self.fuse = nn.Sequential(
            ChannelAttention(cat_dims, 16),
            nn.Conv2d(cat_dims, out_dim, 1),
            nn.Conv2d(out_dim, out_dim, 7, padding=3, groups=out_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Hardswish(inplace=True)
        )

    def forward(self, *features):
        # downsample
        out = [
            self.downs[idx](feat) for idx, feat in enumerate(features)
        ]
        # concat
        out = torch.cat(
            out, 1
        )
        # fuse
        out = self.fuse(out)

        return out
