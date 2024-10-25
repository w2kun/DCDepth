import numpy as np
import torch
import torch.nn as nn
from scipy.fft import dct
from torch.nn.init import trunc_normal_


def patchify(img: torch.Tensor, patch_size: int):
    """
    Cut image into patches
    :param img: (b, h, w)
    :param patch_size:
    :return: (b, h', w', p, p)
    """
    b, h, w = img.shape
    assert ((h % patch_size) == 0) and ((w % patch_size) == 0)

    img = img.view(b, h // patch_size, patch_size, w // patch_size, patch_size).permute(0, 1, 3, 2, 4)

    return img


def unpatchify(img: torch.Tensor):
    """
    Reshape a patchified image
    :param img: (b, h // p, w // p, p, p)
    :return:
    """
    b, hdp, wdp, p, p = img.shape

    img = img.permute(0, 1, 3, 2, 4).reshape(b, hdp * p, wdp * p)

    return img


class ChannelAttention(nn.Module):
    """
    An adapted SE-Module
    """

    def __init__(self, in_dim: int, reduction: int):
        super().__init__()

        r_dim = max(in_dim // reduction, 16)
        self.attention = nn.Sequential(
            nn.Linear(in_dim, r_dim),
            nn.Hardswish(inplace=True),
            nn.Linear(r_dim, in_dim),
            nn.Sigmoid()
        )

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, 0., 0.02)
                    nn.init.constant_(m.bias, 0.)

    def forward(self, x: torch.Tensor):
        """
        Squeeze
        :param x: (b, c, h, w)
        :return:
        """
        attention = self.attention(x.mean(dim=(2, 3)))[..., None, None]  # (b, c, 1, 1)
        return x * attention


class DCT2(nn.Module):
    """
    Discrete cosine transformation module
    """

    def __init__(self, patch_size: int):
        super().__init__()

        assert 2 <= patch_size <= 128
        self.patch_size = patch_size

        T = dct(np.eye(patch_size), axis=0, norm='ortho').astype(np.float32)
        self.register_buffer('_dct_mtx', torch.from_numpy(T).view(self.patch_size, self.patch_size),
                             persistent=True)

    def transform(self, img: torch.Tensor):
        """
        Transform an image in spatial domain to frequency domain
        :param img: (..., p, p)
        :return:
        """
        assert img.shape[-2:] == (self.patch_size, self.patch_size)

        return self._dct_mtx @ img @ self._dct_mtx.transpose(0, 1)

    def inv_transform(self, img: torch.Tensor):
        """
        Transform an image in frequency domain to spatial domain
        :param img: (..., p, p)
        :return:
        """
        assert img.shape[-2:] == (self.patch_size, self.patch_size)

        return self._dct_mtx.transpose(0, 1) @ img @ self._dct_mtx

    def forward(self):
        raise NotImplementedError


def radial_coords(size: int, nd: int):
    """
    Generate radial coords
    :param size: side length
    :param nd: 1 or 2
    :return:
    """
    assert 1 < size < 50
    assert nd in [1, 2]

    results = []

    for s in range(2 * size - 1):
        temp = []
        j = 0 if s < size else s - size + 1

        while j <= min(s, size - 1):
            coord = (s - j, j)
            if nd == 2:
                temp.append(
                    coord
                )
            else:
                temp.append(coord[0] * size + coord[1])
            j += 1

        results.append(
            torch.tensor(temp, dtype=torch.int64)
        )

    return results


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128 + 192):
        super(SepConvGRU, self).__init__()

        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


def get_act(act: str):
    if act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'hard_swish':
        return nn.Hardswish(inplace=True)
    else:
        raise ValueError(f'Unknown act function: {act}.')


def get_norm(norm: str, in_dim: int, num_groups: int = 4):
    if norm == 'bn':
        return nn.BatchNorm2d(in_dim)
    elif norm == 'ln':
        return nn.LayerNorm(in_dim)
    elif norm == 'in':
        return nn.InstanceNorm2d(in_dim)
    elif norm == 'gn':
        return nn.GroupNorm(num_groups, in_dim)
    elif norm == 'gn2ln':
        return nn.GroupNorm(1, in_dim)
    else:
        raise ValueError(f'Unknown norm layer: {norm}.')


def conv_act(in_dim: int, out_dim: int, kernel_size: int, stride: int = 1, act: str = 'hard_swish'):
    conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride,
                     (kernel_size - 1) // 2, padding_mode='replicate')
    act = get_act(act)
    return nn.Sequential(conv, act)


def ds_conv(in_dim: int, out_dim: int, kernel_size: int, bias: bool = True):
    dw = nn.Conv2d(in_dim, in_dim, kernel_size, 1, (kernel_size - 1) // 2, padding_mode='replicate', groups=in_dim)
    pw = nn.Conv2d(in_dim, out_dim, 1, bias=bias)
    return nn.Sequential(dw, pw)


# def group_conv_act(in_dim: int, out_dim: int, kernel_size: int, stride: int = 1, groups: int = 1,
#                    act: str = 'hard_swish'):
#     conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride,
#                      (kernel_size - 1) // 2, groups=groups, padding_mode='replicate')
#     act = get_act(act)
#     return nn.Sequential(conv, act)


def conv_norm_act(in_dim: int, out_dim: int, kernel_size: int, stride: int = 1, norm: str = 'gn',
                  act: str = 'hard_swish', num_groups: int = 4):
    conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride,
                     (kernel_size - 1) // 2, padding_mode='replicate', bias=(norm != 'bn'))
    norm = get_norm(norm, out_dim, num_groups)
    act = get_act(act)
    return nn.Sequential(conv, norm, act)
