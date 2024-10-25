from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from models.utils import FrequencySparseRegularity
from networks.util import DCT2, unpatchify, radial_coords, SepConvGRU, conv_act, ChannelAttention, ds_conv

_PADDING_MODE = 'replicate'


class DepthUpdateModule(nn.Module):
    def __init__(self, hidden_dim: int = 128, patch_size: int = 8, scale: float = 1., seq_drop_rate: float = 0.):
        """
        A module that can progressively predict the coefficient of depth in frequency domain
        :param hidden_dim:
        :param patch_size:
        """
        super(DepthUpdateModule, self).__init__()

        assert patch_size in [8]
        self.patch_size = patch_size
        self.scale = scale
        self.seq_drop_rate = seq_drop_rate

        # generate progressive masks
        coords = radial_coords(self.patch_size, 1)
        path_indices = {
            8: [0, 1, 2, 3, 4, 5, [6, 7], [8, 9, 10, 11, 12, 13, 14]]
        }
        self.generate_indices(coords, path_indices[self.patch_size])
        self.n_steps = len(path_indices[self.patch_size]) + 1  # 1-update all components

        # define modules
        self.dct = DCT2(self.patch_size)
        self.in_proj = InputProjection(hidden_dim, self.patch_size)
        self.gru = SepConvGRU(hidden_dim, self.in_proj.out_chs)

        out_dims = [
            getattr(self, f'_indices_{idx}').numel()
            for idx in range(self.n_steps - 1)
        ]
        out_dims.append(self.patch_size ** 2)
        self.heads = DepthHead(hidden_dim, out_dims)

        # frequency regularity
        self.freq_reg = FrequencySparseRegularity(self.patch_size)

    @torch.no_grad()
    def generate_indices(self, coords: List[torch.Tensor], path_indices: list):
        """
        Generate output masks
        :param coords:
        :param path_indices:
        :return:
        """
        paths = self.generate_paths(coords, path_indices)

        # progressively generate indices
        indices = []
        for idx, path in enumerate(paths):
            indices.append(path)

            cum_indices = torch.cat(indices, 0)[None, :, None, None]

            self.register_buffer(f'_indices_{idx}', cum_indices, persistent=True)
            self.register_buffer(f'_cum_indices_{idx + 1}', cum_indices.clone(), persistent=True)

    def scatter_freq(self, freq_map: torch.Tensor, freq_pred: torch.Tensor, idx: int):
        """
        Scatter predicted frequency coefficient to a frequency map. This is a in-place operation
        :param freq_map: (B, p * p, H, W)
        :param freq_pred: (B, lc, H, W)
        :param idx: int
        :return:
        """
        # get path
        coords = getattr(self, f'_indices_{idx}')
        assert coords.numel() == freq_pred.shape[1]
        assert torch.all(freq_map[:, coords, :, :] == 0.)

        # scatter in place
        freq_map.scatter_(1, coords.expand_as(freq_pred), freq_pred)

    def freq_mtx(self, x: torch.Tensor):
        """
        Reshape a tensor to frequency matrix
        :param x: (B, C, H, W)
        :return:
        """
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H, W, self.patch_size, self.patch_size)

    def freq2depth(self, depth_freq: torch.Tensor):
        """
        Convert depth to spatial domain from frequency domain
        :param depth_freq: (b, p * p, h, w)
        :return: depth in log space
        """
        B, _, H, W = depth_freq.shape

        depth = self.freq_mtx(depth_freq)
        depth = self.dct.inv_transform(depth) * (self.scale * self.patch_size * 0.5)  # depth in log space or metric space
        depth = unpatchify(depth).contiguous().unsqueeze(1)

        return depth

    def generate_freq_sequence(self, freq_map: torch.Tensor, idx: int):
        """
        Generate frequency sequence as the input of gru cell
        :param freq_map: (B, C, H, W)
        :param idx:
        :return:
        """
        if idx <= 0:
            return None

        B, _, H, W = freq_map.shape

        # gather frequency values, (B, L, H, W)
        sequence = torch.gather(freq_map.detach(), 1,
                                getattr(self, f'_cum_indices_{idx}').expand(B, -1, H, W))

        return sequence

    def forward(self, gru_hidden: torch.Tensor, max_iters: int = None):
        """
        Progressively predict the coefficient of each component
        :param gru_hidden:
        :param max_iters:
        :return: depth in log space
        """
        B, _, H, W = gru_hidden.shape

        # frequency map
        freq_map = torch.zeros(B, self.patch_size ** 2, H, W).type_as(gru_hidden)

        # store results
        depths = []
        freq_regs = []
        freq_maps = []

        # Progressively predict the dct coefficients and refine the previous coefficients
        if max_iters is None:
            max_iters = self.n_steps
        assert 0 < max_iters <= self.n_steps
        for idx in range(max_iters):
            # update gru_hidden when idx > 0
            if idx > 0:
                # encode depth and depth_freq
                freq_sequence = self.generate_freq_sequence(freq_map, idx)  # detached
                input_features = self.in_proj(depth.detach(), freq_sequence)

                # sequence dropout
                if self.training:
                    # draw drop mask
                    drop_mask = torch.bernoulli(
                        torch.full((B, 1, 1, 1), 1. - self.seq_drop_rate)
                    ).type_as(gru_hidden)

                    # update hidden state
                    gru_hidden = self.gru(gru_hidden, input_features) * drop_mask + gru_hidden * (1. - drop_mask)
                else:
                    # update hidden state
                    gru_hidden = self.gru(gru_hidden, input_features)

            # predict the coefficients
            out = self.heads(gru_hidden, idx)  # (b, o, h, w)

            # scatter to correct position
            if idx < self.n_steps - 1:
                coe_update = torch.zeros(B, self.patch_size ** 2, H, W).type_as(gru_hidden)
                self.scatter_freq(coe_update, out, idx)
            else:
                coe_update = out

            # update predicted coefficients
            freq_map = coe_update + freq_map.detach()

            # compute frequency regularity
            if self.training:
                if idx > 0:
                    freq_reg = self.freq_reg(self.freq_mtx(freq_map))
                else:
                    freq_reg = torch.zeros(1).type_as(gru_hidden)
                freq_regs.append(freq_reg)

            # update current state
            depth = self.freq2depth(freq_map)

            # store intermediate results
            depths.append(depth)
            freq_maps.append(freq_map.detach().clone())

        if self.training:
            return depths, freq_regs
        else:
            return depths, freq_maps

    @staticmethod
    def generate_paths(coords: List[torch.Tensor], indices: list):
        paths = []

        for ids in indices:
            if isinstance(ids, list):
                paths.append(
                    torch.cat(
                        [coords[idx] for idx in ids], dim=0
                    )
                )
            else:
                paths.append(coords[ids])

        return paths


class DepthHead(nn.Module):
    def __init__(self, in_dim: int, out_dims: List[int]):
        super().__init__()

        # define index embedding
        self.idx_embed = nn.ParameterList(
            [
                nn.Parameter(torch.empty(1, in_dim, 1, 1, dtype=torch.float32))
                for _ in out_dims
            ]
        )

        # PPM head
        self.ppm = PyramidPooling([1, 2, 3, 6], in_dim)

        # define convs for each iterative step
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_dim, out_dim, 3, padding=1, padding_mode=_PADDING_MODE)
                for out_dim in out_dims
            ]
        )

        with torch.no_grad():
            for param in self.idx_embed:
                trunc_normal_(param, std=1.0e-4)

            for idx, conv in enumerate(self.convs):
                if idx < len(self.convs) - 1:
                    std = 0.02 * (0.8 ** idx)
                    trunc_normal_(conv.weight, 0., std)
                else:
                    nn.init.constant_(conv.weight, 0.)
                nn.init.constant_(conv.bias, 0.)

    def forward(self, x: torch.Tensor, idx: int):
        return self.convs[idx](self.ppm(x + self.idx_embed[idx]))


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        """
        Perform attention
        :param x: (B, N, C), N[0] is the cls_token when self.mode == 'ca'
        :return:
        """
        B, N, C = x.shape

        # compute q only from the first element
        q = self.q(x[:, 0: 1, :]).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, 1, C)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (2, B, H, N, C)
        k, v = kv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B, H, 1, N)
        attn = attn.softmax(dim=-1)
        x = attn @ v  # (B, H, 1, C)

        x = x.transpose(1, 2).contiguous().view(B, 1, C)
        x = self.proj(x)

        return x


class FrequencyModule(nn.Module):
    def __init__(self, n_dim: int, n_heads: int, n_freq: int):
        super().__init__()

        self.n_dim = n_dim
        self.n_freq = n_freq

        # cls token
        self.cls_token = nn.Parameter(torch.empty(1, 1, n_dim, dtype=torch.float32))

        # pos embed
        self.pos_embed = nn.Parameter(torch.empty(1, n_freq, n_dim, dtype=torch.float32))

        # tokenize
        self.tokenize = nn.Sequential(
            conv_act(1, n_dim // 2, 7),
            conv_act(n_dim // 2, n_dim, 3),
            nn.Conv2d(n_dim, n_dim, 3, padding=1, padding_mode=_PADDING_MODE)
        )

        # attention
        self.attn = CrossAttention(n_dim, n_heads, True)

        # layer norm
        self.norm = nn.LayerNorm(n_dim)

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=1.0e-4)
        trunc_normal_(self.pos_embed, std=.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        """
        Process frequency input
        :param x: (B, L, H, W)
        :return:
        """
        # tokenize
        B, L, H, W = x.shape
        x = self.tokenize(x.view(B * L, 1, H, W)).view(B, L, self.n_dim, H, W)
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, L, self.n_dim)
        # pos embed
        x = x + self.pos_embed[:, : L, :]
        # concat
        x = torch.cat(
            [
                self.cls_token.expand(B * H * W, 1, self.n_dim),
                x
            ], 1
        )  # (B * H * W, L + 1, C)
        # process
        x = x[:, 0: 1, :] + self.attn(self.norm(x))  # (B * H * W, 1, C)

        return x.squeeze(1)  # (B * H * W, C)


class InputProjection(nn.Module):
    def __init__(self, hidden_dim: int, patch_size: int):
        super().__init__()
        self.out_chs = hidden_dim
        mid_dim = 128
        n_heads = mid_dim // 16

        # define depth input layers
        self.depth_in = nn.Sequential(
            conv_act(1, mid_dim // 2, 7, 2),
            conv_act(mid_dim // 2, mid_dim, 3, 2),
            nn.Conv2d(mid_dim, mid_dim, 3, 2, 1, padding_mode=_PADDING_MODE)
        )
        # define depth_freq input layers
        self.depth_freq_in = FrequencyModule(mid_dim, n_heads, patch_size ** 2)

        # feature attention
        self.fuse = nn.Sequential(
            ChannelAttention(mid_dim * 2, 8),
            ds_conv(mid_dim * 2, hidden_dim, 7)
        )

    def forward(self, depth: torch.Tensor, freq_sequence: torch.Tensor):
        """
        Process depth and depth_freq
        :param depth: (b, 1, h, w)
        :param freq_sequence: (b * h * w, L, 1)
        :return:
        """
        #
        # Compute depth feature
        #
        depth_feature = self.depth_in(depth)
        B, _, H, W = depth_feature.shape

        #
        # Compute frequency feature
        #
        sequence = self.depth_freq_in(freq_sequence).view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # concat and attn
        out = self.fuse(
            torch.cat([depth_feature, sequence], 1)
        )

        return out


class PyramidPooling(nn.Module):
    def __init__(self, scales: List[int], n_dim: int):
        """
        PPM module
        :param scales:
        :param n_dim:
        """
        super().__init__()

        self.scales = scales

        self.pool_convs = nn.ModuleList(
            [
                nn.Conv2d(n_dim, n_dim, 1)
                for _ in self.scales
            ]
        )
        self.squeeze = nn.Sequential(
            ds_conv(n_dim * (len(self.scales) + 1), n_dim, 7),
            nn.Hardswish(inplace=True)
        )

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    trunc_normal_(m.weight, 0., 0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # compute pooling size
        if H >= W:
            pool_sizes = [
                (round(H / W * s), s) for s in self.scales
            ]
        else:
            pool_sizes = [
                (s, round(W / H * s)) for s in self.scales
            ]

        # pool
        out = [
            F.adaptive_avg_pool2d(x, s) for s in pool_sizes
        ]

        # conv and interpolate
        out = [
            F.interpolate(conv(px), size=(H, W), mode='bilinear', align_corners=True)
            for conv, px in zip(self.pool_convs, out)
        ]
        # concat
        out = torch.cat([x] + out, 1)
        # squeeze
        out = self.squeeze(out)

        return out
