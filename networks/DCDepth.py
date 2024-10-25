import torch
import torch.nn as nn

from networks.layers import PyramidFeatureFusionV2, DctDownsample
from .newcrf_layers import NewCRF
from .swin_transformer import SwinTransformer
from .depth_update import DepthUpdateModule


class DCDepth(nn.Module):
    """
    Depth network with dct. Replace the PPM head with PFF
    """

    def __init__(self, version=None, pretrained=None, scale: float = 1.0, ape: bool = False,
                 img_size: tuple = None, drop_path_rate: float = 0.2, drop_path_rate_crf: float = 0.,
                 seq_dropout_rate: float = 0., downsample_strategy: str = 'dct', **kwargs):
        super().__init__()

        self.patch_size = 8
        self.scale = scale
        self.img_size = img_size

        window_size = int(version[-2:])

        if version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]
        else:
            raise ValueError(f'Unknown version: {version}.')

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
            pretrain_img_size=img_size,
            ape=ape
        )
        print(f'Backbone cfg: {backbone_cfg}.')

        embed_dim = 512  # Note, different embed_dim

        self.backbone = SwinTransformer(**backbone_cfg)
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.hidden_dim = 192  # 128
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3],
                           num_heads=32, drop_path=drop_path_rate_crf)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2],
                           num_heads=16, drop_path=drop_path_rate_crf)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=self.hidden_dim, window_size=win, v_dim=v_dims[1],
                           num_heads=8, drop_path=drop_path_rate_crf)

        # build depth update module
        self.update = DepthUpdateModule(
            hidden_dim=self.hidden_dim,
            patch_size=self.patch_size,
            scale=self.scale,
            seq_drop_rate=seq_dropout_rate
        )

        self.decoder = PyramidFeatureFusionV2([8, 4, 2, 1], in_channels, embed_dim, downsample_strategy)
        self.project_hidden = nn.Conv2d(self.hidden_dim + in_channels[0], self.hidden_dim, 3, padding=1)
        self.project_context = DctDownsample(2, 5, 2, in_channels[0], in_channels[0])

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        # self.decoder.init_weights()

    def parameters_1x(self):
        """
        Fine-tuning parameters
        :return:
        """
        yield from self.backbone.parameters()

    def parameters_5x(self):
        """
        Training parameters
        :return:
        """
        param_1x = set(self.parameters_1x())
        for param in self.parameters():
            if param not in param_1x:
                yield param

    def forward(self, imgs: torch.Tensor, max_iters: int = None, return_freq_maps: bool = False):
        assert imgs.shape[-2:] == self.img_size, f'Input image size {imgs.shape[-2:]} is not equal to {self.img_size}.'

        feats = self.backbone(imgs)
        pff_out = self.decoder(*feats)

        e3 = self.crf3(feats[-1], pff_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[-2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[-3], e2)

        context = self.project_context(feats[0])
        gru_hidden = torch.tanh(
            self.project_hidden(
                torch.cat([e1, context], 1)
            )
        )

        depths = self.update(gru_hidden, max_iters=max_iters)

        if self.training:
            return depths
        else:
            return depths if return_freq_maps else depths[0]
