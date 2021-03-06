# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
 
from mmseg.ops import resize
from mmseg.models.builder import DEPTHNECK


@DEPTHNECK.register_module()
class DepthMultiLevelNeck(nn.Module):
    """MultiLevelNeck.

    A neck structure connect vit backbone and decoder_heads. For DPT resemble blocks.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        scales (List[float]): Scale factors for each input feature map.
            Default: [0.5, 1, 2, 4]
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scales=[0.5, 1, 2, 4],
                 norm_cfg=None,
                 act_cfg=None):
        super(DepthMultiLevelNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_channel, out_channel in zip(in_channels, out_channels):
            self.lateral_convs.append(
                ConvModule(
                    in_channel,
                    out_channel,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        for in_channel, out_channel in zip(out_channels, out_channels):
            self.convs.append(
                ConvModule(
                    in_channel,
                    out_channel,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = []
        for i in range(self.num_outs):
            if self.scales[i] != 1:
                x_resize = resize(
                    inputs[i], scale_factor=self.scales[i], mode='bilinear')
            else:
                x_resize = inputs[i]
            outs.append(x_resize)
        return tuple(outs)
