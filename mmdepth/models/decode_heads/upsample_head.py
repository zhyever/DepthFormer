# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn.functional import embedding

from mmseg.models.builder import DEPTHHEAD
from .decode_head import DepthBaseDecodeHead
import torch.nn.functional as F

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.convA(torch.cat([up_x, concat_with], dim=1))))

@DEPTHHEAD.register_module()
class UpsampleHead(DepthBaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 up_sample_channels=[2048, 1024, 512, 256, 128],
                 **kwargs):
        super(UpsampleHead, self).__init__(**kwargs)
        # in_channels=[2048, 1024, 512, 256, 64],
        # in_index=[0, 1, 2, 3, 4],
        self.up_sample_channels = up_sample_channels
        self.conv_list = nn.ModuleList()
        up_channel_temp = 0
        for index, (in_channel, up_channel) in enumerate(zip(self.in_channels, up_sample_channels)):
            if index == 0:
                self.conv_list.append(ConvModule(in_channels=in_channel, out_channels=up_channel, kernel_size=1, stride=1, padding=0))
            else:
                self.conv_list.append(UpSample(skip_input=in_channel + up_channel_temp, output_features=up_channel))
            # save earlier fusion target
            up_channel_temp = up_channel

    def forward(self, inputs):
        """Forward function."""
        # inputs order first -> end
        temp_feat_list = []
        for index, feat in enumerate(inputs[::-1]):
            if index == 0:
                temp_feat = self.conv_list[index](feat)
                temp_feat_list.append(temp_feat)
            else:
                skip_feat = feat
                up_feat = temp_feat_list[index-1]
                temp_feat = self.conv_list[index](up_feat, skip_feat)
                temp_feat_list.append(temp_feat)

        output = self.depth_pred(temp_feat_list[-1])
        return output
