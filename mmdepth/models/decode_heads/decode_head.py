# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from mmseg.models.builder import build_depth_loss
from mmseg.models.losses import accuracy


class DepthBaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='SigLoss').
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 conv_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='SigLoss',
                     valid_mask=True,
                     loss_weight=1.0),
                 sampler=None,
                 align_corners=False,
                 min_depth=None,
                 max_depth=None):
        super(DepthBaseDecodeHead, self).__init__()
        
        self.input_transform = input_transform
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_depth_loss(loss_decode)
        self.align_corners = align_corners
        self.min_depth = min_depth
        self.max_depth = max_depth
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_depth = nn.Conv2d(channels, 1, kernel_size=3, padding=1, stride=1)
        self.fp16_enabled = False
        self.relu = nn.ReLU()

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'align_corners={self.align_corners}'
        return s

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, depth_gt, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            depth_gt (Tensor): GT depth
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        depth_pred = self.forward(inputs)
        losses = self.losses(depth_pred, depth_gt)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def depth_pred(self, feat):
        """Prediction each pixel."""
        output = self.relu(self.conv_depth(feat)) + self.min_depth
        return output

    @force_fp32(apply_to=('depth_pred', ))
    def losses(self, depth_pred, depth_gt):
        """Compute segmentation loss."""
        loss = dict()
        depth_pred = resize(
            input=depth_pred,
            size=depth_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        loss['loss_depth'] = self.loss_decode(
            depth_pred,
            depth_gt)
        return loss
