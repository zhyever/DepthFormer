# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import DEPTHLOSS

@DEPTHLOSS.register_module()
class SigLoss(nn.Module):
    """SigLoss.

    Args:
        valid_mask (bool, optional): Whether filter invalid gt
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight

    def sigloss(self, input, target):
        pass

    def forward(self,
                depth_pred,
                depth_gt,
                **kwargs):
        """Forward function."""
        pass
