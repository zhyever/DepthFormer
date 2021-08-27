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
        if self.valid_mask:
            valid_mask = target > 0
            input = input[valid_mask]
            target = target[valid_mask]
            
        g = torch.log(input) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self,
                depth_pred,
                depth_gt,
                **kwargs):
        """Forward function."""

        loss_depth = self.loss_weight * self.sigloss(
            depth_pred,
            depth_gt,
            )
        return loss_depth
