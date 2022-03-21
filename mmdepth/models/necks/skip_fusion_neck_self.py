# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
 
from mmseg.ops import resize
from mmseg.models.builder import DEPTHNECK

import torch
import math

from torch.nn.modules.activation import MultiheadAttention

from mmdepth.models.necks.ops.modules import MSDeformAttn

from mmcv.cnn import build_norm_layer

# position embedding for fusion layer
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, mask):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).contiguous().unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

@DEPTHNECK.register_module()
class DepthFusionMultiLevelNeckSA(nn.Module):
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
                 levels=4,
                 scales=[0.5, 1, 2, 4],
                 norm_cfg=None,
                 act_cfg=None,
                 embedding_dim=64,
                 cross_att=True,
                 self_att=True):
        super(DepthFusionMultiLevelNeckSA, self).__init__()
        assert isinstance(in_channels, list)
        self.cross_att = cross_att
        self.self_att = self_att
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.embedding_dim = embedding_dim

        # TODO: hard code? remove maybe
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
                    in_channel + self.embedding_dim,
                    out_channel,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # for fusion layer. part of hard codes
        self.proj_conv = nn.ModuleList()
        for in_channel, out_channel in zip(in_channels, out_channels):
            self.proj_conv.append(
                nn.Sequential(
                    ConvModule(
                        in_channel,
                        self.embedding_dim,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    nn.BatchNorm2d(self.embedding_dim),
                    nn.ReLU(inplace=True)    
                    ))

        ########################################

        num_feature_levels = levels
        self.feat_pos_embed = PositionEmbeddingSine(self.embedding_dim//2)
        self.query_pos_embed = PositionEmbeddingSine(self.embedding_dim//2)
        self.reference_points = nn.Linear(self.embedding_dim, 2)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, self.embedding_dim))

        self.self_attn = MSDeformAttn(self.embedding_dim, n_levels=num_feature_levels, n_heads=8, n_points=8)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # TODO: hard code? remove maybe
        transformer_feature = inputs

        # deal with transformer skip feats
        inputs = [
            lateral_conv(transformer_feature[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # here, List[outs] saves the skip transformer feats
        masks = []
        src_flattens = []
        mask_flatten = []
        spatial_shapes = []
        lvl_pos_embed_flatten = []
        for i in range(len(transformer_feature)):
            bs, c, h, w = inputs[i].shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            mask = torch.zeros_like(inputs[i][:, 0, :, :]).type(torch.bool)
            masks.append(mask)
            pos = self.feat_pos_embed(inputs[i], mask)
            mask = mask.flatten(1)
            pos_embed = pos.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[i].view(1, 1, -1)

            feat = self.proj_conv[i](inputs[i])
            flatten_feat = feat.flatten(2).transpose(1, 2)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            src_flattens.append(flatten_feat)
        
        src_flatten = torch.cat(src_flattens, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src_flatten.device)
        if self.self_att:
            src = self.self_attn(self.with_pos_embed(src_flatten, lvl_pos_embed_flatten), 
                                reference_points, 
                                src_flatten, 
                                spatial_shapes, 
                                level_start_index, 
                                None)
        else:
            src = src_flatten

        start = 0
        new_feats = []
        for i in range(len(inputs)):
            bs, c, h, w = inputs[i].shape
            end = start + h * w
            feat = src[:, start:end, :].permute(0, 2, 1).contiguous()
            start = end
            feat = feat.reshape(bs, self.embedding_dim, h, w)
            new_feat = torch.cat([inputs[i], feat], dim=1)
            new_feats.append(new_feat)

        outs = []
        for i in range(len(transformer_feature)):
            if self.scales[i] != 1:
                x_resize = resize(
                    new_feats[i], scale_factor=self.scales[i], mode='bilinear')
            else:
                x_resize = new_feats[i]
            x_resize = self.convs[i](x_resize)
            outs.append(x_resize)
        
        return tuple(outs)
