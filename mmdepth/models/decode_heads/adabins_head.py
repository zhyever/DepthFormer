# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn.functional import embedding

from mmseg.models.builder import DEPTHHEAD
from .decode_head import DepthBaseDecodeHead
import torch.nn.functional as F

class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E

        x = self.conv3x3(x)

        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps

class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4, norm=None):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4, norm=norm)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        self.positional_encodings = nn.Parameter(torch.rand(2000, embedding_dim), requires_grad=True)

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2)  # .shape = n,c,s = n, embedding_dim, s
        # embeddings = nn.functional.pad(embeddings, (1,0))  # extra special token at start ?
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.contiguous().view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).contiguous().view(n, cout, h, w)


class UpFusion(nn.Sequential):
    # head input channels, right now channels(need to concate with head input channels), target channels
    def __init__(self, in_channel, up_channel_temp, output_features):
        super(UpFusion, self).__init__()
        self.conv_fusion = ConvModule(in_channel + output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.identify = ConvModule(up_channel_temp, output_features, kernel_size=3, stride=1, padding=1)
        self.final = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, skip_feat):
        x = self.identify(x)
        up_x = F.interpolate(x, size=[skip_feat.size(2), skip_feat.size(3)], mode='bilinear', align_corners=True)
        plus_with = self.conv_fusion(torch.cat([up_x, skip_feat], dim=1))
        res = up_x + self.leakyreluA(plus_with)
        res = self.final(res)
        res = self.leakyreluB(plus_with)
        return res


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
class AdabinsHead(DepthBaseDecodeHead):
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
                 att_fusion=False,
                 **kwargs):
        super(AdabinsHead, self).__init__(**kwargs)
        # in_channels=[2048, 1024, 512, 256, 64],
        # in_index=[0, 1, 2, 3, 4],
        self.att_fusion = att_fusion
        self.up_sample_channels = up_sample_channels
        self.conv_list = nn.ModuleList()
        up_channel_temp = 0
        for index, (in_channel, up_channel) in enumerate(zip(self.in_channels, up_sample_channels)):
            if index == 0:
                self.conv_list.append(ConvModule(in_channels=in_channel, out_channels=up_channel, kernel_size=1, stride=1, padding=0))
            else:
                if self.att_fusion == True: # res up sample
                    self.conv_list.append(UpFusion(in_channel, up_channel_temp, up_channel))
                else:
                    self.conv_list.append(UpSample(skip_input=in_channel + up_channel_temp, output_features=up_channel))
            # save earlier fusion target
            up_channel_temp = up_channel

        # final feat dim = 128
        n_bins = 256

        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=128, norm='linear')
        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0))
        self.softmax = nn.Sequential(nn.Softmax(dim=1))
            

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

        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(temp_feat_list[-1])
        out = self.conv_out(range_attention_maps)
        out = self.softmax(out)

        bin_widths = (self.max_depth - self.min_depth) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.contiguous().view(n, dout, 1, 1)

        output = torch.sum(out * centers, dim=1, keepdim=True)

        return output
