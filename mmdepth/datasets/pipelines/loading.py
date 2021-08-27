# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from mmseg.datasets.builder import PIPELINES

from PIL import Image

@PIPELINES.register_module()
class DepthLoadAnnotations(object):
    """Load annotations for depth estimation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['ann_info']['depth_map'])
        else:
            filename = results['ann_info']['depth_map']


        depth_gt = np.asarray(Image.open(filename), dtype=np.float32) / results['depth_scale']
        
        results['depth_gt'] = depth_gt
        results['depth_ori_shape'] = depth_gt.shape

        results['depth_fields'].append('depth_gt')
        results['depth_fields'].append('depth_scale')
        results['depth_fields'].append('depth_ori_shape')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
