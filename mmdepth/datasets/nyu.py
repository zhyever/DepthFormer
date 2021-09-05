# Copyright (c) OpenMMLab. All rights reserved.
from logging import raiseExceptions
import os.path as osp
import warnings
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmdepth.core import pre_eval_to_metrics, metrics, eval_metrics
# from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose

from mmseg.ops import resize

from PIL import Image

import torch

def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s

@DATASETS.register_module()
class NYUDataset(Dataset):
    """NYU dataset for semantic segmentation. An example of file structure
    is as followed.
    .. code-block:: none
        ├── data
        │   ├── NYU
        │   │   ├── nyu_train.txt
        │   │   ├── nuy_test.txt
        │   │   ├── scenes_xxxx (xxxx. No. of the scenes)
        │   │   │   ├── data_1
        │   │   │   ├── data_2
        │   │   │   |   ...
        │   │   │   |   ...
        |   │   ├── scenes (test set, no scene No.)
        │   │   │   ├── data_1 ...
    split file format:
    input_image: /kitchen_0028b/rgb_00045.jpg
    gt_depth:    /kitchen_0028b/sync_depth_00045.png
    focal:       518.8579
    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.png'
        ann_dir (str, optional): Path to annotation directory. Default: None
        depth_map_suffix (str): Suffix of depth maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
    """
 
    def __init__(self,
                 pipeline,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 depth_scale=1000,
                 garg_crop=False,
                 eigen_crop=True,
                 min_depth=1e-3,
                 max_depth=10):

        self.pipeline = Compose(pipeline)
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.garg_crop = garg_crop
        self.eigen_crop = eigen_crop
        self.min_depth = min_depth
        self.max_depth = max_depth
        

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.data_root, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, data_root, split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """
        self.invalid_depth_num = 0
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info = dict()
                    depth_map = line.strip().split(" ")[1]
                    if depth_map == 'None':
                        self.invalid_depth_num += 1
                        continue
                    img_info['ann'] = dict(depth_map=osp.join(data_root, remove_leading_slash(depth_map)))
                    img_name = line.strip().split(" ")[0]
                    img_info['filename'] = osp.join(data_root, remove_leading_slash(img_name))
                    img_infos.append(img_info)
        else:
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images. Totally {self.invalid_depth_num} invalid pairs are filtered', logger=get_root_logger())
        return img_infos
    
    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['depth_fields'] = []
        results['depth_scale'] = self.depth_scale

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        raise NotImplementedError

    def get_gt_depth_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for img_info in self.img_infos:
            depth_map = img_info['ann']['depth_map']
            depth_map_gt = np.asarray(Image.open(depth_map), dtype=np.float32) / self.depth_scale
            yield depth_map_gt

    # get nyu cropped gt, shape 1, H, W
    def eval_nyu_crop(self, depth_gt):
        depth_cropped = depth_gt[45: 472, 43: 608]
        depth_cropped = np.expand_dims(depth_cropped, axis=0)
        return depth_cropped

    def eval_mask(self, depth_gt):
        depth_gt = np.squeeze(depth_gt)
        valid_mask = np.logical_and(depth_gt > self.min_depth, depth_gt < self.max_depth)
        if self.garg_crop or self.eigen_crop:
            gt_height, gt_width = depth_gt.shape
            eval_mask = np.zeros(valid_mask.shape)

            if self.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif self.eigen_crop:
                eval_mask[45:471, 41:601] = 1

        valid_mask = np.logical_and(valid_mask, eval_mask)
        valid_mask = np.expand_dims(valid_mask, axis=0)
        return valid_mask

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.
        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.
        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        pre_eval_preds = []

        for i, (pred, index) in enumerate(zip(preds, indices)):
            depth_map = self.img_infos[index]['ann']['depth_map']

            depth_map_gt = np.asarray(Image.open(depth_map), dtype=np.float32) / self.depth_scale
            depth_map_gt = self.eval_nyu_crop(depth_map_gt)

            # TODO: delete hack for testing transformer
            # depth_map_gt = torch.tensor(depth_map_gt)
            # depth_map_gt = depth_map_gt.unsqueeze(dim=0)
            # depth_map_gt = resize(input=depth_map_gt, size=(352, 704), mode='nearest')
            # depth_map_gt = depth_map_gt.squeeze(dim=0)
            # depth_map_gt = depth_map_gt.numpy()

            valid_mask = self.eval_mask(depth_map_gt)
            
            eval = metrics(depth_map_gt[valid_mask], pred[valid_mask])
            pre_eval_results.append(eval)

            # save prediction results
            pre_eval_preds.append(pred)

        return pre_eval_results, pre_eval_preds

    def evaluate(self, results, metric='eigen', logger=None, **kwargs):
        """Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]: Default metrics.
        """
        metric = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            gt_depth_maps = self.get_gt_depth_maps()
            ret_metrics = eval_metrics(
                gt_depth_maps,
                results)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results)
        
        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        num_table = len(ret_metrics) // 9
        for i in range(num_table):
            names = ret_metric_names[i*9: i*9 + 9]
            values = ret_metric_values[i*9: i*9 + 9]

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 4)
                for ret_metric, ret_metric_value in zip(names, values)
            })

            # for logger
            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics.items():
            eval_results[key] = value

        return eval_results

# if __name__ == '__main__':
#     from mmseg.datasets.pipelines import Compose, LoadImageFromFile, DepthLoadAnnotations, DepthKBCrop, DepthRandomRotate, \
#     DepthRandomFlip, DepthRandomCrop, PhotoMetricDistortion, DepthDefaultFormatBundle,\
#     MultiScaleFlipAug, Resize, RandomFlip, ImageToTensor, Collect, Normalize, DepthNYUCrop

#     import matplotlib.pyplot as plt
    
#     pipeline = [LoadImageFromFile(), DepthLoadAnnotations(), DepthNYUCrop(depth=True), DepthRandomRotate(prob=0.5, degree=10),
#                 DepthRandomFlip(prob=0.25), DepthRandomCrop(crop_size=(416, 544)), PhotoMetricDistortion(), DepthDefaultFormatBundle()]

#     test_pipeline = [LoadImageFromFile(), DepthKBCrop(), ImageToTensor(keys=['img']), Collect(keys=['img'], 
#                              meta_keys=('filename', 'ori_filename', 'ori_shape', 
#                             'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'))]

#     data = NYUDataset(
#                  pipeline,
#                  split='nyu_train.txt',
#                  data_root="/mnt/10-5-108-187/lizhenyu1/nyu/",
#                  test_mode=False,
#                  depth_scale=256
#                  )
    
#     item = data.__getitem__(1000)
#     for key, val in item.items():
#         print(key)
#         print(val)

#     plt.subplot(2, 1, 1)
#     print(item["img"].data.shape)
#     plt.imshow(item["img"].data.permute(1, 2, 0))
#     plt.subplot(2, 1, 2)
#     plt.imshow(item["depth_gt"].data.squeeze())
#     plt.savefig("mmdepth/project/debug_imgs/test.png")