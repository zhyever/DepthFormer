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
class SUNRGBDDataset(Dataset):
    """SUNRGBD dataset for semantic segmentation. An example of file structure
    is as followed.
    .. code-block:: none
        ├── root_path
        │   ├── SUNRGBD
        │   │   ├── kv1
        │   │   ├── kv2
        │   │   ├── ...
        |   ├── RUNRGBD_val_splits.txt
        │   │
    split file format:
    input_image: SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/image/0000103.jpg 
    gt_depth:    SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/image/0000103.png 
    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
    """


    def __init__(self,
                 pipeline,
                 img_suffix='.jpg',
                 depth_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 depth_scale=1000,
                 min_depth=1e-3,
                 max_depth=10):
        self.pipeline = Compose(pipeline)
        self.img_suffix = img_suffix
        self.depth_map_suffix = depth_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.data_root, self.img_suffix,
                                               self.depth_map_suffix, self.split)
        

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, data_root, img_suffix, depth_map_suffix, split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to data directory
            img_suffix (str): Suffix of images.
            depth_map_suffix (str|None): Suffix of depth maps.
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
            print("Split is None, ERROR")
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

    def format_results(self, results, imgfile_prefix=None, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        results[0] = (results[0] * self.depth_scale)
        # .astype(np.uint16)
        return results

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

    def eval_mask(self, depth_gt):
        depth_gt = np.squeeze(depth_gt)
        valid_mask = np.logical_and(depth_gt > self.min_depth, depth_gt < self.max_depth)
        
        ##########
        gt_height, gt_width = depth_gt.shape
        eval_mask = np.zeros(valid_mask.shape)


        # eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
        #             int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        eval_mask[45:471, 41:601] = 1

        valid_mask = np.logical_and(valid_mask, eval_mask)
        ###########

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

            # self.depth_scale
            depthVisData = np.asarray(Image.open(depth_map, 'r'), np.uint16)
            depthInpaint = np.bitwise_or(np.right_shift(depthVisData, 3), np.left_shift(depthVisData, 16 - 3))
            depthInpaint = depthInpaint.astype(np.single) / 1000
            depthInpaint[depthInpaint > 8] = 8
            pred[pred > 8] = 8
            depthInpaint = depthInpaint.astype(np.float32)
            depth_map_gt = depthInpaint

            # depth_map_gt = np.asarray(Image.open(depth_map), dtype=np.float32) / self.depth_scale
            depth_map_gt = np.expand_dims(depth_map_gt, axis=0)

            valid_mask = self.eval_mask(depth_map_gt)
            
            eval = metrics(depth_map_gt[valid_mask], pred[valid_mask])
            pre_eval_results.append(eval)

            # save prediction results
            pre_eval_preds.append(pred)

        # depthInpaint = np.expand_dims(depthInpaint, axis=0)
        # depthInpaint = np.expand_dims(depthInpaint, axis=0)
        # return pre_eval_results, depthInpaint
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
#     MultiScaleFlipAug, Resize, RandomFlip, ImageToTensor, Collect, Normalize

#     import matplotlib.pyplot as plt
    
#     pipeline = [LoadImageFromFile(), DepthLoadAnnotations(), DepthKBCrop(), DepthRandomRotate(prob=0.5, degree=10),
#                 DepthRandomFlip(prob=0.25), DepthRandomCrop(crop_size=(352, 704)), PhotoMetricDistortion(), DepthDefaultFormatBundle()]

#     test_pipeline = [LoadImageFromFile(), DepthKBCrop(), ImageToTensor(keys=['img']), Collect(keys=['img'], 
#                              meta_keys=('filename', 'ori_filename', 'ori_shape', 
#                             'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'))]

#     data = KITTIDataset(
#                  test_pipeline,
#                  "input",
#                  img_suffix='.png',
#                  ann_dir= "gt_depth",
#                  seg_map_suffix='.png',
#                 #  split='kitti_eigen_train.txt',
#                  split='kitti_eigen_test.txt',
#                  data_root="/mnt/10-5-108-187/xxx/data_depth_annotated/",
#                  test_mode=True,
#                  depth_scale=256
#                  )
    
#     item = data.__getitem__(100)
#     for key, val in item.items():
#         print(key)
#         print(val)

#     plt.subplot(2, 1, 1)
#     print(item["img"][0].data.shape)
#     plt.imshow(item["img"][0].data.permute(1, 2, 0))
#     # plt.subplot(2, 1, 2)
#     # plt.imshow(item["depth_gt"].data.squeeze())
#     plt.savefig("mmdepth/project/debug_imgs/test.png")

    # import os

    # files = os.listdir("/mnt/10-5-108-187/xxx/data_depth_annotated/test_images")
    # with open("/mnt/10-5-108-187/xxx/data_depth_annotated/benchmark_test_split.txt", "w+") as f:
    #     for i in files:
    #         split_str = i
    #         f.write(split_str + "\n")
        