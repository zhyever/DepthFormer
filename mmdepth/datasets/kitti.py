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

from mmdepth.core import pre_eval_to_metrics, total_items_to_metrics, metrics, eval_metrics
# from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose

from mmseg.ops import resize

from PIL import Image

import torch


@DATASETS.register_module()
class KITTIDataset(Dataset):
    """KITTI dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── KITTI
        │   │   ├── kitti_eigen_train.txt
        │   │   ├── kitti_eigen_test.txt
        │   │   ├── data
        │   │   │   ├── date_1
        │   │   │   ├── date_2
        │   │   │   |   ...
        │   │   │   |   ...
        |   │   ├── gt_depth
        │   │   │   ├── date_drive_number_sync

    split file format:
    input_image: 2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.png 
    gt_depth:    2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png 
    focal:       721.5377

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
                 img_dir,
                 img_suffix='.png',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 depth_scale=256):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
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
                    if ann_dir is not None:
                        depth_map = line.strip().split(" ")[1]
                        if depth_map == 'None':
                            self.invalid_depth_num += 1
                            continue
                        img_info['ann'] = dict(depth_map=depth_map)
                    img_name = line.strip().split(" ")[0]
                    img_info['filename'] = img_name
                    img_infos.append(img_info)
        else:
            raise NotImplementedError 

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
        results['img_prefix'] = self.img_dir
        results['depth_prefix'] = self.ann_dir
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
            depth_map = osp.join(self.ann_dir, img_info['ann']['depth_map'])
            depth_map_gt = np.asarray(Image.open(depth_map), dtype=np.float32) / self.depth_scale
            yield depth_map_gt
    
    # get kb cropped gt, shape 1, W, H
    def eval_kb_crop(self, depth_gt):
        height = depth_gt.shape[0]
        width = depth_gt.shape[1]
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        depth_cropped = depth_gt[top_margin: top_margin + 352, left_margin: left_margin + 1216]
        depth_cropped = np.expand_dims(depth_cropped, axis=0)
        return depth_cropped


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

        for i, (pred, index) in enumerate(zip(preds, indices)):
            
            depth_map = osp.join(self.ann_dir,
                               self.img_infos[index]['ann']['depth_map'])
            depth_map_gt = np.asarray(Image.open(depth_map), dtype=np.float32) / self.depth_scale
            depth_map_gt = self.eval_kb_crop(depth_map_gt)
            
            # if i == 1:
            #     import matplotlib.pyplot as plt
            #     plt.subplot(2, 1, 1)
            #     pred = torch.tensor(pred)
            #     plt.imshow(pred.permute(1, 2, 0))
            #     plt.subplot(2, 1, 2)
            #     depth_map_gt = torch.tensor(depth_map_gt)
            #     plt.imshow(depth_map_gt.squeeze())
            #     plt.savefig("mmdepth/project/debug_imgs/test_1.png")
            #     exit(1000)

            # if pred.shape[0] != depth_map_gt.shape[0] or pred.shape[1] != depth_map_gt.shape[1]:
            #     pred_torch = torch.tensor(pred, dtype=torch.float)
            #     pred_torch_resized = resize(
            #                         input=pred_torch.unsqueeze(dim=0),
            #                         size=depth_map_gt.shape,
            #                         mode='bilinear',
            #                         align_corners=True)
            #     pred_torch_resized = pred_torch_resized.squeeze()
            #     pred = pred_torch_resized.numpy()

            pre_eval_results.append(
                metrics(depth_map_gt, pred))

        return pre_eval_results

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
            ret_metrics = pre_eval_to_metrics(results, metric)

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value), 4)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # for logger
        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            summary_table_data.add_column(key, [val])

        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
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
#                  data_root="/mnt/10-5-108-187/lizhenyu1/data_depth_annotated/",
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