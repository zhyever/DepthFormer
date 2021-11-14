from collections import OrderedDict

import mmcv
import numpy as np
import torch

def calculate(gt, pred):
    if gt.shape[0] == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)

    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    if np.isnan(silog):
        silog = 0
        
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel

def metrics(gt, pred, interval1=20, interval2=60, max_depth=80):
    mask = gt > 0
    gt = gt[mask]
    pred = pred[mask]

    a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = calculate(gt, pred)

    # TODO: hack here to eval different distance:
    mask_1 = gt <= interval1
    mask_2 = gt > 0
    mask = np.logical_and(mask_1, mask_2)
    gt_l1 = gt[mask]
    pred_l1 = pred[mask]
    a1_l1, a2_l1, a3_l1, abs_rel_l1, rmse_l1, log_10_l1, rmse_log_l1, silog_l1, sq_rel_l1 = calculate(gt_l1, pred_l1)

    mask_1 = gt <= interval2
    mask_2 = gt > interval1
    mask = np.logical_and(mask_1, mask_2)
    gt_l2 = gt[mask]
    pred_l2 = pred[mask]
    a1_l2, a2_l2, a3_l2, abs_rel_l2, rmse_l2, log_10_l2, rmse_log_l2, silog_l2, sq_rel_l2 = calculate(gt_l2, pred_l2)

    mask_1 = gt <= max_depth
    mask_2 = gt > interval2
    mask = np.logical_and(mask_1, mask_2)
    gt_l3 = gt[mask]
    pred_l3 = pred[mask]
    a1_l3, a2_l3, a3_l3, abs_rel_l3, rmse_l3, log_10_l3, rmse_log_l3, silog_l3, sq_rel_l3 = calculate(gt_l3, pred_l3)

    return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel, \
           a1_l1, a2_l1, a3_l1, abs_rel_l1, rmse_l1, log_10_l1, rmse_log_l1, silog_l1, sq_rel_l1, \
           a1_l2, a2_l2, a3_l2, abs_rel_l2, rmse_l2, log_10_l2, rmse_log_l2, silog_l2, sq_rel_l2, \
           a1_l3, a2_l3, a3_l3, abs_rel_l3, rmse_l3, log_10_l3, rmse_log_l3, silog_l3, sq_rel_l3

# hack for enhance interval evaluation
# def metrics(gt, pred, interval1=20, interval2=60, max_depth=80):
#     mask = gt > 0
#     gt = gt[mask]
#     pred = pred[mask]

#     a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = calculate(gt, pred)

#     temp = []
#     # TODO: hack here to eval different distance:
#     for index, begin in enumerate(range(80)):
#         end = begin + 1

#         mask_1 = gt <= end
#         mask_2 = gt > begin
#         mask = np.logical_and(mask_1, mask_2)
#         gt_l1 = gt[mask]
#         pred_l1 = pred[mask]
#         a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = calculate(gt_l1, pred_l1)
#         temp.extend([a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel])

#     return temp

def eval_metrics(gt, pred):
    mask = gt > 0
    gt = gt[mask]
    pred = pred[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def pre_eval_to_metrics(pre_eval_results):

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    ret_metrics = OrderedDict({})

    level_num = len(pre_eval_results)  // 9
    for i in range(level_num):
        ret_metrics['a1_{}'.format("all" if i==0 else "l_{}".format(i))] = np.nanmean(pre_eval_results[i*9+0])
        ret_metrics['a2_{}'.format("all" if i==0 else "l_{}".format(i))] = np.nanmean(pre_eval_results[i*9+1])
        ret_metrics['a3_{}'.format("all" if i==0 else "l_{}".format(i))] = np.nanmean(pre_eval_results[i*9+2])
        ret_metrics['abs_rel_{}'.format("all" if i==0 else "l_{}".format(i))] = np.nanmean(pre_eval_results[i*9+3])
        ret_metrics['rmse_{}'.format("all" if i==0 else "l_{}".format(i))] = np.nanmean(pre_eval_results[i*9+4])
        ret_metrics['log_10_{}'.format("all" if i==0 else "l_{}".format(i))] = np.nanmean(pre_eval_results[i*9+5])
        ret_metrics['rmse_log_{}'.format("all" if i==0 else "l_{}".format(i))] = np.nanmean(pre_eval_results[i*9+6])
        ret_metrics['silog_{}'.format("all" if i==0 else "l_{}".format(i))] = np.nanmean(pre_eval_results[i*9+7])
        ret_metrics['sq_rel_{}'.format("all" if i==0 else "l_{}".format(i))] = np.nanmean(pre_eval_results[i*9+8])

    ret_metrics = {
        metric: value
        for metric, value in ret_metrics.items()
    }

    return ret_metrics


# hack for curve plot
# def pre_eval_to_metrics(pre_eval_results):

#     # convert list of tuples to tuple of lists, e.g.
#     # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
#     # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
#     pre_eval_results = tuple(zip(*pre_eval_results))
#     ret_metrics = OrderedDict({})

#     level_num = len(pre_eval_results)  // 9
#     for i in range(level_num):
#         ret_metrics['a1_{}'.format(i)] = np.nanmean(pre_eval_results[i*9+0])
#         ret_metrics['a2_{}'.format(i)] = np.nanmean(pre_eval_results[i*9+1])
#         ret_metrics['a3_{}'.format(i)] = np.nanmean(pre_eval_results[i*9+2])
#         ret_metrics['abs_rel_{}'.format(i)] = np.nanmean(pre_eval_results[i*9+3])
#         ret_metrics['rmse_{}'.format(i)] = np.nanmean(pre_eval_results[i*9+4])
#         ret_metrics['log_10_{}'.format(i)] = np.nanmean(pre_eval_results[i*9+5])
#         ret_metrics['rmse_log_{}'.format(i)] = np.nanmean(pre_eval_results[i*9+6])
#         ret_metrics['silog_{}'.format(i)] = np.nanmean(pre_eval_results[i*9+7])
#         ret_metrics['sq_rel_{}'.format(i)] = np.nanmean(pre_eval_results[i*9+8])

#     ret_metrics = {
#         metric: value
#         for metric, value in ret_metrics.items()
#     }

#     print(ret_metrics)
#     # mmcv.dump(ret_metrics, "nfs/DepthFormer/sup_dist_curve/swinconvmf.pkl")

#     return ret_metrics
