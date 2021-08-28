from collections import OrderedDict

import mmcv
import numpy as np
import torch

def metrics(gt, pred):
    mask = gt > 0
    gt = gt[mask]
    pred = pred[mask]

    # TODO: hack here to eval different distance:
    mask_1 = gt < 80
    mask_2 = gt > 60
    mask = np.logical_and(mask_1, mask_2)
    gt = gt[mask]
    pred = pred[mask]
    if gt.shape[0] == 0:
        return None


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
    return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel

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


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 9
    ret_metrics = total_items_to_metrics(total_num = len(pre_eval_results[0]),
                                         a1 = sum(pre_eval_results[0]),
                                         a2 = sum(pre_eval_results[1]),
                                         a3 = sum(pre_eval_results[2]),
                                         abs_rel = sum(pre_eval_results[3]),
                                         rmse = sum(pre_eval_results[4]),
                                         log_10 = sum(pre_eval_results[5]),
                                         rmse_log = sum(pre_eval_results[6]),
                                         silog = sum(pre_eval_results[7]),
                                         sq_rel = sum(pre_eval_results[8]),)

    return ret_metrics


def total_items_to_metrics(total_num,
                           a1,
                           a2,
                           a3,
                           abs_rel,
                           rmse,
                           log_10,
                           rmse_log,
                           silog,
                           sq_rel
                           ):

    ret_metrics = OrderedDict({})
    ret_metrics['a1'] = a1 / total_num
    ret_metrics['a2'] = a2 / total_num
    ret_metrics['a3'] = a3 / total_num
    ret_metrics['abs_rel'] = abs_rel / total_num
    ret_metrics['rmse'] = rmse / total_num
    ret_metrics['log_10'] = log_10 / total_num
    ret_metrics['rmse_log'] = rmse_log / total_num
    ret_metrics['silog'] = silog / total_num
    ret_metrics['sq_rel'] = sq_rel / total_num

    ret_metrics = {
        metric: value
        for metric, value in ret_metrics.items()
    }
    
    return ret_metrics