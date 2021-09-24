import numpy as np
import os
import mmcv

split_name = "benchmark_test_split.txt"
split_path = "data/kitti"
out_path = "nfs/ensemble_res_new"

split_file = os.path.join(split_path, split_name)
file_names = []

with open(split_file, 'r') as f:
    for line in f:
        file_names.append(line[:-1])

ensemble_paths = []
ensemble_paths.append("nfs/lzy/kitti_benchmark_npz_res_best_abs")
ensemble_paths.append("nfs/lzy/kitti_benchmark_npz_res_best_rmse")
ensemble_paths.append("nfs/lzy/kitti_benchmark_npz_res_best_abs_w7")
ensemble_paths.append("nfs/lzy/kitti_benchmark_npz_res_best_rmse_w7")

ensemble_weights = [1, 1, 1, 1]

for index, name in enumerate(file_names):
    ensemble_temp = []
    for ensemble_path in ensemble_paths:
        item_path = os.path.join(ensemble_path, name + ".npy")
        item = np.load(item_path)
        ensemble_temp.append(item)
    
    ensemble_sum = 0
    for res, weight in zip(ensemble_temp, ensemble_weights):
        ensemble_sum += weight * res
    ensemble_res = ensemble_sum / sum(ensemble_weights)

    ensemble_res = ensemble_res.astype(np.uint16)

    out_file = os.path.join(out_path, name)
    mmcv.imwrite(ensemble_res.squeeze(), out_file)

