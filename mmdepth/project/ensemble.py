import numpy as np
import os
import mmcv

def generate_reweight_list(weight, min_val=0.6, max_val=1.0):
    min_score = min(weight)
    max_score = max(weight)
    interval = max_val - min_val
    reweighted_list = []
    for each_score in weight:
        reweight_score = min_val + (each_score - min_score) * (max_val - min_val) / interval
        reweighted_list.append(reweight_score)
    return reweighted_list

split_name = "benchmark_test_split.txt"
split_path = "data/kitti"
out_path = "nfs/ensemble_res"

split_file = os.path.join(split_path, split_name)
file_names = []

with open(split_file, 'r') as f:
    for line in f:
        file_names.append(line[:-1])

ensemble_paths = []
ensemble_paths.append("nfs/kitti_benchmark_res/large_w7_best_silog_relu")
ensemble_paths.append("nfs/kitti_benchmark_res/large_w12_best_silog_relu")
ensemble_paths.append("nfs/kitti_benchmark_res/large_w7_best_silog_sigmoid")
ensemble_paths.append("nfs/kitti_benchmark_res/large_w12_best_silog_sigmoid")

# ensemble_weights = [-7.2797, -7.3422, -7.3942, -7.4487]
# ensemble_weights = generate_reweight_list(ensemble_weights)
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

