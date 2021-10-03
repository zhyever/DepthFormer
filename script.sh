#! /bin/sh

./tools/dist_train.sh mmdepth/configs/new_exps/ablations_convnum/wo_conv4.py 8 --work-dir nfs/ablation_convnum/wo_conv4

./tools/dist_train.sh mmdepth/configs/new_exps/ablations_convnum/wo_conv1.py 8 --work-dir nfs/ablation_convnum/wo_conv1

./tools/dist_train.sh mmdepth/configs/new_exps/ablations_convnum/wo_conv2.py 8 --work-dir nfs/ablation_convnum/wo_conv2

./tools/dist_train.sh mmdepth/configs/new_exps/ablations_convnum/wo_conv3.py 8 --work-dir nfs/ablation_convnum/wo_conv3
