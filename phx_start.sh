#! /bin/sh

spc run pod $1 -N contrastive \
 -i zhyever/mmdetection3d:latest \
 -v "/mnt/10-5-108-187/lizhenyu1:/nfs/lizhenyu1/:rw" \
 -v "/mnt/10-5-108-187/mig:/nfs/share_data/:rw" \
 -v "/dev/shm:/dev/shm:rw" \
 --cmd "bash,-c" \
 --cmd-args "pip install tensorboardx && pip uninstall -y mmsegmentation && pip uninstall -y mmcv-full && pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html  && cd /nfs/lizhenyu1/workspace/python_workspace/mmsegmentation && bash ./tools/dist_train.sh ./mmdepth/configs/resUnet/kitti_res101Unet_baseline.py 8 --work-dir ./nfs/lzy/res101Unet_baseline" \
 --gpus-per-pod 8 --cpus-per-pod 12 --mems-per-pod=128Gi


# spc run pytorch-job $1 -N contrastive \
#  -p elena \
#  -n 1 \
#  -i zhyever/mmdetection3d:latest \
#  -v "/mnt/10-5-108-187/lizhenyu1:/nfs/lizhenyu1/:rw" \
#  -v "/mnt/10-5-108-187/mig:/nfs/share_data/:rw" \
#  -v "/dev/shm:/dev/shm:rw" \
#  --use-spot \
#  --cmd "bash,-c" \
#  --cmd-args "pip install tensorboardx && cd /nfs/lizhenyu1/workspace/python_workspace/Depth_Estimation_Baseline && bash ./dist_train.sh 8 --config ./configs/waymo_abl/resUnet50_kitti_oneten.json  --model_name oneten" \
#  --gpus-per-pod 8 --cpus-per-pod 18 --mems-per-pod=128Gi