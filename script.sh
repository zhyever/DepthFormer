#! /bin/sh


./tools/dist_train.sh mmdepth/configs/new_exps/nyu_convswin_tiny_unet3.py 8 --work-dir nfs/lzy/nyu_convswin_tiny_unet3_1
./tools/dist_train.sh mmdepth/configs/new_exps/nyu_convswin_tiny_unet3.py 8 --work-dir nfs/lzy/nyu_convswin_tiny_unet3_2
./tools/dist_train.sh mmdepth/configs/new_exps/nyu_swin_tiny_baseline.py 8 --work-dir nfs/lzy/nyu_swin_tiny_baseline_1