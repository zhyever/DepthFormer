#! /bin/sh

./tools/dist_train.sh mmdepth/configs/conv_swin/convswin_interaction_fusion_cos20x.py  8 --work-dir nfs/lzy/convswin_interaction_fusion
./tools/dist_train.sh mmdepth/configs/swin/swin_cos20x.py  8 --work-dir nfs/lzy/swin_final
./tools/dist_train.sh mmdepth/configs/conv_swin/convswin_fusion_cos20x.py  8 --work-dir nfs/lzy/convswin_fusion