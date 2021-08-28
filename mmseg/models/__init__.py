# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, HEADS, LOSSES, SEGMENTORS, build_backbone,
                      build_head, build_loss, build_segmentor)
from .decode_heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403

from .builder import (DEPTHER, build_depther, DEPTHHEAD, build_depth_head, DEPTHBACKBONE, build_depth_backbone, DEPTHLOSS, build_depth_loss, DEPTHNECK, build_depth_neck)
# from mmdepth.models.decode_heads import *
from mmdepth.models.depther import *
from mmdepth.models.decode_heads import *
from mmdepth.models.backbones import *
from mmdepth.models.losses import *
from mmdepth.models.necks import *

__all__ = [
    'BACKBONES', 'HEADS', 'LOSSES', 'SEGMENTORS', 'build_backbone',
    'build_head', 'build_loss', 'build_segmentor',

    "DEPTHER", "build_depther", 'DEPTHHEAD', 'build_depth_head', "DEPTHBACKBONE", "build_depth_backbone", 'DEPTHLOSS', 'build_depth_loss', 
    "DEPTHNECK", "build_depth_neck"
]
