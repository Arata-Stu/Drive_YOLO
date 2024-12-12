import os
from typing import Tuple

import math
from omegaconf import DictConfig, open_dict


def dynamically_modify_train_config(config: DictConfig):
    with open_dict(config):
        
        dataset_cfg = config.dataset
        dataset_name = dataset_cfg.name
        assert dataset_name in {'waymo'}

        mdl_cfg = config.model

        #32の倍数になるようにheight widthを調整
        partition_split_32 = mdl_cfg.backbone.partition_split_32
        assert partition_split_32 in (1, 2, 4)
        multiple_of = 32 * partition_split_32

        img_size_map = {
            'waymo': (640, 640)
        }
        mdl_hw = img_size_map[dataset_name]
        dataset_cfg.target_size = mdl_hw
        
        ##データセットのクラス数
        class_len_map = {
            'waymo': 3,
        }
        mdl_cfg.head.num_classes = class_len_map[dataset_name]
        print('num_class', mdl_cfg.head.num_classes)

