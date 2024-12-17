from omegaconf import DictConfig
from torchvision import transforms

from ..data_utils.transform import ImagePad, LabelPad, RandomRotate, Flip, BoxFormatTransform, LabelFilter, RandomZoom
from .waymo.build_dataset import build_waymo_dataset
from .waymo.data_info import ORIG_CLASS, MY_CLASS

def build_dataset(dataset_config: DictConfig, mode: str = 'train'):

    name = dataset_config.name
    use_time = dataset_config.transform_timer
    target_size = (640, 640)

    img_pad = ImagePad(target_size=target_size, mode='constant' ,padding_value=114, timing=use_time)
    label_pad = LabelPad(max_num_labels=150, timing=use_time)
    label_filter = LabelFilter(orig_class=ORIG_CLASS, my_class=MY_CLASS, timing=use_time)
    rotate = RandomRotate(min_angle=-6, max_angle=6, timing=use_time)
    horizontal_flip = Flip(vertical=False, horizontal=True, timing=use_time)
    zoom = RandomZoom(prob_weight=(8, 2), in_scale=(1, 1.5), out_scale=(1, 1.2), center_margin_ratio=0.2, timing=use_time)
    yolo_box_transform = BoxFormatTransform(mode=mode, timing=use_time)

    if mode == 'train':
        _transform = [img_pad, label_filter, horizontal_flip, rotate, zoom, yolo_box_transform, label_pad]
    elif mode == 'val' or mode == 'test':
        _transform = [img_pad, label_filter, yolo_box_transform, label_pad]
    else:
        NotImplementedError
        

    transform = transforms.Compose(_transform)

    if 'waymo' in name:
        dataset = build_waymo_dataset(dataset_config=dataset_config, mode=mode, transform=transform)
    else:
        print(f'{name=} not available')
        raise NotImplementedError
    
    return dataset