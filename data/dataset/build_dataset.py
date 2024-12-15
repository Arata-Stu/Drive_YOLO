from omegaconf import DictConfig
from torchvision import transforms

from ..data_utils.transform import ImagePad, LabelPad, RandomRotate, Flip
from .waymo.build_dataset import build_waymo_dataset
from .waymo.data_info import ORIG_CLASS, MY_CLASS

def build_dataset(dataset_config: DictConfig, mode: str = 'train'):

    name = dataset_config.name
    target_size = (640, 640)

    img_pad = ImagePad(target_size=target_size, mode='constant' ,padding_value=114)
    label_pad = LabelPad(max_num_labels=150)
    rotate = RandomRotate(min_angle=-6, max_angle=6)
    horizontal_flip = Flip(vertical=False, horizontal=True)
    yolo_transform = YOLOXTransform(mode=mode)
    _transform = [img_pad, horizontal_flip, rotate, label_pad, yolo_transform]

    transform = transforms.Compose(_transform)

    if 'waymo' in name:
        dataset = build_waymo_dataset(dataset_config=dataset_config, mode=mode, transform=transform)
    else:
        print(f'{name=} not available')
        raise NotImplementedError
    
    return dataset