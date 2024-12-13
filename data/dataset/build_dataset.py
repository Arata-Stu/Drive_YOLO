from omegaconf import DictConfig
from torchvision import transforms

from ..data_utils.transform import ResizeTransform, PadLabelTransform, FlipTransform, YOLOXTransform, MyFilter
from .waymo.build_dataset import build_waymo_dataset
from .waymo.data_info import ORIG_CLASS, MY_CLASS

def build_dataset(dataset_config: DictConfig, mode: str = 'train'):

    name = dataset_config.name
    target_size = (640, 640)

    resize = ResizeTransform(input_size=target_size)
    label_filter = MyFilter(orig_class=ORIG_CLASS, my_class=MY_CLASS)
    label_pad = PadLabelTransform(max_num_labels=100)
    flip = FlipTransform(flip_horizontal=True, flip_vertical=False)
    yolo_transform = YOLOXTransform(mode=mode)
    _transform = [resize, label_filter, flip, label_pad, yolo_transform]

    transform = transforms.Compose(_transform)

    if 'waymo' in name:
        dataset = build_waymo_dataset(dataset_config=dataset_config, mode=mode, transform=transform)
    else:
        print(f'{name=} not available')
        raise NotImplementedError
    
    return dataset