from omegaconf import DictConfig
from torchvision import transforms

from ..data_utils.transform import ResizeTransform, PadLabelTransform, FlipTransform
from .waymo.build_dataset import build_waymo_dataset

def build_dataset(dataset_config: DictConfig, mode: str = 'train'):

    name = dataset_config.name
    target_size = (640, 640)

    resize = ResizeTransform(target_size=target_size)
    label_pad = PadLabelTransform(max_num_labels=50)
    flip = FlipTransform(flip_horizontal=True, flip_vertical=False)
    _transform = [resize, flip, label_pad]

    transform = transforms.Compose(_transform)

    if 'waymo' in name:
        dataset = build_waymo_dataset(dataset_config=dataset_config, mode=mode, transform=transform)
    else:
        print(f'{name=} not available')
        raise NotImplementedError
    
    return dataset