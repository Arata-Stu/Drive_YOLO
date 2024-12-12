from omegaconf import DictConfig
from torchvision import transforms

from data.data_utils.transform import ResizePaddingTransform
from waymo.build_dataset import build_waymo_dataset

def build_dataset(dataset_config: DictConfig, mode: str = 'train'):

    name = dataset_config.name
    target_size = (640, 640)
    transform = transforms.Compose([ResizePaddingTransform(target_size=target_size)])

    if 'waymo' in name:
        dataset = build_waymo_dataset(dataset_config=dataset_config, mode=mode, transform=transform)
    else:
        print(f'{name=} not available')
        raise NotImplementedError
    
    return dataset