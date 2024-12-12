from omegaconf import DictConfig
from .dataset import WaymoDataset
from typing import Callable


def build_waymo_dataset(dataset_config: DictConfig, mode: str = 'train', transform: Callable = None):
    
    assert mode in ['train', 'val', 'test'], f"mode must be 'train', 'val', or 'test', but got {mode}."
    return WaymoDataset(root_dir=dataset_config.data_dir,
                        mode=mode,
                        transform=transform)