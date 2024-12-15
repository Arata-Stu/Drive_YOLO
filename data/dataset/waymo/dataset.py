import os
import h5py
import torch
from torch.utils.data import ConcatDataset
from typing import Callable, List

from .sequence_dataset import WaymoSequenceDataset


class WaymoConcatDataset(ConcatDataset):
    def __init__(self, root_dir: str, mode: str, transform: Callable = None, cameras: List[str] = None):
        """
        WaymoConcatDatasetクラス（複数のWaymoSequenceDatasetを結合）

        Args:
            root_dir (str): データセットのルートディレクトリ。
            mode (str): データのモード ("train", "val", "test")。
            transform (Callable, optional): 画像に適用する前処理。
            cameras (list, optional): 使用するカメラリスト。
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.cameras = cameras

        # モードに応じたディレクトリを探索
        mode_dir = os.path.join(root_dir, mode)
        if not os.path.exists(mode_dir):
            raise FileNotFoundError(f"ディレクトリ {mode_dir} が存在しません。")

        # 各HDF5ファイルのデータセットを作成
        sequence_datasets = []
        for h5_file in sorted(os.listdir(mode_dir)):
            if h5_file.endswith(".h5"):
                h5_path = os.path.join(mode_dir, h5_file)
                sequence_datasets.append(WaymoSequenceDataset(h5_path=h5_path, transform=transform))

        # ConcatDatasetを初期化
        super().__init__(sequence_datasets)