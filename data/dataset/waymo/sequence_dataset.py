import os
from torch.utils.data import Dataset, ConcatDataset
from typing import Callable

from .camera_dataset import WaymoCameraDataset

class WaymoSequenceDataset(Dataset):
    def __init__(self, sequence_dir, mode: str = "train", transform: Callable = None):
        """
        シーケンス単位のデータセットを初期化します。

        Args:
            sequence_dir (str): シーケンスディレクトリのパス。
            transform (Callable): 画像変換処理。
        """
        self.sequence_dir = sequence_dir
        self.transform = transform

        # 使用するカメラのリスト
        self.cameras = ["FRONT", "FRONT_LEFT", "SIDE_LEFT", "FRONT_RIGHT", "SIDE_RIGHT"]

        # カメラ単位のデータセットを構築
        self.datasets = []
        for camera in self.cameras:
            camera_dir = os.path.join(sequence_dir, "annotations", camera)
            if os.path.exists(camera_dir):
                self.datasets.append(WaymoCameraDataset(sequence_dir, camera, mode, transform))

        # カメラ単位のデータセットを統合
        self.dataset = ConcatDataset(self.datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]