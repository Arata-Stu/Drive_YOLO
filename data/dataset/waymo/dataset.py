import os
from torch.utils.data import ConcatDataset
from typing import Callable

from .sequence_dataset import WaymoSequenceDataset

class WaymoDataset:
    def __init__(self, root_dir, mode: str = 'train', transform: Callable = None):
        """
        全シーケンスを統合したデータセットを初期化します。

        Args:
            root_dir (str): 前処理済みデータのルートディレクトリ。
            mode (str): 使用するデータセットモード ('train', 'test', 'val')。
            transform (Callable): 画像変換処理。
        """
        assert root_dir is not None, "root_dir must be specified."
        assert mode in ['train', 'test', 'val'], f"Invalid mode: {mode}"

        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        mode_dir = os.path.join(root_dir, mode)
        if not os.path.exists(mode_dir):
            raise FileNotFoundError(f"Directory for mode '{mode}' not found: {mode_dir}")

        # シーケンス単位のデータセットを構築
        self.datasets = []
        for sequence_id in os.listdir(mode_dir):
            sequence_dir = os.path.join(mode_dir, sequence_id)
            if os.path.isdir(sequence_dir):
                self.datasets.append(WaymoSequenceDataset(sequence_dir, mode, transform))

        # シーケンス単位のデータセットを統合
        self.dataset = ConcatDataset(self.datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]