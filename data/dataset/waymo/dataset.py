import os
from torch.utils.data import ConcatDataset
from typing import Callable

from .sequence_dataset import WaymoSequenceDataset

class WaymoDataset:
    def __init__(self, root_dir, mode: str ='train', transform: Callable =None):
        """
        全シーケンスを統合したデータセットを初期化します。

        Args:
            root_dir (str): 前処理済みデータのルートディレクトリ。
        """

        assert root_dir is not None, "root_dir must be specified."
        self.root_dir = root_dir
        self.sequence_dirs = [
            os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        ]

        # シーケンスごとのデータセットを作成
        self.datasets = [WaymoSequenceDataset(seq_dir, mode, transform) for seq_dir in self.sequence_dirs]

        # ConcatDatasetで統合
        self.dataset = ConcatDataset(self.datasets)

    def __len__(self):
        """
        統合データセットの全サンプル数を返します。
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        統合データセットからサンプルを取得します。

        Args:
            idx (int): インデックス。

        Returns:
            dict: 統合データセットのサンプル。
        """
        return self.dataset[idx]