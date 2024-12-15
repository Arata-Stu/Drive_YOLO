import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from typing import Callable


class WaymoSequenceDataset(Dataset):
    def __init__(self, h5_path: str, transform: Callable = None):
        """
        Waymoシーケンスデータセットクラス。

        Args:
            h5_path (str): HDF5ファイルのパス。
            transform (Callable, optional): 画像に適用する前処理。
        """
        self.h5_path = h5_path
        self.transform = transform
        self.data_indices = []

        # フレームとカメラデータを収集
        with h5py.File(h5_path, "r") as h5_file:
            self.frames = list(h5_file.keys())
            for frame in self.frames:
                self.data_indices.append(frame)

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        frame_id = self.data_indices[idx]

        with h5py.File(self.h5_path, "r") as h5_file:
            frame_data = h5_file[frame_id]

            # 画像データ
            image = frame_data["image"][:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            # バウンディングボックスデータ
            bboxes = frame_data["bboxes"][:]
            if bboxes.size == 0:  # バウンディングボックスがない場合
                bboxes = np.zeros((0, 5), dtype=np.float32)
            else:
                # 形式変換: cx, cy, w, h, cls -> cls, cx, cy, w, h
                bboxes = np.hstack((bboxes[:, -1:], bboxes[:, :-1])).astype(np.float32)

            bboxes = torch.from_numpy(bboxes).float()

            outputs = {
                "images": image,
                "labels": bboxes,
                "frame_id": frame_id
            }

            # 前処理
            if self.transform:
                outputs = self.transform(outputs)

        return outputs
