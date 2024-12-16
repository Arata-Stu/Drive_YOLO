import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from typing import Callable
import os

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

        # 使用するデバイスを設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # HDF5ファイル名からユニークな数字部分を取得
        self.file_id = int(os.path.splitext(os.path.basename(h5_path))[0])

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
            image = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device)

            # バウンディングボックスデータ
            bboxes = frame_data["bboxes"][:]
            if bboxes.size == 0:  # バウンディングボックスがない場合
                bboxes = np.zeros((0, 5), dtype=np.float32)
            else:
                # 形式変換: cx, cy, w, h, cls -> cls, cx, cy, w, h
                bboxes = np.hstack((bboxes[:, -1:], bboxes[:, :-1])).astype(np.float32)

            bboxes = torch.from_numpy(bboxes).float().to(self.device)

            # frame_idの数値部分を抽出 (_ で区切って数字部分を取得)
            frame_number = int(frame_id.split("_")[-1])  # 'frame_001' -> 1
            unique_id = self.file_id + frame_number

            outputs = {
                "images": image,
                "labels": bboxes,
                "frame_id": frame_id,
                "unique_id": unique_id // 10e6
            }

            # 前処理
            if self.transform:
                outputs = self.transform(outputs)

        return outputs
