import os
import h5py
import torch
from torch.utils.data import Dataset
from typing import Callable, List


class WaymoSequenceDataset(Dataset):
    def __init__(self, h5_path: str, transform: Callable = None, cameras: List[str] = None):
        """
        Waymoシーケンスデータセットクラス（1つのHDF5ファイルを管理）

        Args:
            h5_path (str): HDF5ファイルのパス。
            transform (Callable, optional): 画像に適用する前処理。
            cameras (list, optional): 使用するカメラリスト（例: ["FRONT", "FRONT_LEFT"]）。
        """
        self.h5_path = h5_path
        self.transform = transform
        self.data_indices = []

        # HDF5ファイルを読み込み、フレームとカメラの情報を収集
        with h5py.File(h5_path, "r") as h5_file:
            frames = list(h5_file.keys())
            if cameras is None:
                cameras = list(h5_file[frames[0]].keys())  # 全カメラを自動検出
            self.data_indices.extend([(frame, camera) for frame in frames for camera in cameras])

    def __len__(self):
        """データセットのサイズを返します。"""
        return len(self.data_indices)

    def __getitem__(self, idx):
        """
        指定されたインデックスのデータを取得します。

        Args:
            idx (int): データセット内のインデックス。

        Returns:
            dict: 指定カメラの画像とバウンディングボックスデータ。
        """
        frame, camera = self.data_indices[idx]

        with h5py.File(self.h5_path, "r") as h5_file:
            camera_data = h5_file[frame][camera]
            image = camera_data["image"][:]
            bboxes = camera_data["bboxes"][:]

        # 画像をTensorに変換し、前処理を適用
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # HWC -> CHW
        if self.transform:
            image = self.transform(image)

        # バウンディングボックスをTensorに変換
        bboxes = torch.from_numpy(bboxes).float()

        return {
            "image": image,
            "bboxes": bboxes,
            "camera": camera,
            "frame_id": frame,
        }
