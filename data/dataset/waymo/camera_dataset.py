import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from typing import Callable

class WaymoCameraDataset(Dataset):
    def __init__(self, sequence_dir, camera_name, mode: str = "train" ,transform: Callable = None):
        """
        カメラ位置単位のデータセットを初期化します。

        Args:
            sequence_dir (str): シーケンスディレクトリのパス。
            camera_name (str): カメラの位置 ('FRONT', 'FRONT_LEFT', etc.)。
            transform (Callable): 画像変換処理。
        """
        self.sequence_dir = sequence_dir
        self.camera_name = camera_name
        self.mode = mode
        self.transform = transform

        # アノテーションファイルをロード
        annotation_path = os.path.join(sequence_dir, "annotations", camera_name, "annotations.json")
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found for camera {camera_name}: {annotation_path}")

        with open(annotation_path, "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = annotation["image_path"]
        bboxes = annotation["bboxes"]

        # 画像をロード
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 画像をTensor形式に変換
        image = torch.tensor(image).permute(2, 0, 1).float()

        # バウンディングボックスをTensor形式に変換
        if self.mode == "train":
            ## バウンディングボックスの形式をcls, cx, cy, w, hに変換
            labels = torch.tensor(
                [[
                    bbox["class"],
                    bbox["center_x"],
                    bbox["center_y"],
                    bbox["length"],
                    bbox["width"],
                ] for bbox in bboxes]
            )
        elif self.mode == "test" or self.mode == "val":
            ## バウンディングボックスの形式をx, y, w, hに変換
            labels = torch.tensor(
                [[
                    bbox["center_x"] - bbox["length"] / 2,
                    bbox["center_y"] - bbox["width"] / 2,
                    bbox["length"],
                    bbox["width"],
                    bbox["class"]
                ] for bbox in bboxes]
            )

        # ユニークIDを生成
        sequence_id = os.path.basename(self.sequence_dir)
        sequence_id_int = int(sequence_id)  # シーケンスIDを整数に変換
        unique_id = sequence_id_int * 10**6 + idx  # フレームインデックスを加算

        outputs = {
            "image": image,
            "labels": labels,
            "camera_name": self.camera_name,
            "frame_id": annotation["frame_id"],
            "unique_id": unique_id,
        }

        if self.transform is not None:
            outputs = self.transform(outputs)

        return outputs