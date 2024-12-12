import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from typing import Callable

class WaymoSequenceDataset(Dataset):
    def __init__(self, sequence_dir, mode: str="train", transform: Callable=None):
        """
        1つのシーケンスディレクトリからデータセットを初期化します。

        Args:
            sequence_dir (str): シーケンスディレクトリのパス。
        """
        self.sequence_dir = sequence_dir
        self.annotation_path = os.path.join(sequence_dir, "annotations.json")
        self.transform = transform

        # アノテーションファイルをロード
        with open(self.annotation_path, "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        """
        シーケンス内のフレーム数を返します。
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        フレームの画像とラベルを取得します。

        Args:
            idx (int): インデックス。

        Returns:
            dict: フレームの画像データとラベル。
        """
        annotation = self.annotations[idx]
        image_path = annotation["image_path"]
        bboxes = annotation["bboxes"]

        # OpenCVで画像をロード (HWC形式, BGR -> RGBに変換)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 画像をTensor形式に変換 (CHW形式)
        image = torch.tensor(image).permute(2, 0, 1).float() 

        # バウンディングボックスとクラス情報をTensor形式に変換
        labels = torch.tensor(
            [[bbox["center_x"], bbox["center_y"], bbox["length"], bbox["width"], bbox["class"]] for bbox in bboxes]
        )

        outputs = {
            "image": image,
            "labels": labels,
            "camera_name": annotation["camera_name"],
            "frame_id": annotation["frame_id"],
        }

        if self.transform is not None:
            outputs = self.transform(outputs)

        return outputs