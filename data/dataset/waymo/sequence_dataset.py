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

        assert os.path.exists(sequence_dir), f"Sequence directory not found: {sequence_dir}"
        self.sequence_dir = sequence_dir
        self.sequence_id = os.path.basename(sequence_dir)  # シーケンス番号を取得
        self.annotation_path = os.path.join(sequence_dir, "annotations.json")
        self.transform = transform
        self.mode = mode

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
        if self.mode == 'train':
            labels = torch.tensor(
                [[bbox["class"], bbox["center_x"], bbox["center_y"], bbox["length"], bbox["width"]] for bbox in bboxes]
            )
        elif self.mode in ['test', 'val']:
            labels = torch.tensor(
                [[
                    bbox["center_x"] - bbox["length"] / 2,  # x = center_x - (width / 2)
                    bbox["center_y"] - bbox["width"] / 2,   # y = center_y - (height / 2)
                    bbox["length"],                        # width
                    bbox["width"],                         # height
                    bbox["class"],                         # class
                ] for bbox in bboxes]
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        sequence_id_int = int(self.sequence_id)  # シーケンスIDを整数に変換
        unique_id = sequence_id_int * 10**6 + idx  # フレームインデックスを加算


        outputs = {
            "image": image,
            "labels": labels,
            "camera_name": annotation["camera_name"],
            "frame_id": annotation["frame_id"],
            "unique_id": unique_id,
        }

        if self.transform is not None:
            outputs = self.transform(outputs)

        return outputs