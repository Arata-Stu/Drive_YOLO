import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from typing import Callable

class WaymoSequenceDataset(Dataset):
    def __init__(self, dataset_dir, mode: str="train", transform: Callable=None):
        """
        train, test, validationディレクトリ構造を持つデータセットを初期化します。

        Args:
            dataset_dir (str): データセットディレクトリのパス。
            mode (str): 使用するデータセットモード ('train', 'test', 'val')。
            transform (Callable): 画像変換処理。
        """
        assert os.path.exists(dataset_dir), f"Dataset directory not found: {dataset_dir}"
        assert mode in ['train', 'test', 'val'], f"Invalid mode: {mode}"

        self.dataset_dir = dataset_dir
        self.mode = mode
        self.annotations = []
        self.transform = transform

    
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Directory for mode '{mode}' not found: {self.dataset_dir}")

        # シーケンスごとにアノテーションを収集
        for sequence_dir in os.listdir(self.dataset_dir):
            sequence_path = os.path.join(self.dataset_dir, sequence_dir)
            if not os.path.isdir(sequence_path):
                continue
            annotation_path = os.path.join(sequence_path, "annotations.json")
            if not os.path.exists(annotation_path):
                print(f"Warning: Annotation file not found for sequence {sequence_dir}. Skipping.")
                continue

            with open(annotation_path, "r") as f:
                sequence_annotations = json.load(f)
                for annotation in sequence_annotations:
                    annotation["sequence_dir"] = sequence_path  # シーケンスディレクトリ情報を追加
                self.annotations.extend(sequence_annotations)

    def __len__(self):
        """
        データセット内の総フレーム数を返します。
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
            [[
                bbox["center_x"] - bbox["length"] / 2,  # x = center_x - (width / 2)
                bbox["center_y"] - bbox["width"] / 2,   # y = center_y - (height / 2)
                bbox["length"],                        # width
                bbox["width"],                         # height
                bbox["class"],                         # class
            ] for bbox in bboxes]
        )

        # ユニークIDを生成
        sequence_dir = annotation["sequence_dir"]
        sequence_id = os.path.basename(sequence_dir)
        sequence_id_int = int(sequence_id)  # シーケンスIDを整数に変換
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