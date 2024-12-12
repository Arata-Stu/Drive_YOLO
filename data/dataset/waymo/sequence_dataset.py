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
        self.sequence_dirs = []
        self.annotations = []
        self.transform = transform

        # モードに応じたディレクトリを設定
        mode_dir = os.path.join(dataset_dir, mode)
        if not os.path.exists(mode_dir):
            raise FileNotFoundError(f"Directory for mode '{mode}' not found: {mode_dir}")

        # シーケンスごとにアノテーションを収集
        for sequence_dir in os.listdir(mode_dir):
            sequence_path = os.path.join(mode_dir, sequence_dir)
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
