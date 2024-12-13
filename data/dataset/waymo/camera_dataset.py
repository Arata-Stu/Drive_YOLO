import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from typing import Callable

class WaymoCameraDataset(Dataset):
    def __init__(self, sequence_dir, camera_name, mode: str = "train", transform: Callable = None):
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
        try:
            annotation = self.annotations[idx]
        except IndexError:
            raise ValueError(f"Index {idx} is out of range for annotations of length {len(self.annotations)}.")

        image_path = annotation.get("image_path")
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path not found or invalid: {image_path}")

        # 画像をロード
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at path: {image_path}")
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Error converting image to RGB format at {image_path}: {e}")

        try:
            # 画像をTensor形式に変換
            image = torch.tensor(image.copy()).permute(2, 0, 1).float()
        except Exception as e:
            raise ValueError(f"Error converting image to tensor: {e}")

        # ラベル情報を処理
        bboxes = annotation.get("bboxes", [])
        if not isinstance(bboxes, list):
            raise ValueError(f"Invalid bounding box data for index {idx}: {bboxes}")

        try:
            if len(bboxes) == 0:
                # バウンディングボックスがない場合にゼロ埋めのテンソルを生成
                labels = torch.empty((0, 5), dtype=torch.float32)
            else:
                # バウンディングボックスの形式を cls, cx, cy, w, h に変換
                labels = torch.tensor(
                    [[
                        bbox["class"],
                        bbox["center_x"],
                        bbox["center_y"],
                        bbox["length"],
                        bbox["width"],
                    ] for bbox in bboxes],
                    dtype=torch.float32
                )
        except Exception as e:
            raise ValueError(f"Error processing bounding boxes for index {idx}: {e}")


        # ユニークIDを生成
        sequence_id = os.path.basename(self.sequence_dir)
        try:
            sequence_id_int = int(sequence_id)  # シーケンスIDを整数に変換
        except ValueError as e:
            raise ValueError(f"Invalid sequence ID: {sequence_id}. It must be an integer.")

        # print(sequence_id_int)
        unique_id = int(sequence_id_int * 1_000_000 + idx) // 100000000


        outputs = {
            "image": image,
            "labels": labels,
            "camera_name": self.camera_name,
            "frame_id": annotation["frame_id"],
            "unique_id": unique_id,
        }

        # トランスフォームを適用
        if self.transform is not None:
            try:
                
                outputs = self.transform(outputs)
                # print("After transform:", outputs.get("image", None)..shape)
            except Exception as e:
                raise ValueError(f"Error applying transform to data: {e}")

        return outputs
