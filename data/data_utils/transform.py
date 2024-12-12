import cv2
import numpy as np
import torch

class ResizePaddingTransform:
    def __init__(self, target_size=(640, 640)):
        """
        ターゲットサイズを指定して、リサイズとパディングを行うTransformを初期化。

        Args:
            target_size (tuple): リサイズ後のターゲットサイズ (width, height)。
        """
        self.target_size = target_size

    def __call__(self, sample):
        """
        データローダーのサンプルにリサイズとパディングを適用。

        Args:
            sample (dict): データローダーから受け取るサンプル。
                - "image": 画像データ (numpy.ndarray)。
                - "labels": バウンディングボックス情報 (torch.Tensor)。
                - その他: 他の情報をそのまま保持。

        Returns:
            dict: 変換後のサンプル。
        """
        image = sample["image"]
        labels = sample["labels"]
        target_width, target_height = self.target_size
        original_height, original_width = image.shape[:2]

        # アスペクト比を維持するスケール比率を計算
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        scale_ratio = min(width_ratio, height_ratio)

        # 新しいサイズを計算
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

        # 画像をリサイズ
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # パディングの計算 (右下にパディング)
        pad_top = 0
        pad_left = 0
        pad_bottom = target_height - new_height
        pad_right = target_width - new_width

        # パディングを追加して正方形にする
        padded_image = cv2.copyMakeBorder(
            resized_image,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # 黒でパディング
        )

        # バウンディングボックスの変換
        if labels is not None:
            # [center_x, center_y, width, height, class] の形式
            labels = labels.clone()
            labels[:, 0] = labels[:, 0] * scale_ratio + pad_left  # center_x
            labels[:, 1] = labels[:, 1] * scale_ratio + pad_top   # center_y
            labels[:, 2] = labels[:, 2] * scale_ratio            # width
            labels[:, 3] = labels[:, 3] * scale_ratio            # height

        # サンプルを更新して返す
        sample["image"] = padded_image
        sample["labels"] = labels
        return sample
