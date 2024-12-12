import cv2
import numpy as np
import torch

class ResizePaddingTransform:
    def __init__(self, target_size=(640, 640), mode="train"):
        """
        ターゲットサイズを指定して、リサイズとパディングを行うTransformを初期化。

        Args:
            target_size (tuple): リサイズ後のターゲットサイズ (width, height)。
            mode (str): データのモード ("train", "val", "test")。
        """
        self.target_size = target_size
        assert target_size[0] == target_size[1], "ターゲットサイズは正方形である必要があります。"
        self.mode = mode

    def __call__(self, sample):
        """
        データローダーのサンプルにリサイズとパディングを適用。

        Args:
            sample (dict): データローダーから受け取るサンプル。
                - "image": 画像データ (torch.Tensor)。
                - "labels": バウンディングボックス情報 (torch.Tensor)。
                - その他: 他の情報をそのまま保持。

        Returns:
            dict: 変換後のサンプル。
        """
        # 画像をCHW -> HWCに変換
        image = sample["image"].permute(1, 2, 0).numpy()
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
            labels = labels.clone()
            if self.mode == "train":
                # [class, center_x, center_y, width, height]
                labels[:, 1] = labels[:, 1] * scale_ratio + pad_left  # center_x
                labels[:, 2] = labels[:, 2] * scale_ratio + pad_top   # center_y
                labels[:, 3] = labels[:, 3] * scale_ratio            # width
                labels[:, 4] = labels[:, 4] * scale_ratio            # height
            elif self.mode in ["val", "test"]:
                # [x, y, width, height, class]
                labels[:, 0] = labels[:, 0] * scale_ratio + pad_left  # x
                labels[:, 1] = labels[:, 1] * scale_ratio + pad_top   # y
                labels[:, 2] = labels[:, 2] * scale_ratio            # width
                labels[:, 3] = labels[:, 3] * scale_ratio            # height

        # 画像をHWC -> CHWに変換して戻す
        sample["image"] = torch.tensor(padded_image).permute(2, 0, 1).float()
        sample["labels"] = labels
        return sample


class PadLabelTransform:
    def __init__(self, max_num_labels=50):
        """
        バウンディングボックスの数を最大数にパディングするTransformを初期化。

        Args:
            max_num_labels (int): パディング後のバウンディングボックスの最大数。
        """
        self.max_num_labels = max_num_labels

    def __call__(self, sample):
        """
        データローダーのサンプルにバウンディングボックスのパディングを適用。

        Args:
            sample (dict): 入力サンプル。
                - "image": 画像データ (Tensor)
                - "labels": バウンディングボックス情報 (Tensor)
                - 他の情報: camera_name, frame_idなど

        Returns:
            dict: パディングされたサンプル。
        """
        labels = sample["labels"]
        num_labels = labels.size(0)

        # パディングするためのテンソルを作成
        padded_labels = torch.zeros((self.max_num_labels, labels.size(1)), dtype=labels.dtype)
        padded_labels[:num_labels] = labels

        # パディング後のラベルをサンプルにセット
        sample["labels"] = padded_labels

        return sample

