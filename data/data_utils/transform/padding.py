import time
import numpy as np
from typing import Tuple

class ImagePad:
    def __init__(self, target_size: Tuple[int, int], mode: str = 'constant', padding_value: float = 0, timing: bool = False):
        """
        Args:
            target_size: (height, width) - target size
            mode: 'constant', 'reflect', 'replicate', 'circular'
            padding_value: constant padding 用の値 ('constant' の場合)
            timing: 処理時間を計測するかどうか
        """
        self.target_size = target_size
        self.mode = mode
        self.padding_value = padding_value
        self.timing = timing

    def __call__(self, inputs):
        if self.timing:
            start_time = time.time()

        img = inputs["images"]

        if img is None:
            raise ValueError("Input dictionary must contain 'images' key.")

        if not isinstance(img, np.ndarray):
            raise TypeError("The 'image' must be a NumPy ndarray.")

        _, height, width = img.shape
        target_height, target_width = self.target_size

        # 必要なパディングを計算
        pad_height = max(0, target_height - height)  # 下方向
        pad_width = max(0, target_width - width)    # 右方向

        # パディング指定
        pad = ((0, 0),  # チャンネル方向はそのまま
               (0, pad_height),  # 高さ方向
               (0, pad_width))   # 幅方向

        # np.padのモード変換
        mode = self.mode
        if mode == "replicate":
            mode = "edge"
        elif mode == "circular":
            mode = "wrap"

        # パディング適用
        if mode == "constant":
            padded_img = np.pad(img, pad, mode=mode, constant_values=self.padding_value)
        else:
            padded_img = np.pad(img, pad, mode=mode)

        # パディング後の画像を更新
        inputs["images"] = padded_img

        if self.timing:
            elapsed_time = time.time() - start_time
            print(f"ImagePad processing time: {elapsed_time:.6f} seconds")

        return inputs


class LabelPad:
    def __init__(self, max_num_labels=50, timing: bool = False):
        """
        バウンディングボックスの数を最大数にパディングするTransformを初期化。

        Args:
            max_num_labels (int): パディング後のバウンディングボックスの最大数。
            timing (bool): 処理時間を計測するかどうか
        """
        self.max_num_labels = max_num_labels
        self.timing = timing

    def __call__(self, inputs):
        if self.timing:
            start_time = time.time()

        labels = inputs["labels"]

        if not isinstance(labels, np.ndarray):
            raise TypeError("The 'labels' must be a NumPy ndarray.")

        # ラベルが空の場合
        if labels.size == 0:
            padded_labels = np.zeros((self.max_num_labels, 5), dtype=np.float32)
        else:
            num_labels = labels.shape[0]
            # パディングするためのテンソルを作成
            padded_labels = np.zeros((self.max_num_labels, labels.shape[1]), dtype=labels.dtype)
            padded_labels[:num_labels] = labels

        # パディング後のラベルをサンプルにセット
        inputs["labels"] = padded_labels

        if self.timing:
            elapsed_time = time.time() - start_time
            print(f"LabelPad processing time: {elapsed_time:.6f} seconds")

        return inputs