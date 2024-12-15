import torch 
import torch.nn.functional as F
from typing import Tuple

class ImagePad:
    def __init__(self, target_size: Tuple[int, int], mode: str = 'constant', padding_value: float = 0):
        """
        Args:
            target_size: (height, width) - target size
            mode: 'constant', 'reflect', 'replicate', 'circular' 
            padding_value: constant padding 用の値 ('constant' )
        """
        self.target_size = target_size
        self.mode = mode
        self.padding_value = padding_value

    def __call__(self, inputs):
        """
        inputs: dict
            "images": Tensor [C, H, W] 
        """

        img = inputs["images"]  # 入力テンソル

        if img is None:
            raise ValueError("Input dictionary must contain 'images' key.")

        if not isinstance(img, torch.Tensor):
            raise TypeError("The 'image' must be a PyTorch Tensor.")

        _, height, width = img.shape

        target_height, target_width = self.target_size

        # 必要なパディングの計算
        pad_height = max(0, target_height - height)  # 下方向
        pad_width = max(0, target_width - width)    # 右方向

        # パディング指定 (左, 右, 上, 下)
        pad = (0, pad_width, 0, pad_height)

        # パディングの適用
        if self.mode == 'constant':
            padded_img = F.pad(img, pad, mode=self.mode, value=self.padding_value)
        else:
            padded_img = F.pad(img, pad, mode=self.mode)

        # パディング後のテンソルを更新
        inputs["images"] = padded_img
        return inputs

class LabelPad:
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
                - "labels": バウンディングボックス情報 (Tensor)


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