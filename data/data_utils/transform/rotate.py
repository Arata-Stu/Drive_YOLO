import torch
from torchvision.transforms import functional as F
import math

import random

class RandomRotate:
    def __init__(self, min_angle: float = -10.0, max_angle: float = 10.0):
        """
        Args:
            min_angle (float):
                ランダムに選択する角度の最小値（度単位）。
            max_angle (float):
                ランダムに選択する角度の最大値（度単位）。
        """
        if min_angle > max_angle:
            raise ValueError("min_angle should be less than or equal to max_angle.")
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, inputs: dict) -> dict:
        """
        Args:
            inputs: dict
                - "images": Tensor of shape [C, H, W]
                - "labels": Tensor of shape [N, 5] (format: [cls, cx, cy, w, h])

        Returns:
            dict:
                ランダムな角度で回転後の画像と対応するバウンディングボックスを含む辞書。
        """
        # ランダムな角度を選択
        random_angle = random.uniform(self.min_angle, self.max_angle)

        # Rotateクラスを利用して回転処理
        rotate = Rotate(angle=random_angle)
        return rotate(inputs)


class Rotate:
    def __init__(self, angle: float = 0.0):
        """
        Args:
            angle (float):
                画像を回転させる角度（度単位）。
                正の値で反時計回り、負の値で時計回り。
        """
        self.angle = angle

    def __call__(self, inputs: dict) -> dict:
        """
        Args:
            inputs: dict
                - "images": Tensor of shape [C, H, W]
                - "labels": Tensor of shape [N, 5] (format: [cls, cx, cy, w, h])

        Returns:
            dict:
                回転後の画像と対応するバウンディングボックスを含む辞書。
        """
        image = inputs.get("images")
        labels = inputs.get("labels")

        # 入力チェック
        if image is None:
            raise ValueError("Input dictionary must contain 'images' key.")
        if not isinstance(image, torch.Tensor):
            raise TypeError("'images' must be a PyTorch Tensor.")
        if labels is None:
            raise ValueError("Input dictionary must contain 'labels' key.")
        if not isinstance(labels, torch.Tensor):
            raise TypeError("'labels' must be a PyTorch Tensor.")

        # 画像サイズを取得
        _, H, W = image.shape

        # 画像を回転（PyTorchの機能）
        rotated_image = F.rotate(image, self.angle)

        # ラベルを回転
        rotated_labels = self.rotate_bboxes(labels, self.angle, W, H)

        # 更新した辞書を返す
        inputs["images"] = rotated_image
        inputs["labels"] = rotated_labels
        return inputs

    def rotate_bboxes(self, labels: torch.Tensor, angle: float, img_w: int, img_h: int) -> torch.Tensor:
        """
        軸揃えバウンディングボックス（[cls, cx, cy, w, h]）を画像中心基準で回転させる。
        回転後も axis-aligned として再計算する。

        Args:
            labels (Tensor): 形状 [N, 5]、フォーマット [cls, cx, cy, w, h]
            angle (float): 度単位の回転角。正は反時計、負は時計回り。
            img_w (int): 画像幅
            img_h (int): 画像高さ

        Returns:
            Tensor: 回転後のラベル。形状 [N, 5]、フォーマット [cls, cx, cy, w, h]
        """
        device = labels.device
        # 出力用のテンソル
        rotated_labels = labels.clone()

        # 度 -> ラジアン変換
        theta = -math.radians(angle)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # 画像中心
        cx0 = img_w / 2.0
        cy0 = img_h / 2.0

        # [N, 5]: [cls, cx, cy, w, h]
        cls_  = labels[:, 0]
        cx    = labels[:, 1]
        cy    = labels[:, 2]
        w     = labels[:, 3]
        h     = labels[:, 4]

        # バウンディングボックスの四隅を計算（軸揃え想定）
        # 左上, 右上, 左下, 右下の4点
        # shape: [N, 4, 2]
        corners_x = torch.stack([
            cx - w/2,  # 左
            cx + w/2,  # 右
            cx - w/2,  # 左
            cx + w/2   # 右
        ], dim=1)  # [N, 4]
        corners_y = torch.stack([
            cy - h/2,  # 上
            cy - h/2,  # 上
            cy + h/2,  # 下
            cy + h/2   # 下
        ], dim=1)  # [N, 4]

        # 回転行列を適用（画像中心(cx0, cy0)を基準）
        # (x', y') = (cosθ*(x-cx0) - sinθ*(y-cy0) + cx0, sinθ*(x-cx0) + cosθ*(y-cy0) + cy0)
        corners_x_rot = cos_t * (corners_x - cx0) - sin_t * (corners_y - cy0) + cx0
        corners_y_rot = sin_t * (corners_x - cx0) + cos_t * (corners_y - cy0) + cy0

        # 回転後、軸揃えBBに変換するため min/max を取得
        x_min, _ = torch.min(corners_x_rot, dim=1)
        x_max, _ = torch.max(corners_x_rot, dim=1)
        y_min, _ = torch.min(corners_y_rot, dim=1)
        y_max, _ = torch.max(corners_y_rot, dim=1)

        # 新しい中心座標と幅高さ
        new_cx = (x_min + x_max) / 2.0
        new_cy = (y_min + y_max) / 2.0
        new_w = x_max - x_min
        new_h = y_max - y_min

        # 回転後のラベルを更新
        rotated_labels[:, 0] = cls_
        rotated_labels[:, 1] = new_cx
        rotated_labels[:, 2] = new_cy
        rotated_labels[:, 3] = new_w
        rotated_labels[:, 4] = new_h

        # 画像範囲外に出た場合のクリッピング等はここで必要に応じて処理
        # 例: 0 ~ img_w, 0 ~ img_h に clamp
        rotated_labels[:, 1] = rotated_labels[:, 1].clamp(min=0, max=img_w)
        rotated_labels[:, 2] = rotated_labels[:, 2].clamp(min=0, max=img_h)
        rotated_labels[:, 3] = rotated_labels[:, 3].clamp(min=0, max=img_w)
        rotated_labels[:, 4] = rotated_labels[:, 4].clamp(min=0, max=img_h)

        return rotated_labels
