import cv2
import numpy as np
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
                - "images": ndarray of shape [C, H, W]
                - "labels": ndarray of shape [N, 5] (format: [cls, cx, cy, w, h])

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
                - "images": ndarray of shape [C, H, W]
                - "labels": ndarray of shape [N, 5] (format: [cls, cx, cy, w, h])

        Returns:
            dict:
                回転後の画像と対応するバウンディングボックスを含む辞書。
        """
        image = inputs.get("images")
        labels = inputs.get("labels")

        # 入力チェック
        if image is None:
            raise ValueError("Input dictionary must contain 'images' key.")
        if not isinstance(image, np.ndarray):
            raise TypeError("'images' must be a NumPy ndarray.")
        if labels is None:
            raise ValueError("Input dictionary must contain 'labels' key.")
        if not isinstance(labels, np.ndarray):
            raise TypeError("'labels' must be a NumPy ndarray.")

        # Convert image from [C, H, W] to [H, W, C]
        image = np.transpose(image, (1, 2, 0))

        # 画像サイズを取得
        H, W, _ = image.shape

        # OpenCVで画像を回転
        rotated_image = self.rotate_image(image, self.angle)

        # ラベルを回転
        rotated_labels = self.rotate_bboxes(labels, self.angle, W, H)

        # Convert image back to [C, H, W]
        rotated_image = np.transpose(rotated_image, (2, 0, 1))

        # 更新した辞書を返す
        inputs["images"] = rotated_image
        inputs["labels"] = rotated_labels
        return inputs

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        OpenCVを使用して画像を回転。

        Args:
            image (ndarray): 画像データ [H, W, C]
            angle (float): 回転角度（度単位）

        Returns:
            ndarray: 回転後の画像
        """
        H, W = image.shape[:2]
        # 回転行列を取得
        center = (W / 2, H / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        # 回転を適用
        rotated_image = cv2.warpAffine(image, rotation_matrix, (W, H), borderValue=(114, 114, 114))
        return rotated_image

    def rotate_bboxes(self, labels: np.ndarray, angle: float, img_w: int, img_h: int) -> np.ndarray:
        """
        軸揃えバウンディングボックス（[cls, cx, cy, w, h]）を画像中心基準で回転させる。
        回転後も axis-aligned として再計算する。

        Args:
            labels (ndarray): 形状 [N, 5]、フォーマット [cls, cx, cy, w, h]
            angle (float): 度単位の回転角。正は反時計、負は時計回り。
            img_w (int): 画像幅
            img_h (int): 画像高さ

        Returns:
            ndarray: 回転後のラベル。形状 [N, 5]、フォーマット [cls, cx, cy, w, h]
        """
        # 出力用の配列を作成
        rotated_labels = labels.copy()

        # 度 -> ラジアン変換
        theta = -math.radians(angle)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # 画像中心
        cx0 = img_w / 2.0
        cy0 = img_h / 2.0

        # バウンディングボックスの四隅を計算
        corners_x = np.stack([
            labels[:, 1] - labels[:, 3] / 2,  # 左
            labels[:, 1] + labels[:, 3] / 2,  # 右
            labels[:, 1] - labels[:, 3] / 2,  # 左
            labels[:, 1] + labels[:, 3] / 2   # 右
        ], axis=1)
        corners_y = np.stack([
            labels[:, 2] - labels[:, 4] / 2,  # 上
            labels[:, 2] - labels[:, 4] / 2,  # 上
            labels[:, 2] + labels[:, 4] / 2,  # 下
            labels[:, 2] + labels[:, 4] / 2   # 下
        ], axis=1)

        # 回転行列を適用
        corners_x_rot = cos_t * (corners_x - cx0) - sin_t * (corners_y - cy0) + cx0
        corners_y_rot = sin_t * (corners_x - cx0) + cos_t * (corners_y - cy0) + cy0

        # 回転後、軸揃えBBに変換するため min/max を取得
        x_min = np.min(corners_x_rot, axis=1)
        x_max = np.max(corners_x_rot, axis=1)
        y_min = np.min(corners_y_rot, axis=1)
        y_max = np.max(corners_y_rot, axis=1)

        # 新しい中心座標と幅高さ
        new_cx = (x_min + x_max) / 2.0
        new_cy = (y_min + y_max) / 2.0
        new_w = x_max - x_min
        new_h = y_max - y_min

        # 回転後のラベルを更新
        rotated_labels[:, 1] = np.clip(new_cx, 0, img_w)
        rotated_labels[:, 2] = np.clip(new_cy, 0, img_h)
        rotated_labels[:, 3] = np.clip(new_w, 0, img_w)
        rotated_labels[:, 4] = np.clip(new_h, 0, img_h)

        return rotated_labels
