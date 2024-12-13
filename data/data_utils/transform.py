import cv2
import numpy as np
import torch

class ResizeTransform:
    def __init__(self, input_size=(640, 640), swap=(2, 0, 1)):
        """
        リサイズとパディングを行う前処理Transformを初期化。

        Args:
            input_size (tuple): リサイズ後のターゲットサイズ (height, width)。
            mean (list or None): ピクセル値の平均値 (正規化用)。
            std (list or None): ピクセル値の標準偏差 (正規化用)。
            swap (tuple): チャンネルの順序 (デフォルトは (2, 0, 1))。
        """
        self.input_size = input_size
        self.swap = swap

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
        image = sample["image"].permute(1, 2, 0).numpy()  # CHW -> HWC
        labels = sample["labels"]
        input_height, input_width = self.input_size

        # 元画像のリサイズ比率を計算
        original_height, original_width = image.shape[:2]
        scale_ratio = min(input_height / original_height, input_width / original_width)

        # リサイズ後のサイズを計算
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

        # リサイズされた画像を作成
        resized_img = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        # パディング画像を初期化 (114で埋める)
        padded_img = np.ones((input_height, input_width, 3), dtype=np.float32) * 114.0

        # リサイズ画像を左上から埋め込む
        padded_img[:new_height, :new_width, :] = resized_img

        # チャンネル順序を変更
        padded_img = padded_img.transpose(self.swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

        # バウンディングボックスの変換
        if labels is not None:
            labels = labels.clone()
            # ラベルのスケール変換
            labels[:, 1:] *= scale_ratio  # [center_x, center_y, width, height]

        # 画像をtorch.Tensorに変換
        sample["image"] = torch.tensor(padded_img)
        sample["labels"] = labels

        return sample

class FlipTransform:
    def __init__(self, flip_horizontal=True, flip_vertical=False):
        """
        画像とラベルに対して水平反転および垂直反転を行うクラス。

        Args:
            flip_horizontal (bool): 水平反転を行うかどうか。
            flip_vertical (bool): 垂直反転を行うかどうか。
        """
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical

    def __call__(self, sample):
        """
        サンプルに反転操作を適用。

        Args:
            sample (dict): データローダーから受け取るサンプル。
                - "image": 画像データ (torch.Tensor, CHWフォーマット)。
                - "labels": バウンディングボックス情報 (torch.Tensor)。
                  ラベルフォーマット: [cls, cx, cy, w, h]

        Returns:
            dict: 変換後のサンプル。
        """
        image = sample["image"].numpy()  # torch.Tensor -> numpy (CHW形式)
        labels = sample["labels"]
        C, H, W = image.shape

        # 水平反転
        if self.flip_horizontal:
            image = image[:, :, ::-1]  # 水平方向に反転
            if labels is not None:
                labels = labels.clone()
                labels[:, 1] = W - labels[:, 1]  # cx = W - cx

        # 垂直反転
        if self.flip_vertical:
            image = image[:, ::-1, :]  # 垂直方向に反転
            if labels is not None:
                labels = labels.clone()
                labels[:, 2] = H - labels[:, 2]  # cy = H - cy

        # numpy -> torch.Tensor に戻す
        sample["image"] = torch.tensor(image, dtype=torch.float32)
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

