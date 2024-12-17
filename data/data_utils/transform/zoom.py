import numpy as np
import cv2
import time
from typing import Tuple
import random

class RandomZoom:
    def __init__(
        self, 
        prob_weight=(8, 2), 
        in_scale=(1.0, 1.5), 
        out_scale=(1.0, 1.2), 
        center_margin_ratio=0.2, 
        timing=False
    ):
        """
        Randomly applies zoom in or zoom out based on given probabilities and scales.

        Args:
            prob_weight (tuple): Probabilities of (zoom_in, zoom_out).
            in_scale (tuple): Min and max scale for zoom in.
            out_scale (tuple): Min and max scale for zoom out.
            center_margin_ratio (float): Margin ratio for random center near image center.
            timing (bool): Whether to print processing time.
        """
        self.prob_weight = prob_weight
        self.in_scale = in_scale
        self.out_scale = out_scale
        self.center_margin_ratio = center_margin_ratio
        self.timing = timing

    def _get_random_center(self, image_shape):
        """Generate a random center point within the specified margin range."""
        _, H, W = image_shape
        margin_x = int(W * self.center_margin_ratio)
        margin_y = int(H * self.center_margin_ratio)
        center_x = random.randint(margin_x, W - margin_x)
        center_y = random.randint(margin_y, H - margin_y)
        return center_x, center_y

    def __call__(self, inputs):
        # Decide whether to zoom in or zoom out
        zoom_type = random.choices(["in", "out"], weights=self.prob_weight, k=1)[0]
        
        # Randomly determine the scale within the range
        if zoom_type == "in":
            scale = random.uniform(*self.in_scale)
        else:
            scale = random.uniform(*self.out_scale)
        
        # Randomly choose the center near the image center
        image = inputs.get("images")
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("'images' must be a NumPy ndarray.")
        center = self._get_random_center(image.shape)

        # Apply zoom in or zoom out
        if zoom_type == "in":
            zoom = Zoom_in(scale=scale, center=center, timing=self.timing)
        else:
            zoom = Zoom_out(scale=scale, center=center, timing=self.timing, background_color=(114, 114, 114))

        return zoom(inputs)

class Zoom_in:
    def __init__(self, scale: float, center: Tuple[int, int], timing: bool = False):
        self.scale = scale
        self.center = center
        self.timing = timing

    def __call__(self, inputs) -> dict:
        if self.timing:
            start_time = time.time()

        image = inputs.get("images")
        labels = inputs.get("labels")

        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("'images' must be a NumPy ndarray.")
        if labels is None or not isinstance(labels, np.ndarray):
            raise ValueError("'labels' must be a NumPy ndarray of shape [N, 5]")

        C, H, W = image.shape
        new_H, new_W = int(H / self.scale), int(W / self.scale)

        # ズームウィンドウの位置を計算
        cx, cy = self.center
        x1 = max(0, min(int(cx - new_W // 2), W - new_W))
        y1 = max(0, min(int(cy - new_H // 2), H - new_H))

        # 画像のクロップとリサイズ
        image_transposed = image.transpose(1, 2, 0)  # (H, W, C)
        cropped_image = image_transposed[y1:y1 + new_H, x1:x1 + new_W, :]
        zoomed_image = cv2.resize(cropped_image, (W, H), interpolation=cv2.INTER_CUBIC)
        zoomed_image = zoomed_image.transpose(2, 0, 1)  # (C, H, W)

        # ラベルの座標を調整
        if labels is not None:
            zoomed_labels = labels.copy()
            zoomed_labels[:, 1] = (labels[:, 1] - x1) * self.scale  # cx
            zoomed_labels[:, 2] = (labels[:, 2] - y1) * self.scale  # cy
            zoomed_labels[:, 3] *= self.scale  # w
            zoomed_labels[:, 4] *= self.scale  # h
        else:
            zoomed_labels = None

        inputs['images'] = zoomed_image
        inputs['labels'] = zoomed_labels

        if self.timing:
            elapsed_time = time.time() - start_time
            print(f"Processing time: {elapsed_time:.6f} seconds")

        return inputs

    
class Zoom_out:
    def __init__(self, scale: float, center: Tuple[int, int] = None, timing: bool = False, background_color=(0, 0, 0)):
        self.scale = scale
        self.center = center
        self.timing = timing
        self.background_color = background_color

    def __call__(self, inputs):
        if self.timing:
            start_time = time.time()

        image = inputs.get("images")
        labels = inputs.get("labels")

        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("'images' must be a NumPy ndarray.")
        if labels is None or not isinstance(labels, np.ndarray):
            raise ValueError("'labels' must be a NumPy ndarray of shape [N, 5]")

        C, H, W = image.shape
        new_H, new_W = int(H / self.scale), int(W / self.scale)

        # Resize the image (zoomed out)
        image_transposed = image.transpose(1, 2, 0)  # (H, W, C)
        resized_image = cv2.resize(image_transposed, (new_W, new_H), interpolation=cv2.INTER_CUBIC)

        # Place resized image onto the canvas
        canvas = np.full((H, W, C), self.background_color, dtype=image.dtype)
        cx, cy = self.center if self.center else (W // 2, H // 2)

        # Calculate top-left coordinate for placement
        x1 = max(0, cx - new_W // 2)
        y1 = max(0, cy - new_H // 2)
        x2 = min(W, x1 + new_W)
        y2 = min(H, y1 + new_H)

        # Add resized image to the canvas with boundary checking
        canvas[y1:y2, x1:x2] = resized_image[0:(y2 - y1), 0:(x2 - x1)]

        # Adjust labels for zoom out
        zoomed_labels = labels.copy()

        if labels is not None:
            zoomed_labels[:, 1] = (zoomed_labels[:, 1] / self.scale) + x1  # cx の調整
            zoomed_labels[:, 2] = (zoomed_labels[:, 2] / self.scale) + y1  # cy の調整
            zoomed_labels[:, 3] /= self.scale  # w の調整
            zoomed_labels[:, 4] /= self.scale  # h の調整

            # キャンバス外に出ないようにクリッピング
            zoomed_labels[:, 1] = np.clip(zoomed_labels[:, 1], 0, W)  # cx
            zoomed_labels[:, 2] = np.clip(zoomed_labels[:, 2], 0, H)  # cy
            zoomed_labels[:, 3] = np.clip(zoomed_labels[:, 3], 0, W)  # w
            zoomed_labels[:, 4] = np.clip(zoomed_labels[:, 4], 0, H)  # h
        else:
            zoomed_labels = None

        # Update inputs
        inputs['images'] = canvas.transpose(2, 0, 1)  # (C, H, W)
        inputs['labels'] = zoomed_labels

        if self.timing:
            elapsed_time = time.time() - start_time
            print(f"Processing time: {elapsed_time:.6f} seconds")

        return inputs
