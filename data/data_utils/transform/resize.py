import time
import cv2
import numpy as np
from typing import Tuple

class Resize:
    def __init__(self, target_size: Tuple[int, int], mode: str = "bilinear", timing: bool = False):
        """
        Args:
            target_size: (height, width)
            mode: 'bilinear', 'nearest', 'bicubic', 'area'
            timing: 処理時間を計測するかどうか
        """
        self.target_size = target_size
        self.mode = mode
        self.timing = timing

    def __call__(self, inputs: dict) -> dict:
        if self.timing:
            start_time = time.time()

        img = inputs.get("images")
        labels = inputs.get("labels")

        if img is None:
            raise ValueError("Input dictionary must contain 'images' key.")
        if not isinstance(img, np.ndarray):
            raise TypeError("The 'images' must be a NumPy ndarray.")
        if labels is None:
            raise ValueError("Input dictionary must contain 'labels' key.")
        if not isinstance(labels, np.ndarray):
            raise TypeError("The 'labels' must be a NumPy ndarray.")

        # Convert image from [C, H, W] to [H, W, C]
        img = np.transpose(img, (1, 2, 0))

        # Original image size
        orig_height, orig_width, _ = img.shape

        # Target size
        target_height, target_width = self.target_size

        # Calculate scale factors
        scale_w = target_width / orig_width
        scale_h = target_height / orig_height

        # Resize image using OpenCV
        interpolation = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA
        }.get(self.mode, cv2.INTER_LINEAR)

        resized_img = cv2.resize(img, (target_width, target_height), interpolation=interpolation)

        # Scale bounding box labels
        labels[:, 1] *= scale_w  # cx
        labels[:, 2] *= scale_h  # cy
        labels[:, 3] *= scale_w  # w
        labels[:, 4] *= scale_h  # h

        # Convert image back to [C, H, W]
        resized_img = np.transpose(resized_img, (2, 0, 1))

        # Update inputs dictionary
        inputs["images"] = resized_img
        inputs["labels"] = labels

        if self.timing:
            elapsed_time = time.time() - start_time
            print(f"Resize processing time: {elapsed_time:.6f} seconds")

        return inputs