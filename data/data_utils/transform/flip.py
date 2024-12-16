import numpy as np
import time

class Flip:
    def __init__(self, vertical: bool = False, horizontal: bool = False, timing: bool = False):
        """
        Args:
            vertical: Whether to enable vertical flipping.
            horizontal: Whether to enable horizontal flipping.
            timing: Whether to enable timing for the __call__ method.
        """
        self.vertical = vertical
        self.horizontal = horizontal
        self.timing = timing

    def __call__(self, inputs: dict) -> dict:
        """
        Args:
            inputs: dict
                "images": ndarray of shape [C, H, W].
                "labels": ndarray of shape [N, 5] (5: cls, cx, cy, w, h).

        Returns:
            outputs: dict with the flipped image and updated labels.
        """
        if self.timing:
            start_time = time.time()

        image = inputs.get("images")
        labels = inputs.get("labels")

        if image is None:
            raise ValueError("Input dictionary must contain 'images' key.")
        if not isinstance(image, np.ndarray):
            raise TypeError("The 'images' must be a NumPy ndarray.")
        if labels is None:
            raise ValueError("Input dictionary must contain 'labels' key.")
        if not isinstance(labels, np.ndarray):
            raise TypeError("The 'labels' must be a NumPy ndarray.")

        # Convert image from [C, H, W] to [H, W, C]
        image = np.transpose(image, (1, 2, 0))
        height, width, _ = image.shape

        # Apply vertical flip
        if self.vertical:
            image = np.flip(image, axis=0)  # Flip along the height axis
            # Update the vertical position of bounding boxes
            labels[:, 2] = height - labels[:, 2]  # Update cy (center y)

        # Apply horizontal flip
        if self.horizontal:
            image = np.flip(image, axis=1)  # Flip along the width axis
            # Update the horizontal position of bounding boxes
            labels[:, 1] = width - labels[:, 1]  # Update cx (center x)

        # Convert image back to [C, H, W]
        image = np.transpose(image, (2, 0, 1))

        inputs['images'] = image
        inputs['labels'] = labels

        if self.timing:
            elapsed_time = time.time() - start_time
            print(f"Processing time: {elapsed_time:.6f} seconds")

        return inputs
