import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
from typing import Tuple

class Resize:
    def __init__(self, target_size: Tuple[int, int], mode: InterpolationMode = InterpolationMode.BILINEAR):
        """
        Args:
            target_size: (height, width)
            mode: 'linear', 'bilinear', 'nearest', 'bicubic', 'area'
        """
        self.target_size = target_size
        self.mode = mode

    def __call__(self, inputs: dict) -> dict:
        """
        Args:
            inputs: dict
                "images": Tensor [C, H, W]
                "labels": Tensor [N, 5] (5: cls, cx, cy, w, h)
        Returns:
            dict with resized "images" and updated "labels"
        """
        img = inputs.get("images")
        labels = inputs.get("labels")

        if img is None:
            raise ValueError("Input dictionary must contain 'images' key.")
        if not isinstance(img, torch.Tensor):
            raise TypeError("The 'images' must be a PyTorch Tensor.")
        if labels is None:
            raise ValueError("Input dictionary must contain 'labels' key.")
        if not isinstance(labels, torch.Tensor):
            raise TypeError("The 'labels' must be a PyTorch Tensor.")

        # Original image size
        _, orig_height, orig_width = img.shape

        # Target size
        target_height, target_width = self.target_size

        # Calculate scale factors and new size while maintaining aspect ratio
        scale = min(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize image while maintaining aspect ratio
        resized_img = F.resize(img,  # Add batch dimension
                                size=(new_height, new_width),
                                interpolation=self.mode,
                                antialias=False)

        
        labels[:, 1] = labels[:, 1] * scale 
        labels[:, 2] = labels[:, 2] * scale 
        labels[:, 3] *= scale 
        labels[:, 4] *= scale  

        # Update inputs dictionary
        inputs["images"] = resized_img
        inputs["labels"] = labels
        return inputs
