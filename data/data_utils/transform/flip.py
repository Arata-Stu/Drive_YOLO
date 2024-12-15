import torch

class Flip:

    def __init__(self, vertical: bool = False, horizontal: bool = False):
        """
        Args:
            vertical: Whether to enable vertical flipping.
            horizontal: Whether to enable horizontal flipping.
        """
        self.vertical = vertical
        self.horizontal = horizontal
        

    def __call__(self, inputs: dict) -> dict:
        """
        Args:
            inputs: dict
                "images": Tensor of shape [C, H, W].
                "labels": Tensor of shape [N, 5] (5: cls, cx, cy, w, h).

        Returns:
            outputs: dict with the flipped image and updated labels.
        """
        image = inputs.get("images")
        labels = inputs.get("labels")

        if image is None:
            raise ValueError("Input dictionary must contain 'image' key.")
        if not isinstance(image, torch.Tensor):
            raise TypeError("The 'image' must be a PyTorch Tensor.")
        if labels is None:
            raise ValueError("Input dictionary must contain 'labels' key.")
        if not isinstance(labels, torch.Tensor):
            raise TypeError("The 'labels' must be a PyTorch Tensor.")

        _, height, width = image.shape

        # Apply vertical flip
        if self.vertical:
            image = torch.flip(image, dims=[1])  # Flip along the height dimension
            # Update the vertical position of bounding boxes
            labels[:, 2] = height - labels[:, 2]  # Update cy (center y)

        # Apply horizontal flip
        if self.horizontal:
            image = torch.flip(image, dims=[2])  # Flip along the width dimension
            # Update the horizontal position of bounding boxes
            labels[:, 1] = width - labels[:, 1]  # Update cx (center x)

        inputs['images'] = image
        inputs['labels'] = labels
        return inputs
