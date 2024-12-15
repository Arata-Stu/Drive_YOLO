import torch

class YOLOXTransform:
    def __init__(self, mode: str = "train"):
        self.mode = mode

    def __call__(self, sample):
        if self.mode == "train":
            # trainモードでは何もしない
            return sample

        elif self.mode in ["val", "test"]:
            # val/testモードではラベルを変換
            transformed_labels = []
            for label in sample["labels"]:
                cls, cx, cy, w, h = label
                # バウンディングボックスを [x, y, w, h, cls] に変換
                x = cx - w / 2
                y = cy - h / 2
                transformed_labels.append([x, y, w, h, cls])

            # ラベルを更新
            sample["labels"] = torch.tensor(transformed_labels, dtype=torch.float32)

        return sample