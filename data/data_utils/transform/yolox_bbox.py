import torch

class BoxFormatTransform:
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
    
class LabelFilter:
    def __init__(self, orig_class, my_class):
        """
        特定のクラスをフィルタリングし、ラベルを整列するTransform。

        Args:
            orig_class (dict): 元のクラスIDマッピング (例: {"unknown": 0, ...})。
            my_class (dict): 新しいクラスIDマッピング (例: {"vehicle": 0, ...})。
        """
        self.orig_class = orig_class
        self.my_class = my_class
        self.allowed_classes = [orig_class[name] for name in my_class.keys()]
        self.class_mapping = {orig_class[name]: my_class[name] for name in my_class.keys()}

    def __call__(self, inputs):
        """
        フィルタリングとクラス整列を適用。

        Args:
            sample (dict): 入力サンプル。
                - "image": 画像データ (torch.Tensor)。
                - "labels": バウンディングボックス情報 (torch.Tensor)。
                  ラベルフォーマット: [cls, cx, cy, w, h]

        Returns:
            dict: フィルタリング後のサンプル。
        """
        labels = inputs["labels"]
        if labels is not None:
            # 指定したクラスのみ抽出
            mask = torch.isin(labels[:, 0], torch.tensor(self.allowed_classes, dtype=labels.dtype))
            filtered_labels = labels[mask]

            # クラスIDを新しいIDに整列
            for old_class, new_class in self.class_mapping.items():
                filtered_labels[filtered_labels[:, 0] == old_class, 0] = new_class

            inputs["labels"] = filtered_labels

        return inputs