import time
import numpy as np

class BoxFormatTransform:
    def __init__(self, mode: str = "train", timing: bool = False):
        self.mode = mode
        self.timing = timing

    def __call__(self, inputs):
        if self.timing:
            start_time = time.time()

        if self.mode == "train":
            # trainモードでは何もしない
            result = inputs

        elif self.mode in ["val", "test"]:
            # val/testモードではラベルを変換
            labels = inputs["labels"]
            transformed_labels = []
            for label in labels:
                cls, cx, cy, w, h = label
                # バウンディングボックスを [x, y, w, h, cls] に変換
                x = cx - w / 2
                y = cy - h / 2
                transformed_labels.append([x, y, w, h, cls])

            # ラベルを更新
            inputs["labels"] = np.array(transformed_labels, dtype=np.float32)
            
        if self.timing:
            elapsed_time = time.time() - start_time
            print(f"BoxFormatTransform processing time: {elapsed_time:.6f} seconds")

        return inputs


class LabelFilter:
    def __init__(self, orig_class, my_class, timing: bool = False):
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

        self.timing = timing

    def __call__(self, inputs):
        """
        フィルタリングとクラス整列を適用。

        Args:
            inputs (dict): 入力サンプル。
                - "image": 画像データ (NumPy ndarray)。
                - "labels": バウンディングボックス情報 (NumPy ndarray)。
                  ラベルフォーマット: [cls, cx, cy, w, h]

        Returns:
            dict: フィルタリング後のサンプル。
        """

        if self.timing:
            start_time = time.time()

        labels = inputs["labels"]

        if labels is not None and len(labels) > 0:
            # 指定したクラスのみ抽出
            mask = np.isin(labels[:, 0], self.allowed_classes)
            filtered_labels = labels[mask]

            # クラスIDを新しいIDに整列
            for old_class, new_class in self.class_mapping.items():
                filtered_labels[filtered_labels[:, 0] == old_class, 0] = new_class

            inputs["labels"] = filtered_labels

        if self.timing:
            elapsed_time = time.time() - start_time
            print(f"LabelFilter processing time: {elapsed_time:.6f} seconds")

        return inputs
