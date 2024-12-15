import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import math


from data.dataset.waymo.data_info import MY_CLASS   

def visualize_sequence_from_dataset(dataset, frame_indices: list, mode='train', cols=4):
    """
    データセットから指定されたフレームを可視化し、クラスごとに異なる色で描画します。

    Args:
        dataset: 可視化するデータセット。
        frame_indices (list): 可視化するフレームのインデックスリスト。
        mode (str): 'train', 'val', 'test' などのモード。座標形式の違いに対応。
        cols (int): 1行あたりのフレーム数。
    """
    # クラスごとの色を定義（固定色リスト）
    class_colors = [
        (255, 0, 0),    # クラス0: 赤
        (0, 255, 0),    # クラス1: 緑
        (0, 0, 255),    # クラス2: 青
    ]

    # クラス名を取得
    class_names = {v: k for k, v in MY_CLASS.items()}

    # 行数と列数を計算
    num_frames = len(frame_indices)
    rows = math.ceil(num_frames / cols)

    # プロットを作成
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    axes = axes.flatten()  # 1次元リストに変換

    for i, idx in enumerate(frame_indices):
        data = dataset[idx]
        events = data['images']  # (ch, h, w)
        labels = data['labels']
        
        # フレーム変換と描画
        frame = events  # (ch, h, w)
        img_uint = np.transpose(frame.numpy(), (1, 2, 0)).astype('uint8').copy()

        # バウンディングボックスの描画
        for bbox in labels:
            if torch.all(bbox == 0):  # 無効なバウンディングボックスはスキップ
                continue

            # `mode` に応じたバウンディングボックスの座標取得
            if mode == 'train':
                cls, cx, cy, w, h = bbox
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)
            elif mode in ['val', 'test']:
                x1, y1, w, h, cls = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)

            # クラスに応じて色を選択
            class_id = int(cls)
            color = class_colors[class_id % len(class_colors)]  # クラスIDに基づく色

            # クラス名を取得
            class_name = class_names.get(class_id, "unknown")

            # バウンディングボックスを描画
            img_uint = cv2.rectangle(img_uint, (x1, y1), (x2, y2), color, 2)
            img_uint = cv2.putText(img_uint, f"{class_name} ({class_id})", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # サブプロットに描画
        axes[i].imshow(img_uint)
        axes[i].axis('off')
        axes[i].set_title(f"Frame {idx}", fontsize=12)

    # 余ったサブプロットを非表示にする
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)  # プロット間の間隔を調整
    plt.show()
