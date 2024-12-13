import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

def visualize_sequence_from_dataset(dataset, frame_indices: list, mode='train'):
   

    # 指定されたインデックスで処理
    num_frames = len(frame_indices)
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))

    for i, idx in enumerate(frame_indices):
        data = dataset[idx]
        events = data['image']  # (ch, h, w)
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

            # バウンディングボックスを描画
            img_uint = cv2.rectangle(img_uint, (x1, y1), (x2, y2), (255, 0, 0), 2)
            img_uint = cv2.putText(img_uint, f"Cls: {int(cls)}", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # サブプロットに描画
        axes[i].imshow(img_uint)
        axes[i].axis('off')
        axes[i].set_title(f"Frame {idx}")

    plt.tight_layout()
    plt.show()
