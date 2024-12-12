import tensorflow as tf
import os
import json
from waymo_open_dataset import dataset_pb2 as open_dataset

class WaymoSequencePreprocessor:
    def __init__(self, output_dir):
        """
        初期化: 出力ディレクトリを設定します。

        Args:
            output_dir (str): 抽出データを保存するディレクトリ。
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def parse_tfrecord(self, filename):
        """
        TFRecordファイルをパースしてフレームを生成します。
        
        Args:
            filename (str): TFRecordファイルのパス。
            
        Returns:
            generator: パースされたフレーム。
        """
        dataset = tf.data.TFRecordDataset(filename, compression_type='')
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            yield frame

    def preprocess(self, filenames):
        """
        TFRecordファイルをシーケンスごとに処理し、画像とアノテーションを保存します。
        
        Args:
            filenames (list): TFRecordファイルのパスリスト。
        """
        for filename in filenames:
            # ファイル名からIDを抽出
            base_name = os.path.basename(filename)
            sequence_id = base_name.split('_')[0].replace('segment-', '')

            print(f"Processing sequence {sequence_id}: {filename}")

            # シーケンス用ディレクトリを作成
            sequence_dir = os.path.join(self.output_dir, sequence_id)
            os.makedirs(sequence_dir, exist_ok=True)
            os.makedirs(os.path.join(sequence_dir, "images"), exist_ok=True)

            annotations = []

            # フレームごとに処理
            for frame_idx, frame in enumerate(self.parse_tfrecord(filename)):
                frame_id = f"frame_{frame_idx:04d}"
                for camera_image in frame.images:
                    camera_name = open_dataset.CameraName.Name.Name(camera_image.name)

                    if camera_name not in ["FRONT", "FRONT_LEFT", "SIDE_LEFT", "FRONT_RIGHT", "SIDE_RIGHT"]:
                        continue
                    
                    # 画像を保存
                    image_path = os.path.join(sequence_dir, "images", f"{frame_id}_{camera_name}.jpg")
                    with open(image_path, "wb") as img_file:
                        img_file.write(camera_image.image)

                    # バウンディングボックスを収集
                    bboxes = []
                    for camera_labels in frame.camera_labels:
                        if camera_labels.name != camera_image.name:
                            continue
                        for label in camera_labels.labels:
                            bboxes.append({
                                "center_x": label.box.center_x,
                                "center_y": label.box.center_y,
                                "length": label.box.length,
                                "width": label.box.width,
                                "class": getattr(label, "type", -1)  # クラス情報
                            })

                    # フレームアノテーションを保存
                    annotations.append({
                        "frame_id": frame_id,
                        "camera_name": camera_name,
                        "image_path": image_path,
                        "bboxes": bboxes
                    })

            # シーケンスのアノテーションを保存
            with open(os.path.join(sequence_dir, "annotations.json"), "w") as anno_file:
                json.dump(annotations, anno_file, indent=4)

# 使用例
output_dir = "./processed_waymo"
preprocessor = WaymoSequencePreprocessor(output_dir)
tfrecord_files = [
    "segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord",
]
preprocessor.preprocess(tfrecord_files)
