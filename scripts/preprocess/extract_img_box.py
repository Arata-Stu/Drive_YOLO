import argparse
import tensorflow as tf
import os
import cv2
import numpy as np
from waymo_open_dataset import dataset_pb2 as open_dataset
from colorama import Fore, Style
from tqdm import tqdm

MODE = {"training": "train", "testing": "test", "validation": "val"}

class WaymoSequencePreprocessor:
    def __init__(self, output_dir, image_size):
        """
        初期化: 出力ディレクトリと画像サイズを設定します。

        Args:
            output_dir (str): 抽出データを保存するディレクトリ。
            image_size (int): 出力画像のサイズ（正方形）。
        """
        self.output_dir = output_dir
        self.image_size = image_size
        os.makedirs(output_dir, exist_ok=True)

    def parse_tfrecord(self, filename):
        """
        TFRecordファイルをパースしてフレームごとにデータを生成します。

        Args:
            filename (str): TFRecordファイルのパス。

        Yields:
            open_dataset.Frame: パースされたフレームオブジェクト。
        """
        try:
            dataset = tf.data.TFRecordDataset(filename, compression_type='')
            for data in dataset:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                yield frame
        except tf.errors.DataLossError as e:
            print(f"{Fore.RED}DataLossError: {filename} is corrupted. Skipping...{Style.RESET_ALL}")
            return

    def resize_image_without_padding(self, image, bboxes):
        """
        アスペクト比を維持しつつ画像をリサイズし、パディングを行いません。
        バウンディングボックスの座標はスケーリングします。

        Args:
            image (np.ndarray): 元の画像。
            bboxes (list): 元のバウンディングボックス。

        Returns:
            np.ndarray: リサイズ後の画像。
            list: リサイズ後のバウンディングボックス。
        """
        original_height, original_width = image.shape[:2]
        target_size = self.image_size

        # アスペクト比を維持してリサイズ
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        resized_image = cv2.resize(image, (new_width, new_height))

        # バウンディングボックスのスケーリング
        resized_bboxes = []
        for bbox in bboxes:
            resized_bboxes.append({
                "center_x": bbox["center_x"] * scale,
                "center_y": bbox["center_y"] * scale,
                "length": bbox["length"] * scale,
                "width": bbox["width"] * scale,
                "class": bbox["class"]
            })

        return resized_image, resized_bboxes

    def preprocess_to_tfrecord(self, input_dir, output_dir):
        """
        入力ディレクトリ内のTFRecordをフィルタリングし、新しいTFRecordに変換して保存します。

        Args:
            input_dir (str): TFRecordファイルが格納されたディレクトリ。
            output_dir (str): フィルタリング後のTFRecordを保存するディレクトリ。
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        subsets = ['training', 'testing', 'validation']
        for subset in subsets:
            subset_input_dir = os.path.join(input_dir, subset)
            if not os.path.exists(subset_input_dir):
                print(f"Warning: {subset_input_dir} does not exist. Skipping.")
                continue

            subset_output_dir = os.path.join(output_dir, MODE[subset])
            os.makedirs(subset_output_dir, exist_ok=True)

            tfrecord_files = [
                os.path.join(subset_input_dir, f) for f in os.listdir(subset_input_dir) if f.endswith(".tfrecord")
            ]

            for filename in tqdm(tfrecord_files, desc=f"Processing {subset} sequences"):
                base_name = os.path.basename(filename)
                sequence_id = base_name.split('_')[0].replace('segment-', '')

                # 正式な出力ファイルと一時ファイルのパス
                sequence_output_path = os.path.join(subset_output_dir, f"{sequence_id}.tfrecord")
                temp_output_path = sequence_output_path + ".tmp"

                # 再開機能: 正式ファイルが存在する場合はスキップ
                if os.path.exists(sequence_output_path):
                    print(f"Skipping already processed file: {sequence_output_path}")
                    continue

                # 再開時: 途中ファイルが存在する場合は削除
                if os.path.exists(temp_output_path):
                    print(f"Removing incomplete temporary file: {temp_output_path}")
                    os.remove(temp_output_path)

                print(f"Processing sequence {sequence_id} in {subset}: {filename}")

                try:
                    with tf.io.TFRecordWriter(temp_output_path) as writer:
                        for frame_idx, frame in enumerate(self.parse_tfrecord(filename)):
                            frame_id = f"frame_{frame_idx:04d}"
                            for camera_image in frame.images:
                                camera_name = open_dataset.CameraName.Name.Name(camera_image.name)
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
                                            "class": getattr(label, "type", -1)
                                        })

                                # 画像リサイズ（パディングなし）
                                image_np = np.frombuffer(camera_image.image, dtype=np.uint8)
                                decoded_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                                resized_image, resized_bboxes = self.resize_image_without_padding(decoded_image, bboxes)

                                # TFRecordにシリアライズして保存
                                _, buffer = cv2.imencode('.jpg', resized_image)
                                example = self.serialize_example(
                                    image_bytes=buffer.tobytes(),
                                    camera_name=camera_name,
                                    frame_id=frame_id,
                                    bboxes=resized_bboxes
                                )
                                writer.write(example.SerializeToString())

                    # 処理完了後に一時ファイルをリネーム
                    os.rename(temp_output_path, sequence_output_path)
                    print(f"Successfully saved: {sequence_output_path}")

                except Exception as e:
                    # エラー発生時に一時ファイルを削除
                    if os.path.exists(temp_output_path):
                        os.remove(temp_output_path)
                    print(f"Error processing {filename}: {e}")


    def serialize_example(self, image_bytes, camera_name, frame_id, bboxes):
        """
        データを新しいTFRecord形式にシリアライズします。

        Args:
            image_bytes (bytes): 画像データ。
            camera_name (str): カメラの名前。
            frame_id (str): フレームID。
            bboxes (list): バウンディングボックスの情報。

        Returns:
            tf.train.Example: シリアライズされたExample。
        """
        # バウンディングボックス情報を配列形式で保存
        bbox_data = []
        for bbox in bboxes:
            bbox_data.extend([bbox["center_x"], bbox["center_y"], bbox["length"], bbox["width"], bbox["class"]])

        feature = {
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
            "camera_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[camera_name.encode('utf-8')])),
            "frame_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[frame_id.encode('utf-8')])),
            "bboxes": tf.train.Feature(float_list=tf.train.FloatList(value=bbox_data))
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waymo TFRecord Preprocessor")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing TFRecord files")
    parser.add_argument("-o", "--output", required=True, help="Output directory to save processed data")
    parser.add_argument("-s", "--size", type=int, required=True, help="Output image size (square)")

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    image_size = args.size

    preprocessor = WaymoSequencePreprocessor(output_dir, image_size)
    preprocessor.preprocess_to_tfrecord(input_dir, output_dir)
