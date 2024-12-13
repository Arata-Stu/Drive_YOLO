import argparse
import tensorflow as tf
import os
import json
from waymo_open_dataset import dataset_pb2 as open_dataset
from colorama import Fore, Style
from tqdm import tqdm

MODE = {"training": "train", "testing": "test", "validation": "val"}

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
        try:
            dataset = tf.data.TFRecordDataset(filename, compression_type='')
            for data in dataset:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                yield frame
        except tf.errors.DataLossError as e:
            print(f"{Fore.RED}DataLossError: {filename} is corrupted. Skipping...{Style.RESET_ALL}")
            return

    def preprocess(self, input_dir, output_dir):
        """
        入力ディレクトリ内のすべてのTFRecordファイルを処理します。

        Args:
            input_dir (str): TFRecordファイルが格納されたディレクトリ。
            output_dir (str): 処理後のデータを保存するディレクトリ。
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # train, test, validationの各ディレクトリを検出
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
                # ファイル名からIDを抽出
                base_name = os.path.basename(filename)
                sequence_id = base_name.split('_')[0].replace('segment-', '')

                print(f"Processing sequence {sequence_id} in {subset}: {filename}")

                # シーケンス用ディレクトリを作成
                sequence_dir = os.path.join(subset_output_dir, sequence_id)
                os.makedirs(sequence_dir, exist_ok=True)

                camera_annotations = {  # カメラごとのアノテーションを格納
                    "FRONT": [],
                    "FRONT_LEFT": [],
                    "SIDE_LEFT": [],
                    "FRONT_RIGHT": [],
                    "SIDE_RIGHT": []
                }

                # フレームごとに処理
                for frame_idx, frame in enumerate(self.parse_tfrecord(filename)):
                    frame_id = f"frame_{frame_idx:04d}"
                    for camera_image in frame.images:
                        camera_name = open_dataset.CameraName.Name.Name(camera_image.name)

                        if camera_name not in camera_annotations.keys():
                            continue

                        # カメラごとのディレクトリを作成
                        camera_dir = os.path.join(sequence_dir, "images", camera_name)
                        os.makedirs(camera_dir, exist_ok=True)

                        # 画像を保存
                        image_path = os.path.join(camera_dir, f"{frame_id}.jpg")
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
                        camera_annotations[camera_name].append({
                            "frame_id": frame_id,
                            "camera_name": camera_name,
                            "image_path": image_path,
                            "bboxes": bboxes
                        })

                # カメラごとのアノテーションを保存
                for camera_name, annotations in camera_annotations.items():
                    camera_anno_path = os.path.join(sequence_dir, "annotations", camera_name)
                    os.makedirs(camera_anno_path, exist_ok=True)
                    anno_file_path = os.path.join(camera_anno_path, "annotations.json")
                    with open(anno_file_path, "w") as anno_file:
                        json.dump(annotations, anno_file, indent=4)


if __name__ == "__main__":
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description="Waymo TFRecord Preprocessor")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing TFRecord files")
    parser.add_argument("-o", "--output", required=True, help="Output directory to save processed data")

    args = parser.parse_args()

    # 入力と出力ディレクトリの指定
    input_dir = args.input
    output_dir = args.output

    # 前処理実行
    preprocessor = WaymoSequencePreprocessor(output_dir)
    preprocessor.preprocess(input_dir, output_dir)