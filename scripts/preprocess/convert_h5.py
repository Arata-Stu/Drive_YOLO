import h5py
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm


class TFRecordToH5Converter:
    def __init__(self, tfrecord_dir, output_h5_path):
        """
        初期化: 入力ディレクトリと出力ファイルを設定します。

        Args:
            tfrecord_dir (str): TFRecordファイルが格納されたディレクトリ。
            output_h5_path (str): HDF5ファイルの保存先パス。
        """
        self.tfrecord_dir = tfrecord_dir
        self.output_h5_path = output_h5_path

    def parse_tfrecord(self, serialized_example):
        """
        シリアライズされたTFRecordをパースします。

        Args:
            serialized_example (bytes): シリアライズされたTFRecordデータ。

        Returns:
            dict: パースされたデータ。
        """
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "camera_name": tf.io.FixedLenFeature([], tf.string),
            "frame_id": tf.io.FixedLenFeature([], tf.string),
            "bboxes": tf.io.VarLenFeature(tf.float32),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)

        # データの復元
        image = tf.image.decode_jpeg(example["image"])
        bboxes = tf.sparse.to_dense(example["bboxes"])
        bboxes = tf.reshape(bboxes, [-1, 5])  # [center_x, center_y, length, width, class]
        return {
            "image": image.numpy(),
            "bboxes": bboxes.numpy(),
        }

    def convert_to_h5(self):
        """
        TFRecordからHDF5にデータを変換して保存します。
        """
        with h5py.File(self.output_h5_path, "w") as h5_file:
            tfrecord_files = [
                os.path.join(self.tfrecord_dir, f) for f in os.listdir(self.tfrecord_dir) if f.endswith(".tfrecord")
            ]

            for tfrecord_file in tqdm(tfrecord_files, desc="Converting TFRecords to H5"):
                dataset = tf.data.TFRecordDataset(tfrecord_file)
                for idx, raw_record in enumerate(dataset):
                    data = self.parse_tfrecord(raw_record.numpy())

                    # フレームIDとしてファイル名＋インデックスを利用
                    frame_id = f"{os.path.basename(tfrecord_file)}_frame_{idx}"

                    # HDF5にデータを保存
                    group = h5_file.create_group(frame_id)
                    group.create_dataset("image", data=data["image"], compression="gzip")
                    group.create_dataset("bboxes", data=data["bboxes"], compression="gzip")

        print(f"HDF5ファイルにデータを保存しました: {self.output_h5_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert processed TFRecords to HDF5 format")
    parser.add_argument("-i", "--input", required=True, help="Directory containing processed TFRecord files")
    parser.add_argument("-o", "--output", required=True, help="Output HDF5 file path")

    args = parser.parse_args()

    converter = TFRecordToH5Converter(args.input, args.output)
    converter.convert_to_h5()
