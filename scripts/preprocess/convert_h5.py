import os
import h5py
import tensorflow as tf
from tqdm import tqdm
from multiprocessing import Pool, get_context

class TFRecordToH5Converter:
    def __init__(self, input_dir, output_dir):
        """
        初期化: 入力ディレクトリと出力HDF5ディレクトリを設定します。

        Args:
            input_dir (str): TFRecordファイルが格納されたディレクトリ。
            output_dir (str): HDF5ファイルの保存先ディレクトリ。
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

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

    def process_sequence(self, args):
        """
        1つのTFRecordファイルを処理してHDF5ファイルに変換します。

        Args:
            args (tuple): (tfrecord_file, output_subdir) のタプル。

        Returns:
            None
        """
        tfrecord_file, output_subdir = args
        sequence_id = os.path.basename(tfrecord_file).split(".")[0]
        output_path = os.path.join(output_subdir, f"{sequence_id}.h5")

        # HDF5に変換
        with h5py.File(output_path, "w") as h5_file:
            dataset = tf.data.TFRecordDataset(tfrecord_file)
            for idx, raw_record in enumerate(dataset):
                data = self.parse_tfrecord(raw_record.numpy())
                frame_id = f"frame_{idx:04d}"

                # フレームごとに保存
                frame_group = h5_file.create_group(frame_id)
                frame_group.create_dataset("image", data=data["image"], compression="gzip")
                frame_group.create_dataset("bboxes", data=data["bboxes"], compression="gzip")

        print(f"HDF5ファイルを保存しました: {output_path}")

    def convert_to_h5(self, num_workers):
        """
        TFRecordからHDF5への変換を並列処理で行います。

        Args:
            num_workers (int): 並列処理に使用するワーカー数。
        """
        subsets = ["train", "val", "test"]
        tasks = []

        # 各サブセットの処理
        for subset in subsets:
            input_subdir = os.path.join(self.input_dir, subset)
            output_subdir = os.path.join(self.output_dir, subset)
            os.makedirs(output_subdir, exist_ok=True)

            if not os.path.exists(input_subdir):
                print(f"Warning: {input_subdir} does not exist. Skipping.")
                continue

            # 各ファイルを処理タスクに追加
            tfrecord_files = [
                os.path.join(input_subdir, f) for f in os.listdir(input_subdir) if f.endswith(".tfrecord")
            ]
            tasks.extend([(file, output_subdir) for file in tfrecord_files])

        # 並列処理でシーケンスを処理
        with get_context("spawn").Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap(self.process_sequence, tasks), total=len(tasks), desc="Processing sequences"))


if __name__ == "__main__":
    import argparse
    import multiprocessing

    # `spawn` モードを明示的に設定
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Convert processed Waymo TFRecords to HDF5 format (mode-based)")
    parser.add_argument("-i", "--input", required=True, help="Directory containing processed TFRecord files")
    parser.add_argument("-o", "--output", required=True, help="Directory to save HDF5 files")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(), help="Number of parallel workers (default: CPU count)")

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    num_workers = args.workers

    converter = TFRecordToH5Converter(input_dir, output_dir)
    converter.convert_to_h5(num_workers)
