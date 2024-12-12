import os
import argparse
import tensorflow as tf

def validate_tfrecord(file_path):
    """
    指定したTFRecordファイルを検証し、破損しているかどうかを確認する。
    """
    try:
        dataset = tf.data.TFRecordDataset(file_path, compression_type='')
        for _ in dataset:
            pass  # 全てのレコードを読み込む
        return True  # ファイルは正常
    except tf.errors.DataLossError:
        return False  # ファイルが破損している

def check_tfrecords(directory, memo_file="corrupted_files.txt"):
    """
    ディレクトリ内の全てのTFRecordファイルを検証し、破損ファイルを記録する。
    """
    corrupted_files = []
    total_files = 0

    # ディレクトリ内のすべてのファイルを検証
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".tfrecord"):
                total_files += 1
                file_path = os.path.join(root, file)
                print(f"Checking {file_path} ...")
                if not validate_tfrecord(file_path):
                    print(f"\033[91m[ERROR] Corrupted file: {file_path}\033[0m")
                    corrupted_files.append(file)

    # メモファイルに記録
    if corrupted_files:
        with open(memo_file, "w") as f:
            for file in corrupted_files:
                f.write(f"{file}\n")
        print(f"\033[93mCorrupted files recorded in {memo_file}\033[0m")
    else:
        print("\033[92mAll files are valid.\033[0m")

    print(f"Checked {total_files} TFRecord files in total.")
    return corrupted_files

def generate_download_script(corrupted_files, output_script="download_corrupted_files.sh"):
    """
    破損ファイルを再ダウンロードするためのスクリプトを生成する。
    """
    with open(output_script, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("gsutil -m cp -r \\\n")
        for file in corrupted_files:
            f.write(f"  \"gs://waymo_open_dataset_v_1_4_3/individual_files/validation/{file}\" \\\n")
        f.write("  ./\n")
    print(f"\033[93mDownload script generated: {output_script}\033[0m")
    print("Run the script with:\n  bash download_corrupted_files.sh")

if __name__ == "__main__":
    # コマンドライン引数を定義
    parser = argparse.ArgumentParser(description="Validate and record corrupted TFRecord files.")
    parser.add_argument(
        "-d", "--directory", required=True, help="Directory containing TFRecord files to validate."
    )
    parser.add_argument(
        "-m", "--memo_file", default="corrupted_files.txt", help="File to record corrupted TFRecord files."
    )
    parser.add_argument(
        "-s", "--script", default="download_corrupted_files.sh", help="Script to download corrupted files."
    )
    args = parser.parse_args()

    # ディレクトリを検証
    if not os.path.isdir(args.directory):
        print(f"\033[91m[ERROR] The directory '{args.directory}' does not exist.\033[0m")
        exit(1)

    # 検証を実行
    corrupted_files = check_tfrecords(args.directory, args.memo_file)

    # ダウンロードスクリプトを生成
    if corrupted_files:
        generate_download_script(corrupted_files, args.script)
