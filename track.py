import sys
sys.path.append('./../../')

from omegaconf import OmegaConf
from modules.fetch import fetch_data_module, fetch_model_module

import torch
import os
import argparse

def get_config_paths_from_ckpt(ckpt_path):
    # ckpt_pathからベースディレクトリを取得
    train_dir = os.path.dirname(ckpt_path)
    
    # merged_config.yaml のパスを自動で取得
    config_path = os.path.join(train_dir, "merged_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path}が見つかりません")
    
    return config_path, train_dir

def main(ckpt_path):
    # 設定ファイルとベースディレクトリを取得
    config_path, train_dir = get_config_paths_from_ckpt(ckpt_path)

    save_dir = os.path.join(train_dir, 'test')
    os.makedirs(save_dir, exist_ok=True)  # 保存ディレクトリを作成

    # YAML ファイルを読み込んで OmegaConf に変換
    merged_conf = OmegaConf.load(config_path)

    # データモジュールとモデルモジュールのインスタンスを作成
    data = fetch_data_module(merged_conf)
    data.setup('test')
    model = fetch_model_module(merged_conf)
    model.setup('test')

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])

    
    

    

    

if __name__ == '__main__':
    # argparseでコマンドライン引数を取得
    parser = argparse.ArgumentParser(description="Model testing script")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the checkpoint file")
    
    # 引数をパース
    args = parser.parse_args()

    # ckpt_pathをmainに渡す
    main(args.ckpt_path)
