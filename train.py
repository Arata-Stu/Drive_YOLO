import sys
sys.path.append('./../../')

import yaml
from omegaconf import OmegaConf
from modules.fetch import fetch_data_module, fetch_model_module

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
import os
import datetime

def main(merged_conf):
    base_save_dir = './result'
    
    # 実行時のタイムスタンプを付与して、一意のディレクトリ名を生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # dataset.name, model.name, ev_representation, ev_delta_t を取得
    dataset_name = merged_conf.dataset.name
    model_name = merged_conf.model.name
    
    # ディレクトリの階層構造を作成
    save_dir = os.path.join(base_save_dir, dataset_name, model_name, timestamp, 'train')
    os.makedirs(save_dir, exist_ok=True)  # 保存ディレクトリを作成
    
    # 統合されたconfigを保存
    merged_config_path = os.path.join(save_dir, 'merged_config.yaml')
    with open(merged_config_path, 'w') as f:
        yaml.dump(OmegaConf.to_container(merged_conf, resolve=True), f)
    
    # データモジュールとモデルモジュールのインスタンスを作成
    data = fetch_data_module(merged_conf)
    data.setup('fit')
    model = fetch_model_module(merged_conf)
    model.setup('fit')
    
    # コールバックの設定
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir,
            filename='{epoch:02d}-{val_AP:.2f}',
            monitor='val_AP',
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # TensorBoard Loggerもsave_dirに対応させる
    logger = pl_loggers.TensorBoardLogger(
        save_dir=save_dir,
        name='',
        version='',
    )

    # トレーナーを設定
    train_cfg = merged_conf.experiment.training
    trainer = pl.Trainer(
        max_epochs=train_cfg.max_epochs,
        max_steps=train_cfg.max_steps,
        logger=logger,
        callbacks=callbacks,
        accelerator='gpu',
        precision=train_cfg.precision,
        devices=[0],
        benchmark=True,
        profiler='simple',
    )

    # モデルの学習を実行
    trainer.fit(model, datamodule=data)

if __name__ == '__main__':
    import argparse

    # コマンドライン引数から設定ファイルのパスを取得し、事前にマージして `merged_conf` を作成
    parser = argparse.ArgumentParser(description='Train a model with specified YAML config files')
    parser.add_argument('--model', type=str, required=True, help='Path to model configuration file')
    parser.add_argument('--exp', type=str, required=True, help='Path to experiment configuration file')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset configuration file')

    args = parser.parse_args()

    # 個別のconfigファイルをロードしてマージ
    model_config = OmegaConf.load(args.model)
    exp_config = OmegaConf.load(args.exp)
    dataset_config = OmegaConf.load(args.dataset)
    merged_conf = OmegaConf.merge(model_config, exp_config, dataset_config)

    # 統合された `merged_conf` を `main` に渡す
    main(merged_conf)
