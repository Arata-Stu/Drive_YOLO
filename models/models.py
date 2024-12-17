import os
import torch
from .detector.yolox import YOLOXDetector

def get_detector(model_cfg):
    model = YOLOXDetector(model_cfg)

    if model_cfg.pretrained:
        # チェックポイントファイルのパスを取得
        ckpt_path = os.path.join('./config/model/', model_cfg.weight)
        ckpt = torch.load(ckpt_path)
        
        # ヘッド部分のキーを除外
        model_state_dict = ckpt['model']
        backbone_neck_state_dict = {
            k: v for k, v in model_state_dict.items() 
            if not k.startswith('head')  # head部分のキーをフィルタリング
        }
        
        # 残った部分だけを読み込む
        model.load_state_dict(backbone_neck_state_dict, strict=False)
    
    return model
