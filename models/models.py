import os 
import torch
from .detector.yolox import YOLOXDetector

def get_detector(model_cfg):
    model = YOLOXDetector(model_cfg)

    if model_cfg.weight is not None:
        ckpt_path = os.path.join('./config/model/', model_cfg.weight)
        ckpt = torch.load(ckpt_path)
        model.load_weights(ckpt['model'])

    return model