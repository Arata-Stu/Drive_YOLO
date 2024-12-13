import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..build import build_backbone, build_head, build_neck

class YOLOXDetector(nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        neck_cfg = model_cfg.neck
        head_cfg = model_cfg.head

        self.backbone = build_backbone(backbone_cfg)

        in_channels = self.backbone.get_stage_dims(neck_cfg.in_stages)
        print('inchannels:', in_channels)
        self.neck = build_neck(neck_cfg, in_channels=in_channels)
        
        strides = self.backbone.get_strides(neck_cfg.in_stages)
        print('strides:', strides)
        self.head = build_head(head_cfg, in_channels=in_channels, strides=strides)

    def forward(self, x, targets=None):
        backbone_features = self.backbone(x)
        neck_outs = self.neck(backbone_features)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                neck_outs, targets)
            
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            outputs = outputs["total_loss"]
        else:
            outputs = self.head(neck_outs)
      
        return outputs