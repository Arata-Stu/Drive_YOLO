from detector.yolox import YOLOXDetector

def get_detector(model_cfg):
    return YOLOXDetector(model_cfg)