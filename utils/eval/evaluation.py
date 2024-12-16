from .io.box_filtering import filter_boxes
from .metrics.coco_eval import evaluate_detection


def evaluate_list(result_boxes_list,
                  gt_boxes_list,
                  height: int,
                  width: int,
                  camera: str = 'waymo',
                  return_aps: bool = True):
    
    assert camera in {'waymo'}

    if camera == 'waymo':
        classes = ("vehicle", "pedestrian", "cyclist")
    else:
        raise NotImplementedError
    
    min_box_diag_map = {'waymo' : 0}
    min_box_side_map = {'waymo' : 0}

    min_box_diag = min_box_diag_map[camera]
    min_box_side = min_box_side_map[camera]
    
    filter_boxes_fn = lambda x: filter_boxes(x, min_box_diag, min_box_side)

    print("filtering boxes")
    gt_boxes_list = map(filter_boxes_fn, gt_boxes_list)
    # NOTE: We also filter the prediction to follow the prophesee protocol of evaluation.
    result_boxes_list = map(filter_boxes_fn, result_boxes_list)

    print("evaluating")

    return evaluate_detection(gt_boxes_list, result_boxes_list,
                              height=height, width=width,
                              classes=classes, return_aps=return_aps)
