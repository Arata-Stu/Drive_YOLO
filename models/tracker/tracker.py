import torch
from omegaconf import DictConfig
from functools import partial
from argparse import Namespace

from models.yolox.utils.boxes import postprocess
from ..ByteTrack.tracker.byte_tracker import BYTETracker

class Tracker:

    def __init__(self, full_config: DictConfig):
        """
        Initialize the tracker
        Args:
            track_thresh
            track_buffer
        """

        self.info_imgs = [640, 640]
        args = Namespace()
        args.track_thresh = 0.6
        args.track_buffer = 30

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.tracker = BYTETracker(args=args, frame_rate=30)

        self.post_process = partial(postprocess,
                                    num_classes=full_config.model.head.num_classes,
                                    conf_thre=full_config.model.postprocess.conf_thre,
                                    nms_thre=full_config.model.postprocess.nms_thre,
                                    class_agnostic=False)

    def track(self, model, dataloader):
        model.to(self.device)
        model.eval()

        with torch.no_grad():

            for batch_idx, batch in enumerate(dataloader):
                image = batch['image'].to(self.device)

                predictions = model(image)
                processed_preds = self.post_process(prediction=predictions)

                if processed_preds[0] is not None:
                    
                    online_targets = self.tracker.update(processed_preds[0], self.info_imgs, self.img_size)
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > 1.6
                        if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                    # save results
                    results.append((frame_id, online_tlwhs, online_ids, online_scores))



        pass