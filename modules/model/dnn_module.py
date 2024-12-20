import torch
import torch.nn as nn
import pytorch_lightning as pl
from functools import partial
from omegaconf import DictConfig

from models.models import get_detector
from utils.eval.evaluator import Evaluator
from models.yolox.utils.boxes import postprocess
from utils.eval.io.box_loading import to_prophesee


class DNNModule(pl.LightningModule):

    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config

        if self.full_config.dataset.name == "waymo":
            from data.dataset.waymo.data_info import MY_CLASS 
            self.height, self.width = 640, 640
        else:
            NotImplementedError(f"Dataset {self.full_config.dataset.name} not supported.")
        
        
        self.classes = MY_CLASS  # クラスを保持
        self.model = get_detector(model_cfg=full_config.model)

        self.post_process = partial(postprocess,
                                    num_classes=full_config.model.head.num_classes,
                                    conf_thre=full_config.model.postprocess.conf_thre,
                                    nms_thre=full_config.model.postprocess.nms_thre,
                                    class_agnostic=False)


       
    def setup(self, stage):
        self.started_training = True
        
        if stage == 'fit':
            self.started_training = False
            self.val_evaluator = Evaluator(dataset=self.full_config.dataset.name)
        elif stage == 'test':
            self.test_evaluator = Evaluator(dataset=self.full_config.dataset.name)
        
        
    def forward(self, x, targets=None):
        return self.model(x, targets)
    
    def training_step(self, batch, batch_idx):
        self.started_training = True
        self.model.train()
        imgs = batch['images'].to(dtype=self.dtype)  
        labels = batch['labels'].to(dtype=self.dtype)  
        labels.requires_grad = False

        loss = self.model(imgs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def on_train_epoch_end(self):
        #訓練を再開した時の、ckptコールバックのエラーを回避するため
        # if 'val_AP' not in self.trainer.callback_metrics:
        #     self.log('val_AP', 0.0, on_epoch=True, prog_bar=True, logger=True)
        
        pass

    def validation_step(self, batch, batch_idx):
        
        self.model.eval()
        
        imgs = batch['images'].to(dtype=self.dtype)  
        labels = batch['labels'].to(dtype=self.dtype)  
        data_id = batch['unique_id']
        labels.requires_grad = False

        preds = self.model(imgs)

        processed_preds = self.post_process(prediction=preds)

        loaded_labels_proph, yolox_preds_proph = to_prophesee(loaded_label_tensor=labels, 
                                                              label_timestamps=data_id, 
                                                              yolox_pred_list=processed_preds)

        if self.started_training:
            self.val_evaluator.add_labels(loaded_labels_proph)
            self.val_evaluator.add_predictions(yolox_preds_proph)

        
        return 

    def on_validation_epoch_end(self):
        if self.started_training:
            if self.val_evaluator.has_data():
                metrics = self.val_evaluator.evaluate_buffer(img_height=self.height,
                                                            img_width=self.width)
                for k, v in metrics.items():
                    # val_APのみプログレスバーに表示する
                    prog_bar_flag = True if k == "AP" else False
                    self.log(f'val_{k}', v, on_epoch=True, prog_bar=prog_bar_flag, logger=True)
                self.val_evaluator.reset_buffer()

    def test_step(self, batch, batch_idx):
        self.model.eval()
        
        imgs = batch['images'].to(dtype=self.dtype)  
        labels = batch['labels'].to(dtype=self.dtype)  
        data_id = batch['unique_id']
        labels.requires_grad = False

        preds = self.model(imgs)

        processed_preds = self.post_process(prediction=preds)

        loaded_labels_proph, yolox_preds_proph = to_prophesee(loaded_label_tensor=labels, 
                                                              label_timestamps=data_id, 
                                                              yolox_pred_list=processed_preds)


        if self.started_training:
            self.test_evaluator.add_labels(loaded_labels_proph)
            self.test_evaluator.add_predictions(yolox_preds_proph)

        return 

    def on_test_epoch_end(self):
        if self.started_training:
            if self.test_evaluator.has_data():
                metrics = self.test_evaluator.evaluate_buffer(img_height=self.height,
                                                            img_width=self.width)
                for k, v in metrics.items():
                    # val_APのみプログレスバーに表示する
                    prog_bar_flag = True if k == "AP" else False
                    self.log(f'test_{k}', v, on_epoch=True, prog_bar=prog_bar_flag, logger=True)
                self.test_evaluator.reset_buffer()
        
        
    def configure_optimizers(self):
        lr = self.full_config.experiment.training.learning_rate
        weight_decay = self.full_config.experiment.training.weight_decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.full_config.experiment.training.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        # Here we interpret the final lr as max_lr/final_div_factor.
        # Note that Pytorch OneCycleLR interprets it as initial_lr/final_div_factor:
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}



