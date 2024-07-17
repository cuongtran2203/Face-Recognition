import pytorch_lightning as pl
import torch
import cv2
import numpy as np
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
import pytorch_lightning as pl



class Combineloss(torch.nn.Module):
    def __init__(self,alpha=0.4) -> None:
        super(Combineloss,self).__init__()
        self.fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE)
        self.bn = torch.nn.BCEWithLogitsLoss()
        self.alpha = alpha
    def forward(self,y,y_pred):
        total_loss = self.alpha * self.fn(y_pred,y) + (1-self.alpha) * self.bn(y_pred.sigmoid(),y.sigmoid())
        return total_loss
    
class Segmentation_model(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes
        )

        # # preprocessing parameteres for image
        # params = smp.encoders.get_preprocessing_params(encoder_name)
        # self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.save_hyperparameters()
        # for image segmentation dice loss could be the best first choice
        self.loss_fn = Combineloss(alpha=0.4)
        

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std
       
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        # print(batch['image'])
        image,label = batch
        # print("mask shape:",mask.shape)
        # print("image shape:",image.shape)
        logits_mask = self.forward(image)
        
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)
        

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
       
        
     
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dataset_recall = smp.metrics.recall(tp,fp,fn,tn,reduction="micro")
        dataset_p = smp.metrics.precision(tp,fp,fn,tn,reduction="micro")
        dataset_acc = smp.metrics.accuracy(tp,fp,fn,tn,reduction="micro")
        

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_dataset_recall": dataset_recall,
            f"{stage}_dataset_p": dataset_p,
            f"{stage}_dataset_acc": dataset_acc
            
            
            
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)