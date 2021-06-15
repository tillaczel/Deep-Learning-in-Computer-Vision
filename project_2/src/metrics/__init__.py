import torchmetrics
import pytorch_lightning as pl
from collections import Iterable

from project_2.src.metrics.dice import Dice
from project_2.src.metrics.iou import IoU


class Metrics(pl.LightningModule):
    def __init__(self, main_metrics: Iterable = ("sensitivity", "specificity", "iou", "dice", "acc")):
        super().__init__()
        # TODO: instance average those?
        self.train_acc = torchmetrics.Accuracy(multiclass=False)
        self.val_acc = torchmetrics.Accuracy(multiclass=False)

        self.train_sensitivity = torchmetrics.Recall(multiclass=False)
        self.val_sensitivity = torchmetrics.Recall(multiclass=False)
        self.train_specificity = torchmetrics.Specificity(multiclass=False)
        self.val_specificity = torchmetrics.Specificity(multiclass=False)

        self.train_iou = IoU()
        self.val_iou = IoU()

        self.train_dice = Dice()
        self.val_dice = Dice()

        self.metrics = ["acc", "sensitivity", "specificity", "iou", "dice"]
        self.main_metrics = main_metrics


def calc_all_metrics(probs, labels, mode='train'):
    metrics = Metrics()
    for metric_name in metrics.metrics:
        metric = getattr(metrics, f"{mode}_{metric_name}")
        print(metric(probs, labels))

