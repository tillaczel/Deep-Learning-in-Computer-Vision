import torchmetrics
import pytorch_lightning as pl
from collections import Iterable

from project_2.src.metrics.accuracy import Accuracy
from project_2.src.metrics.dice import Dice
from project_2.src.metrics.iou import IoU
from project_2.src.metrics.precision import Precision
from project_2.src.metrics.sensitivity import Sensitivity
from project_2.src.metrics.specificity import Specificity


class Metrics(pl.LightningModule):
    def __init__(self, main_metrics: Iterable = ("sensitivity", "specificity", "iou", "dice", "acc")):
        super().__init__()
        # TODO: instance average those?
        self.train_acc = Accuracy(multiclass=False)
        self.val_acc = Accuracy(multiclass=False)

        self.train_sensitivity = Sensitivity(multiclass=False)
        self.val_sensitivity = Sensitivity(multiclass=False)
        self.train_specificity = Specificity(multiclass=False)
        self.val_specificity = Specificity(multiclass=False)
        self.train_precision = Precision(multiclass=False)
        self.val_precision = Precision(multiclass=False)

        self.train_iou = IoU()
        self.val_iou = IoU()

        self.train_dice = Dice()
        self.val_dice = Dice()

        self.metrics = ["acc", "sensitivity", "specificity", "iou", "dice", "precision"]
        self.main_metrics = main_metrics


def calc_all_metrics(probs, labels, mode='train'):
    metrics = Metrics()
    results = dict()
    for metric_name in metrics.metrics:
        metric = getattr(metrics, f"{mode}_{metric_name}")
        results[metric_name] = float(metric(probs, labels).cpu().numpy())
    del metrics
    return results


