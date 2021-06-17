import torchmetrics
import pytorch_lightning as pl


class Metrics(pl.LightningModule):
    def __init__(self, main_metrics: list = None):
        super().__init__()
        self.main_metrics = main_metrics


def calc_all_metrics(probs, labels, mode='train'):
    metrics = Metrics()
    results = dict()
    for metric_name in metrics.metrics:
        metric = getattr(metrics, f"{mode}_{metric_name}")
        results[metric_name] = float(metric(probs, labels).cpu().numpy())
    return results


