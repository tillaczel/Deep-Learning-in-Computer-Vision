import torch
from torchmetrics import Metric

def calculate_iou(preds, target, threshold=0.5, spatial_dim=(2,3)):
    preds_binary = preds >= threshold
    intersection = torch.sum((preds_binary.bool() & target.bool()).int(), dim=spatial_dim)
    union = torch.sum((preds_binary.bool() | target.bool()).int(), dim=spatial_dim)
    iou_per_sample = (intersection + 1e-10) / (union + 1e-10)  # case 0/0 -> 1/1
    return iou_per_sample

class IoU(Metric):
    def __init__(self, average='macro', multiclass=False, num_labels=1,
                 threshold=0.5, label_dim=1, spatial_dim=(2,3), dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.threshold = threshold
        assert average in ['macro', 'none']
        self.average = average # macro or none
        self.label_dim = label_dim
        self.spatial_dim = spatial_dim
        self.multiclass = multiclass
        self.num_classes = num_labels
        self.add_state("iou_sum", default=torch.zeros(num_labels), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        # assert preds.shape == target.shape
        iou_per_sample = calculate_iou(preds, target, threshold=self.threshold, spatial_dim=self.spatial_dim)
        self.iou_sum += torch.sum(iou_per_sample, dim=0) # sum over batch
        self.total += target.shape[0]

    def compute(self):
        if self.average == 'macro':
            return torch.mean(self.iou_sum) / self.total
        else:
            return self.iou_sum / self.total