import torch
from torchmetrics import Metric

def calculate_accuracy(preds, target, threshold=0.5, spatial_dim=(2,3)):
    preds_binary = preds >= threshold
    target = target >= threshold
    intersection = torch.sum((preds_binary.bool() & target.bool()).int(), dim=spatial_dim)
    union = torch.sum((preds_binary.bool() | target.bool()).int(), dim=spatial_dim)
    accuracy_per_sample = (intersection + 1e-10) / (union + 1e-10)  # case 0/0 -> 1/1
    return accuracy_per_sample

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
        self.add_state("accuracy_sum", default=torch.zeros(num_labels), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        # assert preds.shape == target.shape
        accuracy_per_sample = calculate_accuracy(preds, target, threshold=self.threshold, spatial_dim=self.spatial_dim)
        self.accuracy_sum += torch.sum(accuracy_per_sample, dim=0) # sum over batch
        self.total += target.shape[0]

    def compute(self):
        if self.average == 'macro':
            return torch.mean(self.accuracy_sum) / self.total
        else:
            return self.accuracy_sum / self.total