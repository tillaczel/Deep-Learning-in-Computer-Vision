import torch
from torchmetrics import Metric

def calculate_dice(preds, target, threshold=0.5, spatial_dim=(2,3)):
    # return (batch_size - 2 * torch.sum((y_real * sig_pred + eps) / (y_real + sig_pred + eps)) / 65536) / batch_size
    preds_binary = preds >= threshold
    target = target >= threshold
    intersection = torch.sum((preds_binary.bool() & target.bool()).int(), dim=spatial_dim)
    summation = torch.sum(preds_binary.bool().int(), dim=spatial_dim) + torch.sum(target.bool().int(), dim=spatial_dim)
    dice_per_sample = (2 * intersection + 1e-10) / (summation + 1e-10)  # case 0/0 -> 1/1
    return dice_per_sample

class Dice(Metric):
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
        self.add_state("dice_sum", default=torch.zeros(num_labels), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        dice_per_sample = calculate_dice(preds, target, threshold=self.threshold, spatial_dim=self.spatial_dim)
        self.dice_sum += torch.sum(dice_per_sample, dim=0) # sum over batch
        self.total += target.shape[0] # batch

    def compute(self):
        if self.average == 'macro':
            return torch.mean(self.dice_sum) / self.total
        else:
            return self.dice_sum / self.total