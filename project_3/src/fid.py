import torch
from torchmetrics import Metric

# def calculate_fid(act1, act2):
#     # calculate mean and covariance statistics
#     mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
#     mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
#     # calculate sum squared difference between means
#     ssdiff = numpy.sum((mu1 - mu2)**2.0)
#     # calculate sqrt of product between cov
#     covmean = sqrtm(sigma1.dot(sigma2))
#     # check and correct imaginary numbers from sqrt
#     if iscomplexobj(covmean):
#         covmean = covmean.real
#     # calculate score
#     fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
#     return fid
#
# class FrechetInceptionDistance(Metric):
#     def __init__(self, dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)
#
#         self.add_state("fid_sum", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
#
#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         # preds, target = self._input_format(preds, target)
#         # assert preds.shape == target.shape
#         accuracy_per_sample = calculate_fid(preds, target, threshold=self.threshold, spatial_dim=self.spatial_dim)
#         self.accuracy_sum += torch.sum(accuracy_per_sample, dim=0) # sum over batch
#         self.total += target.shape[0]
#
#     def compute(self):
#         return self.fid_sum / self.total