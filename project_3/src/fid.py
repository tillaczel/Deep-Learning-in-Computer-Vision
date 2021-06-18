import torch
from torchmetrics import Metric

# def calculate_fid(act1, act2):
#     # calculate mean and covariance statistics
#     mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
#     mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
#     # calculate sum squared difference between means
#     ssdiff = torch.sum((mu1 - mu2)**2.0)
#     # calculate sqrt of product between cov
#     covmean = sqrtm(sigma1.dot(sigma2))
#     # check and correct imaginary numbers from sqrt
#     if iscomplexobj(covmean):
#         covmean = covmean.real
#     # calculate score
#     fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
#     return fid

class FrechetInceptionDistance(Metric):
    def __init__(self, inception, inception_normalize, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.inception = inception
        self.inception_normalize = inception_normalize
        self.add_state("fid_sum", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        org = self.inception_normalize(target)
        org = self.inception(org)

        pred = self.inception_normalize(preds)
        pred = self.inception(pred)

        print('mu_pred', torch.mean(pred))
        print('mu_org', torch.mean(org))
        # TODO: covariance??

        self.total += target.shape[0]

    def compute(self):
        return self.fid_sum / self.total