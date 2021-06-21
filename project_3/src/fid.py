import torch
from scipy import linalg
from torchmetrics import Metric

from torch.nn.functional import adaptive_avg_pool2d
import numpy as np


# def calculate_fid(act1, act2):
#     # https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
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

def get_activations_step(model, images):
    # todo: normalize and resize??

    with torch.no_grad():
        pred = model(images)  # [0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    # if pred.size(2) != 1 or pred.size(3) != 1:
    #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    #
    # pred = pred.squeeze(3).squeeze(2).cpu().numpy()
    pred = pred.cpu().numpy()
    return pred


def get_statistics(activations):
    act = np.concatenate(activations)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


class FrechetInceptionDistance(Metric):
    def __init__(self, inception, inception_normalize, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.inception = inception
        self.inception_normalize = inception_normalize
        self.add_state("activations_org", default=[], dist_reduce_fx="cat")
        self.add_state("activations_pred", default=[], dist_reduce_fx="cat")
        # self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        # org = self.inception_normalize(target)
        # org = self.inception(org)
        # TODO: is the model right?
        org = get_activations_step(self.inception, target)
        pred = get_activations_step(self.inception, preds)
        #
        # pred = self.inception_normalize(preds)
        # pred = self.inception(pred)

        # print('mu_pred', torch.mean(pred))
        # print('mu_org', torch.mean(org))
        # TODO: covariance??
        # self.activations_org.append(org.detach().cpu())
        # self.activations_pred.append(pred.detach().cpu())
        self.activations_org.append(org)
        self.activations_pred.append(pred)
        del org, pred

    def compute(self):
        """
        Adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
        """
        mu1, sigma1 = get_statistics(self.activations_org)
        mu2, sigma2 = get_statistics(self.activations_pred)

        eps = 1e-6

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        fid = (diff.dot(diff) + np.trace(sigma1)
               + np.trace(sigma2) - 2 * tr_covmean)

        return fid
