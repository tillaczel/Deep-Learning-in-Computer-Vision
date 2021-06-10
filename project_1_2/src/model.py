import torchvision.models as models
from torch.nn import Module, Sequential, Conv2d


class Model(Module):

    def __init__(self, pretrained: bool = False, in_dim: int = 2048, out_dim: int = 256):
        super(Model, self).__init__()
        # self.resnet = Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-1])
        self.resnet = Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-2]) # maybe ?
        self.conv = Conv2d(in_dim, out_dim, (1, 1), bias=True)

    def forward(self, x):
        x = self.resnet(x)
        x = self.conv(x)
        return x
