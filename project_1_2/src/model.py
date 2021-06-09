import torchvision.models as models
from torch.nn import Module, Sequential, Linear, Softmax


class Model(Module):

    def __init__(self, pretrained: bool = False, in_dim: int = 2048, out_dim: int = 256):
        super(Model, self).__init__()
        self.resnet = Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-1])
        self.linear = Sequential((in_features=in_dim, out_features=out_dim, bias=True), Softmax(dim=1))

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
