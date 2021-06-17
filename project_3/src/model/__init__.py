from torch import nn

from project_2.src.model.unet_parts import DoubleConv, Down, Up, OutConv

def dropout_on(layer):
    children = layer.children()
    # recurse over all children
    for child in children:
        dropout_on(child)

    if isinstance(layer, (nn.Dropout,
                          nn.Dropout2d,
                          nn.Dropout3d,
                          nn.AlphaDropout)):
        layer.train()


class Model(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_rate=0, bilinear=True):
        super(Model, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate

        self.inc = DoubleConv(n_channels, 64, dropout_rate=dropout_rate)
        self.down1 = Down(64, 128, dropout_rate=dropout_rate)
        self.down2 = Down(128, 256, dropout_rate=dropout_rate)
        self.down3 = Down(256, 512, dropout_rate=dropout_rate)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout_rate=dropout_rate)
        self.up1 = Up(1024, 512 // factor, bilinear=bilinear, dropout_rate=dropout_rate)
        self.up2 = Up(512, 256 // factor, bilinear=bilinear, dropout_rate=dropout_rate)
        self.up3 = Up(256, 128 // factor, bilinear=bilinear, dropout_rate=dropout_rate)
        self.up4 = Up(128, 64, bilinear=bilinear, dropout_rate=dropout_rate)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def eval_with_dropout(self):
        self.eval()
        dropout_on(self)