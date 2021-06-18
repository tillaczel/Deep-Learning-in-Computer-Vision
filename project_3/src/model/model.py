import torch
import torch.nn as nn
import torch.nn.functional as F

norm_layer = nn.InstanceNorm2d


class ResBlock(nn.Module):
    def __init__(self, f):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(f, f, 3, 1, 1), norm_layer(f), nn.ReLU(),
                                  nn.Conv2d(f, f, 3, 1, 1))
        self.norm = norm_layer(f)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x) + x))


class Generator(nn.Module):
    def __init__(self, f=64, blocks=6):
        super(Generator, self).__init__()
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(3, f, 7, 1, 0), norm_layer(f), nn.ReLU(True),
                  nn.Conv2d(f, 2 * f, 3, 2, 1), norm_layer(2 * f), nn.ReLU(True),
                  nn.Conv2d(2 * f, 4 * f, 3, 2, 1), norm_layer(4 * f), nn.ReLU(True)]
        for i in range(int(blocks)):
            layers.append(ResBlock(4 * f))
        layers.extend([  # Uses a subpixel convolution (PixelShuffle) for upsamling
            nn.ConvTranspose2d(4 * f, 4 * 2 * f, 3, 1, 1), nn.PixelShuffle(2), norm_layer(2 * f), nn.ReLU(True),
            nn.ConvTranspose2d(2 * f, 4 * f, 3, 1, 1), nn.PixelShuffle(2), norm_layer(f), nn.ReLU(True),
            nn.ReflectionPad2d(3), nn.Conv2d(f, 3, 7, 1, 0),
            nn.Tanh()])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            #nn.InstanceNorm2d(128), #https://github.com/Lornatang/CycleGAN-PyTorch/blob/master/cyclegan_pytorch/models.py doesnt use instancenorm everywhere unlike mortens slides
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        x = self.disc(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x
