import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, use_BN=False):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_BN:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, use_BN=False):
        super(DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_BN=use_BN)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, use_BN=False):
        super(UpSample, self).__init__()
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_BN)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_BN=use_BN)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diff_x = x2.size()[3] - x1.size()[3]
        diff_y = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_BN=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, use_BN=use_BN)
        self.down1 = DownSample(64, 128, use_BN=use_BN)
        self.down2 = DownSample(128, 256, use_BN=use_BN)
        self.down3 = DownSample(256, 512, use_BN=use_BN)
        factor = 2 if bilinear else 1
        self.down4 = DownSample(512, 1024 // factor, use_BN=use_BN)
        self.up1 = UpSample(1024, 512 // factor, bilinear, use_BN=use_BN)
        self.up2 = UpSample(512, 256 // factor, bilinear, use_BN=use_BN)
        self.up3 = UpSample(256, 128 // factor, bilinear, use_BN=use_BN)
        self.up4 = UpSample(128, 64, bilinear, use_BN=use_BN)
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
        return self.outc(x)


if __name__ == '__main__':
    X = torch.randn((1, 1, 200, 200))
    print(X.shape)
    model = UNet(1, 1, False)
    print(model(X).shape)
    print(summary(model, input_size=(1, 200, 200), device='cpu'))
    model = UNet(1, 1, False, use_BN=True)
    print(summary(model, input_size=(1, 200, 200), device='cpu'))


