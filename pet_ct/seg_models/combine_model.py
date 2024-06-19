import torch

from unet_parts import *

from kitenet_model import KiteNet
from unet_model import UNet


class Comebined(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, bilinear=False):
        super(Comebined, self).__init__()
        factor = 2 if bilinear else 1
        self.model1 = KiteNet(n_channels, n_classes)
        self.model2 = UNet(n_channels, n_classes)
        self.up1 = (Up(64, 64 // factor, bilinear))
        self.up2 = (Up(64, 64 // factor, bilinear))
        self.outc = (OutConv(128, n_classes))

    def forward(self, x1, x2):
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x2 = self.up1(x2)
        x2 = self.up2(x2)
        x = torch.cat([x1, x2], 1)
        logits = self.outc(x)

        return logits

# device = torch.device('cpu')
# model = Comebined().to(device)
# x1 = torch.randn(1,1,128,128).to(device)
# x2 = torch.randn(1,1,512,512).to(device)
# y = model(x1,x2)
# print(y)

