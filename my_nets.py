import torch
import torch.nn as nn
import torchvision


class ResXF(nn.Module):
    def __init__(self, qr='qr_label_456.tensor', device=None):
        super(ResXF, self).__init__()
        self.qr = torch.load(qr).float().to(device)
        resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        self.FR1 = nn.Sequential(nn.Conv2d(2048, 1024, 3, bias=False), nn.BatchNorm2d(1024))
        self.FR2 = nn.Sequential(nn.Conv2d(1024, 2048, 3, bias=False), nn.BatchNorm2d(2048))
        self.end = nn.Sequential(nn.Conv2d(2048, 912, 1, bias=False), nn.BatchNorm2d(912))
        self.CR1 = nn.Sequential(nn.Conv2d(2048, 1024, 3, bias=False), nn.BatchNorm2d(1024))
        self.CR2 = nn.Sequential(nn.Conv2d(1024, 2048, 3, bias=False), nn.BatchNorm2d(2048))
        self.CB = nn.Sequential(nn.Conv2d(2048, 2048, 1, bias=False), nn.BatchNorm2d(2048), nn.ReLU(),
                                nn.Conv2d(2048, 1024, 3, bias=False), nn.BatchNorm2d(1024), nn.ReLU(),
                                nn.AdaptiveMaxPool2d((1, 1)), nn.Conv2d(1024, 456, 1))
        # self.CB2 = nn.Sequential(nn.Conv2d(456, 1024, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(),
        #                          nn.Conv2d(1024, 456, 1))
        self.CB2 = nn.Sequential(nn.Linear(456, 1024), nn.ReLU(),
                                 nn.Linear(1024, 456))

    def forward(self, x):
        x = self.features(x)
        x2 = self.CR1(x)
        x = self.FR1(x) + x2
        x2 = self.CR2(x2)
        x = self.FR2 + x2
        y = self.end(x)
        y = y.view(-1, 456, 128) * self.qr
        y = torch.sum(y, dim=2)
        z = (self.CB(x)).view(-1, 456) + self.CB2(y)
        # x = self.ResX(x)
        # y = self.FC(x) + 0.1 * x
        return y, z