import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
   

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False, groups=1):
        super(Bottleneck, self).__init__()
        stride = 2 if down_sample else 1
        mid_channels = out_channels // 4
        self.shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=stride, activation=False) if in_channels != out_channels else nn.Identity()
        self.conv = nn.Sequential(*[
            Conv(in_channels, mid_channels, kernel_size=1, stride=1),
            Conv(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=groups),
            Conv(mid_channels, out_channels, kernel_size=1, stride=1, activation=False)
        ])

    def forward(self, x):
        y = self.conv(x) + self.shortcut(x)
        return F.relu(y, inplace=True)


class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.stem = nn.Sequential(*[
            Conv(3, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.stages = nn.Sequential(*[
            self._make_stage(64, 256, down_sample=False, num_blocks=3),
            self._make_stage(256, 512, down_sample=True, num_blocks=4),
            self._make_stage(512, 1024, down_sample=True, num_blocks=6),
            self._make_stage(1024, 2048, down_sample=True, num_blocks=3)
        ])
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.flattern = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(2048, num_classes)


    @staticmethod
    def _make_stage(in_channels, out_channels, down_sample, num_blocks):
        layers = [Bottleneck(in_channels, out_channels, down_sample=down_sample)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels, out_channels, down_sample=False))
        return nn.Sequential(*layers)
    
    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        features = self.flatten(x)
        if return_features:
            return features
        else:
            x = self.fc(features)
            return x
    
class FusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FusionModel, self).__init__()
        self.resnet_meander = Resnet50(num_classes=num_classes)
        self.resnet_spiral = Resnet50(num_classes=num_classes)
        
        for param in self.resnet_meander.parameters():
            param.requires_grad = False
        for param in self.resnet_spiral.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x_meander, x_spiral):
        features_meander = self.resnet_meander(x_meander, return_features=True)
        features_spiral = self.resnet_spiral(x_spiral, return_features=True)
        
        fused_features = torch.cat((features_meander, features_spiral), dim=1)
        
        output = self.classifier(fused_features)
        return output
