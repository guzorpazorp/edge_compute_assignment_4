"""SqueezeNet 1.1 (Iandola et al., 2016), CIFAR-adapted.

Changes vs. original: 3x3 stride-1 stem, BatchNorm inside Fire modules (plain SqueezeNet is
hard to train from scratch at CIFAR learning rates) and no ReLU on the final logits
(non-negative logits interfere with softmax/KD). Pooling gives 32 -> 16 -> 8 -> 4.
KD feature tap: output of fire5 (256ch, 8x8).
"""
import torch
import torch.nn as nn

from src.models.mobilenetv2 import _init_weights


def conv_bn_relu(in_ch, out_ch, k):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class Fire(nn.Module):
    def __init__(self, in_ch, squeeze, expand1x1, expand3x3):
        super().__init__()
        self.squeeze = conv_bn_relu(in_ch, squeeze, 1)
        self.expand1x1 = conv_bn_relu(squeeze, expand1x1, 1)
        self.expand3x3 = conv_bn_relu(squeeze, expand3x3, 3)

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)


class SqueezeNet(nn.Module):
    feat_channels = 256

    def __init__(self, num_classes=100, dropout=0.5):
        super().__init__()
        self.low = nn.Sequential(
            conv_bn_relu(3, 64, 3),                                   # 32x32
            nn.MaxPool2d(3, 2, 1),                                    # 16x16
            Fire(64, 16, 64, 64), Fire(128, 16, 64, 64),
            nn.MaxPool2d(3, 2, 1),                                    # 8x8
            Fire(128, 32, 128, 128), Fire(256, 32, 128, 128),        # -> feature tap
        )
        self.high = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),                                    # 4x4
            Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256), Fire(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Conv2d(512, num_classes, 1),
                                        nn.AdaptiveAvgPool2d(1), nn.Flatten())
        _init_weights(self)

    def forward(self, x, return_feats=False):
        feat = self.low(x)
        logits = self.classifier(self.high(feat))
        return (logits, feat) if return_feats else logits
