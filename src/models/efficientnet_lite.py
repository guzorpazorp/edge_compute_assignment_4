"""EfficientNet-Lite0, CIFAR-adapted.

Lite variants (vs. EfficientNet-B0): no squeeze-and-excitation, ReLU6 instead of Swish,
and fixed stem (32ch) / head (1280ch) when scaling. Lite0 = width 1.0, depth 1.0.
CIFAR changes: stem stride 1 and the first stride-2 stage (24ch) uses stride 1,
giving 32 -> 16 -> 8 -> 4. KD feature tap: output of the 112-channel stage (8x8).
"""
import torch.nn as nn

from src.models.mobilenetv2 import _init_weights, conv_bn


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride, expand):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        layers = [conv_bn(in_ch, hidden, 1)] if expand != 1 else []
        layers += [conv_bn(hidden, hidden, k, stride, groups=hidden),
                   conv_bn(hidden, out_ch, 1, act=False)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class EfficientNetLite0(nn.Module):
    # (expansion, kernel, first stride, out channels, repeats)
    CFG = [(1, 3, 1, 16, 1), (6, 3, 1, 24, 2), (6, 5, 2, 40, 2), (6, 3, 2, 80, 3),
           (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), (6, 3, 1, 320, 1)]
    feat_channels = 112

    def __init__(self, num_classes=100, dropout=0.2):
        super().__init__()
        self.stem = conv_bn(3, 32, 3, 1)
        in_ch, stages = 32, []
        for e, k, s, c, n in self.CFG:
            stages.append(nn.Sequential(*[MBConv(in_ch if i == 0 else c, c, k, s if i == 0 else 1, e)
                                          for i in range(n)]))
            in_ch = c
        self.low = nn.Sequential(*stages[:5])
        self.high = nn.Sequential(*stages[5:])
        self.head = conv_bn(320, 1280, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
        _init_weights(self)

    def forward(self, x, return_feats=False):
        feat = self.low(self.stem(x))
        x = self.pool(self.head(self.high(feat))).flatten(1)
        logits = self.classifier(x)
        return (logits, feat) if return_feats else logits
