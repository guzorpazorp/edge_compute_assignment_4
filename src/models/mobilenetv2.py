"""MobileNetV2 (Sandler et al., 2018), CIFAR-adapted.

Changes vs. ImageNet version: stem stride 1 and the first stride-2 stage (24ch) uses
stride 1, giving 32 -> 16 -> 8 -> 4 spatial resolution instead of collapsing 32x32 to 1x1.
KD feature tap: output of the 96-channel stage (8x8).
"""
import torch.nn as nn


def conv_bn(in_ch, out_ch, k, stride=1, groups=1, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, stride, k // 2, groups=groups, bias=False),
              nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        layers = [conv_bn(in_ch, hidden, 1)] if expand != 1 else []
        layers += [conv_bn(hidden, hidden, 3, stride, groups=hidden),  # depthwise
                   conv_bn(hidden, out_ch, 1, act=False)]               # linear bottleneck
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class MobileNetV2(nn.Module):
    # (expansion t, out channels c, repeats n, first stride s)
    CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
    feat_channels = 96

    def __init__(self, num_classes=100, dropout=0.2):
        super().__init__()
        self.stem = conv_bn(3, 32, 3, 1)
        in_ch, stages = 32, []
        for t, c, n, s in self.CFG:
            blocks = [InvertedResidual(in_ch if i == 0 else c, c, s if i == 0 else 1, t) for i in range(n)]
            stages.append(nn.Sequential(*blocks))
            in_ch = c
        self.low = nn.Sequential(*stages[:5])   # up to the 96ch stage -> feature tap
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


def _init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
