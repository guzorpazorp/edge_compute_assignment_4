"""Automated architecture generator: architecture dict (see space.py) -> nn.Module."""
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
}


class ConvBlock(nn.Module):
    """Standard kxk conv -> BN -> act."""

    def __init__(self, in_ch, out_ch, k, stride, act):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(in_ch, out_ch, k, stride, k // 2, bias=False),
                                  nn.BatchNorm2d(out_ch), ACTIVATIONS[act]())
        self.use_res = stride == 1 and in_ch == out_ch

    def forward(self, x):
        return x + self.body(x) if self.use_res else self.body(x)


class DWSepBlock(nn.Module):
    """Depthwise separable: depthwise kxk -> BN -> act -> pointwise 1x1 -> BN -> act."""

    def __init__(self, in_ch, out_ch, k, stride, act):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, stride, k // 2, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), ACTIVATIONS[act](),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), ACTIVATIONS[act](),
        )
        self.use_res = stride == 1 and in_ch == out_ch

    def forward(self, x):
        return x + self.body(x) if self.use_res else self.body(x)


BLOCKS = {"conv": ConvBlock, "dwsep": DWSepBlock}


class NASNet(nn.Module):
    def __init__(self, arch, num_classes=10):
        super().__init__()
        self.arch = arch
        first_act = arch["stages"][0]["act"]
        ch = arch["stem_channels"]
        layers = [nn.Conv2d(3, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch), ACTIVATIONS[first_act]()]
        for st in arch["stages"]:
            for b in range(st["blocks"]):
                layers.append(BLOCKS[st["block"]](ch, st["channels"], st["kernel"],
                                                  st["stride"] if b == 0 else 1, st["act"]))
                ch = st["channels"]
        if arch["head_channels"]:
            last_act = arch["stages"][-1]["act"]
            layers += [nn.Conv2d(ch, arch["head_channels"], 1, bias=False),
                       nn.BatchNorm2d(arch["head_channels"]), ACTIVATIONS[last_act]()]
            ch = arch["head_channels"]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


def build_model(arch, num_classes=10):
    return NASNet(arch, num_classes)
