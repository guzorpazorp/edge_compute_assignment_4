"""ResNet-50 teacher, CIFAR-adapted.

Parameter names mirror torchvision's ResNet so ImageNet weights can be loaded directly
(see load_imagenet_weights). The 7x7 stem is kept (so its pretrained weights still fit)
but with stride 1 and no max-pool, so 32x32 inputs are not downsampled 4x at the stem.

Spatial sizes for a 32x32 input: layer1 32x32 (256ch) -> layer2 16x16 (512)
-> layer3 8x8 (1024, KD feature tap) -> layer4 4x4 (2048).
"""
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, width, stride=1):
        super().__init__()
        out_ch = width * self.expansion
        self.conv1 = nn.Conv2d(in_ch, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                                            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNet50(nn.Module):
    feat_channels = 1024  # layer3 output, used for feature distillation

    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 1, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.in_ch = 64
        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, width, blocks, stride):
        layers = [Bottleneck(self.in_ch, width, stride)]
        self.in_ch = width * Bottleneck.expansion
        layers += [Bottleneck(self.in_ch, width) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x, return_feats=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer2(self.layer1(x))
        feat = self.layer3(x)
        x = self.avgpool(self.layer4(feat)).flatten(1)
        logits = self.fc(x)
        return (logits, feat) if return_feats else logits


def load_imagenet_weights(model: ResNet50):
    """Copy torchvision's ImageNet-1k (V2) ResNet-50 weights into our model, except `fc`."""
    from torchvision.models import ResNet50_Weights, resnet50

    src = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).state_dict()
    src = {k: v for k, v in src.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(src, strict=False)
    assert set(missing) == {"fc.weight", "fc.bias"} and not unexpected, (missing, unexpected)
    print(f"Loaded ImageNet weights into ResNet-50 ({len(src)} tensors, fc re-initialised)")
    return model
