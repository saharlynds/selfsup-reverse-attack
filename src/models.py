"""Backbones that return (logits, penultimate features), and the contrastive head G."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet18(nn.Module):
    def __init__(self, num_classes=10, width=64):
        super().__init__()
        self.in_planes = width
        self.conv1 = nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(width, 2, 1)
        self.layer2 = self._make_layer(width * 2, 2, 2)
        self.layer3 = self._make_layer(width * 4, 2, 2)
        self.layer4 = self._make_layer(width * 8, 2, 2)
        self.bn = nn.BatchNorm2d(width * 8)
        self.feat_dim = width * 8
        self.linear = nn.Linear(self.feat_dim, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(PreActBlock(self.in_planes, planes, s))
            self.in_planes = planes * PreActBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4).flatten(1)
        return self.linear(out), out


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, width=32):
        super().__init__()
        w = width

        def block(cin, cout, pool=True):
            layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return layers

        self.features = nn.Sequential(
            *block(3, w), *block(w, w * 2), *block(w * 2, w * 4), *block(w * 4, w * 4)
        )
        self.feat_dim = w * 4
        self.linear = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out), out


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, feats):
        return self.net(feats)


def build_backbone(arch: str, num_classes: int, width: int = None):
    if arch.lower().startswith("robustbench:"):
        from .external import build_robustbench_backbone

        return build_robustbench_backbone(arch)
    arch = arch.lower()
    if arch in ("preactresnet18", "resnet18", "prn18"):
        return PreActResNet18(num_classes=num_classes, width=width or 64)
    if arch in ("smallcnn", "small"):
        return SmallCNN(num_classes=num_classes, width=width or 32)
    raise ValueError(f"unknown arch: {arch}")
