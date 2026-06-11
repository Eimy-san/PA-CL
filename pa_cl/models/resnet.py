"""CIFAR-style ResNet-18 for continual learning on Split-CIFAR-100.

This is the "CIFAR" variant of ResNet (not the ImageNet variant): the
stem uses a single 3x3 conv with no max-pool, so the spatial resolution
shrinks 32 -> 16 -> 8 -> 4 across the four BasicBlock stages. The
backbone is shared across all tasks; the classifier head is a single
nn.Linear (in_features=512, out_features=num_classes) sized for the
full class-incremental output space at construction time.

Named layers exposed to PA-CL hooks (use these in `layer_names`):
    'stem_act'    : post-ReLU activations after the 3x3 stem (32x32)
    'layer1.out'  : post-ReLU activations after BasicBlock stage 1
    'layer2.out'  : post-ReLU activations after BasicBlock stage 2
    'layer3.out'  : post-ReLU activations after BasicBlock stage 3
    'layer4.out'  : post-ReLU activations after BasicBlock stage 4 (penultimate spatial)
    'avgpool.out' : post-avgpool, flat 512-d (= penultimate features)

The PA-CL rank loss is most informative on 'avgpool.out' (the
representation that the classifier consumes), but the convolutional
trajectories are useful for the rank-trajectory figure.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
        # Named tap for PA-CL: the post-activation output of the block.
        self.out_relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.out_relu(out)
        return out


class _Tap(nn.Module):
    """Identity layer that exposes a named hook target. Pure pass-through."""
    def forward(self, x):
        return x


class CIFARResNet(nn.Module):
    """ResNet-N variant for 32x32 inputs (CIFAR), shared classifier head."""

    def __init__(self, block, num_blocks, num_classes: int = 100,
                 base_width: int = 64):
        super().__init__()
        self.in_planes = base_width
        self.stem = conv3x3(3, base_width)
        self.stem_bn = nn.BatchNorm2d(base_width)
        self.stem_act = _Tap()                       # PA-CL hook target
        self.layer1 = self._make_layer(block, base_width,     num_blocks[0], stride=1)
        self.layer1_tap = _Tap()
        self.layer2 = self._make_layer(block, base_width * 2, num_blocks[1], stride=2)
        self.layer2_tap = _Tap()
        self.layer3 = self._make_layer(block, base_width * 4, num_blocks[2], stride=2)
        self.layer3_tap = _Tap()
        self.layer4 = self._make_layer(block, base_width * 8, num_blocks[3], stride=2)
        self.layer4_tap = _Tap()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool_tap = _Tap()                    # post-pool, post-flatten
        self.head = nn.Linear(base_width * 8 * block.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder forward: returns the post-avgpool 512-d feature vector."""
        out = F.relu(self.stem_bn(self.stem(x)), inplace=True)
        out = self.stem_act(out)
        out = self.layer1(out); out = self.layer1_tap(out)
        out = self.layer2(out); out = self.layer2_tap(out)
        out = self.layer3(out); out = self.layer3_tap(out)
        out = self.layer4(out); out = self.layer4_tap(out)
        out = self.avgpool(out).flatten(1)
        out = self.avgpool_tap(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def cifar_resnet18(num_classes: int = 100, base_width: int = 64) -> CIFARResNet:
    return CIFARResNet(BasicBlock, [2, 2, 2, 2],
                       num_classes=num_classes, base_width=base_width)


def cifar_resnet32(num_classes: int = 100, base_width: int = 16) -> CIFARResNet:
    """The thinner ResNet-32 used in older CL papers; cheaper, sometimes more
    sensitive to plasticity loss. Provided for ablation studies."""
    return CIFARResNet(BasicBlock, [5, 5, 5, 5],
                       num_classes=num_classes, base_width=base_width)