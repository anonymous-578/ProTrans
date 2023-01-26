from typing import Tuple, List, Optional

import torch
from torch import Tensor
import torchvision.models
from torchvision.models.resnet import ResNet, Bottleneck


__all__ = [
    "resnet50",
]


class MyResnet50(ResNet):
    def __init__(self, cfg):
        super(MyResnet50, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.cfg = cfg
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.feature_dim = self.fc.weight.shape[1]

        self._device = self.fc.weight.data.device

        self.register_buffer(
            'prototypes',
            torch.zeros((self.num_classes, self.feature_dim), device=self._device, dtype=torch.float, requires_grad=False)
        )

    @torch.no_grad()
    def set_prototypes(self, mean_per_classes: Tensor, cov_per_classes: Optional[Tensor], sample_proto: bool):
        assert self.prototypes.shape == mean_per_classes.shape

        if sample_proto:
            for i in range(len(mean_per_classes)):
                if torch.any(cov_per_classes[i]):
                    try:
                        self.prototypes[i] = torch.distributions.multivariate_normal.MultivariateNormal(mean_per_classes[i], cov_per_classes[i]).sample()
                    except ValueError as ve:
                        self.prototypes[i] = mean_per_classes[i]
                        print(ve)
                else:
                    self.prototypes[i] = mean_per_classes[i]
        else:
            self.prototypes = mean_per_classes

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        out = self.fc(feature)
        out = out

        return feature, out


def resnet50(cfg, pretrained=False):
    model = MyResnet50(cfg)
    if pretrained:
        model.load_state_dict(torchvision.models.resnet50(pretrained=True).state_dict(), strict=False)

    return model
