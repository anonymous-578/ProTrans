import torch
import torch.nn as nn
import torchvision.models as models

import protrans.models as pmodels


def build_model(cfg):

    if cfg.PROTRANS.ENABLE:
        model = pmodels.__dict__[cfg.MODEL.ARCH](cfg, pretrained=cfg.MODEL.ARCH_PRETRAINED)
    else:
        model = models.__dict__[cfg.MODEL.ARCH](pretrained=cfg.MODEL.ARCH_PRETRAINED)

    # brute-force replacement
    model.fc = nn.Linear(model.fc.in_features, cfg.MODEL.NUM_CLASSES, bias=cfg.MODEL.FC.BIAS)

    model = model.cuda(device=torch.cuda.current_device())

    return model
