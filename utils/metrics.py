# reference: https://github.com/facebookresearch/moco/main_moco.py

import torch
import torch.nn.functional as F
from torch import Tensor


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def dist_cos(t1: Tensor, t2: Tensor, dim):
    assert t1.ndim == t2.ndim and t1.ndim in (1, 2)

    t1 = F.normalize(t1, dim=dim)
    t2 = F.normalize(t2, dim=dim)
    dist = 1. - torch.pow(
        torch.linalg.vector_norm(t1 - t2, dim=dim), 2
    ) / 2.

    return dist
