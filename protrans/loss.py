import torch
import torch.nn.functional as F


def aggr_sep(features: torch.Tensor, labels: torch.Tensor, proto: torch.Tensor, tmp: float):
    assert isinstance(proto, torch.Tensor) and proto.requires_grad is False
    assert features.ndim == 2
    assert proto.ndim == 2

    num_classes = proto.shape[0]
    features = F.normalize(features, dim=-1)
    proto = F.normalize(proto, dim=-1)
    cos_mtx = torch.matmul(features, proto.t()) / tmp
    loss_aggr = -torch.masked_select(cos_mtx, mask=torch.nn.functional.one_hot(labels, num_classes=num_classes).to(dtype=torch.bool)).mean()
    loss_sep = torch.log(torch.sum(torch.exp(cos_mtx), dim=1)).mean()

    return loss_aggr, loss_sep
