# reference: https://github.com/facebookresearch/SlowFast
import torch
import utils.lr_policy as lr_policy


def construct_optimizer(model, cfg):
    optim_params = get_param_groups(model, cfg)

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr, fc_lr_ratio):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for idx, param_group in enumerate(optimizer.param_groups):
        # idx 0: backbone, idx 1: classifier
        param_group["lr"] = new_lr if idx == 0 else fc_lr_ratio * new_lr


def get_param_groups(model, cfg):
    param_groups = []
    clf = 'fc'
    param_backbone = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(clf)]
    param_classifier = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(clf)]
    param_groups.extend([
        {'params': param_backbone, 'lr': cfg.SOLVER.BASE_LR},
        {'params': param_classifier, 'lr': cfg.SOLVER.FC_LR_RATIO * cfg.SOLVER.BASE_LR}
    ])

    return param_groups
