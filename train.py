# adapted from: https://github.com/facebookresearch/SlowFast
# adapted from: https://github.com/facebookresearch/moco/main_moco.py
import time

import torch
import torch.nn as nn

import models.optimizer as optim
import protrans.loss as ploss
import protrans.utils as putils
from datasets import loader
from models.build import build_model
from utils.misc import set_seeds, mkdir
from utils.meters import AverageMeter, ProgressMeter
from utils.metrics import accuracy


def train(cfg):
    # set random seed
    set_seeds(cfg.SEED)

    # Create the train and val (test) loaders.
    train_loader = loader.construct_loader(cfg, cfg.TRAIN.SPLIT)
    val_loader = loader.construct_loader(cfg, cfg.VAL.SPLIT)
    if cfg.PROTRANS.ENABLE:
        train_loader_fe = loader.construct_loader(cfg, cfg.TRAIN.SPLIT, fe=True)
    else:
        train_loader_fe = None

    # build model
    model = build_model(cfg)

    # construct the optimizer
    optimizer = optim.construct_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss().cuda()

    # best top1 accuracy
    best_top1 = 0.0

    for cur_epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.MAX_EPOCH):

        if cfg.PROTRANS.ENABLE and (cur_epoch == 0):
            penult_layer = 'avgpool'
            features, labels, = putils.get_features_labels(model, train_loader_fe, {penult_layer: 'penul'}, flatten=True)

            print(f"Number of train data with sampling rates {cfg.DATA.SAMPLING_RATES}: {len(labels)}")
            print(f"Number of included classes: {len(torch.unique(labels))}")

            feat = features[penult_layer]
            assert not feat.requires_grad

            # compute sample mean, sample covariance
            mean_per_class, cov_per_class = putils.get_mean_cov(feat, labels, cfg.MODEL.NUM_CLASSES, cfg.PROTRANS.SAMPLE_PROTO, cfg.PROTRANS.ALPHA)
            model.set_prototypes(mean_per_class, cov_per_class, cfg.PROTRANS.SAMPLE_PROTO)

        # Train for one epoch.
        train_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            cur_epoch,
            cfg,
        )

        # Evaluate the model on validation set.
        is_best = False
        if (cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH) or (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0:
            top1_acc = eval_epoch(val_loader, model, cur_epoch, cfg, best_top1)
            if top1_acc > best_top1:
                is_best = True
                best_top1 = top1_acc

        # Save a checkpoint.
        if is_best:
            checkpoint = {
                "epoch": cur_epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg.dump(),
            }
            with open(mkdir(cfg.TRAIN.CHECKPOINT_DIR) / 'checkpoint_best.pth', "wb") as f:
                torch.save(checkpoint, f)


def train_epoch(
        train_loader,
        model,
        criterion,
        optimizer,
        cur_epoch,
        cfg,
):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(cur_epoch)
    )
    all_features = []
    all_labels = []

    # switch to train mode
    model.train()

    data_size = len(train_loader)

    start = time.time()

    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - start)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr, cfg.SOLVER.FC_LR_RATIO)

        # move data to the current GPU
        inputs, labels = inputs.cuda(), labels.cuda()

        if cfg.PROTRANS.ENABLE:
            features, outputs = model(inputs)
        else:
            outputs = model(inputs)
            features = None

        loss = criterion(outputs, labels)

        if cfg.PROTRANS.ENABLE:
            loss_ce = loss
            loss_aggr, loss_sep = ploss.aggr_sep(features, labels, model.prototypes, cfg.PROTRANS.T)

            loss = cfg.PROTRANS.LAMBDA_CE * loss_ce + \
                cfg.PROTRANS.LAMBDA_AGGR * loss_aggr + \
                cfg.PROTRANS.LAMBDA_SEP * loss_sep

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # gather outputs, labels
        if cfg.PROTRANS.ENABLE:
            all_features.append(features.clone().detach())
            all_labels.append(labels.clone().detach())

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if cur_iter % cfg.TRAIN.PRINT_FREQ == 0:
            progress.display(cur_iter)

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

    if cfg.PROTRANS.ENABLE:
        all_features = torch.concat(all_features, dim=0)
        all_labels = torch.concat(all_labels, dim=0)

        mean_per_class, cov_per_class = putils.get_mean_cov(all_features, all_labels, cfg.MODEL.NUM_CLASSES, cfg.PROTRANS.SAMPLE_PROTO, cfg.PROTRANS.ALPHA)
        # update prototypes
        model.set_prototypes(mean_per_class, cov_per_class, cfg.PROTRANS.SAMPLE_PROTO)


@torch.no_grad()
def eval_epoch(val_loader, model, cur_epoch, cfg, best_top1):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, top1, top5],
        prefix="Validation epoch[{}]".format(cur_epoch)
    )
    all_features = []
    all_preds = []
    all_labels = []

    model.eval()

    start = time.time()
    for cur_iter, (inputs, labels) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - start)

        # move data to the current GPU
        inputs, labels = inputs.cuda(), labels.cuda()

        if cfg.PROTRANS.ENABLE:
            features, outputs = model(inputs)
            all_features.append(features)
        else:
            outputs = model(inputs)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        all_preds.append(outputs)
        all_labels.append(labels)

        if cur_iter % cfg.TRAIN.PRINT_FREQ == 0 or (cur_iter + 1) == len(val_loader):
            progress.display(cur_iter)

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

    if top1.avg > best_top1:
        with open(mkdir(cfg.RESULT_DIR) / "best_top1_acc.txt", 'w') as f:
            f.write(f"Val/Best_Top1_acc: {top1.avg}\tEpoch: {cur_epoch}")

    return top1.avg
