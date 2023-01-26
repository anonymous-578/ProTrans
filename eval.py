import os

import torch

from utils.misc import set_seeds
from models.build import build_model
from datasets import loader
from utils.meters import AverageMeter, ProgressMeter
from utils.metrics import accuracy


@torch.no_grad()
def evaluate(cfg):
    # set random seed
    set_seeds(cfg.SEED)

    # build model
    model = build_model(cfg)

    model_path = os.path.join(cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")
    if os.path.isfile(model_path):
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")

        state_dict = checkpoint['model_state']
        msg = model.load_state_dict(state_dict, strict=True)
        assert set(msg.missing_keys) == set()

        print(f"Loaded pre-trained model from {model_path}")
    else:
        print("=> no checkpoint found at '{}'".format(cfg.PRETRAINED))

    # Create the test loader.
    test_loader = loader.construct_loader(cfg, cfg.TEST.SPLIT)

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [top1, top5],
        prefix="Test"
    )

    model.eval()

    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # move data to the current GPU
        inputs, labels = inputs.cuda(), labels.cuda()

        if cfg.PROTRANS.ENABLE:
            features, outputs = model(inputs)
        else:
            outputs = model(inputs)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if (cur_iter + 1) == len(test_loader):
            progress.display(cur_iter)
    with open(os.path.join(cfg.RESULT_DIR, "test.txt"), "w") as f:
        f.write(f"Test/Top1_acc: {top1.avg}")
