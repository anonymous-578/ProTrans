import os

from utils.parser import parse_args, load_config
from train import train
from eval import evaluate


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)

    # select specific devices
    if cfg.VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.VISIBLE_DEVICES)

    if cfg.TRAIN.ENABLE:
        train(cfg)
    if cfg.TEST.ENABLE:
        evaluate(cfg)
