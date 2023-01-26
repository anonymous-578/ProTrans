import torch
from torch.utils.data import DataLoader

from datasets.build import build_dataset


def construct_loader(cfg, split, fe=False):
    assert split in ["train", "val", "test", "trainval"]
    if split in ["train", "trainval"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = True
        drop_last = False if fe else cfg.DATA_LOADER.DROP_LAST
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = False
        drop_last = False
    else:  # split in ["test"]
        # test dataset is same as train dataset
        dataset_name = cfg.TRAIN.DATASET
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = False
        drop_last = False

    dataset = build_dataset(dataset_name, cfg, split, fe)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last
    )

    return loader
