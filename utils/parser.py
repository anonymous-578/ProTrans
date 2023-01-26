# adapted from: https://github.com/facebookresearch/SlowFast

"""Argument parser functions."""

import argparse
import sys

from config import get_cfg_defaults


def parse_args():
    parser = argparse.ArgumentParser(
        description="A Prototype Aggregation for Inductive Transfer Learning"
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="./configs/sup/my/sup_my_aircraft.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    # Setup cfg.
    cfg = get_cfg_defaults()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    return cfg
