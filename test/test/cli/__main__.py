import argparse
from pathlib import Path

import test.cli.train


parser = argparse.ArgumentParser(prog="test")

subcmd = parser.add_subparsers(dest="command")
subcmd.required = True

Formatter = argparse.ArgumentDefaultsHelpFormatter

train = subcmd.add_parser("train", help="trains super-resolution model", formatter_class=Formatter)
train.add_argument("--dataset", type=Path, help="path to directory for loading data from")
train.add_argument("--model", type=Path, required=True, help="file to save trained model to")
train.add_argument("--num-workers", type=int, default=0, help="number of parallel workers")
train.add_argument("--batch-size", type=int, default=64, help="number of items per batch")
train.add_argument("--checkpoint", type=str, required=False, help="path to a model checkpoint (to retrain)")
train.add_argument("--resume", type=bool, default=False, help="resume training or fine-tuning (if checkpoint)")
train.set_defaults(main=test.cli.train.main)

args = parser.parse_args()
args.main(args)
