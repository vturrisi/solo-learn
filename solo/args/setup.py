import argparse
import pytorch_lightning as pl

from solo.args import methods as methods_args
from solo.args.dataset import augmentations_args, dataset_args
from solo.args.train import encoder_args, general_train_args, optizer_args, scheduler_args
from solo.args.utils import additional_setup_contrastive, additional_setup_linear
from solo.methods import METHODS


def parse_args_contrastive():
    parser = argparse.ArgumentParser()

    # add shared arguments
    dataset_args(parser)
    augmentations_args(parser)
    general_train_args(parser)
    encoder_args(parser)
    optizer_args(parser)
    scheduler_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # add method-specific arguments
    subparser = parser.add_subparsers(dest="method")
    for name in METHODS:
        method_parser = subparser.add_parser(name)
        method_args = name + "_args"

        assert method_args in methods_args.__dict__, f"Missing args function {method_args}"

        methods_args.__dict__[method_args](method_parser)

    # parse args
    args = parser.parse_args()

    # prepare arguments with additional setup
    additional_setup_contrastive(args)

    return args


def parse_args_linear():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_feature_extractor", type=str)

    # add shared arguments
    dataset_args(parser)
    general_train_args(parser)
    encoder_args(parser)
    optizer_args(parser)
    scheduler_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # parse args
    args = parser.parse_args()
    additional_setup_linear(args)

    return args
