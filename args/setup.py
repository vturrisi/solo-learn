import argparse

from .dataset import augmentations_args, dataset_args
from .methods import (
    barlow_args,
    byol_args,
    mocov2plus_args,
    nnclr_args,
    simclr_args,
    simsiam_args,
    swav_args,
    vicreg_args,
)
from .train import encoder_args, general_train_args, optizer_args, scheduler_args
from .utils import additional_setup_contrastive, additional_setup_linear


def parse_args_contrastive():
    parser = argparse.ArgumentParser()

    # add shared arguments
    dataset_args(parser)
    augmentations_args(parser)
    general_train_args(parser)
    encoder_args(parser)
    optizer_args(parser)
    scheduler_args(parser)

    # add method-specific arguments
    subparser = parser.add_subparsers(dest="method")
    method_parser = subparser.add_parser("barlow_twins")
    barlow_args(method_parser)

    method_parser = subparser.add_parser("byol")
    byol_args(method_parser)

    method_parser = subparser.add_parser("mocov2plus")
    mocov2plus_args(method_parser)

    method_parser = subparser.add_parser("nnclr")
    nnclr_args(method_parser)

    method_parser = subparser.add_parser("simclr")
    simclr_args(method_parser)

    method_parser = subparser.add_parser("simsiam")
    simsiam_args(method_parser)

    method_parser = subparser.add_parser("swav")
    swav_args(method_parser)

    method_parser = subparser.add_parser("vicreg")
    vicreg_args(method_parser)

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

    args = parser.parse_args()
    additional_setup_linear(args)

    return args
