import argparse
import os
import sys

from . import methods as methods_addargs_funcs
from .dataset import augmentations_args, dataset_args
from .train import encoder_args, general_train_args, optizer_args, scheduler_args
from .utils import additional_setup_contrastive, additional_setup_linear

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from methods import METHODS


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
    for name in METHODS:
        method_parser = subparser.add_parser(name)
        method_addarg_func = name + "_args"

        assert (
            method_addarg_func in methods_addargs_funcs.__dict__.keys()
        ), f"Missing addargs function {method_addarg_func}"

        methods_addargs_funcs.__dict__[method_addarg_func](method_parser)

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
