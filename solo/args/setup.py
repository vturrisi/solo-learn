import argparse
import pytorch_lightning as pl

from solo.args.dataset import augmentations_args, dataset_args
from solo.args.utils import additional_setup_contrastive, additional_setup_linear
from solo.methods import METHODS


def parse_args_contrastive():
    parser = argparse.ArgumentParser()

    # add shared arguments
    dataset_args(parser)
    augmentations_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # base model
    parser = METHODS["base"].add_model_specific_args(parser)

    # add method-specific arguments
    parser.add_argument("--method", type=str)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    parser = METHODS[temp_args.method].add_model_specific_args(parser)

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

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # linear model
    parser = METHODS["linear"].add_model_specific_args(parser)

    # parse args
    args = parser.parse_args()
    additional_setup_linear(args)

    return args
