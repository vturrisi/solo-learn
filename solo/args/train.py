def encoder_args(parser):
    SUPPORTED_NETWORKS = ["resnet18", "resnet50"]

    parser.add_argument("--encoder", choices=SUPPORTED_NETWORKS, type=str)
    parser.add_argument("--zero_init_residual", action="store_true")


def general_train_args(parser):
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--classifier_lr", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--num_workers", type=int, default=4)

    # wandb
    parser.add_argument("--name")
    parser.add_argument("--project")
    parser.add_argument("--entity", default=None, type=str)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--offline", action="store_true")


def optizer_args(parser):
    SUPPORTED_OPTIMIZERS = ["sgd", "adam"]

    parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True)
    parser.add_argument("--lars", action="store_true")
    parser.add_argument("--exclude_bias_n_norm", action="store_true")


def scheduler_args(parser):
    SUPPORTED_SCHEDULERS = [
        "reduce",
        "cosine",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    parser.add_argument("--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce")
    parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
