def dataset_args(parser):
    SUPPORTED_DATASETS = [
        "cifar10",
        "cifar100",
        "stl10",
        "imagenet",
        "imagenet100",
    ]

    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, type=str, required=True)

    # dataset path
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)

    # cropping
    parser.add_argument("--multicrop", action="store_true")
    parser.add_argument("--n_crops", type=int, default=2)
    parser.add_argument("--n_small_crops", type=int, default=6)

    # augmentations
    parser.add_argument("--brightness", type=float, required=True)
    parser.add_argument("--contrast", type=float, required=True)
    parser.add_argument("--saturation", type=float, required=True)
    parser.add_argument("--hue", type=float, required=True)
    parser.add_argument("--gaussian_prob", type=float, default=0.5)
    parser.add_argument("--solarization_prob", type=float, default=0)
    parser.add_argument("--min_scale_crop", type=float, default=0.08)
    parser.add_argument("--asymmetric_augmentations", action="store_true")

    # dali (imagenet-100/imagenet only)
    parser.add_argument("--dali", action="store_true")
    parser.add_argument("--dali_device", type=str, default="gpu")
    parser.add_argument("--last_batch_fill", action="store_true")
