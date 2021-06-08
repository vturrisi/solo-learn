def additional_setup_contrastive(args):
    args.transform_kwargs = {}
    if args.dataset == "cifar10":
        args.n_classes = 10
    elif args.dataset == "cifar100":
        args.n_classes = 100
    elif args.dataset == "stl10":
        args.n_classes = 10
    elif args.dataset == "imagenet":
        args.n_classes = 1000
    else:
        args.n_classes = 100

    if args.asymmetric_augmentations:
        args.transform_kwargs = [
            dict(
                brightness=args.brightness,
                contrast=args.contrast,
                saturation=args.saturation,
                hue=args.hue,
                gaussian_prob=1.0,
                solarization_prob=0.0,
                min_scale_crop=args.min_scale_crop,
            ),
            dict(
                brightness=args.brightness,
                contrast=args.contrast,
                saturation=args.saturation,
                hue=args.hue,
                gaussian_prob=0.1,
                solarization_prob=0.2,
                min_scale_crop=args.min_scale_crop,
            ),
        ]
    elif not args.multicrop:
        args.transform_kwargs = dict(
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
            gaussian_prob=args.gaussian_prob,
            solarization_prob=args.solarization_prob,
            min_scale_crop=args.min_scale_crop,
        )
    else:
        args.transform_kwargs = dict(
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
            gaussian_prob=args.gaussian_prob,
            solarization_prob=args.solarization_prob,
        )

    args.cifar = True if args.dataset in ["cifar10", "cifar100"] else False

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    # adjust lr according to batch size
    args.lr = args.lr * args.batch_size * len(args.gpus) / 256


def additional_setup_linear(args):
    if args.dataset == "cifar10":
        args.n_classes = 10
    elif args.dataset == "cifar100":
        args.n_classes = 100
    elif args.dataset == "stl10":
        args.n_classes = 10
    elif args.dataset == "imagenet":
        args.n_classes = 1000
    elif args.dataset == "imagenet100":
        args.n_classes = 100

    args.cifar = True if args.dataset in ["cifar10", "cifar100"] else False

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9
