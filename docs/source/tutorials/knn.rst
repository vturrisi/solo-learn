After training a model, sometimes you need to run K-NN evaluation. Doing so is straightforward with the newly added `main_knn.py`.
However, we will go through all the details on this tutorial.

First, you start by importing the needed libraries.

.. code-block:: python
    import json
    import os
    from pathlib import Path
    from typing import Tuple

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from solo.args.setup import parse_args_knn
    from solo.methods import METHODS
    from solo.utils.classification_dataloader import (
        prepare_dataloaders,
        prepare_datasets,
        prepare_transforms,
    )
    from solo.utils.knn import WeightedKNNClassifier

Then, we create a helper function that is responsible for iterating a dataloader and extracting all the features using a pretrained model.

.. code-block:: python
    @torch.no_grad()
    def extract_features(loader: DataLoader, model: nn.Module) -> Tuple(torch.Tensor):
        """Extract features from a data loader using a model.

        Args:
            loader (DataLoader): dataloader for a dataset.
            model (nn.Module): torch module used to extract features.

        Returns:
            Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
        """

        # set model to eval
        model.eval()
        backbone_features, proj_features, labels = [], [], []
        for im, lab in tqdm(loader):
            im = im.cuda(non_blocking=True)
            lab = lab.cuda(non_blocking=True)
            outs = model(im)
            backbone_features.append(outs["feats"].detach())
            proj_features.append(outs["z"])
            labels.append(lab)
        # return model back to train
        model.train()
        # features extracted by the backbone
        backbone_features = torch.cat(backbone_features)
        # features after the projection head
        proj_features = torch.cat(proj_features)
        # labels for evaluation
        labels = torch.cat(labels)
        return backbone_features, proj_features, labels

Finally, we create another helper function that will build a WeightedKNNClassifier object and compute the accuracies.

.. code-block:: python
    @torch.no_grad()
    def run_knn(
        train_features: torch.Tensor,
        train_targets: torch.Tensor,
        test_features: torch.Tensor,
        test_targets: torch.Tensor,
        k: int,
        T: float,
        distance_fx: str,
    ) -> Tuple[float]:
        """Runs offline knn on a train and a test dataset.

        Args:
            train_features (torch.Tensor, optional): train features.
            train_targets (torch.Tensor, optional): train targets.
            test_features (torch.Tensor, optional): test features.
            test_targets (torch.Tensor, optional): test targets.
            k (int): number of neighbors.
            T (float): temperature for the exponential. Only used with cosine
                distance.
            distance_fx (str): distance function.

        Returns:
            Tuple[float]: tuple containing the the knn acc@1 and acc@5 for the model.
        """

        # build knn object
        knn = WeightedKNNClassifier(
            k=k,
            T=T,
            distance_fx=distance_fx,
        )

        # add features
        knn(
            train_features=train_features,
            train_targets=train_targets,
            test_features=test_features,
            test_targets=test_targets,
        )

        # compute
        acc1, acc5 = knn.compute()

        # free up memory
        del knn

        return acc1, acc5

We then define the needed arguments.

.. code-block:: python
    kwargs = {
        "dataset": "imagenet100",
        "data_dir": "/datasets",
        "train_dir": "imagenet-100/train",
        "val_dir": "imagenet-100/val",
        "batch_size": 16,
        "num_workers": 10,
        # path to a directory containing the model and the args
        "pretrained_checkpoint_dir": "PATH_TO_PRETRAINED_MODEL_DIR",
        "k": [1, 2, 5, 10, 20, 50, 100, 200],
        "temperature": [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 1],
        "feature_type": ["backbone", "projector"],
        "distance_function": ["euclidean", "cosine"],
    }


Load the paths, prepare the model and the data

.. code-block:: python
    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"
    ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)

    # build the model
    model = METHODS[method_args["method"]].load_from_checkpoint(
        ckpt_path, strict=False, **method_args
    )
    model.cuda()

    # prepare data
    _, T = prepare_transforms(args.dataset)
    train_dataset, val_dataset = prepare_datasets(
        args.dataset,
        T_train=T,
        T_val=T,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        download=True,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


Extract features for both the train dataset and the test dataset.

.. code-block:: python
    # extract train features
    train_features_bb, train_features_proj, train_targets = extract_features(train_loader, model)
    train_features = {"backbone": train_features_bb, "projector": train_features_proj}

    # extract test features
    test_features_bb, test_features_proj, test_targets = extract_features(val_loader, model)
    test_features = {"backbone": test_features_bb, "projector": test_features_proj}

And finally run K-NN for all possible combinations of parameters.

.. code-block:: python
    for feat_type in args.feature_type:
        print(f"\n### {feat_type.upper()} ###")
        for k in args.k:
            for distance_fx in args.distance_function:
                temperatures = args.temperature if distance_fx == "cosine" else [None]
                for T in temperatures:
                    print("---")
                    print(f"Running k-NN with params: distance_fx={distance_fx}, k={k}, T={T}...")
                    acc1, acc5 = run_knn(
                        train_features=train_features[feat_type],
                        train_targets=train_targets,
                        test_features=test_features[feat_type],
                        test_targets=test_targets,
                        k=k,
                        T=T,
                        distance_fx=distance_fx,
                    )
                    print(f"Result: acc@1={acc1}, acc@5={acc5}")

Note that the same can be accomplished by running the following bash file.

.. code-block:: bash
    python3 ../../../main_knn.py \
        --dataset imagenet100 \
        --data_dir /datasets \
        --train_dir imagenet-100/train \
        --val_dir imagenet-100/val \
        --batch_size 16 \
        --num_workers 10 \
        --pretrained_checkpoint_dir PATH_TO_PRETRAINED_MODEL_DIR \
        --k 1 2 5 10 20 50 100 200 \
        --temperature 0.01 0.02 0.05 0.07 0.1 0.2 0.5 1 \
        --feature_type backbone projector \
        --distance_function euclidean cosine
