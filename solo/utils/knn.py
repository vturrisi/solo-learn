# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Sequence

import torch
import torch.nn.functional as F
import faiss
from faiss.contrib import torch_utils
from torchmetrics.metric import Metric


class WeightedKNNClassifier(Metric):
    def __init__(
        self,
        k: int = 20,
        T: float = 0.07,
        distance_fx: str = "cosine",
        epsilon: float = 0.00001,
        index_to_gpu: bool = False,
        approx: bool = True,
        nlist: int = 100,
        m: int = 32,
        dist_sync_on_step: bool = False,
    ):
        """Implements the weighted k-NN classifier used for evaluation.

        Args:
            k (int, optional): number of neighbors. Defaults to 20.
            T (float, optional): temperature for the exponential. Only used with cosine
                distance. Defaults to 0.07.
            distance_fx (str, optional): Distance function. Accepted arguments: "cosine" or
                "euclidean". Defaults to "cosine".
            epsilon (float, optional): Small value for numerical stability. Only used with
                euclidean distance. Defaults to 0.00001.
            dist_sync_on_step (bool, optional): whether to sync distributed values at every
                step. Defaults to False.
        """

        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)

        self.k = k
        self.T = T
        self.distance_fx = distance_fx
        self.epsilon = epsilon
        self.index_to_gpu = index_to_gpu
        self.approx = approx
        self.nlist = nlist
        self.m = m

        assert self.distance_fx in [
            "cosine",
            "euclidean",
        ], "Only cosine and euclidean distances are supported."

        self.add_state("train_features", default=[], persistent=False)
        self.add_state("train_targets", default=[], persistent=False)
        self.add_state("test_features", default=[], persistent=False)
        self.add_state("test_targets", default=[], persistent=False)

    def update(
        self,
        train_features: torch.Tensor = None,
        train_targets: torch.Tensor = None,
        test_features: torch.Tensor = None,
        test_targets: torch.Tensor = None,
    ):
        """Updates the memory banks. If train (test) features are passed as input, the
        corresponding train (test) targets must be passed as well.

        Args:
            train_features (torch.Tensor, optional): a batch of train features. Defaults to None.
            train_targets (torch.Tensor, optional): a batch of train targets. Defaults to None.
            test_features (torch.Tensor, optional): a batch of test features. Defaults to None.
            test_targets (torch.Tensor, optional): a batch of test targets. Defaults to None.
        """
        assert (train_features is None) == (train_targets is None)
        assert (test_features is None) == (test_targets is None)

        if train_features is not None:
            assert train_features.size(0) == train_targets.size(0)
            self.train_features.append(train_features.detach())
            self.train_targets.append(train_targets.detach())

        if test_features is not None:
            assert test_features.size(0) == test_targets.size(0)
            self.test_features.append(test_features.detach())
            self.test_targets.append(test_targets.detach())

    @torch.no_grad()
    def fit(self, X_train: torch.Tensor):

        # make sure that X_train is float32
        X_train = X_train.float()

        # select index according to distance function
        if self.distance_fx == "cosine":
            X_train = F.normalize(X_train, dim=1)
            self.index = faiss.IndexFlatIP(X_train.size(1))
        elif self.distance_fx == "euclidean":
            self.index = faiss.IndexFlatL2(X_train.size(1))

        # optionally use approximate distances
        if self.approx:
            self.index = faiss.IndexIVFPQ(self.index, X_train.size(1), self.nlist, self.m, 8)

        # make sure the index and the data are on the right device
        if self.index_to_gpu:
            assert X_train.is_cuda
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            X_train = X_train.cpu()

        if self.approx:
            self.index.train(X_train)

        # add training samples to the index
        self.index.add(X_train)

    @torch.no_grad()
    def predict(self, X_test: torch.Tensor, y_train: torch.Tensor):
        num_classes = torch.unique(y_train).numel()

        # make sure that X_test is float32
        X_test = X_test.float()

        # make sure X_test is on the right device
        device = X_test.device
        if not self.index_to_gpu:
            X_test = X_test.cpu()

        # lookup neighbors in the index
        distances, indices = self.index.search(X_test, k=self.k)

        # replace -1s with random indices
        mask = indices == -1
        indices[mask] = torch.randint(
            low=0,
            high=y_train.size(0),
            size=indices.size(),
            dtype=indices.dtype,
            device=indices.device,
        )[mask]

        # move results to gpu
        if not self.index_to_gpu:
            distances = distances.to(device)
            indices = indices.to(device)
            X_test = X_test.to(device)

        # similarities from distances
        if self.distance_fx == "cosine":
            similarities = distances
        elif self.distance_fx == "euclidean":
            similarities = 1 / (distances + self.epsilon)

        # compute predictions using similarity as weight
        candidates = y_train.view(1, -1).expand(X_test.size(0), -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)
        retrieval_one_hot = torch.zeros(X_test.size(0) * self.k, num_classes, device=device)
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

        # temperature scaled softmax for cosine similarity
        if self.distance_fx == "cosine":
            similarities = similarities.clone().div_(self.T).exp_()

        # vote and predict
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(X_test.size(0), -1, num_classes),
                similarities.view(X_test.size(0), -1, 1),
            ),
            1,
        )
        _, preds = probs.sort(1, True)
        return preds

    @torch.no_grad()
    def compute_accuracy(self, preds, y_test):
        correct = preds.eq(y_test.data.view(-1, 1))
        top1 = correct.narrow(1, 0, 1).sum().item() / y_test.size(0)
        top5 = correct.narrow(1, 0, min(5, self.k, correct.size(-1))).sum().item() / y_test.size(0)
        return top1, top5

    @torch.no_grad()
    def compute(self) -> Sequence[float]:
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.

        Returns:
            Sequence[float]: k-NN accuracy @1 and @5.
        """

        # collect features and targets
        train_features = torch.cat(self.train_features)
        train_targets = torch.cat(self.train_targets)
        test_features = torch.cat(self.test_features)
        test_targets = torch.cat(self.test_targets)

        # build the index
        self.fit(train_features)

        # use the index to predict
        preds = self.predict(test_features, train_targets)

        # compute accuracy
        acc = self.compute_accuracy(preds, test_targets)

        # reset the states
        self.reset()

        return acc
