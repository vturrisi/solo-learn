import torch
from torchmetrics.metric import Metric


class WeightedKNNClassifier(Metric):
    def __init__(
        self,
        k=20,
        T=0.07,
        num_chunks=100,
        distance_fx="cosine",
        epsilon=0.00001,
        dist_sync_on_step=False,
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)

        self.k = k
        self.T = T
        self.num_chunks = num_chunks
        self.distance_fx = distance_fx
        self.epsilon = epsilon

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
        assert (train_features is None) == (train_targets is None)
        assert (test_features is None) == (test_targets is None)

        if train_features is not None:
            assert train_features.size(0) == train_targets.size(0)
            self.train_features.append(train_features)
            self.train_targets.append(train_targets)

        if test_features is not None:
            assert test_features.size(0) == test_targets.size(0)
            self.test_features.append(test_features)
            self.test_targets.append(test_targets)

    @torch.no_grad()
    def compute(self):
        train_features = torch.cat(self.train_features)
        train_targets = torch.cat(self.train_targets)
        test_features = torch.cat(self.test_features)
        test_targets = torch.cat(self.test_targets)

        top1, top5, total = 0.0, 0.0, 0
        num_classes = torch.unique(test_targets).numel()
        num_test_images = test_targets.size(0)
        chunk_size = max(1, num_test_images // self.num_chunks)
        k = min(self.k, train_targets.size(0))
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        for idx in range(0, num_test_images, chunk_size):
            # get the features for test images
            features = test_features[idx : min((idx + chunk_size), num_test_images), :]
            targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
            batch_size = targets.shape[0]

            # calculate the dot product and compute top-k neighbors
            if self.distance_fx == "cosine":
                similarity = torch.mm(features, train_features.t())
            elif self.distance_fx == "euclidean":
                similarity = 1 / (torch.cdist(features, train_features) + self.epsilon)
            else:
                raise NotImplementedError

            distances, indices = similarity.topk(k, largest=True, sorted=True)
            candidates = train_targets.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            if self.distance_fx == "cosine":
                distances = distances.clone().div_(self.T).exp_()

            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    distances.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = (
                top5 + correct.narrow(1, 0, min(5, k)).sum().item()
            )  # top5 does not make sense if k < 5
            total += targets.size(0)

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total

        self.reset()

        return top1, top5
