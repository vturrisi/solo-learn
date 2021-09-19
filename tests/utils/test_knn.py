import torch
from torch.nn import functional as F

from solo.utils.knn import WeightedKNNClassifier


def test_knn():
    num_samples_train = 100
    num_samples_test = 20
    num_classes = 10
    neighbors = 5
    features_dim = 16

    # test distances
    knn = WeightedKNNClassifier(k=neighbors, distance_fx="cosine", num_chunks=2)
    knn.update(
        train_features=F.normalize(torch.randn(num_samples_train, features_dim)),
        train_targets=torch.arange(end=num_classes).repeat(num_samples_train // num_classes),
        test_features=F.normalize(torch.randn(num_samples_test, features_dim)),
        test_targets=torch.arange(end=num_classes).repeat(num_samples_test // num_classes),
    )
    acc1, acc5 = knn.compute()
    assert acc1 >= 0 and acc1 <= 100
    assert acc5 >= 0 and acc5 <= 100
    assert acc5 >= acc1

    # test distances
    knn = WeightedKNNClassifier(k=neighbors, distance_fx="euclidean", num_chunks=2)
    knn.update(
        train_features=torch.randn(num_samples_train, features_dim),
        train_targets=torch.arange(end=num_classes).repeat(num_samples_train // num_classes),
        test_features=torch.randn(num_samples_test, features_dim),
        test_targets=torch.arange(end=num_classes).repeat(num_samples_test // num_classes),
    )
    acc1, acc5 = knn.compute()
    assert acc1 >= 0 and acc1 <= 100
    assert acc5 >= 0 and acc5 <= 100
    assert acc5 >= acc1
