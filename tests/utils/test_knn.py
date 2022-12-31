# Copyright 2023 solo-learn development team.

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

import torch
import torch.nn.functional as F
from solo.utils.knn import WeightedKNNClassifier


def test_knn():
    num_samples_train = 100
    num_samples_test = 20
    max_distance_matrix_size = num_samples_train * num_samples_test // 10
    num_classes = 10
    neighbors = 5
    features_dim = 16

    # test distances
    knn = WeightedKNNClassifier(
        k=neighbors, distance_fx="cosine", max_distance_matrix_size=max_distance_matrix_size
    )
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
    knn = WeightedKNNClassifier(
        k=neighbors, distance_fx="euclidean", max_distance_matrix_size=max_distance_matrix_size
    )
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
