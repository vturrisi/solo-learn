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

from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.sparse import csr_matrix


class KMeans:
    def __init__(
        self,
        world_size: int,
        rank: int,
        num_large_crops: int,
        dataset_size: int,
        proj_features_dim: int,
        num_prototypes: int,
        kmeans_iters: int = 10,
    ):
        """Class that performs K-Means on the hypersphere.

        Args:
            world_size (int): world size.
            rank (int): rank of the current process.
            num_large_crops (int): number of crops.
            dataset_size (int): total size of the dataset (number of samples).
            proj_features_dim (int): number of dimensions of the projected features.
            num_prototypes (int): number of prototypes.
            kmeans_iters (int, optional): number of iterations for the k-means clustering.
                Defaults to 10.
        """
        self.world_size = world_size
        self.rank = rank
        self.num_large_crops = num_large_crops
        self.dataset_size = dataset_size
        self.proj_features_dim = proj_features_dim
        self.num_prototypes = num_prototypes
        self.kmeans_iters = kmeans_iters

    @staticmethod
    def get_indices_sparse(data: np.ndarray):
        cols = np.arange(data.size)
        M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
        return [np.unravel_index(row.data, data.shape) for row in M]

    def cluster_memory(
        self,
        local_memory_index: torch.Tensor,
        local_memory_embeddings: torch.Tensor,
    ) -> Sequence[Any]:
        """Performs K-Means clustering on the hypersphere and returns centroids and
        assignments for each sample.

        Args:
            local_memory_index (torch.Tensor): memory bank cointaining indices of the
                samples.
            local_memory_embeddings (torch.Tensor): memory bank cointaining embeddings
                of the samples.

        Returns:
            Sequence[Any]: assignments and centroids.
        """
        j = 0
        device = local_memory_embeddings.device
        assignments = -torch.ones(len(self.num_prototypes), self.dataset_size).long()
        centroids_list = []
        with torch.no_grad():
            for i_K, K in enumerate(self.num_prototypes):
                # run distributed k-means

                # init centroids with elements from memory bank of rank 0
                centroids = torch.empty(K, self.proj_features_dim).to(device, non_blocking=True)
                if self.rank == 0:
                    random_idx = torch.randperm(len(local_memory_embeddings[j]))[:K]
                    assert len(random_idx) >= K, "please reduce the number of centroids"
                    centroids = local_memory_embeddings[j][random_idx]
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(centroids, 0)

                for n_iter in range(self.kmeans_iters + 1):

                    # E step
                    dot_products = torch.mm(local_memory_embeddings[j], centroids.t())
                    _, local_assignments = dot_products.max(dim=1)

                    # finish
                    if n_iter == self.kmeans_iters:
                        break

                    # M step
                    where_helper = self.get_indices_sparse(local_assignments.cpu().numpy())
                    counts = torch.zeros(K).to(device, non_blocking=True).int()
                    emb_sums = torch.zeros(K, self.proj_features_dim).to(device, non_blocking=True)
                    for k in range(len(where_helper)):
                        if len(where_helper[k][0]) > 0:
                            emb_sums[k] = torch.sum(
                                local_memory_embeddings[j][where_helper[k][0]],
                                dim=0,
                            )
                            counts[k] = len(where_helper[k][0])
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(counts)
                        dist.all_reduce(emb_sums)
                    mask = counts > 0
                    centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

                    # normalize centroids
                    centroids = F.normalize(centroids, dim=1, p=2)

                centroids_list.append(centroids)

                if dist.is_available() and dist.is_initialized():
                    # gather the assignments
                    assignments_all = torch.empty(
                        self.world_size,
                        local_assignments.size(0),
                        dtype=local_assignments.dtype,
                        device=local_assignments.device,
                    )
                    assignments_all = list(assignments_all.unbind(0))

                    dist_process = dist.all_gather(
                        assignments_all, local_assignments, async_op=True
                    )
                    dist_process.wait()
                    assignments_all = torch.cat(assignments_all).cpu()

                    # gather the indexes
                    indexes_all = torch.empty(
                        self.world_size,
                        local_memory_index.size(0),
                        dtype=local_memory_index.dtype,
                        device=local_memory_index.device,
                    )
                    indexes_all = list(indexes_all.unbind(0))
                    dist_process = dist.all_gather(indexes_all, local_memory_index, async_op=True)
                    dist_process.wait()
                    indexes_all = torch.cat(indexes_all).cpu()

                else:
                    assignments_all = local_assignments
                    indexes_all = local_memory_index

                # log assignments
                assignments[i_K][indexes_all] = assignments_all

                # next memory bank to use
                j = (j + 1) % self.num_large_crops

        return assignments, centroids_list
