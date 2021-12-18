import torch
from solo.utils.kmeans import KMeans


def test_kmeans():
    k = [30]
    kmeans = KMeans(1, 0, 1, 500, 128, k)
    local_memory_index = torch.arange(0, 500)
    local_memory_embeddings = torch.randn((1, 500, 128))
    assignments, centroids_list = kmeans.cluster_memory(local_memory_index, local_memory_embeddings)
    assert assignments.size() == (1, 500)
    assert len(assignments.unique()) == 30
    assert centroids_list[0].size() == (30, 128)
