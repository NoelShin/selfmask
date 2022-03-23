import numpy as np
from sklearn import cluster
import torch
from clusterings.base_clustering import BaseClustering
from clusterings.kmeans import KMeansClustering


class SpectralClustering(BaseClustering):
    def __init__(self, *args, **kwargs):
        super(SpectralClustering, self).__init__(*args, **kwargs)
        if self.use_gpu:
            self.kmeans = KMeansClustering(*args, **kwargs)

    @staticmethod
    def _compute_adjacency_matrix(features: torch.Tensor) -> torch.Tensor:
        if len(features.shape) == 3:
            # assume patch_tokens do not have a batch dimension (i.e. n_dims x h x w).
            # patch_tokens = patch_tokens.permute(1, 2, 0)
            # print("The shape of given features is 3-d. We assume batch_size x dim x hw.")
            features = features.permute(0, 2, 1)  # b x hw x dim
        elif len(features.shape) == 4:
            b, n_dims, h, w = features.shape
            features = features.reshape(b, n_dims, h * w)
            features = features.permute(0, 2, 1)  # b x hw x dim
        else:
            raise ValueError(f"Invalid shape for features {features.shape}.")

        # L2 normalize across the embedding axis.
        features /= torch.linalg.norm(features, ord=2, dim=-1, keepdim=True)
        adj_mat = features @ features.permute(0, 2, 1)  # b x hw x hw

        # ensure the range of the adjacency matrix to be [0, 1] by normalization.
        adj_mat -= adj_mat.amin(dim=(1, 2), keepdim=True)
        adj_mat = adj_mat / adj_mat.amax(dim=(1, 2), keepdim=True)
        return adj_mat

    @staticmethod
    def _compute_degree_matrix(adj_mat: torch.Tensor) -> torch.Tensor:
        b, hw, hw = adj_mat.shape
        degrees = adj_mat.sum(dim=-1, keepdim=False)

        # make a diagonal matrix with each element being the corresponding degree.
        degree_mat = torch.zeros(b, hw, hw, device=adj_mat.device)
        degree_mat.as_strided(degrees.size(), [degree_mat.stride(0), degree_mat.size(2) + 1]).copy_(degrees)
        return degree_mat

    def _compute_graph_laplacian(self, adj_mat: torch.Tensor) -> torch.Tensor:
        degree_mat = self._compute_degree_matrix(adj_mat)
        return degree_mat - adj_mat

    @torch.no_grad()
    def forward(self, features: torch.Tensor, k: int, **kwargs) -> np.ndarray:
        """
        :param patch_tokens: (B x) n_dims x H x W
        :param k:
        :param kwargs:
        :return:
        """
        # if len(features.shape) == 3:
        #     # if patch_tokens do not have a batch dimension, add one.
        #     features = features.unsqueeze(dim=0)
        # assert len(features.shape) == 4, f"Check features' shape: {features.shape}"
        if len(features.shape) == 3:
            b, n_dims, hw = features.shape
        elif len(features.shape) == 4:
            b, n_dims, h, w = features.shape
        else:
            raise ValueError(f"A shape of features should be 3D or 4D, got {features.shape}.")

        adj_mats: torch.Tensor = self._compute_adjacency_matrix(features)

        batch_clusters: list = list()
        if self.use_gpu:
            # use torch library for eigen decomposition and sklearn for k-means on the computed eigen vectors.
            gl = self._compute_graph_laplacian(adj_mats)
            d = self._compute_degree_matrix(adj_mats)

            # solve generalized eigenvalues and eigenvectors
            eigen_vals, eigen_vectors = torch.lobpcg(A=gl, B=d, k=k, largest=False)

            # index k eigen vectors with lowest eigen values.
            k_eigen_vectors = eigen_vectors[..., :k].cpu().numpy()

            # loop over each batch to run K-means (couldn't find how to do batch-mode k-means with faiss)
            for kev in k_eigen_vectors:
                clusters = cluster.KMeans(k, n_init=self.n_init, max_iter=self.max_iter).fit_predict(kev)
                batch_clusters.append(clusters)

        else:
            # use sklearn.cluster.SpectralClustering.
            adj_mats = adj_mats.cpu().numpy()
            assert not np.isnan(adj_mats).any()

            sc = cluster.SpectralClustering(
                k,
                n_init=self.n_init,
                affinity="precomputed",
                assign_labels="kmeans"
            )

            batch_clusters: list = list()
            for i in range(b):
                clusters = sc.fit_predict(adj_mats[i])
                batch_clusters.append(clusters)

        if len(features.shape) == 3:
            return np.array(batch_clusters, dtype=np.uint8).reshape(b, hw)
        else:
            return np.array(batch_clusters, dtype=np.uint8).reshape(b, h, w)
