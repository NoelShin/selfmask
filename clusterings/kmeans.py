from typing import Union
from time import time
import numpy as np
from sklearn.decomposition import PCA
import torch
import faiss
from clusterings.base_clustering import BaseClustering


class KMeansClustering(BaseClustering):
    def __init__(self, *args, **kwargs):
        super(KMeansClustering, self).__init__(*args, **kwargs)

    @staticmethod
    def _check_shape(data: np.ndarray):
        """Check the shape of data and reshape to a shape that is acceptable to faiss (i.e., 2 dimensions with the
        channel dimension at the last axis).
        """
        if len(data.shape) == 3:
            c, h, w = data.shape
            data = np.transpose(data, (1, 2, 0))
            data = np.reshape(data, (h * w, c))

        elif len(data.shape) == 4:
            b, c, h, w = data.shape
            data = np.transpose(data, (0, 2, 3, 1))
            data = np.reshape(data, (b * h * w, c))
        return data

    @staticmethod
    def preprocess_features_bak(data: np.ndarray, pca: int, eps: float = 1e-5):
        """Preprocess features with PCA and L2 normalisation.
        Args:
            data (np.array N * ndim): features to preprocess
            pca (int): dim of output
            eps: a small value for preventing normalized data from having a NaN value.
        Returns:
            np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
        """
        st = time()

        _, ndim = data.shape
        assert not np.isnan(data).any(), f"data of shape {data.shape} contains NaN value."
        assert not np.isinf(data).any(), f"data of shape {data.shape} contains NaN value."

        # faiss requires C-contiguous arrays
        if not data.data.c_contiguous:
            data = np.ascontiguousarray(data)

        data = data.astype('float32')

        # Apply PCA-whitening with Faiss
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(data)
        assert mat.is_trained
        data = mat.apply_py(data)

        # L2 normalization
        row_sums = np.linalg.norm(data, axis=1)
        data = data / (row_sums[:, np.newaxis] + eps)
        # print(f"{time() - st:.3f} sec. taken for preprocessing features.")
        return data

    @staticmethod
    def preprocess_features(data: np.ndarray, pca: int, eps: float = 1e-5):
        """Preprocess features with PCA and L2 normalisation.
        Args:
            data (np.array N * ndim): features to preprocess
            pca (int): dim of output
            eps: a small value for preventing normalized data from having a NaN value.
        Returns:
            np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
        """
        data = PCA(n_components=pca).fit_transform(data)

        # L2 normalization
        # row_sums = np.linalg.norm(data, axis=1)
        # data = data / (row_sums[:, np.newaxis] + eps)
        return data

    def _run_kmeans_cpu(self, data: np.ndarray, k: int):
        n_dims = data.shape[-1]

        # faiss requires C-contiguous arrays
        if not data.data.c_contiguous:
            data = np.ascontiguousarray(data)

        kmeans = faiss.Kmeans(d=n_dims, k=k, niter=self.max_iter)
        patch_tokens = data.reshape(-1, n_dims)
        patch_tokens = np.ascontiguousarray(patch_tokens)  # https://github.com/facebookresearch/faiss/issues/459
        kmeans.train(patch_tokens)
        dists, clusters = kmeans.index.search(patch_tokens, 1)
        return clusters.astype(np.uint8)

    def _run_kmeans_gpu(self, data: np.ndarray, k: int, verbose: bool = False):
        """Runs kmeans on 1 GPU.
        Args:
            data: data to be fit and predicted for k-means clustering
            k (int): number of clusters
            verbose: whether to print process
        Returns:
            list: ids of data in each cluster
        """
        st = time()
        assert not np.isnan(data).any(), f"data of shape {data.shape} contains NaN value."
        assert not np.isinf(data).any(), f"data of shape {data.shape} contains inf value."

        # faiss requires C-contiguous arrays
        if not data.data.c_contiguous:
            data = np.ascontiguousarray(data)

        n_data, d = data.shape

        # faiss implementation of k-means
        clus = faiss.Clustering(d, k)

        # Change faiss seed at each k-means so that the randomly picked initialization centroids do not correspond to
        # the same feature ids from an epoch to another.
        # clus.seed = np.random.randint(1234)

        clus.niter = self.max_iter
        clus.max_points_per_centroid = 10000000
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)

        # perform the training
        clus.train(data, index)
        _, nearest_centroids = index.search(data, 1)
        stats = clus.iteration_stats
        losses = np.array([
            stats.at(i).obj for i in range(stats.size())
        ])

        if verbose:
            print(f'k-means loss evolution: {losses.tolist()}')
            print(f"{time() - st:.3f} sec. taken for clustering features.")
        return np.array([int(n[0]) for n in nearest_centroids]), losses[-1]

    @staticmethod
    def _restore_shape(clusters: np.ndarray, original_shape):
        if len(original_shape) == 2:
            return clusters

        elif len(original_shape) == 3:
            h, w = original_shape[1:]
            return clusters.reshape(h, w)

        elif len(original_shape) == 4:
            b, _, h, w = original_shape
            return clusters.reshape(b, h, w)

    def forward(self, features: Union[np.ndarray, torch.Tensor], k: int, **kwargs):
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        original_shape = features.shape
        features = self._check_shape(features)
        try:
            pca = kwargs.pop("pca")
        except KeyError:
            pca = None

        if pca is not None:
            features = self.preprocess_features(features, pca=pca)

        clusters = getattr(self, f"_run_kmeans_{'gpu' if self.use_gpu else 'cpu'}")(features, k, **kwargs)[0]
        clusters = self._restore_shape(clusters, original_shape)
        return clusters

