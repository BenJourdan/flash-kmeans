
from __future__ import annotations

from typing import Optional
import warnings
from flash_kmeans.torch_fallback import euclid_assign_torch_native_chunked, batch_kmeans_Euclid_torch_native
import torch

try:
    from flash_kmeans.kmeans_triton_impl import batch_mini_batch_kmeans_Euclid, batch_kmeans_Euclid 
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton
    from flash_kmeans.kmeans_large import kmeans_largeN, kmeans_largeN_assign
    _HAS_TRITON_IMPL = True
except Exception:
    _HAS_TRITON_IMPL = False


def _require_triton_cuda():
    if not _HAS_TRITON_IMPL:
        raise RuntimeError(
            "flash_kmeans Triton kernels are not available. "
            "Ensure the package modules are importable."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the Triton-backed k-means implementation.")


def _normalize_input(data: torch.Tensor):
    if data.ndim == 2:
        N, D = data.shape
        B = None
        x_b = data.unsqueeze(0)
    elif data.ndim == 3:
        B, N, D = data.shape
        x_b = data
    else:
        raise ValueError("data must be of shape (n_samples, n_features) or (batch_size, n_samples, n_features)")
    return x_b, B, N, D


def _validate_feature_dim(D: int, expected_d: int):
    if D != expected_d:
        raise ValueError(f"Expected input with feature dimension d={expected_d}, but got D={D}.")


def _prepare_init_centroids(
    init_centroids: Optional[torch.Tensor],
    *,
    batch_size: int,
    k: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
):
    if init_centroids is None:
        return None

    if init_centroids.ndim == 2:
        if init_centroids.shape != (k, d):
            raise ValueError(f"init_centroids must have shape ({k}, {d}) or ({batch_size}, {k}, {d}).")
        centroids_b = init_centroids.unsqueeze(0).expand(batch_size, -1, -1)
    elif init_centroids.ndim == 3:
        if init_centroids.shape != (batch_size, k, d):
            raise ValueError(f"init_centroids must have shape ({k}, {d}) or ({batch_size}, {k}, {d}).")
        centroids_b = init_centroids
    else:
        raise ValueError("init_centroids must be a rank-2 or rank-3 tensor.")

    return centroids_b.to(device=device, dtype=dtype, copy=False)


def _compute_inertia_b(x_b: torch.Tensor, centroids_b: torch.Tensor, cluster_ids_b: torch.Tensor) -> torch.Tensor:
    assigned_centroids = torch.gather(
        centroids_b,
        dim=1,
        index=cluster_ids_b.long().unsqueeze(-1).expand(-1, -1, centroids_b.shape[-1]),
    )
    return ((x_b - assigned_centroids) ** 2).sum(dim=(-1, -2))


class FlashKMeans:
    """
    Fast batched K-Means clustering implemented with Triton GPU kernels.

    Parameters
    ----------
    d : int
        Feature dimensionality (n_features).
    k : int
        Number of clusters. (n_clusters)
    niter : int, default=25
        Maximum iterations.
    tol : float | None, default=1e-8
        Convergence tolerance on centroid shift. Use `None` to disable early stopping.
    use_triton : bool, default=True
        Whether to use triton implementation. If False, falls back to PyTorch implementation.
    seed : int, default=0
        Random seed for centroid initialization.
    chunk_size_data : int, default=32768
        Only used when fallback to PyTorch implementation.
        Chunk size along the data dimension for assignment/update steps.
    chunk_size_centroids : int, default=1024
        Only used when fallback to PyTorch implementation.
        Chunk size along the centroid dimension for assignment/update steps.
    chunk_size_data_cpu : int, default=1048576
        Only when n_samples is too large to fit into GPU memory, this parameter controls
        the chunk size of n_samples when copying data from CPU to GPU in chunks.
    verbose : bool, default=False
        Whether to print per-iteration info.
    dtype : torch.dtype, optional
        Compute Data type for algorithm.
    device : torch.device | None
        Target device. Defaults to "cuda:0" when available.
        Currently, only CUDA devices are supported.
    """

    def __init__(
        self,
        d: int,
        k: int,
        niter: int = 25,
        tol: Optional[float] = 1e-8,
        use_triton: bool = True,
        seed: int = 0,
        chunk_size_data: int = 32768,
        chunk_size_centroids: int = 1024,
        chunk_size_data_cpu: int = 1048576,
        verbose: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        self.d = int(d)
        self.k = int(k)
        self.niter = int(niter)
        self.tol = None if tol is None else float(tol)
        self.use_triton = bool(use_triton)
        self.seed = int(seed)
        self.chunk_size_data = int(chunk_size_data)
        self.chunk_size_centroids = int(chunk_size_centroids)
        self.chunk_size_data_cpu = int(chunk_size_data_cpu)
        self.verbose = bool(verbose)
        self.dtype = dtype

        if self.use_triton:
            try:
                _require_triton_cuda()
            except RuntimeError as e:
                warnings.warn(f"Falling back to PyTorch implementation: {e}", RuntimeWarning)
                self.use_triton = False

        # default device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.centroids_b = None
        self.cluster_ids_b = None
        self._batch_size = None
        self.n_iter_ = None
        self.inertia_b = None
        self.inertia_ = None


    def train(self, data: torch.Tensor, *, init_centroids: Optional[torch.Tensor] = None):
        """
        Fit KMeans on data and store centroids.

        Parameters
        ----------
        data : torch.Tensor
            Accepts Shape:
            - (n_samples, n_features)
            - (batch_size, n_samples, n_features)

            if data is from GPU, it will process directly on GPU.
            if data is from CPU, it will copy & process data on GPU by chunk_size_data_cpu.
        init_centroids : torch.Tensor | None, optional
            Optional initial centroids with shape `(k, d)` or `(B, k, d)`.
            A rank-2 tensor is broadcast across the batch dimension.

        """
        x_b, B, N, D = _normalize_input(data)
        _validate_feature_dim(D, self.d)

        # Set random seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if data.device.type == "cpu" and N > self.chunk_size_data_cpu:
            # handle for large N on CPU
            assert B is None, "Batched data with large N on CPU is not supported yet."
            assert self.use_triton, "process large N data requires triton implementation." 
            cluster_ids_b, centroids_b  = kmeans_largeN(
                x_b[0],
                self.k,
                max_iters=self.niter,
                tol=self.tol,
                init_centroids=init_centroids,
                verbose=self.verbose,
                dtype=self.dtype,
                BLOCK_N=self.chunk_size_data_cpu,
            )
            centroids_b.unsqueeze_(0)
            cluster_ids_b.unsqueeze_(0)
            iters_run = None
            inertia_b = None
        else:
            # Ensure CUDA + dtype
            compute_dtype = self.dtype or x_b.dtype
            x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False)
            init_centroids_b = _prepare_init_centroids(
                init_centroids,
                batch_size=x_b.shape[0],
                k=self.k,
                d=self.d,
                device=self.device,
                dtype=compute_dtype,
            )

            if self.use_triton:
                # Run batched Triton KMeans (Euclidean)
                cluster_ids_b, centroids_b, iters_run = batch_kmeans_Euclid(
                    x_b,
                    self.k,
                    max_iters=self.niter,
                    tol=self.tol,
                    init_centroids=init_centroids_b,
                    verbose=self.verbose,
                )
            else:
                # Run batched PyTorch KMeans (Euclidean)
                cluster_ids_b, centroids_b, iters_run = batch_kmeans_Euclid_torch_native(
                    x_b,
                    self.k,
                    max_iters=self.niter,
                    tol=self.tol,
                    init_centroids=init_centroids_b,
                    verbose=self.verbose,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
                )
            inertia_b = _compute_inertia_b(x_b, centroids_b, cluster_ids_b)
 
        self.centroids_b = centroids_b
        self.cluster_ids_b = cluster_ids_b
        self._batch_size = B
        self.n_iter_ = iters_run
        self.inertia_b = inertia_b
        self.inertia_ = None if inertia_b is None else (inertia_b[0].item() if B is None else inertia_b)

    def fit(self, data: torch.Tensor, *, init_centroids: Optional[torch.Tensor] = None):
        """
        Fit KMeans on data and return `self`.

        Parameters
        ----------
        data : torch.Tensor
            Training data with shape `(n_samples, n_features)` or
            `(batch_size, n_samples, n_features)`.
        init_centroids : torch.Tensor | None, optional
            Optional initial centroids with shape `(k, d)` or `(B, k, d)`.
        """
        self.train(data, init_centroids=init_centroids)
        return self

    def predict(self, data: torch.Tensor) -> torch.LongTensor:
        """
        Assign each point to the nearest centroid using the Triton assign kernel.

        Parameters
        ----------
        data : torch.Tensor
            Accepts Shape:
            - (n_samples, n_features)
            - (batch_size, n_samples, n_features)

        If model was trained batched (batch_size>1), prediction must be provided with the same batch_size.
        """

        if self.centroids_b is None:
            raise RuntimeError("Model not trained. Call train() or fit() first.")

        # Normalize input shape
        x_b, B, N, D = _normalize_input(data)
        _validate_feature_dim(D, self.d)

        if B != self._batch_size:
            raise ValueError(
                f"Model was trained with batch size B={self._batch_size}, "
                f"but predict received B={B}. Provide matching batch size."
            )
        
        if data.device.type == "cpu" and N > self.chunk_size_data_cpu:
            # handle for large N on CPU
            assert B is None, "Batched data with large N on CPU is not supported yet."
            assert self.use_triton, "process large N data requires triton implementation." 
            labels = kmeans_largeN_assign(
                x_b[0],
                self.centroids_b[0],
                dtype=self.dtype,
                BLOCK_N=self.chunk_size_data_cpu,
            )
            return labels  # (N,)
    
        # Prepare tensors for kernel call
        compute_dtype = self.dtype or x_b.dtype 
        x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False)
 
        x_sq = (x_b ** 2).sum(dim=-1)

        if self.use_triton:
            # Call Triton assignment kernel
            labels_b = euclid_assign_triton(x_b, self.centroids_b, x_sq)
        else:
            # Call PyTorch assignment fallback
            labels_b = euclid_assign_torch_native_chunked(
                x_b,
                self.centroids_b,
                x_sq,
                chunk_size_N=self.chunk_size_data,
                chunk_size_K=self.chunk_size_centroids,
            )

        if B is None:
            return labels_b.squeeze(0)  # (N,)
        return labels_b  # (B, N)

    def fit_predict(self, data: torch.Tensor, *, init_centroids: Optional[torch.Tensor] = None) -> torch.tensor:
        """
        Fit KMeans on data and store centroids.

        Parameters
        ----------
        data : torch.Tensor
            Input data for clustering.
            data shape accepts:
            - (n_samples, n_features)
            - (batch_size, n_samples, n_features)
        init_centroids : torch.Tensor | None, optional
            Optional initial centroids with shape `(k, d)` or `(B, k, d)`.

        
        Returns
        -------
        labels : torch.LongTensor (int64)
            Shape depending on input:
            - (n_samples,) if input was (n_samples, n_features)
            - (batch_size, n_samples) if input was (batch_size, n_samples, n_features)

        """
        # cluster_ids: (B, N)
        self.train(data, init_centroids=init_centroids)
        return self.cluster_ids_b.squeeze(0) if self._batch_size is None else self.cluster_ids_b


class FlashMiniBatchKMeans:
    """
    Fast IO aware Mini-Batch K-Means clustering implemented with Triton GPU kernels.

    Parameters
    ----------
    d : int
        Feature dimensionality (n_features).
    k : int
        Number of clusters. (n_clusters)
    mini_batch_size : int, default=1024
        Number of samples per mini-batch. Only used for mini-batch k-means iterations.
    learning_rate : str, default="adaptive"
        Learning rate schedule for updating centroids in mini-batch k-means.
        "adaptive" (default): adaptive ema-style update of Schwartzman (2023)
            "Mini-batch k-means terminates within O(d/ɛ) iterations".,
        "classic": classic constant learning rate of Sully.
    epochs : int, default=10
        Maximum number of epochs. One epoch corresponds to iterating through the entire dataset once in mini-batch k-means. 
    tol : float | None, default=1e-8
        Convergence tolerance on epoch-level inertia improvement. Use `None` to disable early stopping.
    use_triton : bool, default=True
        Whether to use triton implementation. If False, falls back to PyTorch implementation.
    seed : int, default=0
        Random seed for centroid initialization.
    chunk_size_data : int, default=32768
        Only used when fallback to PyTorch implementation.
        Chunk size along the data dimension for assignment/update steps.
    chunk_size_centroids : int, default=1024
        Only used when fallback to PyTorch implementation.
        Chunk size along the centroid dimension for assignment/update steps.
    chunk_size_data_cpu : int, default=1048576
        Only when n_samples is too large to fit into GPU memory, this parameter controls
        the chunk size of n_samples when copying data from CPU to GPU in chunks.
    verbose : bool, default=False
        Whether to print per-iteration info.
    dtype : torch.dtype, optional
        Compute Data type for algorithm.
    device : torch.device | None
        Target device. Defaults to "cuda:0" when available.
        Currently, only CUDA devices are supported.
    """

    def __init__(
        self,
        d: int,
        k: int,
        mini_batch_size: int = 1024,
        learning_rate: str = "adaptive",
        epochs: int = 10,
        tol: Optional[float] = 1e-8,
        use_triton: bool = True,
        seed: int = 0,
        chunk_size_data: int = 32768,
        chunk_size_centroids: int = 1024,
        chunk_size_data_cpu: int = 1048576,
        verbose: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        self.d = int(d)
        self.k = int(k)
        self.mini_batch_size = int(mini_batch_size)
        self.learning_rate = learning_rate
        self.epochs = int(epochs)
        self.tol = None if tol is None else float(tol)
        self.use_triton = bool(use_triton)
        self.seed = int(seed)
        self.chunk_size_data = int(chunk_size_data)
        self.chunk_size_centroids = int(chunk_size_centroids)
        self.chunk_size_data_cpu = int(chunk_size_data_cpu)
        self.verbose = bool(verbose)
        self.dtype = dtype

        if self.use_triton:
            try:
                _require_triton_cuda()
            except RuntimeError as e:
                warnings.warn(f"Falling back to PyTorch implementation: {e}", RuntimeWarning)
                self.use_triton = False

        # default device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.centroids_b = None
        self.cluster_ids_b = None
        self.cluster_counts_b = None
        self._batch_size = None
        self.n_iter_ = None
        self.n_steps_ = None
        self.inertia_b = None
        self.inertia_ = None
        self._seed_call_count = 0

        # assert mini-batch size is valid for mini-batch k-means
        if self.mini_batch_size is not None and self.mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be a positive integer for mini-batch k-means.")

        # assert learning rate is valid for mini-batch k-means
        if self.learning_rate not in ["adaptive", "classic"]:
            raise ValueError("learning_rate must be either 'adaptive' or 'classic' for mini-batch k-means.")

    def _seed_rng(self, *, reset: bool):
        if reset:
            self._seed_call_count = 0

        seed = self.seed + self._seed_call_count
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self._seed_call_count += 1

    def _fit_impl(
        self,
        data: torch.Tensor,
        *,
        init_centroids: Optional[torch.Tensor],
        cluster_counts_b: Optional[torch.Tensor],
        epochs: int,
        reset_state: bool,
    ):
        """
        Internal shared fit implementation for the mini-batch estimator.

        Parameters
        ----------
        data : torch.Tensor
            Accepts Shape:
            - (n_samples, n_features)
            - (batch_size, n_samples, n_features)

            if data is from GPU, it will process directly on GPU.
            if data is from CPU, it will copy & process data on GPU by chunk_size_data_cpu.
        init_centroids : torch.Tensor | None
            Optional initial centroids with shape `(k, d)` or `(B, k, d)`.
        cluster_counts_b : torch.Tensor | None
            Optional persistent per-cluster counts of shape `(B, k)` used to resume
            classic mini-batch training.
        epochs : int
            Number of epochs to execute for this call.
        reset_state : bool
            Whether to reset estimator state and RNG sequence before fitting.

        """
        x_b, B, N, D = _normalize_input(data)
        _validate_feature_dim(D, self.d)
        self._seed_rng(reset=reset_state)

        if not reset_state and self._batch_size != B:
            raise ValueError(
                f"Model state was created with batch size B={self._batch_size}, "
                f"but partial_fit received B={B}. Provide matching batch size."
            )

        if data.device.type == "cpu" and N > self.chunk_size_data_cpu:
            # handle for large N on CPU
            assert B is None, "Batched data with large N on CPU is not supported yet."
            assert self.use_triton, "process large N data requires triton implementation." 

            raise NotImplementedError("Mini-batch k-means for large N on CPU is not implemented yet.")
            # cluster_ids_b, centroids_b  = kmeans_largeN(
            #     x_b[0],
            #     self.k,
            #     max_iters=self.niter,
            #     tol=self.tol,
            #     verbose=self.verbose,
            #     dtype=self.dtype,
            #     BLOCK_N=self.chunk_size_data_cpu,
            # )
            # centroids_b.unsqueeze_(0)
            # cluster_ids_b.unsqueeze_(0)
        else:
            # Ensure CUDA + dtype
            compute_dtype = self.dtype or x_b.dtype
            x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False)
            init_centroids_b = _prepare_init_centroids(
                init_centroids,
                batch_size=x_b.shape[0],
                k=self.k,
                d=self.d,
                device=self.device,
                dtype=compute_dtype,
            )
            init_cluster_counts_b = None
            if cluster_counts_b is not None:
                init_cluster_counts_b = cluster_counts_b.to(device=self.device, dtype=torch.float32, copy=False)

            if self.use_triton:
                # Run batched Triton KMeans (Euclidean)
                cluster_ids_b, centroids_b, epochs_run, cluster_counts_b = batch_mini_batch_kmeans_Euclid(
                    x_b,
                    self.k,
                    mini_batch_size=self.mini_batch_size,
                    epochs=epochs,
                    learning_rate=self.learning_rate,
                    tol=self.tol,
                    init_centroids=init_centroids_b,
                    init_cluster_counts=init_cluster_counts_b,
                    return_cluster_counts=True,
                    verbose=self.verbose,
                )
            else:
                raise NotImplementedError("Mini-batch k-means with PyTorch fallback is not implemented yet.")
            inertia_b = _compute_inertia_b(x_b, centroids_b, cluster_ids_b)

        self.centroids_b = centroids_b
        self.cluster_ids_b = cluster_ids_b
        self.cluster_counts_b = cluster_counts_b
        self._batch_size = B
        batches_per_epoch = (N + self.mini_batch_size - 1) // self.mini_batch_size
        if reset_state or self.n_iter_ is None:
            self.n_iter_ = int(epochs_run)
            self.n_steps_ = int(epochs_run * batches_per_epoch)
        else:
            self.n_iter_ += int(epochs_run)
            self.n_steps_ += int(epochs_run * batches_per_epoch)
        self.inertia_b = inertia_b
        self.inertia_ = inertia_b[0].item() if B is None else inertia_b

    def train(self, data: torch.Tensor, *, init_centroids: Optional[torch.Tensor] = None):
        """
        Fit mini-batch KMeans on data, reset estimator state, and store centroids.

        Parameters
        ----------
        data : torch.Tensor
            Training data with shape `(n_samples, n_features)` or
            `(batch_size, n_samples, n_features)`.
        init_centroids : torch.Tensor | None, optional
            Optional initial centroids with shape `(k, d)` or `(B, k, d)`.
            A rank-2 tensor is broadcast across the batch dimension.
        """
        self._fit_impl(
            data,
            init_centroids=init_centroids,
            cluster_counts_b=None,
            epochs=self.epochs,
            reset_state=True,
        )

    def partial_fit(
        self,
        data: torch.Tensor,
        *,
        init_centroids: Optional[torch.Tensor] = None,
        epochs: Optional[int] = None,
    ):
        """
        Continue mini-batch training from the current estimator state.

        Parameters
        ----------
        data : torch.Tensor
            Training data with shape `(n_samples, n_features)` or
            `(batch_size, n_samples, n_features)`.
        init_centroids : torch.Tensor | None, optional
            Optional initial centroids with shape `(k, d)` or `(B, k, d)`.
            This may only be provided on the first call, before model state exists.
        epochs : int | None, optional
            Number of epochs to run for this update. Defaults to `self.epochs`.

        Returns
        -------
        FlashMiniBatchKMeans
            The fitted estimator instance.
        """
        run_epochs = self.epochs if epochs is None else int(epochs)
        if run_epochs <= 0:
            raise ValueError("epochs must be a positive integer for partial_fit.")

        if self.centroids_b is None:
            init_centroids_b = init_centroids
            cluster_counts_b = None
            reset_state = True
        else:
            if init_centroids is not None:
                raise ValueError("init_centroids cannot be provided after the model has already been fitted.")
            init_centroids_b = self.centroids_b
            cluster_counts_b = self.cluster_counts_b
            reset_state = False

        self._fit_impl(
            data,
            init_centroids=init_centroids_b,
            cluster_counts_b=cluster_counts_b,
            epochs=run_epochs,
            reset_state=reset_state,
        )
        return self

    def fit(self, data: torch.Tensor, *, init_centroids: Optional[torch.Tensor] = None):
        """
        Fit mini-batch KMeans on data and return `self`.

        Parameters
        ----------
        data : torch.Tensor
            Training data with shape `(n_samples, n_features)` or
            `(batch_size, n_samples, n_features)`.
        init_centroids : torch.Tensor | None, optional
            Optional initial centroids with shape `(k, d)` or `(B, k, d)`.
        """
        self.train(data, init_centroids=init_centroids)
        return self

    def predict(self, data: torch.Tensor) -> torch.LongTensor:
        """
        Assign each point to the nearest centroid using the Triton assign kernel.

        Parameters
        ----------
        data : torch.Tensor
            Accepts Shape:
            - (n_samples, n_features)
            - (batch_size, n_samples, n_features)

        If model was trained batched (batch_size>1), prediction must be provided with the same batch_size.
        """

        if self.centroids_b is None:
            raise RuntimeError("Model not trained. Call train() or fit() first.")

        # Normalize input shape
        x_b, B, N, D = _normalize_input(data)
        _validate_feature_dim(D, self.d)

        if B != self._batch_size:
            raise ValueError(
                f"Model was trained with batch size B={self._batch_size}, "
                f"but predict received B={B}. Provide matching batch size."
            )
        
        if data.device.type == "cpu" and N > self.chunk_size_data_cpu:
            # handle for large N on CPU
            assert B is None, "Batched data with large N on CPU is not supported yet."
            assert self.use_triton, "process large N data requires triton implementation." 
            raise NotImplementedError("Mini-batch k-means for large N on CPU is not implemented yet.")
            # labels = kmeans_largeN_assign(
            #     x_b[0],
            #     self.centroids_b[0],
            #     dtype=self.dtype,
            #     BLOCK_N=self.chunk_size_data_cpu,
            # )
            # return labels  # (N,)
    
        # Prepare tensors for kernel call
        compute_dtype = self.dtype or x_b.dtype 
        x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False)
 
        x_sq = (x_b ** 2).sum(dim=-1)

        if self.use_triton:
            # Call Triton assignment kernel
            labels_b = euclid_assign_triton(x_b, self.centroids_b, x_sq)
        else:
            # Call PyTorch assignment fallback
            raise NotImplementedError("Mini-batch k-means with PyTorch fallback is not implemented yet.")
            # labels_b = euclid_assign_torch_native_chunked(
            #     x_b,
            #     self.centroids_b,
            #     x_sq,
            #     chunk_size_N=self.chunk_size_data,
            #     chunk_size_K=self.chunk_size_centroids,
            # )

        if B is None:
            return labels_b.squeeze(0)  # (N,)
        return labels_b  # (B, N)

    def fit_predict(self, data: torch.Tensor, *, init_centroids: Optional[torch.Tensor] = None) -> torch.tensor:
        """
        Fit mini-batch KMeans on data and return final labels for that data.

        Parameters
        ----------
        data : torch.Tensor
            Input data for clustering.
            data shape accepts:
            - (n_samples, n_features)
            - (batch_size, n_samples, n_features)
        init_centroids : torch.Tensor | None, optional
            Optional initial centroids with shape `(k, d)` or `(B, k, d)`.

        
        Returns
        -------
        labels : torch.LongTensor (int64)
            Shape depending on input:
            - (n_samples,) if input was (n_samples, n_features)
            - (batch_size, n_samples) if input was (batch_size, n_samples, n_features)

        """
        # cluster_ids: (B, N)
        self.train(data, init_centroids=init_centroids)
        return self.cluster_ids_b.squeeze(0) if self._batch_size is None else self.cluster_ids_b
