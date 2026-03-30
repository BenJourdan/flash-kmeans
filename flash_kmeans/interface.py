
from __future__ import annotations

from typing import Optional
import warnings
from flash_kmeans._minibatch import (
    MiniBatchTrainingState,
    initialize_minibatch_state,
    reset_minibatch_state_progress,
    run_mini_batch_training,
)
from flash_kmeans.torch_fallback import euclid_assign_torch_native_chunked, batch_kmeans_Euclid_torch_native
from flash_kmeans.initialization import (
    kmeans_parallel_init_centroids,
    kmeans_plusplus_init_centroids,
    sample_random_init_centroids,
)
import torch

try:
    from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid
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


def _validate_init_spec(
    init: Optional[str | torch.Tensor],
    *,
    k: int,
    d: int,
):
    if init is None:
        return
    if isinstance(init, str):
        if init not in {"random", "kmeans++", "kmeans||"}:
            raise ValueError("init must be one of {'random', 'kmeans++', 'kmeans||'} or a tensor of centroids.")
        return
    if not isinstance(init, torch.Tensor):
        raise TypeError("init must be either a string strategy or a torch.Tensor.")
    if init.ndim == 2:
        if init.shape != (k, d):
            raise ValueError(f"init tensor must have shape ({k}, {d}) or (B, {k}, {d}).")
        return
    if init.ndim == 3:
        if init.shape[1:] != (k, d):
            raise ValueError(f"init tensor must have shape ({k}, {d}) or (B, {k}, {d}).")
        return
    raise ValueError("init tensor must be a rank-2 or rank-3 tensor.")


def _validate_init_tuning(
    *,
    init_oversampling_factor: float,
    init_rounds: int,
):
    if init_oversampling_factor <= 0:
        raise ValueError("init_oversampling_factor must be positive.")
    if init_rounds <= 0:
        raise ValueError("init_rounds must be a positive integer.")


def _timed_init_centroids(
    x_b: torch.Tensor,
    *,
    init: Optional[str | torch.Tensor],
    init_centroids: Optional[torch.Tensor],
    batch_size: int,
    k: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    init_oversampling_factor: float,
    init_rounds: int,
) -> tuple[Optional[torch.Tensor], Optional[str], float]:
    effective_init: Optional[str | torch.Tensor]
    if init_centroids is not None:
        effective_init = init_centroids
    else:
        effective_init = init

    if effective_init is None:
        return None, None, 0.0

    if isinstance(effective_init, torch.Tensor):
        prepared = _prepare_init_centroids(
            effective_init,
            batch_size=batch_size,
            k=k,
            d=d,
            device=device,
            dtype=dtype,
        )
        return prepared, "tensor", 0.0

    if effective_init not in {"random", "kmeans++", "kmeans||"}:
        raise ValueError("init must resolve to 'random', 'kmeans++', 'kmeans||', or a tensor of centroids.")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    if start is not None:
        start.record()
    else:
        import time
        start_time = time.perf_counter()

    if effective_init == "random":
        centroids_b = sample_random_init_centroids(x_b, k, seed=seed)
    elif effective_init == "kmeans++":
        centroids_b = kmeans_plusplus_init_centroids(x_b, k, seed=seed)
    else:
        centroids_b = kmeans_parallel_init_centroids(
            x_b,
            k,
            oversampling_factor=init_oversampling_factor,
            rounds=init_rounds,
            seed=seed,
        )

    if end is not None:
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = float(start.elapsed_time(end))
    else:
        elapsed_ms = float((time.perf_counter() - start_time) * 1000.0)

    prepared = _prepare_init_centroids(
        centroids_b,
        batch_size=batch_size,
        k=k,
        d=d,
        device=device,
        dtype=dtype,
    )
    return prepared, effective_init, elapsed_ms


def _compute_inertia_b(x_b: torch.Tensor, centroids_b: torch.Tensor, cluster_ids_b: torch.Tensor) -> torch.Tensor:
    assigned_centroids = torch.gather(
        centroids_b,
        dim=1,
        index=cluster_ids_b.long().unsqueeze(-1).expand(-1, -1, centroids_b.shape[-1]),
    )
    return ((x_b - assigned_centroids) ** 2).sum(dim=(-1, -2))


def _move_to_compute_device(
    x_b: torch.Tensor,
    *,
    device: torch.device,
    dtype: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.dtype]:
    compute_dtype = dtype or x_b.dtype
    return x_b.to(device=device, dtype=compute_dtype, copy=False), compute_dtype


def _assign_euclidean_labels(
    x_b: torch.Tensor,
    centroids_b: torch.Tensor,
    *,
    use_triton: bool,
    chunk_size_data: int,
    chunk_size_centroids: int,
    use_heuristic: bool = True,
) -> torch.LongTensor:
    x_sq = (x_b ** 2).sum(dim=-1)
    if use_triton:
        return euclid_assign_triton(x_b, centroids_b, x_sq, use_heuristic=use_heuristic)
    return euclid_assign_torch_native_chunked(
        x_b,
        centroids_b,
        x_sq,
        chunk_size_N=chunk_size_data,
        chunk_size_K=chunk_size_centroids,
    )


def _require_minibatch_backend(device: torch.device, use_triton: bool):
    if not use_triton:
        raise RuntimeError("FlashMiniBatchKMeans requires Triton/CUDA; use_triton=False is not supported.")
    _require_triton_cuda()
    if device.type != "cuda":
        raise RuntimeError("FlashMiniBatchKMeans requires a CUDA device.")


def _normalize_mini_batch_budget(
    *,
    epochs: Optional[int],
    iterations: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    if epochs is not None and iterations is not None:
        raise ValueError("epochs and iterations are mutually exclusive.")

    if epochs is None and iterations is None:
        return 100, None

    if epochs is not None:
        epochs = int(epochs)
        if epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        return epochs, None

    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be a positive integer.")
    return None, iterations


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
    init : {"random", "kmeans++", "kmeans||"} | torch.Tensor | None, default=None
        Initialization strategy or explicit initial centroids. If a tensor is provided,
        it must have shape `(k, d)` or `(B, k, d)`. `fit(..., init_centroids=...)`
        overrides this constructor-level value. `None` preserves the implementation's
        default random initialization path.
    init_oversampling_factor : float, default=2.0
        Oversampling factor used when `init="kmeans||"`.
    init_rounds : int, default=5
        Number of candidate-sampling rounds used when `init="kmeans||"`.
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
        init: Optional[str | torch.Tensor] = None,
        init_oversampling_factor: float = 2.0,
        init_rounds: int = 5,
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
        _validate_init_spec(init, k=self.k, d=self.d)
        _validate_init_tuning(
            init_oversampling_factor=float(init_oversampling_factor),
            init_rounds=int(init_rounds),
        )
        self.init = init
        self.init_oversampling_factor = float(init_oversampling_factor)
        self.init_rounds = int(init_rounds)
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
        self.init_centroids_b = None
        self.init_time_ms_ = None
        self.init_strategy_ = None


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
            A rank-2 tensor is broadcast across the batch dimension. If omitted,
            the estimator uses the constructor-level `init` strategy or tensor.

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
            init_centroids_b, init_strategy, init_time_ms = _timed_init_centroids(
                x_b,
                init=self.init,
                init_centroids=init_centroids,
                batch_size=x_b.shape[0],
                k=self.k,
                d=self.d,
                device=x_b.device,
                dtype=x_b.dtype,
                seed=self.seed,
                init_oversampling_factor=self.init_oversampling_factor,
                init_rounds=self.init_rounds,
            )
            cluster_ids_b, centroids_b  = kmeans_largeN(
                x_b[0],
                self.k,
                max_iters=self.niter,
                tol=self.tol,
                init_centroids=None if init_centroids_b is None else init_centroids_b[0],
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
            x_b, compute_dtype = _move_to_compute_device(
                x_b,
                device=self.device,
                dtype=self.dtype,
            )
            init_centroids_b, init_strategy, init_time_ms = _timed_init_centroids(
                x_b,
                init=self.init,
                init_centroids=init_centroids,
                batch_size=x_b.shape[0],
                k=self.k,
                d=self.d,
                device=self.device,
                dtype=compute_dtype,
                seed=self.seed,
                init_oversampling_factor=self.init_oversampling_factor,
                init_rounds=self.init_rounds,
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
        self.init_centroids_b = init_centroids_b
        self.init_time_ms_ = init_time_ms
        self.init_strategy_ = init_strategy

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
            If omitted, the estimator uses the constructor-level `init` strategy or tensor.
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
        x_b, _ = _move_to_compute_device(
            x_b,
            device=self.device,
            dtype=self.dtype,
        )
        labels_b = _assign_euclidean_labels(
            x_b,
            self.centroids_b,
            use_triton=self.use_triton,
            chunk_size_data=self.chunk_size_data,
            chunk_size_centroids=self.chunk_size_centroids,
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
            If omitted, the estimator uses the constructor-level `init` strategy or tensor.

        
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
    epochs : int | None, default=None
        Number of completed epochs to execute per `fit` / `partial_fit` call.
        Mutually exclusive with `iterations`. If neither is provided, defaults to `100`.
    iterations : int | None, default=None
        Number of mini-batch iterations to execute per `fit` / `partial_fit` call.
        Mutually exclusive with `epochs`.
    tol : float | None, default=1e-8
        Convergence tolerance. In epoch mode it is checked on epoch-level inertia
        improvement. In iteration mode it is checked on mini-batch inertia improvement.
    use_triton : bool, default=True
        Retained for constructor compatibility. Mini-batch training requires Triton/CUDA
        and fails fast at `fit` / `partial_fit` / `predict` time when disabled.
    seed : int, default=0
        Random seed for centroid initialization.
    chunk_size_data : int, default=32768
        Retained for constructor compatibility with older mini-batch call sites.
    chunk_size_centroids : int, default=1024
        Retained for constructor compatibility with older mini-batch call sites.
    chunk_size_data_cpu : int, default=1048576
        Retained for constructor compatibility with older mini-batch call sites.
    verbose : bool, default=False
        Whether to print per-iteration info.
    use_heuristic : bool, default=True
        Whether Triton mini-batch kernels should use the heuristic launch config.
    init : {"random", "kmeans++", "kmeans||"} | torch.Tensor | None, default=None
        Initialization strategy or explicit initial centroids. If a tensor is provided,
        it must have shape `(k, d)` or `(B, k, d)`. `fit(..., init_centroids=...)`
        overrides this constructor-level value. `None` preserves the implementation's
        default random initialization path.
    init_oversampling_factor : float, default=2.0
        Oversampling factor used when `init="kmeans||"`.
    init_rounds : int, default=5
        Number of candidate-sampling rounds used when `init="kmeans||"`.
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
        epochs: Optional[int] = None,
        iterations: Optional[int] = None,
        tol: Optional[float] = 1e-8,
        use_triton: bool = True,
        seed: int = 0,
        chunk_size_data: int = 32768,
        chunk_size_centroids: int = 1024,
        chunk_size_data_cpu: int = 1048576,
        verbose: bool = False,
        use_heuristic: bool = True,
        init: Optional[str | torch.Tensor] = None,
        init_oversampling_factor: float = 2.0,
        init_rounds: int = 5,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        self.d = int(d)
        self.k = int(k)
        self.mini_batch_size = int(mini_batch_size)
        self.learning_rate = learning_rate
        self.epochs, self.iterations = _normalize_mini_batch_budget(
            epochs=epochs,
            iterations=iterations,
        )
        self.tol = None if tol is None else float(tol)
        self.use_triton = bool(use_triton)
        self.seed = int(seed)
        self.chunk_size_data = int(chunk_size_data)
        self.chunk_size_centroids = int(chunk_size_centroids)
        self.chunk_size_data_cpu = int(chunk_size_data_cpu)
        self.verbose = bool(verbose)
        self.use_heuristic = bool(use_heuristic)
        _validate_init_spec(init, k=self.k, d=self.d)
        _validate_init_tuning(
            init_oversampling_factor=float(init_oversampling_factor),
            init_rounds=int(init_rounds),
        )
        self.init = init
        self.init_oversampling_factor = float(init_oversampling_factor)
        self.init_rounds = int(init_rounds)
        self.dtype = dtype

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
        self.n_epochs_ = None
        self.last_fit_n_iter_ = None
        self.last_fit_n_epochs_ = None
        self.stopped_early_ = None
        self.inertia_b = None
        self.inertia_ = None
        self._seed_call_count = 0
        self.init_centroids_b = None
        self.init_time_ms_ = None
        self.init_strategy_ = None
        self._num_samples = None
        self._training_state: Optional[MiniBatchTrainingState] = None
        self._mini_batch_generator = None

        # assert mini-batch size is valid for mini-batch k-means
        if self.mini_batch_size is not None and self.mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be a positive integer for mini-batch k-means.")

        # assert learning rate is valid for mini-batch k-means
        if self.learning_rate not in ["adaptive", "classic"]:
            raise ValueError("learning_rate must be either 'adaptive' or 'classic' for mini-batch k-means.")

    def set_training_budget(
        self,
        *,
        epochs: Optional[int] = None,
        iterations: Optional[int] = None,
    ):
        """Update the budget used by subsequent fit/partial_fit calls."""
        self.epochs, self.iterations = _normalize_mini_batch_budget(
            epochs=epochs,
            iterations=iterations,
        )
        return self

    def reset_online_progress(
        self,
        *,
        reset_counters: bool = False,
        reset_cluster_counts: bool = False,
    ):
        """
        Reset per-call traversal/stall bookkeeping while preserving centroids.

        This is useful when reusing the estimator across evolving datasets, where the
        previous call's permutation cursor and inertia history are no longer meaningful.
        Optionally clear accumulated cluster counts as well when the learning-rate
        schedule should restart on a new dataset.
        """
        if self._training_state is not None:
            self._training_state = reset_minibatch_state_progress(
                self._training_state,
                reset_counters=reset_counters,
                reset_cluster_counts=reset_cluster_counts,
            )

        if reset_counters:
            self.n_iter_ = 0
            self.n_epochs_ = 0

        if reset_cluster_counts and self.cluster_counts_b is not None:
            self.cluster_counts_b = torch.zeros_like(self.cluster_counts_b)

        self.last_fit_n_iter_ = 0
        self.last_fit_n_epochs_ = 0
        self.stopped_early_ = False
        return self

    def _seed_rng(self, *, reset: bool) -> int:
        if reset:
            self._seed_call_count = 0

        seed = self.seed + self._seed_call_count
        self._seed_call_count += 1
        return seed

    def _make_training_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        return generator

    def _reset_training_state(
        self,
        *,
        batch_size: Optional[int],
        num_samples: int,
        internal_batch_size: int,
        init_centroids_b: Optional[torch.Tensor],
    ):
        self._batch_size = batch_size
        self._num_samples = int(num_samples)
        self.n_iter_ = 0
        self.n_epochs_ = 0
        self.stopped_early_ = False
        self._training_state = initialize_minibatch_state(
            batch_size=internal_batch_size,
            n_clusters=self.k,
            device=self.device,
            centroids=init_centroids_b,
        )

    def _invalidate_full_state(self):
        self.cluster_ids_b = None
        self.inertia_b = None
        self.inertia_ = None

    def _refresh_full_state(self, x_b: torch.Tensor, batch_size: Optional[int]):
        cluster_ids_b = _assign_euclidean_labels(
            x_b,
            self.centroids_b,
            use_triton=True,
            chunk_size_data=self.chunk_size_data,
            chunk_size_centroids=self.chunk_size_centroids,
            use_heuristic=self.use_heuristic,
        )
        inertia_b = _compute_inertia_b(x_b, self.centroids_b, cluster_ids_b)
        self.cluster_ids_b = cluster_ids_b
        self.inertia_b = inertia_b
        self.inertia_ = inertia_b[0].item() if batch_size is None else inertia_b

    def _fit_impl(
        self,
        data: torch.Tensor,
        *,
        init_centroids: Optional[torch.Tensor],
        reset_state: bool,
        refresh_full_state: bool,
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
            If omitted, the estimator uses the constructor-level `init` strategy or tensor.
        reset_state : bool
            Whether to reset estimator state and RNG sequence before fitting.
        refresh_full_state : bool
            Whether to populate full-data labels and inertia for the training data.
        """
        x_b, B, N, D = _normalize_input(data)
        _validate_feature_dim(D, self.d)
        _require_minibatch_backend(self.device, self.use_triton)

        if not reset_state:
            if self._batch_size != B:
                raise ValueError(
                    f"Model state was created with batch size B={self._batch_size}, "
                    f"but partial_fit received B={B}. Provide matching batch size."
                )
            if self._num_samples != N:
                raise ValueError(
                    f"Model state was created with n_samples={self._num_samples}, "
                    f"but partial_fit received n_samples={N}. Provide matching n_samples."
                )
            if self._training_state is None:
                raise RuntimeError("Model state is missing. Call fit() or partial_fit() to initialize it.")

        x_b, compute_dtype = _move_to_compute_device(
            x_b,
            device=self.device,
            dtype=self.dtype,
        )
        if reset_state:
            fit_seed = self._seed_rng(reset=True)
            init_centroids_b, init_strategy, init_time_ms = _timed_init_centroids(
                x_b,
                init=self.init,
                init_centroids=init_centroids,
                batch_size=x_b.shape[0],
                k=self.k,
                d=self.d,
                device=self.device,
                dtype=compute_dtype,
                seed=fit_seed,
                init_oversampling_factor=self.init_oversampling_factor,
                init_rounds=self.init_rounds,
            )
            self._mini_batch_generator = self._make_training_generator(fit_seed)
            self._reset_training_state(
                batch_size=B,
                num_samples=N,
                internal_batch_size=x_b.shape[0],
                init_centroids_b=init_centroids_b,
            )
            self.init_centroids_b = init_centroids_b
            self.init_time_ms_ = init_time_ms
            self.init_strategy_ = init_strategy
        else:
            if self._mini_batch_generator is None:
                self._mini_batch_generator = self._make_training_generator(self.seed)
            if self._training_state.centroids is None and self.centroids_b is not None:
                self._training_state.centroids = self.centroids_b.to(
                    device=self.device,
                    dtype=compute_dtype,
                    copy=False,
                )

        prev_completed_epochs = int(self._training_state.completed_epochs)
        prev_completed_iterations = int(self._training_state.completed_iterations)
        self._training_state = run_mini_batch_training(
            x_b,
            self._training_state,
            mini_batch_size=self.mini_batch_size,
            iterations=self.iterations,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            tol=self.tol,
            verbose=self.verbose,
            use_heuristic=self.use_heuristic,
            generator=self._mini_batch_generator,
        )

        self.centroids_b = self._training_state.centroids
        self.cluster_counts_b = self._training_state.cluster_counts
        self._batch_size = B
        self._num_samples = N
        self.stopped_early_ = bool(self._training_state.stopped_early)
        self.n_epochs_ = int(self._training_state.completed_epochs)
        self.n_iter_ = int(self._training_state.completed_iterations)
        self.last_fit_n_epochs_ = self.n_epochs_ - prev_completed_epochs
        self.last_fit_n_iter_ = self.n_iter_ - prev_completed_iterations

        if refresh_full_state:
            self._refresh_full_state(x_b, B)
        else:
            self._invalidate_full_state()

    def train(
        self,
        data: torch.Tensor,
        *,
        init_centroids: Optional[torch.Tensor] = None,
    ):
        """
        Fit mini-batch KMeans on data, reset estimator state, and store centroids.

        Parameters
        ----------
        data : torch.Tensor
            Training data with shape `(n_samples, n_features)` or
            `(batch_size, n_samples, n_features)`.
        init_centroids : torch.Tensor | None, optional
            Optional initial centroids with shape `(k, d)` or `(B, k, d)`.
            A rank-2 tensor is broadcast across the batch dimension. If omitted,
            the estimator uses the constructor-level `init` strategy or tensor.
        """
        self._fit_impl(
            data,
            init_centroids=init_centroids,
            reset_state=True,
            refresh_full_state=True,
        )

    def partial_fit(
        self,
        data: torch.Tensor,
        *,
        init_centroids: Optional[torch.Tensor] = None,
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
            If omitted on the first call, the estimator uses the constructor-level
            `init` strategy or tensor.

        Returns
        -------
        FlashMiniBatchKMeans
            The fitted estimator instance.
        """
        if self.centroids_b is None:
            init_centroids_b = init_centroids
            reset_state = True
        else:
            if init_centroids is not None:
                raise ValueError("init_centroids cannot be provided after the model has already been fitted.")
            init_centroids_b = None
            reset_state = False

        self._fit_impl(
            data,
            init_centroids=init_centroids_b,
            reset_state=reset_state,
            refresh_full_state=False,
        )
        return self

    def fit(
        self,
        data: torch.Tensor,
        *,
        init_centroids: Optional[torch.Tensor] = None,
    ):
        """
        Fit mini-batch KMeans on data and return `self`.

        Parameters
        ----------
        data : torch.Tensor
            Training data with shape `(n_samples, n_features)` or
            `(batch_size, n_samples, n_features)`.
        init_centroids : torch.Tensor | None, optional
            Optional initial centroids with shape `(k, d)` or `(B, k, d)`.
            If omitted, the estimator uses the constructor-level `init` strategy or tensor.
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
        _require_minibatch_backend(self.device, self.use_triton)

        if B != self._batch_size:
            raise ValueError(
                f"Model was trained with batch size B={self._batch_size}, "
                f"but predict received B={B}. Provide matching batch size."
            )

        x_b, _ = _move_to_compute_device(
            x_b,
            device=self.device,
            dtype=self.dtype,
        )
        labels_b = _assign_euclidean_labels(
            x_b,
            self.centroids_b,
            use_triton=True,
            chunk_size_data=self.chunk_size_data,
            chunk_size_centroids=self.chunk_size_centroids,
            use_heuristic=self.use_heuristic,
        )

        if B is None:
            return labels_b.squeeze(0)  # (N,)
        return labels_b  # (B, N)

    def fit_predict(
        self,
        data: torch.Tensor,
        *,
        init_centroids: Optional[torch.Tensor] = None,
    ) -> torch.tensor:
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
            If omitted, the estimator uses the constructor-level `init` strategy or tensor.

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
