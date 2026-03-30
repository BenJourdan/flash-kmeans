import itertools

import pytest
import torch

import flash_kmeans
import flash_kmeans.interface as interface_mod
from flash_kmeans.interface import FlashKMeans, FlashMiniBatchKMeans


MODEL_CLASSES = (FlashKMeans, FlashMiniBatchKMeans)


def _feature_dim(model_cls, default: int) -> int:
    if model_cls is FlashMiniBatchKMeans:
        return 16
    return default


def _make_separated_blobs(batch_size=None, n_per_cluster=40, d=2):
    base_centers_2d = torch.tensor(
        [[-6.0, -6.0], [0.0, 6.0], [6.0, -2.0]],
        dtype=torch.float32,
    )
    if d < base_centers_2d.shape[1]:
        raise ValueError("d must be at least 2 for the synthetic blob generator.")

    base_centers = torch.zeros((base_centers_2d.shape[0], d), dtype=torch.float32)
    base_centers[:, :2] = base_centers_2d

    def _one_batch(batch_idx: int):
        centers = base_centers + torch.tensor(
            [0.5 * batch_idx, -0.75 * batch_idx] + [0.0] * (d - 2),
            dtype=torch.float32,
        )

        points = []
        labels = []
        for cluster_idx, center in enumerate(centers):
            generator = torch.Generator().manual_seed(1000 + 17 * batch_idx + cluster_idx)
            points.append(center + 0.35 * torch.randn(n_per_cluster, d, generator=generator))
            labels.append(torch.full((n_per_cluster,), cluster_idx, dtype=torch.int64))

        data = torch.cat(points, dim=0)
        target = torch.cat(labels, dim=0)
        perm = torch.randperm(
            data.shape[0],
            generator=torch.Generator().manual_seed(2000 + batch_idx),
        )
        return data[perm], target[perm], centers

    if batch_size is None:
        return _one_batch(0)

    batches = [_one_batch(batch_idx) for batch_idx in range(batch_size)]
    data, target, centers = zip(*batches)
    return torch.stack(data), torch.stack(target), torch.stack(centers)


def _best_label_accuracy(predicted: torch.Tensor, target: torch.Tensor, k: int) -> float:
    best = 0.0
    predicted = predicted.to(torch.int64).cpu()
    target = target.to(torch.int64).cpu()

    for perm in itertools.permutations(range(k)):
        mapped = predicted.clone()
        for src, dst in enumerate(perm):
            mapped[predicted == src] = dst
        best = max(best, (mapped == target).float().mean().item())

    return best


def _best_centroid_error(centroids: torch.Tensor, expected_centers: torch.Tensor) -> float:
    best = float("inf")
    centroids = centroids.to(torch.float32).cpu()
    expected_centers = expected_centers.to(torch.float32).cpu()

    for perm in itertools.permutations(range(expected_centers.shape[0])):
        candidate = centroids[list(perm)]
        error = (candidate - expected_centers).norm(dim=-1).mean().item()
        best = min(best, error)

    return best


def _make_model(model_cls, d: int, k: int, seed: int = 0):
    common_kwargs = {
        "d": d,
        "k": k,
        "tol": None,
        "seed": seed,
    }

    if model_cls is FlashMiniBatchKMeans:
        if not (interface_mod._HAS_TRITON_IMPL and torch.cuda.is_available()):
            pytest.skip("FlashMiniBatchKMeans tests require Triton/CUDA; torch fallback is not implemented.")
        return model_cls(
            **common_kwargs,
            mini_batch_size=32,
            epochs=20,
            use_triton=True,
            device=torch.device("cuda:0"),
        )

    return model_cls(
        **common_kwargs,
        niter=20,
        use_triton=False,
        device=torch.device("cpu"),
    )


def _allow_minibatch_cpu(monkeypatch):
    monkeypatch.setattr(interface_mod, "_require_minibatch_backend", lambda device, use_triton: None)


def _fake_minibatch_assign(x_b, centroids_b, x_sq, use_heuristic=True):
    return torch.zeros((x_b.shape[0], x_b.shape[1]), dtype=torch.int64, device=x_b.device)


def _make_fake_minibatch_runner(cluster_counts: torch.Tensor, captured: dict | None = None):
    def fake_run_minibatch(
        x_b,
        state,
        *,
        mini_batch_size,
        learning_rate,
        iterations,
        epochs,
        tol,
        verbose,
        use_heuristic,
        generator,
    ):
        if captured is not None:
            captured["x_shape"] = tuple(x_b.shape)
            captured["n_clusters"] = state.cluster_counts.shape[1]
            captured["mini_batch_size"] = mini_batch_size
            captured["epochs"] = epochs
            captured["iterations"] = iterations
            captured["learning_rate"] = learning_rate
            captured["init_cluster_counts"] = state.cluster_counts.detach().cpu().clone()
            if state.centroids is not None:
                captured["init_centroids"] = state.centroids.detach().cpu().clone()

        batches_per_epoch = (x_b.shape[1] + mini_batch_size - 1) // mini_batch_size
        iterations_run = iterations if iterations is not None else epochs * batches_per_epoch
        epochs_run = epochs if epochs is not None else iterations_run // batches_per_epoch

        state.centroids = None if state.centroids is None else state.centroids.clone()
        state.cluster_counts = cluster_counts.to(device=x_b.device)
        state.current_epoch_perm = None
        state.current_epoch_cursor = 0
        state.current_epoch_inertia = torch.zeros(x_b.shape[0], device=x_b.device, dtype=torch.float32)
        state.previous_epoch_inertia = None
        state.completed_iterations += int(iterations_run)
        state.completed_epochs += int(epochs_run)
        state.stopped_early = False
        return state

    return fake_run_minibatch


@pytest.mark.parametrize("model_cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_fit_predict_recovers_well_separated_clusters(model_cls):
    data, target, centers = _make_separated_blobs(d=_feature_dim(model_cls, 2))
    seed = 1 if model_cls is interface_mod.FlashMiniBatchKMeans else 7
    model = _make_model(model_cls, d=data.shape[-1], k=centers.shape[0], seed=seed)

    labels = model.fit_predict(data)
    predicted = model.predict(data)

    assert labels.shape == target.shape
    assert torch.equal(labels, predicted)
    assert torch.unique(labels).numel() == centers.shape[0]

    accuracy = _best_label_accuracy(labels, target, centers.shape[0])
    centroid_error = _best_centroid_error(model.centroids_b.squeeze(0), centers)

    assert accuracy >= 0.95
    assert centroid_error <= 0.75


@pytest.mark.parametrize("model_cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_batched_fit_predict_recovers_clusters(model_cls):
    data, target, centers = _make_separated_blobs(batch_size=2, d=_feature_dim(model_cls, 2))
    seed = 1 if model_cls is interface_mod.FlashMiniBatchKMeans else 11
    model = _make_model(model_cls, d=data.shape[-1], k=centers.shape[1], seed=seed)

    labels = model.fit_predict(data)
    predicted = model.predict(data)

    assert labels.shape == target.shape
    assert torch.equal(labels, predicted)
    assert model.centroids_b.shape == centers.shape

    for batch_idx in range(data.shape[0]):
        accuracy = _best_label_accuracy(labels[batch_idx], target[batch_idx], centers.shape[1])
        centroid_error = _best_centroid_error(model.centroids_b[batch_idx], centers[batch_idx])

        assert torch.unique(labels[batch_idx]).numel() == centers.shape[1]
        assert accuracy >= 0.95
        assert centroid_error <= 0.85


@pytest.mark.parametrize("model_cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_fit_is_deterministic_for_fixed_seed(model_cls):
    d = _feature_dim(model_cls, 4)
    data = torch.randn(90, d, generator=torch.Generator().manual_seed(123))

    first = _make_model(model_cls, d=data.shape[-1], k=3, seed=19)
    second = _make_model(model_cls, d=data.shape[-1], k=3, seed=19)

    first_labels = first.fit_predict(data)
    second_labels = second.fit_predict(data)

    assert torch.equal(first_labels, second_labels)
    assert torch.allclose(first.centroids_b, second.centroids_b)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_predict_rejects_batch_size_mismatch(model_cls):
    d = _feature_dim(model_cls, 3)
    data = torch.randn(2, 24, d, generator=torch.Generator().manual_seed(321))
    wrong_shape = torch.randn(24, d, generator=torch.Generator().manual_seed(654))

    model = _make_model(model_cls, d=data.shape[-1], k=3, seed=23)
    model.fit(data)

    with pytest.raises(ValueError, match="batch size"):
        model.predict(wrong_shape)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_fit_rejects_invalid_rank(model_cls):
    d = _feature_dim(model_cls, 6)
    invalid = torch.randn(3, 4, 5, d, generator=torch.Generator().manual_seed(777))
    model = _make_model(model_cls, d=d, k=3, seed=29)

    with pytest.raises(ValueError, match="data must be of shape"):
        model.fit(invalid)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_predict_requires_fit(model_cls):
    d = _feature_dim(model_cls, 4)
    data = torch.randn(12, d, generator=torch.Generator().manual_seed(888))
    model = _make_model(model_cls, d=d, k=3, seed=31)

    with pytest.raises(RuntimeError, match="Model not trained"):
        model.predict(data)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_fit_rejects_feature_dim_mismatch(model_cls):
    d = _feature_dim(model_cls, 4)
    data = torch.randn(32, d + 1, generator=torch.Generator().manual_seed(889))
    model = _make_model(model_cls, d=d, k=3, seed=37)

    with pytest.raises(ValueError, match="feature dimension"):
        model.fit(data)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_fit_rejects_invalid_init_centroid_shape(model_cls):
    d = _feature_dim(model_cls, 5)
    data = torch.randn(40, d, generator=torch.Generator().manual_seed(890))
    init_centroids = torch.randn(3, d + 1, generator=torch.Generator().manual_seed(891))
    model = _make_model(model_cls, d=d, k=3, seed=41)

    with pytest.raises(ValueError, match="init_centroids"):
        model.fit(data, init_centroids=init_centroids)


def test_exact_fit_forwards_broadcasted_init_centroids(monkeypatch):
    data = torch.randn(2, 12, 4, generator=torch.Generator().manual_seed(892))
    init_centroids = torch.randn(3, 4, generator=torch.Generator().manual_seed(893))
    captured = {}

    def fake_batch_kmeans(x_b, n_clusters, max_iters, tol, init_centroids, verbose):
        captured["x_shape"] = tuple(x_b.shape)
        captured["n_clusters"] = n_clusters
        captured["init_centroids"] = init_centroids.detach().cpu().clone()
        labels = torch.zeros((x_b.shape[0], x_b.shape[1]), dtype=torch.int64, device=x_b.device)
        return labels, init_centroids.clone(), 1

    monkeypatch.setattr(interface_mod, "_require_triton_cuda", lambda: None)
    monkeypatch.setattr(interface_mod, "batch_kmeans_Euclid", fake_batch_kmeans)

    model = FlashKMeans(
        d=data.shape[-1],
        k=init_centroids.shape[0],
        niter=1,
        tol=None,
        seed=43,
        use_triton=True,
        device=torch.device("cpu"),
    )
    model.fit(data, init_centroids=init_centroids)

    assert captured["x_shape"] == tuple(data.shape)
    assert captured["n_clusters"] == init_centroids.shape[0]
    assert captured["init_centroids"].shape == (data.shape[0],) + init_centroids.shape
    for batch_idx in range(data.shape[0]):
        assert torch.allclose(captured["init_centroids"][batch_idx], init_centroids)


def test_minibatch_fit_forwards_broadcasted_init_centroids(monkeypatch):
    data = torch.randn(2, 12, 16, generator=torch.Generator().manual_seed(894))
    init_centroids = torch.randn(3, 16, generator=torch.Generator().manual_seed(895))
    cluster_counts = torch.full((2, 3), 4.0, dtype=torch.float32)
    captured = {}

    _allow_minibatch_cpu(monkeypatch)
    monkeypatch.setattr(interface_mod, "run_mini_batch_training", _make_fake_minibatch_runner(cluster_counts, captured))
    monkeypatch.setattr(interface_mod, "euclid_assign_triton", _fake_minibatch_assign)

    model = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=init_centroids.shape[0],
        mini_batch_size=4,
        epochs=2,
        tol=None,
        seed=47,
        use_triton=True,
        device=torch.device("cpu"),
    )
    model.fit(data, init_centroids=init_centroids)

    assert captured["x_shape"] == tuple(data.shape)
    assert captured["n_clusters"] == init_centroids.shape[0]
    assert captured["mini_batch_size"] == 4
    assert captured["epochs"] == 2
    assert captured["iterations"] is None
    assert captured["learning_rate"] == "adaptive"
    assert captured["init_centroids"].shape == (data.shape[0],) + init_centroids.shape
    assert torch.equal(captured["init_cluster_counts"], torch.zeros_like(cluster_counts))
    for batch_idx in range(data.shape[0]):
        assert torch.allclose(captured["init_centroids"][batch_idx], init_centroids)
    assert torch.allclose(model.cluster_counts_b.cpu(), cluster_counts)


def test_exact_fit_uses_constructor_tensor_init_when_fit_init_missing(monkeypatch):
    data = torch.randn(2, 12, 4, generator=torch.Generator().manual_seed(896))
    init_centroids = torch.randn(3, 4, generator=torch.Generator().manual_seed(897))
    captured = {}

    def fake_batch_kmeans(x_b, n_clusters, max_iters, tol, init_centroids, verbose):
        captured["init_centroids"] = init_centroids.detach().cpu().clone()
        labels = torch.zeros((x_b.shape[0], x_b.shape[1]), dtype=torch.int64, device=x_b.device)
        return labels, init_centroids.clone(), 1

    monkeypatch.setattr(interface_mod, "_require_triton_cuda", lambda: None)
    monkeypatch.setattr(interface_mod, "batch_kmeans_Euclid", fake_batch_kmeans)

    model = FlashKMeans(
        d=data.shape[-1],
        k=init_centroids.shape[0],
        niter=1,
        tol=None,
        seed=61,
        init=init_centroids,
        use_triton=True,
        device=torch.device("cpu"),
    )
    model.fit(data)

    assert captured["init_centroids"].shape == (data.shape[0],) + init_centroids.shape
    for batch_idx in range(data.shape[0]):
        assert torch.allclose(captured["init_centroids"][batch_idx], init_centroids)
    assert model.init_strategy_ == "tensor"
    assert model.init_time_ms_ == 0.0


def test_exact_fit_argument_init_overrides_constructor_init(monkeypatch):
    data = torch.randn(2, 12, 4, generator=torch.Generator().manual_seed(898))
    constructor_init = torch.randn(3, 4, generator=torch.Generator().manual_seed(899))
    fit_init = torch.randn(3, 4, generator=torch.Generator().manual_seed(900))
    captured = {}

    def fake_batch_kmeans(x_b, n_clusters, max_iters, tol, init_centroids, verbose):
        captured["init_centroids"] = init_centroids.detach().cpu().clone()
        labels = torch.zeros((x_b.shape[0], x_b.shape[1]), dtype=torch.int64, device=x_b.device)
        return labels, init_centroids.clone(), 1

    monkeypatch.setattr(interface_mod, "_require_triton_cuda", lambda: None)
    monkeypatch.setattr(interface_mod, "batch_kmeans_Euclid", fake_batch_kmeans)

    model = FlashKMeans(
        d=data.shape[-1],
        k=constructor_init.shape[0],
        niter=1,
        tol=None,
        seed=63,
        init=constructor_init,
        use_triton=True,
        device=torch.device("cpu"),
    )
    model.fit(data, init_centroids=fit_init)

    for batch_idx in range(data.shape[0]):
        assert torch.allclose(captured["init_centroids"][batch_idx], fit_init)


def test_exact_fit_uses_constructor_kmeanspp_init(monkeypatch):
    data = torch.randn(2, 12, 4, generator=torch.Generator().manual_seed(901))
    generated_init = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(902))
    captured = {}

    def fake_kmeanspp(x_b, n_clusters, seed):
        captured["kmeanspp_shape"] = tuple(x_b.shape)
        captured["kmeanspp_seed"] = seed
        assert n_clusters == generated_init.shape[1]
        return generated_init.to(device=x_b.device, dtype=x_b.dtype)

    def fake_batch_kmeans(x_b, n_clusters, max_iters, tol, init_centroids, verbose):
        captured["init_centroids"] = init_centroids.detach().cpu().clone()
        labels = torch.zeros((x_b.shape[0], x_b.shape[1]), dtype=torch.int64, device=x_b.device)
        return labels, init_centroids.clone(), 1

    monkeypatch.setattr(interface_mod, "_require_triton_cuda", lambda: None)
    monkeypatch.setattr(interface_mod, "kmeans_plusplus_init_centroids", fake_kmeanspp)
    monkeypatch.setattr(interface_mod, "batch_kmeans_Euclid", fake_batch_kmeans)

    model = FlashKMeans(
        d=data.shape[-1],
        k=generated_init.shape[1],
        niter=1,
        tol=None,
        seed=67,
        init="kmeans++",
        use_triton=True,
        device=torch.device("cpu"),
    )
    model.fit(data)

    assert captured["kmeanspp_shape"] == tuple(data.shape)
    assert captured["kmeanspp_seed"] == 67
    assert torch.allclose(captured["init_centroids"], generated_init)
    assert model.init_strategy_ == "kmeans++"
    assert model.init_time_ms_ is not None
    assert model.init_time_ms_ >= 0.0


def test_minibatch_fit_uses_constructor_tensor_init_when_fit_init_missing(monkeypatch):
    data = torch.randn(2, 12, 16, generator=torch.Generator().manual_seed(903))
    init_centroids = torch.randn(3, 16, generator=torch.Generator().manual_seed(904))
    cluster_counts = torch.full((2, 3), 2.0, dtype=torch.float32)
    captured = {}

    _allow_minibatch_cpu(monkeypatch)
    monkeypatch.setattr(interface_mod, "run_mini_batch_training", _make_fake_minibatch_runner(cluster_counts, captured))
    monkeypatch.setattr(interface_mod, "euclid_assign_triton", _fake_minibatch_assign)

    model = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=init_centroids.shape[0],
        mini_batch_size=4,
        epochs=2,
        tol=None,
        seed=71,
        init=init_centroids,
        use_triton=True,
        device=torch.device("cpu"),
    )
    model.fit(data)

    assert captured["init_centroids"].shape == (data.shape[0],) + init_centroids.shape
    for batch_idx in range(data.shape[0]):
        assert torch.allclose(captured["init_centroids"][batch_idx], init_centroids)
    assert model.init_strategy_ == "tensor"
    assert model.init_time_ms_ == 0.0


def test_minibatch_constructor_budget_validation():
    with pytest.raises(ValueError, match="mutually exclusive"):
        FlashMiniBatchKMeans(
            d=16,
            k=3,
            mini_batch_size=4,
            epochs=1,
            iterations=1,
            tol=None,
            use_triton=False,
            device=torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        FlashMiniBatchKMeans(
            d=16,
            k=3,
            mini_batch_size=4,
            epochs=0,
            tol=None,
            use_triton=False,
            device=torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        FlashMiniBatchKMeans(
            d=16,
            k=3,
            mini_batch_size=4,
            iterations=0,
            tol=None,
            use_triton=False,
            device=torch.device("cpu"),
        )

    model = FlashMiniBatchKMeans(
        d=16,
        k=3,
        mini_batch_size=4,
        tol=None,
        use_triton=False,
        device=torch.device("cpu"),
    )
    assert model.epochs == 100
    assert model.iterations is None


def test_top_level_package_no_longer_exports_low_level_minibatch_function():
    assert "batch_mini_batch_kmeans_Euclid" not in flash_kmeans.__all__
    assert not hasattr(flash_kmeans, "batch_mini_batch_kmeans_Euclid")


def test_minibatch_partial_fit_accumulates_cluster_counts_and_iterations():
    if not (interface_mod._HAS_TRITON_IMPL and torch.cuda.is_available()):
        pytest.skip("FlashMiniBatchKMeans tests require Triton/CUDA; torch fallback is not implemented.")

    data, _, centers = _make_separated_blobs(d=16)
    model = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=centers.shape[0],
        mini_batch_size=32,
        learning_rate="classic",
        epochs=1,
        tol=None,
        seed=5,
        use_triton=True,
        device=torch.device("cuda:0"),
    )
    batches_per_epoch = (data.shape[0] + model.mini_batch_size - 1) // model.mini_batch_size

    result = model.partial_fit(data)
    assert result is model
    assert model.cluster_counts_b is not None
    assert model.cluster_counts_b.shape == (1, centers.shape[0])
    assert torch.allclose(
        model.cluster_counts_b.sum(dim=1).cpu(),
        torch.tensor([data.shape[0]], dtype=torch.float32),
    )
    assert model.n_iter_ == batches_per_epoch
    assert model.n_epochs_ == 1
    assert model.cluster_ids_b is None
    assert model.inertia_b is None
    assert model.inertia_ is None

    model.partial_fit(data)

    assert torch.allclose(
        model.cluster_counts_b.sum(dim=1).cpu(),
        torch.tensor([2 * data.shape[0]], dtype=torch.float32),
    )
    assert model.n_iter_ == 2 * batches_per_epoch
    assert model.n_epochs_ == 2
    predicted = model.predict(data)
    assert predicted.shape == (data.shape[0],)
    assert model.cluster_ids_b is None
    assert model.inertia_b is None
    assert model.inertia_ is None


def test_minibatch_iteration_budget_only_increments_epoch_counter_at_epoch_boundary():
    if not (interface_mod._HAS_TRITON_IMPL and torch.cuda.is_available()):
        pytest.skip("FlashMiniBatchKMeans tests require Triton/CUDA; torch fallback is not implemented.")

    data, _, centers = _make_separated_blobs(d=16)
    model = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=centers.shape[0],
        mini_batch_size=32,
        learning_rate="classic",
        iterations=1,
        tol=None,
        seed=79,
        use_triton=True,
        device=torch.device("cuda:0"),
    )
    batches_per_epoch = (data.shape[0] + model.mini_batch_size - 1) // model.mini_batch_size

    model.partial_fit(data)

    assert model.n_iter_ == 1
    assert model.n_epochs_ == 0
    assert model.cluster_counts_b is not None
    assert torch.allclose(
        model.cluster_counts_b.sum(dim=1).cpu(),
        torch.tensor([32.0], dtype=torch.float32),
    )
    assert model.cluster_ids_b is None
    assert model.inertia_b is None
    assert model.inertia_ is None

    for _ in range(batches_per_epoch - 1):
        model.partial_fit(data)

    assert model.n_iter_ == batches_per_epoch
    assert model.n_epochs_ == 1
    assert torch.allclose(
        model.cluster_counts_b.sum(dim=1).cpu(),
        torch.tensor([data.shape[0]], dtype=torch.float32),
    )


def test_minibatch_partial_fit_iteration_chunking_matches_single_budgeted_call():
    if not (interface_mod._HAS_TRITON_IMPL and torch.cuda.is_available()):
        pytest.skip("FlashMiniBatchKMeans tests require Triton/CUDA; torch fallback is not implemented.")

    data, _, centers = _make_separated_blobs(d=16)
    init_centroids = data[:centers.shape[0]].clone()
    mini_batch_size = 32
    iterations = ((data.shape[0] + mini_batch_size - 1) // mini_batch_size) + 1
    stepwise = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=centers.shape[0],
        mini_batch_size=mini_batch_size,
        learning_rate="classic",
        iterations=1,
        tol=None,
        seed=83,
        use_triton=True,
        device=torch.device("cuda:0"),
    )
    budgeted = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=centers.shape[0],
        mini_batch_size=mini_batch_size,
        learning_rate="classic",
        iterations=iterations,
        tol=None,
        seed=83,
        use_triton=True,
        device=torch.device("cuda:0"),
    )

    for _ in range(iterations):
        if stepwise.centroids_b is None:
            stepwise.partial_fit(data, init_centroids=init_centroids)
        else:
            stepwise.partial_fit(data)
    budgeted.partial_fit(data, init_centroids=init_centroids)

    assert torch.equal(stepwise.predict(data), budgeted.predict(data))
    assert torch.allclose(stepwise.centroids_b, budgeted.centroids_b)
    assert torch.allclose(stepwise.cluster_counts_b, budgeted.cluster_counts_b)
    assert stepwise.n_iter_ == budgeted.n_iter_
    assert stepwise.n_epochs_ == budgeted.n_epochs_
    assert stepwise.stopped_early_ == budgeted.stopped_early_
    assert stepwise.cluster_ids_b is None
    assert budgeted.cluster_ids_b is None


def test_minibatch_fit_iteration_budget_matches_one_epoch_fit():
    if not (interface_mod._HAS_TRITON_IMPL and torch.cuda.is_available()):
        pytest.skip("FlashMiniBatchKMeans tests require Triton/CUDA; torch fallback is not implemented.")

    data, _, centers = _make_separated_blobs(d=16)
    init_centroids = data[:centers.shape[0]].clone()
    mini_batch_size = 32
    batches_per_epoch = (data.shape[0] + mini_batch_size - 1) // mini_batch_size
    steps_model = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=centers.shape[0],
        mini_batch_size=mini_batch_size,
        learning_rate="classic",
        iterations=batches_per_epoch,
        tol=None,
        seed=89,
        use_triton=True,
        device=torch.device("cuda:0"),
    )
    epochs_model = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=centers.shape[0],
        mini_batch_size=mini_batch_size,
        learning_rate="classic",
        epochs=1,
        tol=None,
        seed=89,
        use_triton=True,
        device=torch.device("cuda:0"),
    )

    steps_model.fit(data, init_centroids=init_centroids)
    epochs_model.fit(data, init_centroids=init_centroids)

    assert torch.equal(steps_model.cluster_ids_b, epochs_model.cluster_ids_b)
    assert torch.allclose(steps_model.centroids_b, epochs_model.centroids_b)
    assert torch.allclose(steps_model.cluster_counts_b, epochs_model.cluster_counts_b)
    assert steps_model.n_iter_ == batches_per_epoch
    assert steps_model.n_epochs_ == 1
    assert steps_model.n_iter_ == epochs_model.n_iter_
    assert steps_model.n_epochs_ == epochs_model.n_epochs_


def test_minibatch_partial_fit_rejects_reinitialization_after_fit():
    if not (interface_mod._HAS_TRITON_IMPL and torch.cuda.is_available()):
        pytest.skip("FlashMiniBatchKMeans tests require Triton/CUDA; torch fallback is not implemented.")

    data, _, centers = _make_separated_blobs(d=16)
    model = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=centers.shape[0],
        mini_batch_size=32,
        epochs=1,
        tol=None,
        seed=13,
        use_triton=True,
        device=torch.device("cuda:0"),
    )
    init_centroids = data[:centers.shape[0]].clone()

    model.partial_fit(data, init_centroids=init_centroids)

    with pytest.raises(ValueError, match="init_centroids cannot be provided"):
        model.partial_fit(data, init_centroids=init_centroids)


def test_minibatch_batched_partial_fit_tracks_counts_and_lazy_state():
    if not (interface_mod._HAS_TRITON_IMPL and torch.cuda.is_available()):
        pytest.skip("FlashMiniBatchKMeans tests require Triton/CUDA; torch fallback is not implemented.")

    data, _, centers = _make_separated_blobs(batch_size=2, d=16)
    model = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=centers.shape[1],
        mini_batch_size=32,
        learning_rate="classic",
        epochs=1,
        tol=None,
        seed=53,
        use_triton=True,
        device=torch.device("cuda:0"),
    )

    model.partial_fit(data)

    assert model.cluster_counts_b is not None
    assert model.cluster_counts_b.shape == (2, centers.shape[1])
    assert torch.allclose(
        model.cluster_counts_b.sum(dim=1).cpu(),
        torch.full((2,), data.shape[1], dtype=torch.float32),
    )
    assert model.cluster_ids_b is None
    assert model.inertia_b is None
    assert model.inertia_ is None
    assert model.predict(data).shape == (2, data.shape[1])
    assert model.cluster_ids_b is None


def test_minibatch_set_training_budget_switches_between_epochs_and_iterations():
    model = FlashMiniBatchKMeans(
        d=4,
        k=3,
        mini_batch_size=8,
        epochs=2,
        tol=None,
        use_triton=False,
        device=torch.device("cpu"),
    )

    assert model.epochs == 2
    assert model.iterations is None

    model.set_training_budget(iterations=7)

    assert model.epochs is None
    assert model.iterations == 7


def test_minibatch_reset_online_progress_preserves_centroids_and_counts(monkeypatch):
    _allow_minibatch_cpu(monkeypatch)

    cluster_counts = torch.tensor([[12.0, 20.0]], dtype=torch.float32)
    monkeypatch.setattr(interface_mod, "run_mini_batch_training", _make_fake_minibatch_runner(cluster_counts))

    data = torch.randn(10, 4)
    init_centroids = torch.randn(2, 4)
    model = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=2,
        mini_batch_size=8,
        iterations=2,
        tol=None,
        use_triton=False,
        device=torch.device("cpu"),
    )

    model.partial_fit(data, init_centroids=init_centroids)
    model._training_state.current_epoch_perm = torch.arange(data.shape[0], dtype=torch.long)
    model._training_state.current_epoch_cursor = 6
    model._training_state.current_epoch_inertia = torch.ones(1, dtype=torch.float32)
    model._training_state.previous_epoch_inertia = torch.ones(1, dtype=torch.float32)
    model._training_state.previous_batch_inertia = torch.ones(1, dtype=torch.float32)
    model._training_state.completed_epochs = 4
    model._training_state.completed_iterations = 9
    model.n_epochs_ = 4
    model.n_iter_ = 9
    model.last_fit_n_epochs_ = 3
    model.last_fit_n_iter_ = 5
    model.stopped_early_ = True

    centroids_before = model.centroids_b.clone()
    counts_before = model.cluster_counts_b.clone()

    model.reset_online_progress()

    assert torch.equal(model.centroids_b, centroids_before)
    assert torch.equal(model.cluster_counts_b, counts_before)
    assert model._training_state.current_epoch_perm is None
    assert model._training_state.current_epoch_cursor == 0
    assert torch.equal(model._training_state.current_epoch_inertia, torch.zeros(1, dtype=torch.float32))
    assert model._training_state.previous_epoch_inertia is None
    assert model._training_state.previous_batch_inertia is None
    assert model._training_state.completed_epochs == 4
    assert model._training_state.completed_iterations == 9
    assert model.n_epochs_ == 4
    assert model.n_iter_ == 9
    assert model.last_fit_n_epochs_ == 0
    assert model.last_fit_n_iter_ == 0
    assert model.stopped_early_ is False


def test_minibatch_reset_online_progress_can_clear_cluster_counts(monkeypatch):
    _allow_minibatch_cpu(monkeypatch)

    cluster_counts = torch.tensor([[12.0, 20.0]], dtype=torch.float32)
    monkeypatch.setattr(interface_mod, "run_mini_batch_training", _make_fake_minibatch_runner(cluster_counts))

    data = torch.randn(10, 4)
    init_centroids = torch.randn(2, 4)
    model = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=2,
        mini_batch_size=8,
        iterations=2,
        tol=None,
        use_triton=False,
        device=torch.device("cpu"),
    )

    model.partial_fit(data, init_centroids=init_centroids)
    model._training_state.completed_epochs = 4
    model._training_state.completed_iterations = 9
    model.n_epochs_ = 4
    model.n_iter_ = 9

    model.reset_online_progress(reset_counters=True, reset_cluster_counts=True)

    assert torch.equal(model.centroids_b, model._training_state.centroids)
    assert torch.equal(model.cluster_counts_b, torch.zeros_like(cluster_counts))
    assert torch.equal(model._training_state.cluster_counts, torch.zeros_like(cluster_counts))
    assert model._training_state.completed_epochs == 0
    assert model._training_state.completed_iterations == 0
    assert model.n_epochs_ == 0
    assert model.n_iter_ == 0


def test_minibatch_fit_restores_full_state_after_partial_fit():
    if not (interface_mod._HAS_TRITON_IMPL and torch.cuda.is_available()):
        pytest.skip("FlashMiniBatchKMeans tests require Triton/CUDA; torch fallback is not implemented.")

    data, _, centers = _make_separated_blobs(d=16, n_per_cluster=80)
    model = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=centers.shape[0],
        mini_batch_size=32,
        learning_rate="classic",
        epochs=1,
        tol=None,
        seed=59,
        use_triton=True,
        device=torch.device("cuda:0"),
    )

    model.partial_fit(data)
    assert model.cluster_ids_b is None
    assert model.inertia_b is None
    assert model.inertia_ is None

    model.fit(data)

    assert model.cluster_ids_b is not None
    assert model.cluster_ids_b.shape == (1, data.shape[0])
    assert model.inertia_b is not None
    assert model.inertia_b.shape == (1,)
    assert model.inertia_ is not None
