import itertools

import pytest
import torch

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

    def fake_batch_minibatch(
        x_b,
        n_clusters,
        mini_batch_size,
        epochs,
        learning_rate,
        tol,
        init_centroids,
        init_cluster_counts,
        return_cluster_counts,
        verbose,
    ):
        captured["x_shape"] = tuple(x_b.shape)
        captured["n_clusters"] = n_clusters
        captured["mini_batch_size"] = mini_batch_size
        captured["epochs"] = epochs
        captured["learning_rate"] = learning_rate
        captured["init_centroids"] = init_centroids.detach().cpu().clone()
        captured["init_cluster_counts"] = init_cluster_counts
        captured["return_cluster_counts"] = return_cluster_counts
        labels = torch.zeros((x_b.shape[0], x_b.shape[1]), dtype=torch.int64, device=x_b.device)
        return labels, init_centroids.clone(), epochs, cluster_counts.to(device=x_b.device)

    monkeypatch.setattr(interface_mod, "_require_triton_cuda", lambda: None)
    monkeypatch.setattr(interface_mod, "batch_mini_batch_kmeans_Euclid", fake_batch_minibatch)

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
    assert captured["learning_rate"] == "adaptive"
    assert captured["init_centroids"].shape == (data.shape[0],) + init_centroids.shape
    assert captured["init_cluster_counts"] is None
    assert captured["return_cluster_counts"] is True
    for batch_idx in range(data.shape[0]):
        assert torch.allclose(captured["init_centroids"][batch_idx], init_centroids)
    assert torch.allclose(model.cluster_counts_b.cpu(), cluster_counts)


def test_minibatch_partial_fit_accumulates_cluster_counts_and_steps():
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

    result = model.partial_fit(data)
    assert result is model
    assert model.cluster_counts_b is not None
    assert model.cluster_counts_b.shape == (1, centers.shape[0])
    assert torch.allclose(
        model.cluster_counts_b.sum(dim=1).cpu(),
        torch.tensor([data.shape[0]], dtype=torch.float32),
    )
    assert model.n_iter_ == 1
    assert model.n_steps_ == 4

    model.partial_fit(data)

    assert torch.allclose(
        model.cluster_counts_b.sum(dim=1).cpu(),
        torch.tensor([2 * data.shape[0]], dtype=torch.float32),
    )
    assert model.n_iter_ == 2
    assert model.n_steps_ == 8
    assert model.predict(data).shape == (data.shape[0],)


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


def test_minibatch_batched_partial_fit_tracks_counts_and_inertia():
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
    assert model.inertia_b is not None
    assert model.inertia_b.shape == (2,)
    assert model.predict(data).shape == (2, data.shape[1])


def test_minibatch_partial_fit_updates_inertia_attributes():
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
    first_inertia = float(model.inertia_)

    assert model.inertia_b is not None
    assert model.inertia_b.shape == (1,)

    model.partial_fit(data, epochs=3)
    second_inertia = float(model.inertia_)

    assert second_inertia <= first_inertia + 1e-5
