import pytest
import torch

import flash_kmeans.interface as interface_mod
from flash_kmeans.initialization import kmeans_parallel_init_centroids, kmeans_plusplus_init_centroids
from flash_kmeans.interface import FlashKMeans, FlashMiniBatchKMeans


def _reference_kmeanspp(data: torch.Tensor, n_clusters: int, *, seed: int | None = None) -> torch.Tensor:
    if data.ndim == 2:
        x_b = data.unsqueeze(0)
        squeeze = True
    elif data.ndim == 3:
        x_b = data
        squeeze = False
    else:
        raise ValueError("data must be rank-2 or rank-3")

    B, N, _ = x_b.shape
    generator = None
    if seed is not None:
        generator = torch.Generator(device=x_b.device)
        generator.manual_seed(int(seed))

    work_x_b = x_b.to(dtype=torch.float32, copy=False)
    centroids_b = torch.empty((B, n_clusters, x_b.shape[-1]), device=x_b.device, dtype=x_b.dtype)
    batch_index = torch.arange(B, device=x_b.device)

    first_idx = torch.randint(N, (B,), device=x_b.device, generator=generator)
    centroids_b[:, 0] = x_b[batch_index, first_idx]

    first_centroids = work_x_b[batch_index, first_idx]
    min_dist_sq = ((work_x_b - first_centroids.unsqueeze(1)) ** 2).sum(dim=-1)

    for centroid_idx in range(1, n_clusters):
        weight_sums = min_dist_sq.sum(dim=1, keepdim=True)
        uniform_weights = torch.full_like(min_dist_sq, 1.0 / float(N))
        probs = torch.where(weight_sums > 0, min_dist_sq / weight_sums.clamp_min(1e-12), uniform_weights)
        next_idx = torch.multinomial(probs, num_samples=1, replacement=False, generator=generator).squeeze(1)
        centroids_b[:, centroid_idx] = x_b[batch_index, next_idx]
        next_centroids = work_x_b[batch_index, next_idx]
        next_dist_sq = ((work_x_b - next_centroids.unsqueeze(1)) ** 2).sum(dim=-1)
        min_dist_sq = torch.minimum(min_dist_sq, next_dist_sq)

    return centroids_b.squeeze(0) if squeeze else centroids_b


def _reference_sample_fixed_count_from_mass(
    mass_b: torch.Tensor,
    selected_mask_b: torch.Tensor,
    *,
    num_samples: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if num_samples == 0:
        return torch.empty((mass_b.shape[0], 0), device=mass_b.device, dtype=torch.long)

    masked_mass_b = mass_b.to(dtype=torch.float32, copy=False).masked_fill(selected_mask_b, 0.0)
    remaining_b = (~selected_mask_b).to(dtype=torch.float32)
    remaining_sums_b = remaining_b.sum(dim=1, keepdim=True)
    uniform_probs_b = remaining_b / remaining_sums_b.clamp_min(1.0)
    mass_sums_b = masked_mass_b.sum(dim=1, keepdim=True)
    probs_b = torch.where(
        mass_sums_b > 0,
        masked_mass_b / mass_sums_b.clamp_min(1e-12),
        uniform_probs_b,
    )
    return torch.multinomial(probs_b, num_samples=num_samples, replacement=False, generator=generator)


def _reference_weighted_kmeanspp_indices(
    x_b: torch.Tensor,
    n_clusters: int,
    *,
    base_weights: torch.Tensor | None = None,
    uniform_first_center: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    B, N, _ = x_b.shape
    work_x_b = x_b.to(dtype=torch.float32, copy=False)
    batch_index = torch.arange(B, device=x_b.device)
    selected_mask_b = torch.zeros((B, N), device=x_b.device, dtype=torch.bool)
    chosen_idx_b = torch.empty((B, n_clusters), device=x_b.device, dtype=torch.long)
    weight_b = torch.ones((B, N), device=x_b.device, dtype=torch.float32) if base_weights is None else base_weights.to(dtype=torch.float32)

    if uniform_first_center:
        first_idx_b = torch.randint(N, (B,), device=x_b.device, generator=generator)
    else:
        first_idx_b = _reference_sample_fixed_count_from_mass(
            weight_b,
            selected_mask_b,
            num_samples=1,
            generator=generator,
        ).squeeze(1)
    chosen_idx_b[:, 0] = first_idx_b
    selected_mask_b[batch_index, first_idx_b] = True

    first_centers_b = work_x_b[batch_index, first_idx_b]
    min_dist_sq_b = ((work_x_b - first_centers_b.unsqueeze(1)) ** 2).sum(dim=-1)
    min_dist_sq_b[batch_index, first_idx_b] = 0.0

    for centroid_idx in range(1, n_clusters):
        next_idx_b = _reference_sample_fixed_count_from_mass(
            weight_b * min_dist_sq_b,
            selected_mask_b,
            num_samples=1,
            generator=generator,
        ).squeeze(1)
        chosen_idx_b[:, centroid_idx] = next_idx_b
        selected_mask_b[batch_index, next_idx_b] = True

        next_centers_b = work_x_b[batch_index, next_idx_b]
        next_dist_sq_b = ((work_x_b - next_centers_b.unsqueeze(1)) ** 2).sum(dim=-1)
        next_dist_sq_b[batch_index, next_idx_b] = 0.0
        min_dist_sq_b = torch.minimum(min_dist_sq_b, next_dist_sq_b)

    return chosen_idx_b


def _reference_kmeans_parallel_fixed_count(
    data: torch.Tensor,
    n_clusters: int,
    *,
    oversampling_factor: float,
    rounds: int,
    seed: int | None = None,
) -> torch.Tensor:
    if data.ndim == 2:
        x_b = data.unsqueeze(0)
        squeeze = True
    elif data.ndim == 3:
        x_b = data
        squeeze = False
    else:
        raise ValueError("data must be rank-2 or rank-3")

    B, N, D = x_b.shape
    work_x_b = x_b.to(dtype=torch.float32, copy=False)
    ell = max(1, int(torch.ceil(torch.tensor(float(oversampling_factor) * n_clusters)).item()))
    max_candidates = min(N, 1 + rounds * ell)
    batch_index = torch.arange(B, device=x_b.device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=x_b.device)
        generator.manual_seed(int(seed))

    selected_mask_b = torch.zeros((B, N), device=x_b.device, dtype=torch.bool)
    candidate_idx_b = torch.empty((B, max_candidates), device=x_b.device, dtype=torch.long)

    first_idx_b = torch.randint(N, (B,), device=x_b.device, generator=generator)
    candidate_idx_b[:, 0] = first_idx_b
    selected_mask_b[batch_index, first_idx_b] = True

    first_centers_b = work_x_b[batch_index, first_idx_b]
    min_dist_sq_b = ((work_x_b - first_centers_b.unsqueeze(1)) ** 2).sum(dim=-1)
    min_dist_sq_b[batch_index, first_idx_b] = 0.0

    offset = 1
    for _ in range(rounds):
        if offset >= max_candidates:
            break

        draw_count = min(ell, max_candidates - offset)
        new_idx_b = _reference_sample_fixed_count_from_mass(
            min_dist_sq_b,
            selected_mask_b,
            num_samples=draw_count,
            generator=generator,
        )
        candidate_idx_b[:, offset:offset + draw_count] = new_idx_b
        selected_mask_b.scatter_(1, new_idx_b, True)

        new_centers_b = torch.gather(
            work_x_b,
            dim=1,
            index=new_idx_b.unsqueeze(-1).expand(-1, -1, D),
        )
        chunk_min_dist_sq_b = torch.full((B, N), float("inf"), device=x_b.device, dtype=torch.float32)
        for center_offset in range(draw_count):
            dist_sq_b = ((work_x_b - new_centers_b[:, center_offset].unsqueeze(1)) ** 2).sum(dim=-1)
            chunk_min_dist_sq_b = torch.minimum(chunk_min_dist_sq_b, dist_sq_b)

        min_dist_sq_b = torch.minimum(min_dist_sq_b, chunk_min_dist_sq_b)
        min_dist_sq_b.scatter_(1, new_idx_b, 0.0)
        offset += draw_count

    candidate_idx_b = candidate_idx_b[:, :offset]
    candidate_work_b = torch.gather(
        work_x_b,
        dim=1,
        index=candidate_idx_b.unsqueeze(-1).expand(-1, -1, D),
    )
    dist_sq_b = ((work_x_b.unsqueeze(2) - candidate_work_b.unsqueeze(1)) ** 2).sum(dim=-1)
    best_idx_b = dist_sq_b.argmin(dim=-1)
    candidate_weights_b = torch.zeros((B, candidate_idx_b.shape[1]), device=x_b.device, dtype=torch.float32)
    candidate_weights_b.scatter_add_(1, best_idx_b, torch.ones_like(best_idx_b, dtype=torch.float32))

    num_candidate_centroids = min(n_clusters, candidate_idx_b.shape[1])
    chosen_candidate_local_idx_b = _reference_weighted_kmeanspp_indices(
        torch.gather(
            x_b,
            dim=1,
            index=candidate_idx_b.unsqueeze(-1).expand(-1, -1, D),
        ),
        num_candidate_centroids,
        base_weights=candidate_weights_b,
        uniform_first_center=False,
        generator=generator,
    )
    chosen_orig_idx_b = torch.gather(candidate_idx_b, dim=1, index=chosen_candidate_local_idx_b)

    if num_candidate_centroids < n_clusters:
        chosen_mask_b = torch.zeros((B, N), device=x_b.device, dtype=torch.bool)
        chosen_mask_b.scatter_(1, chosen_orig_idx_b, True)
        fill_idx_b = _reference_sample_fixed_count_from_mass(
            torch.zeros((B, N), device=x_b.device, dtype=torch.float32),
            chosen_mask_b,
            num_samples=n_clusters - num_candidate_centroids,
            generator=generator,
        )
        chosen_orig_idx_b = torch.cat([chosen_orig_idx_b, fill_idx_b], dim=1)

    centroids_b = torch.gather(
        x_b,
        dim=1,
        index=chosen_orig_idx_b.unsqueeze(-1).expand(-1, -1, D),
    )
    return centroids_b.squeeze(0) if squeeze else centroids_b


def _assert_centroids_are_sampled_from_input(centroids: torch.Tensor, data: torch.Tensor):
    if data.ndim == 2:
        for centroid in centroids:
            assert bool((data == centroid).all(dim=1).any().item())
        return

    for batch_idx in range(data.shape[0]):
        for centroid in centroids[batch_idx]:
            assert bool((data[batch_idx] == centroid).all(dim=1).any().item())


@pytest.mark.parametrize(
    "data",
    [
        torch.randn(32, 4, generator=torch.Generator().manual_seed(11)),
        torch.randn(2, 32, 4, generator=torch.Generator().manual_seed(12)),
    ],
)
def test_kmeans_plusplus_matches_reference_for_fixed_seed(data):
    expected = _reference_kmeanspp(data, 5, seed=17)
    actual = kmeans_plusplus_init_centroids(data, 5, seed=17)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "data",
    [
        torch.randn(64, 8, generator=torch.Generator().manual_seed(21)),
        torch.randn(2, 64, 8, generator=torch.Generator().manual_seed(22)),
    ],
)
def test_kmeans_parallel_is_deterministic_and_samples_input_points(data):
    first = kmeans_parallel_init_centroids(data, 6, oversampling_factor=2.0, rounds=4, seed=19)
    second = kmeans_parallel_init_centroids(data, 6, oversampling_factor=2.0, rounds=4, seed=19)

    assert torch.equal(first, second)
    _assert_centroids_are_sampled_from_input(first, data)


def test_kmeans_parallel_handles_identical_points_and_short_candidate_pool():
    data = torch.ones(16, 5, dtype=torch.float32)

    centroids = kmeans_parallel_init_centroids(
        data,
        6,
        oversampling_factor=0.1,
        rounds=1,
        seed=23,
    )

    assert centroids.shape == (6, 5)
    assert torch.isfinite(centroids).all()
    assert torch.allclose(centroids, torch.ones_like(centroids))


@pytest.mark.parametrize(
    "data",
    [
        torch.randn(24, 4, generator=torch.Generator().manual_seed(25)),
        torch.randn(2, 24, 4, generator=torch.Generator().manual_seed(26)),
    ],
)
def test_kmeans_parallel_matches_fixed_count_reference_on_cpu(data):
    expected = _reference_kmeans_parallel_fixed_count(
        data,
        5,
        oversampling_factor=1.5,
        rounds=2,
        seed=27,
    )
    actual = kmeans_parallel_init_centroids(
        data,
        5,
        oversampling_factor=1.5,
        rounds=2,
        seed=27,
    )

    assert torch.equal(actual, expected)


def test_kmeans_parallel_rank2_matches_rank3_batch_of_one():
    data = torch.randn(1, 32, 6, generator=torch.Generator().manual_seed(28))

    rank2 = kmeans_parallel_init_centroids(data[0], 7, oversampling_factor=1.25, rounds=3, seed=29)
    rank3 = kmeans_parallel_init_centroids(data, 7, oversampling_factor=1.25, rounds=3, seed=29)

    assert torch.equal(rank2, rank3.squeeze(0))


def test_kmeans_parallel_candidate_pool_cap_returns_unique_points_when_input_rows_are_unique():
    data = torch.arange(40, dtype=torch.float32).view(10, 4)

    centroids = kmeans_parallel_init_centroids(
        data,
        6,
        oversampling_factor=2.0,
        rounds=8,
        seed=30,
    )

    assert centroids.shape == (6, 4)
    assert centroids.unique(dim=0).shape[0] == 6


def test_kmeans_parallel_validates_parameters():
    data = torch.randn(16, 4, generator=torch.Generator().manual_seed(24))

    with pytest.raises(ValueError, match="oversampling_factor"):
        kmeans_parallel_init_centroids(data, 4, oversampling_factor=0.0, rounds=3)

    with pytest.raises(ValueError, match="rounds"):
        kmeans_parallel_init_centroids(data, 4, oversampling_factor=2.0, rounds=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kmeans_parallel_cuda_batched_is_deterministic_and_samples_input_points():
    generator = torch.Generator(device="cuda")
    generator.manual_seed(33)
    data = torch.randn((4, 64, 8), device="cuda", generator=generator)

    first = kmeans_parallel_init_centroids(data, 6, oversampling_factor=2.0, rounds=4, seed=34)
    second = kmeans_parallel_init_centroids(data, 6, oversampling_factor=2.0, rounds=4, seed=34)

    assert torch.equal(first, second)
    _assert_centroids_are_sampled_from_input(first, data)


def test_flash_kmeans_uses_kmeans_parallel_constructor_init(monkeypatch):
    data = torch.randn(2, 20, 4, generator=torch.Generator().manual_seed(31))
    generated_init = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(32))
    captured: dict[str, object] = {}

    def fake_kmeans_parallel(x_b, n_clusters, *, oversampling_factor, rounds, seed):
        captured["shape"] = tuple(x_b.shape)
        captured["n_clusters"] = n_clusters
        captured["oversampling_factor"] = oversampling_factor
        captured["rounds"] = rounds
        captured["seed"] = seed
        return generated_init.to(device=x_b.device, dtype=x_b.dtype)

    def fake_batch_kmeans(x_b, n_clusters, max_iters, tol, init_centroids, verbose):
        labels = torch.zeros((x_b.shape[0], x_b.shape[1]), dtype=torch.int64, device=x_b.device)
        return labels, init_centroids.clone(), 1

    monkeypatch.setattr(interface_mod, "_require_triton_cuda", lambda: None)
    monkeypatch.setattr(interface_mod, "kmeans_parallel_init_centroids", fake_kmeans_parallel)
    monkeypatch.setattr(interface_mod, "batch_kmeans_Euclid", fake_batch_kmeans)

    model = FlashKMeans(
        d=data.shape[-1],
        k=generated_init.shape[1],
        niter=1,
        tol=None,
        seed=37,
        init="kmeans||",
        init_oversampling_factor=3.5,
        init_rounds=7,
        use_triton=True,
        device=torch.device("cpu"),
    )
    model.fit(data)

    assert captured == {
        "shape": tuple(data.shape),
        "n_clusters": generated_init.shape[1],
        "oversampling_factor": 3.5,
        "rounds": 7,
        "seed": 37,
    }
    assert model.init_strategy_ == "kmeans||"


def test_flash_minibatch_uses_kmeans_parallel_constructor_init(monkeypatch):
    data = torch.randn(2, 16, 16, generator=torch.Generator().manual_seed(41))
    generated_init = torch.randn(2, 3, 16, generator=torch.Generator().manual_seed(42))
    cluster_counts = torch.full((2, 3), 4.0, dtype=torch.float32)
    captured: dict[str, object] = {}

    def fake_kmeans_parallel(x_b, n_clusters, *, oversampling_factor, rounds, seed):
        captured["shape"] = tuple(x_b.shape)
        captured["n_clusters"] = n_clusters
        captured["oversampling_factor"] = oversampling_factor
        captured["rounds"] = rounds
        captured["seed"] = seed
        return generated_init.to(device=x_b.device, dtype=x_b.dtype)

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
        batches_per_epoch = (x_b.shape[1] + mini_batch_size - 1) // mini_batch_size
        state.centroids = state.centroids.clone()
        state.cluster_counts = cluster_counts.to(device=x_b.device)
        state.current_epoch_perm = None
        state.current_epoch_cursor = 0
        state.current_epoch_inertia = torch.zeros(x_b.shape[0], device=x_b.device, dtype=torch.float32)
        state.previous_epoch_inertia = None
        state.completed_iterations += epochs * batches_per_epoch
        state.completed_epochs += epochs
        state.stopped_early = False
        return state

    def fake_assign(x_b, centroids_b, x_sq, use_heuristic=True):
        return torch.zeros((x_b.shape[0], x_b.shape[1]), dtype=torch.int64, device=x_b.device)

    monkeypatch.setattr(interface_mod, "_require_minibatch_backend", lambda device, use_triton: None)
    monkeypatch.setattr(interface_mod, "kmeans_parallel_init_centroids", fake_kmeans_parallel)
    monkeypatch.setattr(interface_mod, "run_mini_batch_training", fake_run_minibatch)
    monkeypatch.setattr(interface_mod, "euclid_assign_triton", fake_assign)

    model = FlashMiniBatchKMeans(
        d=data.shape[-1],
        k=generated_init.shape[1],
        mini_batch_size=4,
        epochs=2,
        tol=None,
        seed=43,
        init="kmeans||",
        init_oversampling_factor=4.0,
        init_rounds=6,
        use_triton=True,
        device=torch.device("cpu"),
    )
    model.fit(data)

    assert captured == {
        "shape": tuple(data.shape),
        "n_clusters": generated_init.shape[1],
        "oversampling_factor": 4.0,
        "rounds": 6,
        "seed": 43,
    }
    assert model.init_strategy_ == "kmeans||"
