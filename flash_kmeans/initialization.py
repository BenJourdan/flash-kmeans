from __future__ import annotations

import math
from typing import Optional

import torch

try:
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton

    _HAS_TRITON_ASSIGN = True
except Exception:  # pragma: no cover
    euclid_assign_triton = None
    _HAS_TRITON_ASSIGN = False


_DISTANCE_CHUNK_SIZE = 256


def _normalize_data(data: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if data.ndim == 2:
        return data.unsqueeze(0), True
    if data.ndim == 3:
        return data, False
    raise ValueError("data must be of shape (n_samples, n_features) or (batch_size, n_samples, n_features)")


def _make_generator(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _validate_cluster_count(num_samples: int, n_clusters: int):
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    if n_clusters > num_samples:
        raise ValueError("n_clusters must be less than or equal to the number of samples.")


def _validate_kmeans_parallel_params(oversampling_factor: float, rounds: int):
    if oversampling_factor <= 0:
        raise ValueError("oversampling_factor must be positive.")
    if rounds <= 0:
        raise ValueError("rounds must be a positive integer.")


def _normalize_weights(
    weights: torch.Tensor,
    *,
    batch_size: int,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    if weights.ndim == 1:
        if weights.shape != (num_samples,):
            raise ValueError(f"weights must have shape ({num_samples},) or ({batch_size}, {num_samples}).")
        weights_b = weights.unsqueeze(0).expand(batch_size, -1)
    elif weights.ndim == 2:
        if weights.shape != (batch_size, num_samples):
            raise ValueError(f"weights must have shape ({num_samples},) or ({batch_size}, {num_samples}).")
        weights_b = weights
    else:
        raise ValueError("weights must be a rank-1 or rank-2 tensor.")

    weights_b = weights_b.to(device=device, dtype=torch.float32, copy=False)
    if bool((weights_b < 0).any().item()):
        raise ValueError("weights must be non-negative.")
    return weights_b


def _gather_points(x_b: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        x_b,
        dim=1,
        index=idx_b.long().unsqueeze(-1).expand(-1, -1, x_b.shape[-1]),
    )


def _batched_point_dist_sq(
    x_b: torch.Tensor,
    x_sq_b: torch.Tensor,
    centers_b: torch.Tensor,
) -> torch.Tensor:
    center_sq_b = (centers_b * centers_b).sum(dim=-1, keepdim=True)
    cross_b = torch.bmm(x_b, centers_b.unsqueeze(-1)).squeeze(-1)
    return (x_sq_b + center_sq_b - 2.0 * cross_b).clamp_min_(0.0)


def _sample_indices_from_mass(
    mass_b: torch.Tensor,
    selected_mask_b: torch.Tensor,
    *,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    return _sample_fixed_count_indices_from_mass(
        mass_b,
        selected_mask_b,
        num_samples=1,
        generator=generator,
    ).squeeze(1)


def _sample_fixed_count_indices_from_mass(
    mass_b: torch.Tensor,
    selected_mask_b: torch.Tensor,
    *,
    num_samples: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    if num_samples == 0:
        return torch.empty((mass_b.shape[0], 0), device=mass_b.device, dtype=torch.long)

    mass_b = mass_b.to(dtype=torch.float32, copy=False)
    mass_b = mass_b.masked_fill(selected_mask_b, 0.0)
    mass_sums_b = mass_b.sum(dim=1, keepdim=True)

    remaining_b = (~selected_mask_b).to(dtype=torch.float32)
    remaining_sums_b = remaining_b.sum(dim=1, keepdim=True)
    if bool((remaining_sums_b < float(num_samples)).any().item()):
        raise ValueError("Cannot sample more points than remain unselected.")
    uniform_probs_b = remaining_b / remaining_sums_b.clamp_min(1.0)

    probs_b = torch.where(
        mass_sums_b > 0,
        mass_b / mass_sums_b.clamp_min(1e-12),
        uniform_probs_b,
    )
    return torch.multinomial(probs_b, num_samples=num_samples, replacement=False, generator=generator)


def _kmeanspp_sample_indices(
    x_b: torch.Tensor,
    n_clusters: int,
    *,
    base_weights: Optional[torch.Tensor] = None,
    uniform_first_center: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    B, N, _ = x_b.shape
    _validate_cluster_count(N, n_clusters)

    work_x_b = x_b.to(dtype=torch.float32, copy=False)
    x_sq_b = (work_x_b * work_x_b).sum(dim=-1)
    if base_weights is None:
        base_weights_b = torch.ones((B, N), device=x_b.device, dtype=torch.float32)
    else:
        base_weights_b = _normalize_weights(
            base_weights,
            batch_size=B,
            num_samples=N,
            device=x_b.device,
        )

    batch_index = torch.arange(B, device=x_b.device)
    selected_mask_b = torch.zeros((B, N), device=x_b.device, dtype=torch.bool)
    chosen_idx_b = torch.empty((B, n_clusters), device=x_b.device, dtype=torch.long)

    if uniform_first_center:
        first_idx_b = torch.randint(N, (B,), device=x_b.device, generator=generator)
    else:
        first_idx_b = _sample_indices_from_mass(base_weights_b, selected_mask_b, generator=generator)
    chosen_idx_b[:, 0] = first_idx_b
    selected_mask_b[batch_index, first_idx_b] = True

    first_centers_b = work_x_b[batch_index, first_idx_b]
    min_dist_sq_b = _batched_point_dist_sq(work_x_b, x_sq_b, first_centers_b)
    min_dist_sq_b[batch_index, first_idx_b] = 0.0

    for centroid_idx in range(1, n_clusters):
        sampling_mass_b = base_weights_b * min_dist_sq_b
        next_idx_b = _sample_indices_from_mass(sampling_mass_b, selected_mask_b, generator=generator)
        chosen_idx_b[:, centroid_idx] = next_idx_b
        selected_mask_b[batch_index, next_idx_b] = True

        next_centers_b = work_x_b[batch_index, next_idx_b]
        next_dist_sq_b = _batched_point_dist_sq(work_x_b, x_sq_b, next_centers_b)
        next_dist_sq_b[batch_index, next_idx_b] = 0.0
        min_dist_sq_b = torch.minimum(min_dist_sq_b, next_dist_sq_b)

    return chosen_idx_b


def _can_use_triton_assign(x_b: torch.Tensor, centers_b: torch.Tensor) -> bool:
    return bool(
        _HAS_TRITON_ASSIGN
        and x_b.is_cuda
        and centers_b.is_cuda
        and x_b.device == centers_b.device
        and x_b.shape[-1] >= 16
    )


def _chunked_assign_min_dist_sq_b(
    work_x_b: torch.Tensor,
    x_sq_b: torch.Tensor,
    work_centers_b: torch.Tensor,
    *,
    chunk_size: int = _DISTANCE_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    if work_centers_b.shape[1] == 0:
        raise ValueError("centers must contain at least one point.")

    B, N, _ = work_x_b.shape
    best_dist_sq_b = torch.full((B, N), float("inf"), device=work_x_b.device, dtype=torch.float32)
    best_idx_b = torch.zeros((B, N), device=work_x_b.device, dtype=torch.long)

    for start in range(0, work_centers_b.shape[1], chunk_size):
        center_chunk_b = work_centers_b[:, start:start + chunk_size].to(dtype=torch.float32, copy=False)
        center_sq_b = (center_chunk_b * center_chunk_b).sum(dim=-1)
        cross_b = torch.bmm(work_x_b, center_chunk_b.transpose(1, 2))
        dist_sq_b = (x_sq_b.unsqueeze(-1) + center_sq_b.unsqueeze(1) - 2.0 * cross_b).clamp_min_(0.0)
        chunk_best_dist_b, chunk_best_idx_b = dist_sq_b.min(dim=-1)
        better_b = chunk_best_dist_b < best_dist_sq_b
        best_dist_sq_b = torch.where(better_b, chunk_best_dist_b, best_dist_sq_b)
        best_idx_b = torch.where(better_b, chunk_best_idx_b + start, best_idx_b)

    return best_idx_b, best_dist_sq_b


def _triton_assign_min_dist_sq_b(
    x_b: torch.Tensor,
    work_x_b: torch.Tensor,
    x_sq_b: torch.Tensor,
    centers_b: torch.Tensor,
    work_centers_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if centers_b.shape[1] == 0:
        raise ValueError("centers must contain at least one point.")

    center_sq_b = (work_centers_b * work_centers_b).sum(dim=-1)
    best_idx_b = euclid_assign_triton(
        x_b,
        centers_b,
        x_sq_b,
        c_sq=center_sq_b,
    ).long()
    assigned_centers_b = _gather_points(work_centers_b, best_idx_b)
    assigned_center_sq_b = torch.gather(center_sq_b, dim=1, index=best_idx_b)
    cross_b = (work_x_b * assigned_centers_b).sum(dim=-1)
    best_dist_sq_b = (x_sq_b + assigned_center_sq_b - 2.0 * cross_b).clamp_min_(0.0)
    return best_idx_b, best_dist_sq_b


def _assign_min_dist_sq_b(
    x_b: torch.Tensor,
    work_x_b: torch.Tensor,
    x_sq_b: torch.Tensor,
    centers_b: torch.Tensor,
    work_centers_b: torch.Tensor,
    *,
    chunk_size: int = _DISTANCE_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _can_use_triton_assign(x_b, centers_b):
        return _triton_assign_min_dist_sq_b(
            x_b,
            work_x_b,
            x_sq_b,
            centers_b,
            work_centers_b,
        )

    return _chunked_assign_min_dist_sq_b(
        work_x_b,
        x_sq_b,
        work_centers_b,
        chunk_size=chunk_size,
    )


def _update_min_dist_sq_b(
    x_b: torch.Tensor,
    work_x_b: torch.Tensor,
    x_sq_b: torch.Tensor,
    centers_b: torch.Tensor,
    work_centers_b: torch.Tensor,
    *,
    current_min_dist_sq_b: Optional[torch.Tensor] = None,
    chunk_size: int = _DISTANCE_CHUNK_SIZE,
) -> torch.Tensor:
    if centers_b.shape[1] == 0:
        if current_min_dist_sq_b is None:
            return torch.full((x_b.shape[0], x_b.shape[1]), float("inf"), device=x_b.device, dtype=torch.float32)
        return current_min_dist_sq_b.to(device=x_b.device, dtype=torch.float32, copy=False)

    new_min_dist_sq_b = _assign_min_dist_sq_b(
        x_b,
        work_x_b,
        x_sq_b,
        centers_b,
        work_centers_b,
        chunk_size=chunk_size,
    )[1]
    if current_min_dist_sq_b is None:
        return new_min_dist_sq_b
    return torch.minimum(
        current_min_dist_sq_b.to(device=x_b.device, dtype=torch.float32, copy=False),
        new_min_dist_sq_b,
    )


def _candidate_weights_b(
    x_b: torch.Tensor,
    work_x_b: torch.Tensor,
    x_sq_b: torch.Tensor,
    candidates_b: torch.Tensor,
    candidate_work_b: torch.Tensor,
    *,
    chunk_size: int = _DISTANCE_CHUNK_SIZE,
) -> torch.Tensor:
    if candidates_b.shape[1] == 0:
        raise ValueError("candidates must contain at least one point.")

    best_idx_b = _assign_min_dist_sq_b(
        x_b,
        work_x_b,
        x_sq_b,
        candidates_b,
        candidate_work_b,
        chunk_size=chunk_size,
    )[0]
    weights_b = torch.zeros_like(candidate_work_b[..., 0], dtype=torch.float32)
    weights_b.scatter_add_(1, best_idx_b, torch.ones_like(best_idx_b, dtype=torch.float32))
    return weights_b


def _kmeans_parallel_sample_indices(
    x_b: torch.Tensor,
    n_clusters: int,
    *,
    oversampling_factor: float,
    rounds: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    B, N, _ = x_b.shape
    work_x_b = x_b.to(dtype=torch.float32, copy=False)
    x_sq_b = (work_x_b * work_x_b).sum(dim=-1)
    ell = max(1, int(math.ceil(float(oversampling_factor) * n_clusters)))
    max_candidates = min(N, 1 + int(rounds) * ell)
    batch_index = torch.arange(B, device=x_b.device)

    first_idx_b = torch.randint(N, (B,), device=x_b.device, generator=generator)
    candidate_idx_b = torch.empty((B, max_candidates), device=x_b.device, dtype=torch.long)
    candidate_idx_b[:, 0] = first_idx_b
    selected_mask_b = torch.zeros((B, N), device=x_b.device, dtype=torch.bool)
    selected_mask_b[batch_index, first_idx_b] = True

    first_centers_b = x_b[batch_index, first_idx_b].unsqueeze(1)
    first_work_centers_b = work_x_b[batch_index, first_idx_b].unsqueeze(1)
    min_dist_sq_b = _update_min_dist_sq_b(
        x_b,
        work_x_b,
        x_sq_b,
        first_centers_b,
        first_work_centers_b,
    )
    min_dist_sq_b[batch_index, first_idx_b] = 0.0

    offset = 1
    for _ in range(int(rounds)):
        if offset >= max_candidates:
            break

        draw_count = min(ell, max_candidates - offset)
        new_idx_b = _sample_fixed_count_indices_from_mass(
            min_dist_sq_b,
            selected_mask_b,
            num_samples=draw_count,
            generator=generator,
        )
        candidate_idx_b[:, offset:offset + draw_count] = new_idx_b
        selected_mask_b.scatter_(1, new_idx_b, True)

        new_centers_b = _gather_points(x_b, new_idx_b)
        new_work_centers_b = _gather_points(work_x_b, new_idx_b)
        min_dist_sq_b = _update_min_dist_sq_b(
            x_b,
            work_x_b,
            x_sq_b,
            new_centers_b,
            new_work_centers_b,
            current_min_dist_sq_b=min_dist_sq_b,
        )
        min_dist_sq_b.scatter_(1, new_idx_b, 0.0)
        offset += draw_count

    candidate_idx_b = candidate_idx_b[:, :offset]
    candidate_points_b = _gather_points(x_b, candidate_idx_b)
    candidate_work_b = _gather_points(work_x_b, candidate_idx_b)
    candidate_weights_b = _candidate_weights_b(
        x_b,
        work_x_b,
        x_sq_b,
        candidate_points_b,
        candidate_work_b,
    )

    num_candidate_centroids = min(n_clusters, candidate_idx_b.shape[1])
    chosen_candidate_local_idx_b = _kmeanspp_sample_indices(
        candidate_points_b,
        num_candidate_centroids,
        base_weights=candidate_weights_b,
        uniform_first_center=False,
        generator=generator,
    )
    chosen_orig_idx_b = torch.gather(candidate_idx_b, dim=1, index=chosen_candidate_local_idx_b)

    if num_candidate_centroids < n_clusters:
        chosen_mask_b = torch.zeros((B, N), device=x_b.device, dtype=torch.bool)
        chosen_mask_b.scatter_(1, chosen_orig_idx_b, True)
        fill_idx_b = _sample_fixed_count_indices_from_mass(
            torch.zeros((B, N), device=x_b.device, dtype=torch.float32),
            chosen_mask_b,
            num_samples=n_clusters - num_candidate_centroids,
            generator=generator,
        )
        chosen_orig_idx_b = torch.cat([chosen_orig_idx_b, fill_idx_b], dim=1)

    return chosen_orig_idx_b


def _kmeans_parallel_sample_indices_one_batch(
    x: torch.Tensor,
    work_x: torch.Tensor,
    x_sq: torch.Tensor,
    n_clusters: int,
    *,
    oversampling_factor: float,
    rounds: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    del work_x, x_sq
    return _kmeans_parallel_sample_indices(
        x.unsqueeze(0),
        n_clusters,
        oversampling_factor=oversampling_factor,
        rounds=rounds,
        generator=generator,
    ).squeeze(0)


def sample_random_init_centroids(
    data: torch.Tensor,
    n_clusters: int,
    *,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample random initial centroids from the input data without replacement.

    Parameters
    ----------
    data : torch.Tensor
        Input data of shape `(n_samples, n_features)` or `(batch_size, n_samples, n_features)`.
    n_clusters : int
        Number of centroids to sample per batch item.
    seed : int | None, optional
        Optional RNG seed used for deterministic sampling.

    Returns
    -------
    torch.Tensor
        Initial centroids with shape `(n_clusters, n_features)` for rank-2 inputs or
        `(batch_size, n_clusters, n_features)` for rank-3 inputs.
    """
    x_b, squeeze = _normalize_data(data)
    B, N, _ = x_b.shape
    _validate_cluster_count(N, n_clusters)

    generator = _make_generator(x_b.device, seed)
    centroids = []
    for batch_idx in range(B):
        idx = torch.randperm(N, device=x_b.device, generator=generator)[:n_clusters]
        centroids.append(x_b[batch_idx, idx])

    centroids_b = torch.stack(centroids, dim=0)
    return centroids_b.squeeze(0) if squeeze else centroids_b


def kmeans_plusplus_init_centroids(
    data: torch.Tensor,
    n_clusters: int,
    *,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute exact batched k-means++ initial centroids on the same device as `data`.

    Parameters
    ----------
    data : torch.Tensor
        Input data of shape `(n_samples, n_features)` or `(batch_size, n_samples, n_features)`.
    n_clusters : int
        Number of centroids to sample per batch item.
    seed : int | None, optional
        Optional RNG seed used for deterministic sampling.

    Returns
    -------
    torch.Tensor
        Initial centroids with shape `(n_clusters, n_features)` for rank-2 inputs or
        `(batch_size, n_clusters, n_features)` for rank-3 inputs.
    """
    x_b, squeeze = _normalize_data(data)
    generator = _make_generator(x_b.device, seed)
    chosen_idx_b = _kmeanspp_sample_indices(
        x_b,
        n_clusters,
        uniform_first_center=True,
        generator=generator,
    )
    centroids_b = _gather_points(x_b, chosen_idx_b)
    return centroids_b.squeeze(0) if squeeze else centroids_b


def kmeans_parallel_init_centroids(
    data: torch.Tensor,
    n_clusters: int,
    *,
    oversampling_factor: float = 2.0,
    rounds: int = 5,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute batched fixed-count k-means||-style initial centroids on the same device as `data`.

    Parameters
    ----------
    data : torch.Tensor
        Input data of shape `(n_samples, n_features)` or `(batch_size, n_samples, n_features)`.
    n_clusters : int
        Number of centroids to sample per batch item.
    oversampling_factor : float, default=2.0
        Oversampling factor relative to `n_clusters`; each round samples
        `ceil(oversampling_factor * n_clusters)` new candidates without replacement.
    rounds : int, default=5
        Number of fixed-count candidate-sampling rounds.
    seed : int | None, optional
        Optional RNG seed used for deterministic sampling.

    Returns
    -------
    torch.Tensor
        Initial centroids with shape `(n_clusters, n_features)` for rank-2 inputs or
        `(batch_size, n_clusters, n_features)` for rank-3 inputs.
    """
    x_b, squeeze = _normalize_data(data)
    _, N, _ = x_b.shape
    _validate_cluster_count(N, n_clusters)
    _validate_kmeans_parallel_params(float(oversampling_factor), int(rounds))

    generator = _make_generator(x_b.device, seed)
    chosen_idx_b = _kmeans_parallel_sample_indices(
        x_b,
        n_clusters,
        oversampling_factor=float(oversampling_factor),
        rounds=int(rounds),
        generator=generator,
    )
    centroids_b = _gather_points(x_b, chosen_idx_b)
    return centroids_b.squeeze(0) if squeeze else centroids_b
