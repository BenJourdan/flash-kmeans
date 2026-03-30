from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:
    from flash_kmeans.assign_euclid_triton import mini_batch_euclid_assign_triton
    from flash_kmeans.centroid_update_triton import triton_centroid_update_sorted_mini_batch_euclid

    _HAS_MINIBATCH_IMPL = True
except Exception:  # pragma: no cover
    mini_batch_euclid_assign_triton = None
    triton_centroid_update_sorted_mini_batch_euclid = None
    _HAS_MINIBATCH_IMPL = False


def _mini_batch_euclid_step(
    x_mb: torch.Tensor,
    x_mb_sq: torch.Tensor,
    centroids: torch.Tensor,
    cluster_counts: torch.Tensor,
    learning_rate: str,
    *,
    use_heuristic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _HAS_MINIBATCH_IMPL:
        raise RuntimeError("Mini-batch Triton kernels are not available.")

    cluster_ids_mb, batch_inertia = mini_batch_euclid_assign_triton(
        x_mb,
        centroids,
        x_mb_sq,
        use_heuristic=use_heuristic,
    )

    mini_batch_counts = torch.zeros_like(cluster_counts, dtype=torch.float32)
    mini_batch_counts.scatter_add_(
        1,
        cluster_ids_mb.long(),
        torch.ones_like(cluster_ids_mb, dtype=torch.float32),
    )

    if learning_rate == "adaptive":
        alpha = torch.sqrt(mini_batch_counts / x_mb.shape[1])
    elif learning_rate == "classic":
        alpha = mini_batch_counts / (cluster_counts + mini_batch_counts)
    else:  # pragma: no cover - validated by public API
        raise ValueError("learning_rate must be either 'adaptive' or 'classic'.")

    centroids_new = triton_centroid_update_sorted_mini_batch_euclid(
        x_mb,
        cluster_ids_mb,
        centroids,
        alpha,
    )
    return centroids_new, batch_inertia, cluster_counts + mini_batch_counts


COMPILE_FLAG = False

try:
    if COMPILE_FLAG:
        _mini_batch_euclid_step_compiled = torch.compile(
            _mini_batch_euclid_step,
            dynamic=True,
            mode="reduce-overhead",
        )
    else:
        _mini_batch_euclid_step_compiled = _mini_batch_euclid_step
except Exception:  # pragma: no cover
    _mini_batch_euclid_step_compiled = _mini_batch_euclid_step


@dataclass
class MiniBatchTrainingState:
    centroids: Optional[torch.Tensor]
    cluster_counts: torch.Tensor
    current_epoch_perm: Optional[torch.Tensor] = None
    current_epoch_cursor: int = 0
    current_epoch_inertia: Optional[torch.Tensor] = None
    previous_epoch_inertia: Optional[torch.Tensor] = None
    previous_batch_inertia: Optional[torch.Tensor] = None
    completed_epochs: int = 0
    completed_iterations: int = 0
    stopped_early: bool = False


def initialize_minibatch_state(
    *,
    batch_size: int,
    n_clusters: int,
    device: torch.device,
    centroids: Optional[torch.Tensor] = None,
) -> MiniBatchTrainingState:
    return MiniBatchTrainingState(
        centroids=centroids,
        cluster_counts=torch.zeros((batch_size, n_clusters), device=device, dtype=torch.float32),
        current_epoch_inertia=torch.zeros(batch_size, device=device, dtype=torch.float32),
    )


def reset_minibatch_state_progress(
    state: MiniBatchTrainingState,
    *,
    reset_counters: bool = False,
    reset_cluster_counts: bool = False,
) -> MiniBatchTrainingState:
    """Reset traversal/stall bookkeeping while preserving centroids, optionally clearing counts."""
    batch_size = state.cluster_counts.shape[0]
    device = state.cluster_counts.device

    state.current_epoch_perm = None
    state.current_epoch_cursor = 0
    state.current_epoch_inertia = torch.zeros(batch_size, device=device, dtype=torch.float32)
    state.previous_epoch_inertia = None
    state.previous_batch_inertia = None
    state.stopped_early = False

    if reset_cluster_counts:
        state.cluster_counts = torch.zeros_like(state.cluster_counts)

    if reset_counters:
        state.completed_epochs = 0
        state.completed_iterations = 0

    return state


def run_mini_batch_training(
    x: torch.Tensor,
    state: MiniBatchTrainingState,
    *,
    mini_batch_size: int,
    learning_rate: str,
    iterations: Optional[int] = None,
    epochs: Optional[int] = None,
    tol: Optional[float] = None,
    verbose: bool = False,
    use_heuristic: bool = True,
    generator: Optional[torch.Generator] = None,
) -> MiniBatchTrainingState:
    """Advance mini-batch Euclidean k-means from the provided mutable state."""
    if (iterations is None) == (epochs is None):
        raise ValueError("Exactly one of iterations or epochs must be provided.")
    if iterations is not None and iterations <= 0:
        raise ValueError("iterations must be a positive integer.")
    if epochs is not None and epochs <= 0:
        raise ValueError("epochs must be a positive integer.")

    B, N, D = x.shape
    x_sq = (x ** 2).sum(dim=-1)
    batches_per_epoch = (N + mini_batch_size - 1) // mini_batch_size
    cluster_counts = state.cluster_counts.to(device=x.device, dtype=torch.float32, copy=False).view(B, -1)

    if state.centroids is None:
        n_clusters = cluster_counts.shape[1]
        indices = torch.randint(0, N, (B, n_clusters), device=x.device, generator=generator)
        centroids = torch.gather(
            x,
            dim=1,
            index=indices[..., None].expand(-1, -1, D),
        )
    else:
        centroids = state.centroids.to(device=x.device, dtype=x.dtype, copy=False).view(B, cluster_counts.shape[1], D)

    current_epoch_inertia = state.current_epoch_inertia
    if current_epoch_inertia is None:
        current_epoch_inertia = torch.zeros(B, device=x.device, dtype=torch.float32)
    else:
        current_epoch_inertia = current_epoch_inertia.to(device=x.device, dtype=torch.float32, copy=False)

    previous_epoch_inertia = state.previous_epoch_inertia
    if previous_epoch_inertia is not None:
        previous_epoch_inertia = previous_epoch_inertia.to(device=x.device, dtype=torch.float32, copy=False)

    previous_batch_inertia = state.previous_batch_inertia
    if previous_batch_inertia is not None:
        previous_batch_inertia = previous_batch_inertia.to(device=x.device, dtype=torch.float32, copy=False)

    current_epoch_perm = state.current_epoch_perm
    if current_epoch_perm is not None:
        current_epoch_perm = current_epoch_perm.to(device=x.device, dtype=torch.long, copy=False)

    current_epoch_cursor = int(state.current_epoch_cursor)
    if current_epoch_cursor < 0 or current_epoch_cursor > N:
        raise ValueError("current_epoch_cursor must be in the inclusive range [0, N].")

    iterations_run = 0
    epochs_run = 0
    stopped_early = False

    while True:
        if iterations is not None and iterations_run >= iterations:
            break
        if epochs is not None and epochs_run >= epochs:
            break

        if current_epoch_perm is None:
            current_epoch_perm = torch.randperm(N, device=x.device, generator=generator)
            current_epoch_cursor = 0

        idx = current_epoch_perm[current_epoch_cursor:current_epoch_cursor + mini_batch_size]
        x_mb = x[:, idx, :]
        x_mb_sq = x_sq[:, idx]

        centroids, batch_inertia, cluster_counts = _mini_batch_euclid_step_compiled(
            x_mb,
            x_mb_sq,
            centroids,
            cluster_counts,
            learning_rate,
            use_heuristic=use_heuristic,
        )
        current_epoch_inertia += batch_inertia
        current_epoch_cursor += idx.shape[0]
        iterations_run += 1

        avg_batch_inertia = batch_inertia / idx.shape[0]
        batch_stalled = False
        if iterations is not None:
            if verbose:
                print(
                    f"Iter {state.completed_iterations + iterations_run - 1}, "
                    f"mean batch inertia: {avg_batch_inertia.mean().item():.6f}"
                )
            if tol is not None and previous_batch_inertia is not None:
                batch_stalled = bool((avg_batch_inertia >= previous_batch_inertia * (1 - tol)).all().item())
            previous_batch_inertia = avg_batch_inertia

        if current_epoch_cursor >= N:
            avg_inertia = current_epoch_inertia / batches_per_epoch
            if epochs is not None and verbose:
                print(
                    f"Iter {state.completed_epochs + epochs_run}, "
                    f"mean inertia: {avg_inertia.mean().item():.6f}"
                )

            epoch_stalled = False
            if epochs is not None and tol is not None and previous_epoch_inertia is not None:
                epoch_stalled = bool((avg_inertia >= previous_epoch_inertia * (1 - tol)).all().item())

            previous_epoch_inertia = avg_inertia
            current_epoch_inertia = torch.zeros(B, device=x.device, dtype=torch.float32)
            current_epoch_perm = None
            current_epoch_cursor = 0
            epochs_run += 1

            if epoch_stalled:
                stopped_early = True
                break

        if batch_stalled:
            stopped_early = True
            break

    state.centroids = centroids
    state.cluster_counts = cluster_counts
    state.current_epoch_perm = current_epoch_perm
    state.current_epoch_cursor = current_epoch_cursor
    state.current_epoch_inertia = current_epoch_inertia
    state.previous_epoch_inertia = previous_epoch_inertia
    state.previous_batch_inertia = previous_batch_inertia
    state.completed_epochs += epochs_run
    state.completed_iterations += iterations_run
    state.stopped_early = stopped_early
    return state
