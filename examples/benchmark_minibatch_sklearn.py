import argparse
import math
import time

import numpy as np
import torch

from flash_kmeans import batch_mini_batch_kmeans_Euclid

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required for this benchmark. "
        "Try: uv run --with scikit-learn python examples/benchmark_minibatch_sklearn.py"
    ) from exc


def _parse_args():
    def _maybe_float(value: str):
        if value.lower() == "none":
            return None
        return float(value)

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark flash-kmeans mini-batch Euclidean training against "
            "scikit-learn MiniBatchKMeans using shared random initialization."
        )
    )
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Number of independent clustering problems.")
    parser.add_argument("--num-points", "-n", type=int, default=100000, help="Number of points per problem.")
    parser.add_argument("--dim", "-d", type=int, default=128, help="Point dimensionality.")
    parser.add_argument("--num-clusters", "-k", type=int, default=1000, help="Number of clusters.")
    parser.add_argument("--mini-batch-size", type=int, default=1024, help="Mini-batch size for both implementations.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs / passes over the data.")
    parser.add_argument("--learning-rate", choices=["classic", "adaptive"], default="classic", help="flash-kmeans learning-rate schedule.")
    parser.add_argument(
        "--tol",
        type=_maybe_float,
        default=0.0,
        help="Relative tolerance. Pass 'none' to disable flash early stopping without using a sentinel.",
    )
    parser.add_argument("--sklearn-max-no-improvement", type=int, default=10, help="sklearn inertia patience when early stopping is enabled.")
    parser.add_argument("--sklearn-reassignment-ratio", type=float, default=0.0, help="sklearn reassignment_ratio. Default 0.0 is closer to flash-kmeans today.")
    parser.add_argument("--disable-early-stop", action="store_true", help="Disable early stopping for both implementations.")
    parser.add_argument("--flash-dtype", choices=["float16", "float32"], default="float32", help="Compute dtype for flash-kmeans input tensors.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed runs.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for data, init, and per-run reproducibility.")
    parser.add_argument("--use-heuristic", action="store_true", help="Use the heuristic Triton config instead of autotune.")
    return parser.parse_args()


def _make_dataset(batch_size: int, num_points: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((batch_size, num_points, dim), dtype=np.float32)
    return np.ascontiguousarray(data)


def _make_shared_init(x_cpu: np.ndarray, num_clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    batch_size, num_points, dim = x_cpu.shape
    init = np.empty((batch_size, num_clusters, dim), dtype=np.float32)
    for b in range(batch_size):
        indices = rng.choice(num_points, size=num_clusters, replace=False)
        init[b] = x_cpu[b, indices]
    return np.ascontiguousarray(init)


def _resolve_flash_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _flash_tol(args):
    return None if args.disable_early_stop else args.tol


def _sklearn_tol(args) -> float:
    if args.disable_early_stop or args.tol is None:
        return 0.0
    return max(args.tol, 0.0)


def _sklearn_max_no_improvement(args):
    if args.disable_early_stop or args.tol is None:
        return None
    return args.sklearn_max_no_improvement


def _run_flash_once(x_gpu: torch.Tensor, init_gpu: torch.Tensor, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    _, _, epochs_run = batch_mini_batch_kmeans_Euclid(
        x_gpu,
        args.num_clusters,
        mini_batch_size=args.mini_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        tol=_flash_tol(args),
        init_centroids=init_gpu,
        verbose=False,
        use_heuristic=args.use_heuristic,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    steps_run = epochs_run * math.ceil(args.num_points / args.mini_batch_size)
    return elapsed_ms, epochs_run, steps_run


def _flash_inertia(
    x_gpu: torch.Tensor,
    cluster_ids: torch.Tensor,
    centroids: torch.Tensor,
    chunk_size_points: int = 4096,
):
    B, N, _ = x_gpu.shape
    cent_sq = (centroids.to(torch.float32) ** 2).sum(dim=-1)
    inertia_b = torch.zeros(B, device=x_gpu.device, dtype=torch.float32)

    for start in range(0, N, chunk_size_points):
        end = min(start + chunk_size_points, N)
        x_chunk = x_gpu[:, start:end, :]
        ids_chunk = cluster_ids[:, start:end].long()
        chosen_centroids = torch.gather(
            centroids,
            dim=1,
            index=ids_chunk.unsqueeze(-1).expand(-1, -1, centroids.shape[-1]),
        )
        chosen_cent_sq = torch.gather(cent_sq, dim=1, index=ids_chunk)
        x_sq_chunk = (x_chunk.to(torch.float32) ** 2).sum(dim=-1)
        cross_chunk = (x_chunk.to(torch.float32) * chosen_centroids.to(torch.float32)).sum(dim=-1)
        inertia_b += (x_sq_chunk + chosen_cent_sq - 2.0 * cross_chunk).clamp_min_(0.0).sum(dim=-1)

    return inertia_b


def _evaluate_flash(x_gpu: torch.Tensor, init_gpu: torch.Tensor, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cluster_ids, centroids, epochs_run = batch_mini_batch_kmeans_Euclid(
        x_gpu,
        args.num_clusters,
        mini_batch_size=args.mini_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        tol=_flash_tol(args),
        init_centroids=init_gpu,
        verbose=False,
        use_heuristic=args.use_heuristic,
    )
    inertia_b = _flash_inertia(x_gpu, cluster_ids, centroids)
    steps_run = epochs_run * math.ceil(args.num_points / args.mini_batch_size)
    return {
        "epochs": float(epochs_run),
        "steps": float(steps_run),
        "inertia_total": float(inertia_b.sum().item()),
        "inertia_mean": float(inertia_b.mean().item()),
    }


def _run_sklearn_once(x_cpu: np.ndarray, init_cpu: np.ndarray, args):
    start = time.perf_counter()
    epochs_run = []
    steps_run = []

    for b in range(args.batch_size):
        model = MiniBatchKMeans(
            n_clusters=args.num_clusters,
            init=init_cpu[b],
            n_init=1,
            max_iter=args.epochs,
            batch_size=args.mini_batch_size,
            compute_labels=True,
            random_state=args.seed,
            tol=_sklearn_tol(args),
            max_no_improvement=_sklearn_max_no_improvement(args),
            init_size=None,
            reassignment_ratio=args.sklearn_reassignment_ratio,
            verbose=0,
        )
        model.fit(x_cpu[b])
        epochs_run.append(model.n_iter_)
        steps_run.append(model.n_steps_)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, float(np.mean(epochs_run)), float(np.mean(steps_run))


def _evaluate_sklearn(x_cpu: np.ndarray, init_cpu: np.ndarray, args):
    epochs_run = []
    steps_run = []
    inertias = []

    for b in range(args.batch_size):
        model = MiniBatchKMeans(
            n_clusters=args.num_clusters,
            init=init_cpu[b],
            n_init=1,
            max_iter=args.epochs,
            batch_size=args.mini_batch_size,
            compute_labels=True,
            random_state=args.seed,
            tol=_sklearn_tol(args),
            max_no_improvement=_sklearn_max_no_improvement(args),
            init_size=None,
            reassignment_ratio=args.sklearn_reassignment_ratio,
            verbose=0,
        )
        model.fit(x_cpu[b])
        epochs_run.append(model.n_iter_)
        steps_run.append(model.n_steps_)
        inertias.append(float(model.inertia_))

    inertias = np.asarray(inertias, dtype=np.float64)
    return {
        "epochs": float(np.mean(epochs_run)),
        "steps": float(np.mean(steps_run)),
        "inertia_total": float(inertias.sum()),
        "inertia_mean": float(inertias.mean()),
    }


def _benchmark(name: str, fn, warmup: int, repeats: int):
    for _ in range(warmup):
        fn()

    times_ms = []
    epochs = []
    steps = []
    for _ in range(repeats):
        elapsed_ms, epoch_count, step_count = fn()
        times_ms.append(elapsed_ms)
        epochs.append(epoch_count)
        steps.append(step_count)

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    mean_epochs = float(np.mean(epochs))
    mean_steps = float(np.mean(steps))

    return {
        "name": name,
        "time_ms_mean": mean_ms,
        "time_ms_std": std_ms,
        "epochs_mean": mean_epochs,
        "steps_mean": mean_steps,
    }


def main():
    args = _parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the flash-kmeans side of this benchmark.")

    if args.batch_size <= 0 or args.num_points <= 0 or args.dim <= 0 or args.num_clusters <= 0:
        raise SystemExit("batch-size, num-points, dim, and num-clusters must all be positive.")
    if args.num_clusters > args.num_points:
        raise SystemExit("num-clusters must be <= num-points so shared random init can sample without replacement.")

    x_cpu = _make_dataset(args.batch_size, args.num_points, args.dim, args.seed)
    init_cpu = _make_shared_init(x_cpu, args.num_clusters, args.seed + 1)

    flash_dtype = _resolve_flash_dtype(args.flash_dtype)
    x_gpu = torch.from_numpy(x_cpu).to(device="cuda", dtype=flash_dtype)
    init_gpu = torch.from_numpy(init_cpu).to(device="cuda", dtype=flash_dtype)

    flash_result = _benchmark(
        "flash_minibatch",
        lambda: _run_flash_once(x_gpu, init_gpu, args),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    sklearn_result = _benchmark(
        "sklearn_minibatch",
        lambda: _run_sklearn_once(x_cpu, init_cpu, args),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    flash_eval = _evaluate_flash(x_gpu, init_gpu, args)
    sklearn_eval = _evaluate_sklearn(x_cpu, init_cpu, args)

    speedup = sklearn_result["time_ms_mean"] / flash_result["time_ms_mean"]

    print("Config")
    print(
        f"  B={args.batch_size} N={args.num_points} D={args.dim} K={args.num_clusters} "
        f"mini_batch_size={args.mini_batch_size} epochs={args.epochs}"
    )
    print(
        f"  flash_dtype={args.flash_dtype} flash_lr={args.learning_rate} "
        f"flash_tol={_flash_tol(args)} sklearn_tol={_sklearn_tol(args)} "
        f"sklearn_max_no_improvement={_sklearn_max_no_improvement(args)} "
        f"sklearn_reassignment_ratio={args.sklearn_reassignment_ratio}"
    )
    print("  shared_init=random sample from data, n_init=1, compute_labels=True")
    print("  note: flash runs on a preloaded CUDA tensor; sklearn runs on the same samples as a NumPy array on CPU")
    print("  note: sklearn early stopping uses center-change plus inertia patience; flash currently uses relative epoch inertia")
    print()

    print("Results")
    for result in (flash_result, sklearn_result):
        print(
            f"  {result['name']}: "
            f"{result['time_ms_mean']:.2f} +/- {result['time_ms_std']:.2f} ms, "
            f"mean_epochs={result['epochs_mean']:.2f}, "
            f"mean_steps={result['steps_mean']:.2f}"
        )
    print(f"  speedup_vs_sklearn={speedup:.2f}x")
    print()
    print("Quality")
    print(
        f"  flash_minibatch: inertia_mean={flash_eval['inertia_mean']:.4f}, "
        f"inertia_total={flash_eval['inertia_total']:.4f}, "
        f"epochs={flash_eval['epochs']:.2f}, steps={flash_eval['steps']:.2f}"
    )
    print(
        f"  sklearn_minibatch: inertia_mean={sklearn_eval['inertia_mean']:.4f}, "
        f"inertia_total={sklearn_eval['inertia_total']:.4f}, "
        f"epochs={sklearn_eval['epochs']:.2f}, steps={sklearn_eval['steps']:.2f}"
    )


if __name__ == "__main__":
    main()
