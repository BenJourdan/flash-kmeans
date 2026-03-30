import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from flash_kmeans.assign_euclid_triton import euclid_assign_triton
from flash_kmeans.centroid_update_triton import triton_centroid_update_sorted_euclid
from flash_kmeans.interface import FlashMiniBatchKMeans
from flash_kmeans.initialization import (
    kmeans_parallel_init_centroids,
    kmeans_plusplus_init_centroids,
    sample_random_init_centroids,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot ARI over cumulative training time for flash full-batch k-means, "
            "and flash mini-batch k-means."
        )
    )
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Number of independent clustering problems.")
    parser.add_argument("--num-points", "-n", type=int, default=100000, help="Number of points per problem.")
    parser.add_argument("--dim", "-d", type=int, default=128, help="Point dimensionality.")
    parser.add_argument("--num-clusters", "-k", type=int, default=1000, help="Number of clusters.")
    parser.add_argument("--mini-batch-size", type=int, default=1024, help="Mini-batch size for both mini-batch methods.")
    parser.add_argument(
        "--fullbatch-epochs",
        "--epochs",
        dest="fullbatch_epochs",
        type=int,
        default=25,
        help="Training budget in full passes over the data for flash full-batch k-means.",
    )
    parser.add_argument(
        "--minibatch-iterations",
        type=int,
        default=None,
        help=(
            "Exact number of mini-batch updates for flash mini-batch k-means. "
            "Defaults to fullbatch_epochs * ceil(num_points / mini_batch_size)."
        ),
    )
    parser.add_argument("--init", choices=["random", "kmeans++", "kmeans||"], default="random", help="Shared initialization strategy computed before training.")
    parser.add_argument(
        "--init-oversampling-factor",
        type=float,
        default=2.0,
        help="Oversampling factor used when --init kmeans||.",
    )
    parser.add_argument(
        "--init-rounds",
        type=int,
        default=5,
        help="Number of candidate-sampling rounds used when --init kmeans||.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        choices=["classic", "adaptive"],
        default=["classic", "adaptive"],
        help="flash mini-batch learning-rate schedules to benchmark. Defaults to both classic and adaptive.",
    )
    parser.add_argument(
        "--flash-dtype",
        choices=["float16", "float32"],
        default="float32",
        help="Compute dtype for flash-kmeans input tensors.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for data, initialization, and mini-batch order.")
    parser.add_argument("--use-heuristic", action="store_true", help="Use the heuristic Triton config instead of autotune.")
    parser.add_argument("--output", type=str, default="ari_trajectory.png", help="Output PNG path for the plot.")
    parser.add_argument("--csv-output", type=str, default=None, help="Optional CSV path for raw trajectory measurements.")
    parser.add_argument("--cluster-std", type=float, default=1.0, help="Standard deviation for Gaussian blobs.")
    parser.add_argument("--center-scale", type=float, default=10.0, help="Scale factor for sampled cluster centers.")
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help=(
            "Optional early-stopping tolerance for the flash trajectories. "
            "Full-batch uses centroid-shift tolerance; mini-batch uses the estimator's tolerance rule."
        ),
    )
    args = parser.parse_args()
    args.learning_rates = list(dict.fromkeys(args.learning_rates))
    return args


def _validate_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_tol(value: float | None) -> float | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError("tol must be non-negative.")
    return float(value)


def _validate_init_controls(oversampling_factor: float, rounds: int):
    if oversampling_factor <= 0:
        raise ValueError("init_oversampling_factor must be positive.")
    if rounds <= 0:
        raise ValueError("init_rounds must be a positive integer.")


def _resolve_minibatch_iterations(
    *,
    fullbatch_epochs: int,
    minibatch_iterations: int | None,
    num_points: int,
    mini_batch_size: int,
) -> tuple[int, int]:
    batches_per_epoch = (num_points + mini_batch_size - 1) // mini_batch_size
    if minibatch_iterations is None:
        return fullbatch_epochs * batches_per_epoch, batches_per_epoch
    return minibatch_iterations, batches_per_epoch


def _resolve_flash_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _resolve_csv_path(output_path: Path, csv_output: str | None) -> Path:
    if csv_output is not None:
        return Path(csv_output)
    return output_path.with_suffix(".csv")


def _make_gaussian_blobs(
    batch_size: int,
    num_points: int,
    dim: int,
    num_clusters: int,
    cluster_std: float,
    center_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_clusters > num_points:
        raise ValueError("num_clusters must be less than or equal to num_points.")

    rng = np.random.default_rng(seed)
    x = np.empty((batch_size, num_points, dim), dtype=np.float32)
    labels = np.empty((batch_size, num_points), dtype=np.int64)

    base_count = num_points // num_clusters
    remainder = num_points % num_clusters
    counts = np.full(num_clusters, base_count, dtype=np.int64)
    counts[:remainder] += 1

    for b in range(batch_size):
        centers = rng.standard_normal((num_clusters, dim), dtype=np.float32) * center_scale
        points = []
        point_labels = []
        for k in range(num_clusters):
            pts = centers[k] + cluster_std * rng.standard_normal((counts[k], dim), dtype=np.float32)
            points.append(pts.astype(np.float32, copy=False))
            point_labels.append(np.full(counts[k], k, dtype=np.int64))

        data_b = np.concatenate(points, axis=0)
        labels_b = np.concatenate(point_labels, axis=0)
        perm = rng.permutation(num_points)
        x[b] = np.ascontiguousarray(data_b[perm])
        labels[b] = labels_b[perm]

    return np.ascontiguousarray(x), np.ascontiguousarray(labels)


def _to_flash_tensor(x_cpu: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(x_cpu).to(device="cuda", dtype=dtype)


def _comb2(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.float64, copy=False)
    return counts * (counts - 1.0) * 0.5


def _adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    labels_true = np.asarray(labels_true, dtype=np.int64)
    labels_pred = np.asarray(labels_pred, dtype=np.int64)

    if labels_true.shape != labels_pred.shape:
        raise ValueError("labels_true and labels_pred must have matching shapes.")
    if labels_true.size < 2:
        return 1.0

    _, true_inverse = np.unique(labels_true, return_inverse=True)
    _, pred_inverse = np.unique(labels_pred, return_inverse=True)
    true_classes = int(true_inverse.max()) + 1
    pred_classes = int(pred_inverse.max()) + 1

    contingency = np.bincount(
        true_inverse * pred_classes + pred_inverse,
        minlength=true_classes * pred_classes,
    ).reshape(true_classes, pred_classes)

    sum_comb = _comb2(contingency).sum()
    sum_true = _comb2(contingency.sum(axis=1)).sum()
    sum_pred = _comb2(contingency.sum(axis=0)).sum()
    total_pairs = labels_true.size * (labels_true.size - 1.0) * 0.5
    if total_pairs == 0.0:
        return 1.0

    expected = (sum_true * sum_pred) / total_pairs
    max_index = 0.5 * (sum_true + sum_pred)
    denominator = max_index - expected
    if denominator == 0.0:
        return 1.0
    return float((sum_comb - expected) / denominator)


def _ari_per_batch(labels_pred: np.ndarray, labels_true: np.ndarray) -> np.ndarray:
    batch_size = labels_true.shape[0]
    aris = np.empty(batch_size, dtype=np.float64)
    for b in range(batch_size):
        aris[b] = _adjusted_rand_index(labels_true[b], labels_pred[b])
    return aris


def _labels_to_numpy(labels: torch.Tensor) -> np.ndarray:
    return labels.long().cpu().numpy()


def _append_measurement(
    rows: list[dict[str, float | int | str]],
    *,
    algorithm: str,
    step_index: int,
    epoch: int,
    batch_index: int,
    time_ms: float,
    aris: np.ndarray,
):
    row: dict[str, float | int | str] = {
        "algorithm": algorithm,
        "step_index": step_index,
        "epoch": epoch,
        "batch_index": batch_index,
        "time_ms": float(time_ms),
        "ari_mean": float(np.mean(aris)),
    }
    for b, ari in enumerate(aris):
        row[f"ari_b{b}"] = float(ari)
    rows.append(row)


def _flash_minibatch_algorithm_name(learning_rate: str) -> str:
    return f"flash_minibatch_{learning_rate}"


def _build_shared_init(
    x_gpu: torch.Tensor,
    init: str,
    num_clusters: int,
    seed: int,
    flash_dtype: torch.dtype,
    init_oversampling_factor: float,
    init_rounds: int,
) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    if init == "random":
        init_gpu = sample_random_init_centroids(x_gpu, num_clusters, seed=seed)
    elif init == "kmeans++":
        init_gpu = kmeans_plusplus_init_centroids(x_gpu, num_clusters, seed=seed)
    else:
        init_gpu = kmeans_parallel_init_centroids(
            x_gpu,
            num_clusters,
            oversampling_factor=init_oversampling_factor,
            rounds=init_rounds,
            seed=seed,
        )
    torch.cuda.synchronize()
    init_ms = (time.perf_counter() - start) * 1000.0
    init_gpu = init_gpu.to(dtype=flash_dtype, copy=False)
    return init_gpu, float(init_ms)


def _make_flash_minibatch_model(
    x_gpu: torch.Tensor,
    init_gpu: torch.Tensor,
    mini_batch_size: int,
    learning_rate: str,
    *,
    tol: float | None,
    use_heuristic: bool,
    seed: int,
) -> FlashMiniBatchKMeans:
    return FlashMiniBatchKMeans(
        d=x_gpu.shape[-1],
        k=init_gpu.shape[1],
        mini_batch_size=mini_batch_size,
        learning_rate=learning_rate,
        iterations=1,
        tol=tol,
        use_triton=True,
        seed=seed,
        verbose=False,
        use_heuristic=use_heuristic,
        init=init_gpu,
        dtype=x_gpu.dtype,
        device=x_gpu.device,
    )


def _warmup_flash(
    x_gpu: torch.Tensor,
    init_gpu: torch.Tensor,
    mini_batch_size: int,
    learning_rates: list[str],
    *,
    tol: float | None,
    use_heuristic: bool,
):
    x_sq = (x_gpu ** 2).sum(dim=-1)
    centroids = init_gpu.clone()

    cluster_ids = euclid_assign_triton(x_gpu, centroids, x_sq, use_heuristic=use_heuristic)
    _ = triton_centroid_update_sorted_euclid(x_gpu, cluster_ids, centroids)

    for learning_rate in learning_rates:
        model = _make_flash_minibatch_model(
            x_gpu,
            init_gpu,
            mini_batch_size,
            learning_rate,
            tol=tol,
            use_heuristic=use_heuristic,
            seed=0,
        )
        model.partial_fit(x_gpu)
    _ = euclid_assign_triton(x_gpu, centroids, x_sq, use_heuristic=use_heuristic)
    torch.cuda.synchronize()


def _run_flash_fullbatch(
    x_gpu: torch.Tensor,
    labels_true: np.ndarray,
    init_gpu: torch.Tensor,
    epochs: int,
    *,
    tol: float | None,
    use_heuristic: bool,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    x_sq = (x_gpu ** 2).sum(dim=-1)
    centroids = init_gpu.clone().contiguous()
    cumulative_ms = 0.0

    for epoch in range(epochs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        cluster_ids = euclid_assign_triton(x_gpu, centroids, x_sq, use_heuristic=use_heuristic)
        centroids_next = triton_centroid_update_sorted_euclid(x_gpu, cluster_ids, centroids)
        torch.cuda.synchronize()
        cumulative_ms += (time.perf_counter() - start) * 1000.0
        center_shift = (centroids_next - centroids).norm(dim=-1).max().item()
        centroids = centroids_next

        aris = _ari_per_batch(_labels_to_numpy(cluster_ids), labels_true)
        _append_measurement(
            rows,
            algorithm="flash_fullbatch",
            step_index=epoch + 1,
            epoch=epoch + 1,
            batch_index=0,
            time_ms=cumulative_ms,
            aris=aris,
        )
        if tol is not None and center_shift < tol:
            break

    return rows


def _run_flash_minibatch(
    x_gpu: torch.Tensor,
    labels_true: np.ndarray,
    init_gpu: torch.Tensor,
    mini_batch_size: int,
    iterations: int,
    learning_rate: str,
    *,
    seed: int,
    tol: float | None,
    use_heuristic: bool,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    _, N, _ = x_gpu.shape
    batches_per_epoch = (N + mini_batch_size - 1) // mini_batch_size
    model = _make_flash_minibatch_model(
        x_gpu,
        init_gpu,
        mini_batch_size,
        learning_rate,
        tol=tol,
        use_heuristic=use_heuristic,
        seed=seed,
    )
    cumulative_ms = 0.0

    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.partial_fit(x_gpu)
        torch.cuda.synchronize()
        cumulative_ms += (time.perf_counter() - start) * 1000.0

        labels = model.predict(x_gpu)
        step_index = int(model.n_iter_)
        batch_index = (step_index - 1) % batches_per_epoch
        epoch_index = ((step_index - 1) // batches_per_epoch) + 1
        aris = _ari_per_batch(_labels_to_numpy(labels), labels_true)
        _append_measurement(
            rows,
            algorithm=_flash_minibatch_algorithm_name(learning_rate),
            step_index=step_index,
            epoch=epoch_index,
            batch_index=batch_index,
            time_ms=cumulative_ms,
            aris=aris,
        )
        if model.stopped_early_:
            break

    return rows

def _write_csv(rows: list[dict[str, float | int | str]], csv_path: Path, batch_size: int):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["algorithm", "step_index", "epoch", "batch_index", "time_ms", "ari_mean"]
    fieldnames.extend(f"ari_b{b}" for b in range(batch_size))

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_plot(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
    *,
    init_name: str,
    init_time_ms: float,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required to render the plot. "
            "Try: uv run --with matplotlib "
            "python examples/benchmark_ari_trajectory.py"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = {
        "flash_fullbatch": {"color": "#1f77b4", "marker": "o", "zorder": 3},
        "flash_minibatch_classic": {"color": "#d62728", "marker": "s", "zorder": 1},
        "flash_minibatch_adaptive": {"color": "#ff9896", "marker": "D", "zorder": 1},
    }

    plt.figure(figsize=(10, 6))
    for algorithm in ("flash_fullbatch", "flash_minibatch_classic", "flash_minibatch_adaptive"):
        series = [row for row in rows if row["algorithm"] == algorithm]
        if not series:
            continue
        times = [float(row["time_ms"]) for row in series]
        aris = [float(row["ari_mean"]) for row in series]
        plt.scatter(
            times,
            aris,
            label=algorithm,
            s=28,
            marker=styles[algorithm]["marker"],
            color=styles[algorithm]["color"],
            alpha=0.9,
            zorder=styles[algorithm]["zorder"],
        )

    plt.xlabel("Cumulative training time (ms)")
    plt.ylabel("Mean ARI")
    plt.title("ARI over Training Time")
    plt.ylim(-0.05, 1.05)
    plt.xscale("log")
    plt.grid(True, alpha=0.25)
    plt.legend(title=f"shared init: {init_name} ({init_time_ms:.2f} ms, excluded)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _summarize(rows: list[dict[str, float | int | str]]):
    print("Trajectory Summary")
    for algorithm in ("flash_fullbatch", "flash_minibatch_classic", "flash_minibatch_adaptive"):
        series = [row for row in rows if row["algorithm"] == algorithm]
        if not series:
            continue
        last = series[-1]
        print(
            f"  {algorithm}: points={len(series)}, "
            f"final_time_ms={float(last['time_ms']):.2f}, final_ari_mean={float(last['ari_mean']):.4f}"
        )


def main():
    args = _parse_args()
    args.fullbatch_epochs = _validate_positive_int("fullbatch_epochs", args.fullbatch_epochs)
    args.minibatch_iterations = _validate_positive_int("minibatch_iterations", args.minibatch_iterations)
    args.tol = _validate_tol(args.tol)
    _validate_init_controls(args.init_oversampling_factor, args.init_rounds)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the flash-kmeans side of this benchmark.")
    if args.num_points % args.mini_batch_size != 0:
        raise AssertionError(
            "benchmark_ari_trajectory.py requires num_points to be divisible by mini_batch_size "
            "so the flash mini-batch trajectory does not include tail-batch Triton compile overhead."
        )
    minibatch_iterations, batches_per_epoch = _resolve_minibatch_iterations(
        fullbatch_epochs=args.fullbatch_epochs,
        minibatch_iterations=args.minibatch_iterations,
        num_points=args.num_points,
        mini_batch_size=args.mini_batch_size,
    )

    output_path = Path(args.output)
    csv_path = _resolve_csv_path(output_path, args.csv_output)
    flash_dtype = _resolve_flash_dtype(args.flash_dtype)

    x_cpu, labels_true = _make_gaussian_blobs(
        batch_size=args.batch_size,
        num_points=args.num_points,
        dim=args.dim,
        num_clusters=args.num_clusters,
        cluster_std=args.cluster_std,
        center_scale=args.center_scale,
        seed=args.seed,
    )
    x_gpu = _to_flash_tensor(x_cpu, flash_dtype)
    init_gpu, init_ms = _build_shared_init(
        x_gpu,
        args.init,
        args.num_clusters,
        args.seed + 1,
        flash_dtype,
        args.init_oversampling_factor,
        args.init_rounds,
    )

    _warmup_flash(
        x_gpu,
        init_gpu,
        args.mini_batch_size,
        args.learning_rates,
        tol=args.tol,
        use_heuristic=args.use_heuristic,
    )

    rows = []
    rows.extend(
        _run_flash_fullbatch(
            x_gpu,
            labels_true,
            init_gpu,
            args.fullbatch_epochs,
            tol=args.tol,
            use_heuristic=args.use_heuristic,
        )
    )
    for learning_rate in args.learning_rates:
        rows.extend(
            _run_flash_minibatch(
                x_gpu,
                labels_true,
                init_gpu,
                args.mini_batch_size,
                minibatch_iterations,
                learning_rate,
                seed=args.seed + 2,
                tol=args.tol,
                use_heuristic=args.use_heuristic,
            )
        )

    _write_csv(rows, csv_path, args.batch_size)
    _write_plot(rows, output_path, init_name=args.init, init_time_ms=init_ms)

    print("Config")
    print(
        f"  B={args.batch_size} N={args.num_points} D={args.dim} K={args.num_clusters} "
        f"mini_batch_size={args.mini_batch_size} "
        f"fullbatch_epochs={args.fullbatch_epochs} minibatch_iterations={minibatch_iterations} tol={args.tol}"
    )
    print(
        f"  flash_dtype={args.flash_dtype} flash_lrs={','.join(args.learning_rates)} "
        f"shared_init={args.init} shared_init_ms={init_ms:.2f} "
        f"cluster_std={args.cluster_std} center_scale={args.center_scale} "
        f"init_oversampling_factor={args.init_oversampling_factor} init_rounds={args.init_rounds}"
    )
    _summarize(rows)
    print(f"Outputs")
    print(f"  plot: {output_path}")
    print(f"  csv: {csv_path}")


if __name__ == "__main__":
    main()
