import torch
import torch.nn.functional as F
from torch.cuda import nvtx
from flash_kmeans.assign_euclid_triton import euclid_assign_triton, mini_batch_euclid_assign_triton, cosine_assign_triton
from flash_kmeans.centroid_update_triton import triton_centroid_update_cosine, triton_centroid_update_euclid, triton_centroid_update_sorted_euclid, triton_centroid_update_sorted_mini_batch_euclid, triton_centroid_update_sorted_cosine
from tqdm import trange

# -------------------- Compiled single-iteration kernels --------------------

# 1. Euclidean
def _euclid_iter(x, x_sq, centroids, use_heuristic=True):
    
    cluster_ids = euclid_assign_triton(x, centroids, x_sq, use_heuristic=use_heuristic)
    centroids_new = triton_centroid_update_sorted_euclid(x, cluster_ids, centroids)

    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids

def _mini_batch_euclid_iter(x_mb, x_mb_sq, centroids, cluster_counts, learning_rate, use_heuristic=True):
    
    # mini-batch_cluster_ids: (B, mini_batch_size), batch_counts: (B, n_clusters)
    cluster_ids_mb, batch_inertia = mini_batch_euclid_assign_triton(x_mb, centroids, x_mb_sq, use_heuristic=use_heuristic)

    # Count how many points in the mini-batch are assigned to each cluster, shape (B, n_clusters)
    mini_batch_counts = torch.zeros_like(cluster_counts, device=cluster_counts.device, dtype=torch.float32)
    mini_batch_counts.scatter_add_(
        1,
        cluster_ids_mb.long(), 
        torch.ones_like(cluster_ids_mb, dtype=torch.float32))
    # ...

    # learning rates to apply to each centroid update, shape (B, n_clusters)
    if learning_rate == "adaptive":
        # update rule is centroid_new = (1-alpha)* centroid_old + alpha * x_mb.mean_of_assigned_points
        # where alpha = sqrt(batch_count / batch_size)
        alpha = torch.sqrt(mini_batch_counts.float() / x_mb.shape[1])  # (B, n_clusters)
    elif learning_rate == "classic":
        # same as above with alpha = batch_count / (cluster_count + batch_count)
        # This corresponds to each point in the mini-batch having equal weight in the update,
        # and the cluster center being the average of all points assigned to it across all mini-batches.
        alpha = mini_batch_counts.float() / (cluster_counts.float() + mini_batch_counts.float())  # (B, n_clusters)
    

    centroids_new = triton_centroid_update_sorted_mini_batch_euclid(x_mb, cluster_ids_mb, centroids, alpha)

    return centroids_new, batch_inertia, cluster_counts + mini_batch_counts # update the global cluster counts with the mini-batch counts for the next iteration

# 2. Cosine
def _cosine_iter(x_norm, centroids):
    # cos_sim = torch.einsum('bnd,bkd->bnk', x_norm, centroids)
    # cluster_ids = cos_sim.argmax(dim=-1)
    cluster_ids = cosine_assign_triton(x_norm, centroids)
    centroids_new = triton_centroid_update_sorted_cosine(x_norm, cluster_ids, centroids)
    # centroids_new = centroids_new.clone()
    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids

# 3. Dot-product
def _dot_iter(x, centroids):
    # sim = torch.einsum('bnd,bkd->bnk', x, centroids)
    # cluster_ids = sim.argmax(dim=-1)
    cluster_ids = cosine_assign_triton(x, centroids)
    centroids_new = triton_centroid_update_sorted_cosine(x, cluster_ids, centroids)
    # centroids_new = centroids_new.clone()
    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids

COMPILE_FLAG = False

try:
    if COMPILE_FLAG:
        _euclid_iter_compiled = torch.compile(_euclid_iter, dynamic=True, mode="reduce-overhead")
        _mini_batch_euclid_iter_compiled = torch.compile(_mini_batch_euclid_iter, dynamic=True, mode="reduce-overhead")
        _cosine_iter_compiled = torch.compile(_cosine_iter, dynamic=True, mode="reduce-overhead")
        _dot_iter_compiled    = torch.compile(_dot_iter,    dynamic=True, mode="reduce-overhead")
    else:
        _euclid_iter_compiled = _euclid_iter
        _mini_batch_euclid_iter_compiled = _mini_batch_euclid_iter
        _cosine_iter_compiled = _cosine_iter
        _dot_iter_compiled    = _dot_iter
except Exception:  # pragma: no cover
    _euclid_iter_compiled = _euclid_iter
    _mini_batch_euclid_iter_compiled = _mini_batch_euclid_iter
    _cosine_iter_compiled = _cosine_iter
    _dot_iter_compiled    = _dot_iter

def batch_kmeans_Euclid(
    x,
    n_clusters,
    max_iters=100,
    tol=None,
    init_centroids=None,
    verbose=False,
    *,
    use_heuristic=True,
):
    """
    Batched KMeans clustering in PyTorch using Euclidean distance.

    Args:
        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
        n_clusters: Number of clusters.
        max_iters: Max number of iterations.
        tol: Relative tolerance for center movement. Use None to disable early stopping.
        init_centroids: Optional tensor of shape (B, n_clusters, D) used as the initial centers.
        verbose: Print loss for each iter.
        use_heuristic: Use heuristic Triton config (skip autotune).
    Returns:
        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
        centroids: (B, n_clusters, D) final cluster centers.
        iterations_run: Number of iterations executed.
    """
    B, N, D = x.shape

    # Pre-compute squared L2 norm of all points (constant during iterations)
    x_sq = (x ** 2).sum(dim=-1)  # (B, N)

    if init_centroids is None:
        # Randomly select initial centers from x
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x,
            dim=1,
            index=indices[..., None].expand(-1, -1, D)
        )  # (B, n_clusters, D)
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)

    for it in range(max_iters):
        # ---- compiled single iteration ----
        centroids_new, center_shift, cluster_ids = _euclid_iter_compiled(
            x, x_sq, centroids, use_heuristic
        )

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if tol is not None and center_shift < tol:
            break
        centroids = centroids_new

    return cluster_ids, centroids, it + 1



def batch_mini_batch_kmeans_Euclid(
    x,
    n_clusters,
    mini_batch_size=1024,
    epochs=100,
    learning_rate="adaptive",
    tol=None,
    init_centroids=None,
    init_cluster_counts=None,
    return_cluster_counts=False,
    verbose=False,
    *,
    use_heuristic=True,
):
    """
    Batched mini-batch KMeans clustering in PyTorch using Euclidean distance.

    Args:
        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
        n_clusters: Number of clusters.
        mini_batch_size: Number of points to use in each mini-batch iteration.
        epochs: Number of epochs.
        learning_rate: Mini-batch centroid update rule. One of {"adaptive", "classic"}.
        tol: Relative tolerance for inertia decrease to declare convergence. Use None to disable early stopping.
        init_centroids: Optional tensor of shape (B, n_clusters, D) used as the initial centers.
        init_cluster_counts: Optional tensor of shape (B, n_clusters) with persistent per-cluster counts.
            Used to resume classic mini-batch training from an existing model state.
        return_cluster_counts: When True, also return the final persistent cluster counts.
        verbose: Print loss for each iter.
        use_heuristic: Use heuristic Triton config (skip autotune).
    Returns:
        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
        centroids: (B, n_clusters, D) final cluster centers.
        epochs_run: Number of epochs executed.
        cluster_counts: (B, n_clusters) float tensor of persistent cluster counts when
            return_cluster_counts=True.
    """
    B, N, D = x.shape

    # Pre-compute squared L2 norm of all points (constant during iterations)
    x_sq = (x ** 2).sum(dim=-1)  # (B, N)

    if init_centroids is None:
        # Randomly select initial centers from x
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x,
            dim=1,
            index=indices[..., None].expand(-1, -1, D)
        )  # (B, n_clusters, D)
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)

    # Holds the count of points assigned to each cluster for the full run
    # used in the classic update rule. (B, n_clusters)
    if init_cluster_counts is None:
        cluster_counts = torch.zeros(B, n_clusters, device=x.device, dtype=torch.float32)
    else:
        cluster_counts = init_cluster_counts.to(device=x.device, dtype=torch.float32, copy=False).view(B, n_clusters)

    old_inertia = None
    for it in range(epochs):
        perm = torch.randperm(N, device=x.device)
        epoch_inertia = torch.zeros(B, device=x.device, dtype=torch.float32)
        batches_per_epoch = (N + mini_batch_size - 1) // mini_batch_size

        for batch_start in range(0, N, mini_batch_size):
            # note torch will clip the last batch if mini_batch_size does not divide N for us.
            idx = perm[batch_start:batch_start + mini_batch_size]
            x_mb = x[:, idx, :]  # (B, mini_batch_size, D)
            x_mb_sq = x_sq[:, idx]  # (B, mini_batch_size)

            # ---- compiled single iteration ----
            centroids_new, batch_inertia, cluster_counts = _mini_batch_euclid_iter_compiled(
                x_mb, x_mb_sq, centroids, cluster_counts, learning_rate, use_heuristic
            )
            centroids = centroids_new  # update centroids after each mini-batch
            epoch_inertia += batch_inertia

        avg_inertia = epoch_inertia / batches_per_epoch
        # 4. Check for convergence
        if verbose:
            print(f"Iter {it}, mean inertia: {avg_inertia.mean().item():.6f}")
        if tol is not None and old_inertia is not None:
            stalled = avg_inertia >= old_inertia * (1 - tol)
            if stalled.all():
                break
        old_inertia = avg_inertia

    cluster_ids = euclid_assign_triton(x, centroids, x_sq, use_heuristic=use_heuristic)

    if return_cluster_counts:
        return cluster_ids, centroids, it + 1, cluster_counts
    return cluster_ids, centroids, it + 1


def batch_kmeans_Cosine(x, n_clusters, max_iters=100, tol=None, init_centroids=None, verbose=False):
    """
    Batched KMeans clustering in PyTorch using Cosine similarity.

    Args:
        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
        n_clusters: Number of clusters.
        max_iters: Max number of iterations.
        tol: Relative tolerance for center movement. Use None to disable early stopping.
        verbose: Print loss for each iter.
    Returns:
        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
        centroids: (B, n_clusters, D) final cluster centers.
    """
    B, N, D = x.shape

    # Normalize input vectors for cosine similarity
    x_norm = F.normalize(x, p=2, dim=-1)  # (B, N, D)

    if init_centroids is None:
        # Randomly select initial centers from x_norm
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x_norm,
            dim=1,
            index=indices[..., None].expand(-1, -1, D)
        ) # (B, n_clusters, D)
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)
    centroids = F.normalize(centroids, p=2, dim=-1)  # Ensure centroids are normalized

    for it in range(max_iters):
        # ---- compiled single iteration ----
        centroids_new, center_shift, cluster_ids = _cosine_iter_compiled(x_norm, centroids)

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if tol is not None and center_shift < tol:
            break
        centroids = centroids_new

    return cluster_ids, centroids, it + 1


def batch_kmeans_Dot(x, n_clusters, max_iters=100, tol=None, init_centroids=None, verbose=False):
    """
    Batched KMeans clustering in PyTorch using raw dot-product as similarity.

    """
    B, N, D = x.shape

    if init_centroids is None:
        # 随机初始化中心
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x,
            dim=1,
            index=indices[..., None].expand(-1, -1, D)
        )
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)

    for it in range(max_iters):
        # ---- compiled single iteration ----
        centroids_new, center_shift, cluster_ids = _dot_iter_compiled(x, centroids)

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it} (dot), center shift: {center_shift.item():.6f}")
        if tol is not None and center_shift < tol:
            break
        centroids = centroids_new

    return cluster_ids, centroids, it + 1


if __name__ == "__main__":
    torch.manual_seed(0)
    
    # 用法示例
    B, N, D = 32, 74256, 128  # 32 个 batch，每个 batch 10 万点，128 维
    dtype = torch.float16
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)  # 大 batch 用 GPU 跑
    n_clusters = 1000
    max_iters = 2

    print("=== Testing Euclidean Distance K-Means ===")
    cluster_ids_euclid, centroids_euclid, n_iters_euclid = batch_kmeans_Euclid(x, n_clusters, max_iters=max_iters, verbose=True)
    print(f"Euclidean - cluster_ids shape: {cluster_ids_euclid.shape}, centroids shape: {centroids_euclid.shape}")

    print("\n=== Testing Cosine Similarity K-Means ===")
    cluster_ids_cosine, centroids_cosine, n_iters_cosine = batch_kmeans_Cosine(x, n_clusters, max_iters=max_iters, verbose=True)
    print(f"Cosine - cluster_ids shape: {cluster_ids_cosine.shape}, centroids shape: {centroids_cosine.shape}")

    print("\n=== Testing Dot-Product K-Means ===")
    cluster_ids_dot, centroids_dot, n_iters_dot = batch_kmeans_Dot(x, n_clusters, max_iters=max_iters, verbose=True)
    print(f"Dot - cluster_ids shape: {cluster_ids_dot.shape}, centroids shape: {centroids_dot.shape}")

    # Profile the time cost with rounds=100
    rounds = 200
    import time

    print(f"\n=== Speed Comparison (averaged over {rounds} rounds) ===")

    # Test Euclidean Distance K-Means
    euclid_start = torch.cuda.Event(enable_timing=True)
    euclid_end = torch.cuda.Event(enable_timing=True)
    euclid_start.record()
    for i in range(rounds):
        cluster_ids_euclid, centroids_euclid, n_iters_euclid = batch_kmeans_Euclid(x, n_clusters, init_centroids=centroids_euclid, max_iters=max_iters, verbose=False)
    euclid_end.record(); torch.cuda.synchronize()
    euclid_time = euclid_start.elapsed_time(euclid_end) / rounds
    euclid_time_per_iter = euclid_time / n_iters_euclid
    print(f"Euclidean Distance K-Means: {euclid_time:.2f} ms per run, total {n_iters_euclid} iterations, {euclid_time_per_iter:.2f} ms per iter")
    print(f"Euclidean Distance TFLOPS: {2 * B * N * D * n_clusters * n_iters_euclid / euclid_time / 1e12:.2f}")
    
    # Test Cosine Similarity K-Means
    cosine_start = torch.cuda.Event(enable_timing=True)
    cosine_end = torch.cuda.Event(enable_timing=True)
    cosine_start.record()
    for i in range(rounds):
        cluster_ids_cosine, centroids_cosine, n_iters_cosine = batch_kmeans_Cosine(x, n_clusters, max_iters=max_iters, init_centroids=centroids_cosine, verbose=False)
    cosine_end.record(); torch.cuda.synchronize()
    cosine_time = cosine_start.elapsed_time(cosine_end) / rounds
    cosine_time_per_iter = cosine_time / n_iters_cosine
    print(f"Cosine Similarity K-Means: {cosine_time:.2f} ms per run, total {n_iters_cosine} iterations, {cosine_time_per_iter:.2f} ms per iter")
    print(f"Cosine Similarity TFLOPS: {2 * B * N * D * n_clusters * n_iters_cosine / cosine_time / 1e12:.2f}")

    # Test Dot-Product K-Means
    dot_start = torch.cuda.Event(enable_timing=True)
    dot_end = torch.cuda.Event(enable_timing=True)
    dot_start.record()
    for i in range(rounds):
        cluster_ids_dot, centroids_dot, n_iters_dot = batch_kmeans_Dot(x, n_clusters, max_iters=max_iters, init_centroids=centroids_dot, verbose=False)
    dot_end.record(); torch.cuda.synchronize()
    dot_time = dot_start.elapsed_time(dot_end) / rounds
    dot_time_per_iter = dot_time / n_iters_dot
    print(f"Dot-Product K-Means: {dot_time:.2f} ms per run, total {n_iters_dot} iterations, {dot_time_per_iter:.2f} ms per iter")
    print(f"Dot-Product TFLOPS: {2 * B * N * D * n_clusters * n_iters_dot / dot_time / 1e12:.2f}")
