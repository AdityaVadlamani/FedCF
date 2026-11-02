"""Differential privacy helpers for FedCF (communication-efficient protocol).

Each client releases its lower/upper coverage terms once per coverage-gap round.
We make the whole run (eps_total, delta_total)-DP as follows:

1. Advanced composition (Dwork, Rothblum & Vadhan 2010) over the T rounds gives a
   per-round budget (eps_0, delta_0). We fall back to basic composition
   (eps_total / T) when that is larger.
2. Replacing one calibration point changes at most m coordinates of a round's
   release: it touches at most 2 groups, i.e. 2 * |Y+| cells in each released
   array. Each coordinate gets (eps_0 / m, delta_0 / m), so the round is
   (eps_0, delta_0)-DP by basic composition over the changed coordinates.
3. Each coordinate gets truncated Laplace noise (Geng et al., AISTATS 2020),
   which is (eps, delta)-DP for sensitivity Delta:
       scale b = Delta / eps, support [-A, A],
       A = (Delta / eps) * ln(1 + (e^eps - 1) / (2 * delta)).
   Shifting by +A makes the noise non-negative. We add it to upper terms and
   subtract it from lower terms, so the server can only overestimate the
   coverage gap and the closeness guarantee stays strict.

Per-coordinate sensitivity is 1 / n_k for both the upper term (alpha+1)/(n_k+1)
and the lower term alpha * n / ((n+1)(n_k+1)) (and for the MLE alpha / n_k).

The above is the "per_round" release. The "histogram" release (bottom of this
file) instead noises one prefix-sum tree of each client's scores, once.
"""

import math
from typing import Tuple

import torch


def advanced_composition_per_round(
    eps_total: float, delta_total: float, num_rounds: int
) -> Tuple[float, float]:
    """Per-round (eps_0, delta_0) so that num_rounds-fold composition is
    (eps_total, delta_total)-DP. Half of delta_total goes to the composition
    slack delta', half to the per-round deltas."""
    T = max(int(num_rounds), 1)
    delta_slack = delta_total / 2
    delta_0 = delta_total / (2 * T)

    def total_eps(e0: float) -> float:
        if e0 > 700:  # e^{e0} overflows; the total is astronomically large
            return math.inf
        return e0 * math.sqrt(2 * T * math.log(1 / delta_slack)) + T * e0 * math.expm1(e0)

    lo, hi = 0.0, eps_total
    for _ in range(200):  # total_eps is increasing in e0
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if total_eps(mid) <= eps_total else (lo, mid)
    eps_adv = lo

    eps_basic = eps_total / T
    if eps_basic >= eps_adv:
        return eps_basic, delta_total / T
    return eps_adv, delta_0


def truncated_laplace_params(
    sensitivity: float, eps: float, delta: float
) -> Tuple[float, float]:
    """(scale b, half-width A) of the (eps, delta)-DP truncated Laplace mechanism."""
    b = sensitivity / eps
    # ln(1 + (e^eps - 1) / (2 delta)), rewritten to avoid overflowing e^eps:
    # = eps - ln(2 delta) + ln(1 - (1 - 2 delta) e^{-eps})
    A = b * (eps - math.log(2 * delta) + math.log1p(-(1 - 2 * delta) * math.exp(-eps)))
    return b, A


def sample_truncated_laplace(shape, b: float, A: float) -> torch.Tensor:
    """Laplace(0, b) truncated to [-A, A] (inverse-CDF sampling), float64."""
    lo_cdf = 0.5 * math.exp(-A / b)  # F(-A)
    u = lo_cdf + (1 - 2 * lo_cdf) * torch.rand(shape, dtype=torch.float64)
    x = torch.where(
        u < 0.5, b * torch.log(2 * u), -b * torch.log(2 * (1 - u))
    )
    return x.clamp(-A, A)


def sample_nonneg_truncated_laplace(shape, b: float, A: float) -> torch.Tensor:
    """Laplace(0, b) truncated to [-A, A], shifted by +A."""
    return (sample_truncated_laplace(shape, b, A) + A).to(torch.float32)


# ---------------------------------------------------------------------------
# One-shot release: a noisy prefix-sum tree (tree mechanism) per histogram.
#
# Each client releases, once, the calibration-score histogram of every
# (filter, group, positive label) over a fixed grid of B bins on [0, max_score].
# The histogram is stored as a binary tree (heap layout, node 1 = root, leaves
# at B..2B-1); every node gets independent truncated Laplace noise in [-A, A].
# A record lands in one leaf per histogram, i.e. in log2(B) + 1 nodes, so
# replacing one record changes at most
#     m = 2 * n_filters * |Y+| * (log2(B) + 1)
# node counts by 1. Each node is (eps/m, delta/m)-DP, so the release is
# (eps, delta)-DP, and every later coverage term is post-processing of it: no
# composition over rounds.
#
# A prefix count #{s <= edge_j} is a sum of at most log2(B) nodes, so
#     noisy_prefix - A * nodes <= true_prefix <= noisy_prefix + A * nodes,
# and rounding lambda down / up to the grid keeps both bounds strict.
# ---------------------------------------------------------------------------


def hist_num_levels(num_bins: int) -> int:
    assert num_bins >= 2 and num_bins & (num_bins - 1) == 0, "num_bins must be a power of 2"
    return int(math.log2(num_bins)) + 1


def hist_edges(num_bins: int, max_score: float) -> torch.Tensor:
    return torch.linspace(0.0, max_score, num_bins + 1, dtype=torch.float64)


def hist_leaf_index(scores: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Leaf j holds edges[j] < s <= edges[j+1] (leaf 0 also holds s <= 0).
    Scores above edges[-1] get index B (dropped)."""
    i = torch.searchsorted(edges, scores.to(torch.float64).contiguous(), right=False)
    return (i - 1).clamp(min=0)


def build_tree(leaf_counts: torch.Tensor) -> torch.Tensor:
    """leaf_counts: (..., B) -> heap tree (..., 2B); index 0 unused."""
    B = leaf_counts.shape[-1]
    tree = torch.zeros(leaf_counts.shape[:-1] + (2 * B,), dtype=torch.float64)
    tree[..., B:] = leaf_counts.to(torch.float64)
    for node in range(B - 1, 0, -1):
        tree[..., node] = tree[..., 2 * node] + tree[..., 2 * node + 1]
    return tree


def tree_prefix(tree: torch.Tensor, j: int) -> Tuple[torch.Tensor, int]:
    """Sum of leaves [0, j) and the number of tree nodes used."""
    B = tree.shape[-1] // 2
    total = torch.zeros(tree.shape[:-1], dtype=torch.float64)
    if j <= 0:
        return total, 0
    if j >= B:
        return tree[..., 1].clone(), 1
    lo, hi, nodes = B, B + j, 0
    while lo < hi:
        if lo & 1:
            total += tree[..., lo]
            lo += 1
            nodes += 1
        if hi & 1:
            hi -= 1
            total += tree[..., hi]
            nodes += 1
        lo >>= 1
        hi >>= 1
    return total, nodes


def snap_to_grid(lmbda: torch.Tensor, num_bins: int, max_score: float) -> torch.Tensor:
    """Round lambda up to the next histogram edge (server side). With lambda on
    an edge, the client's upper and lower bounds use the same prefix, so grid
    rounding adds nothing to the coverage-gap estimate."""
    w = max_score / num_bins
    j = torch.ceil(lmbda.to(torch.float64) / w - 1e-6).clamp(0, num_bins)
    return (j * w).to(lmbda.dtype)


def prefix_count_bounds(
    noisy_tree: torch.Tensor, A: float, edges: torch.Tensor, lmbda: float, n_total: float
) -> Tuple[float, float]:
    """Strict (lower, upper) bounds on #{s <= lmbda} from one noisy tree."""
    B = edges.shape[0] - 1
    # treat lambda within float32 rounding of an edge as that edge
    w = float(edges[1] - edges[0])
    j_near = round(lmbda / w)
    if abs(lmbda - j_near * w) <= 1e-5 * w:
        lmbda = float(edges[min(max(j_near, 0), B)])
    # upper: smallest edge >= lmbda; lower: largest edge <= lmbda
    j_up = int(torch.searchsorted(edges, torch.tensor([lmbda], dtype=torch.float64), right=False))
    j_lo = int(torch.searchsorted(edges, torch.tensor([lmbda], dtype=torch.float64), right=True)) - 1
    if j_up > B:
        upper = float(n_total)
    else:
        s, k = tree_prefix(noisy_tree, max(j_up, 1))
        upper = min(float(n_total), float(s) + A * k)
    if j_lo <= 0:
        lower = 0.0
    else:
        s, k = tree_prefix(noisy_tree, min(j_lo, B))
        lower = max(0.0, float(s) - A * k)
    return lower, upper

