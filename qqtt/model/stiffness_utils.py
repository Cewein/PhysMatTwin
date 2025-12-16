"""
Stiffness Interpolation Utilities.

Provides functions to:
1. Aggregate per-spring stiffness to per-particle stiffness
2. Interpolate per-particle stiffness to per-Gaussian stiffness using KNN

This enables mapping physics material properties to 3D Gaussian Splatting primitives.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import warp as wp


def spring_to_particle_stiffness(
    spring_Y: torch.Tensor,
    springs: torch.Tensor,
    num_particles: int,
    reduction: str = 'mean',
    log_space: bool = True,
) -> torch.Tensor:
    """
    Aggregate per-spring stiffness to per-particle stiffness.

    For each particle, computes the aggregation (mean/max/sum) of stiffness
    values from all springs incident to that particle.

    Args:
        spring_Y: Spring stiffness values (M,) - may be in log space
        springs: Spring connectivity (M, 2) - indices of connected particles
        num_particles: Total number of particles
        reduction: Aggregation method - 'mean', 'max', 'sum'
        log_space: If True, spring_Y is in log space (will exp before aggregating)

    Returns:
        Per-particle stiffness (num_particles,)
    """
    device = spring_Y.device
    M = springs.shape[0]

    # Convert from log space if needed
    if log_space:
        stiffness = torch.exp(spring_Y)
    else:
        stiffness = spring_Y.clone()

    # Initialize accumulators
    particle_stiffness_sum = torch.zeros(num_particles, device=device)
    particle_stiffness_count = torch.zeros(num_particles, device=device)
    particle_stiffness_max = torch.zeros(num_particles, device=device)

    # Get spring endpoints
    idx1 = springs[:, 0]  # (M,)
    idx2 = springs[:, 1]  # (M,)

    # Scatter add stiffness to both endpoints
    particle_stiffness_sum.scatter_add_(0, idx1, stiffness)
    particle_stiffness_sum.scatter_add_(0, idx2, stiffness)

    # Count connections per particle
    ones = torch.ones(M, device=device)
    particle_stiffness_count.scatter_add_(0, idx1, ones)
    particle_stiffness_count.scatter_add_(0, idx2, ones)

    # Avoid division by zero
    particle_stiffness_count = torch.clamp(particle_stiffness_count, min=1.0)

    if reduction == 'mean':
        particle_stiffness = particle_stiffness_sum / particle_stiffness_count
    elif reduction == 'sum':
        particle_stiffness = particle_stiffness_sum
    elif reduction == 'max':
        # For max, we need a different approach
        # Scatter max for both endpoints
        particle_stiffness_max1 = torch.zeros(num_particles, device=device)
        particle_stiffness_max2 = torch.zeros(num_particles, device=device)

        particle_stiffness_max1.scatter_reduce_(0, idx1, stiffness, reduce='amax')
        particle_stiffness_max2.scatter_reduce_(0, idx2, stiffness, reduce='amax')

        particle_stiffness = torch.maximum(particle_stiffness_max1, particle_stiffness_max2)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return particle_stiffness


def particle_to_gaussian_stiffness_knn(
    particle_positions: torch.Tensor,
    particle_stiffness: torch.Tensor,
    gaussian_positions: torch.Tensor,
    k: int = 4,
    weight_type: str = 'inverse_distance',
) -> torch.Tensor:
    """
    Interpolate per-particle stiffness to per-Gaussian using KNN.

    For each Gaussian, finds the k nearest particles and interpolates
    their stiffness values using distance-weighted averaging.

    Args:
        particle_positions: Particle 3D positions (P, 3)
        particle_stiffness: Per-particle stiffness (P,)
        gaussian_positions: Gaussian 3D positions (N, 3)
        k: Number of nearest neighbors
        weight_type: 'inverse_distance', 'uniform', or 'gaussian'

    Returns:
        Per-Gaussian stiffness (N,)
    """
    device = gaussian_positions.device
    N = gaussian_positions.shape[0]
    P = particle_positions.shape[0]

    # Clamp k to available particles
    k = min(k, P)

    # Compute pairwise distances: (N, P)
    # Use chunked computation for memory efficiency on large tensors
    if N * P > 1e8:  # ~400MB for float32
        return _particle_to_gaussian_stiffness_knn_chunked(
            particle_positions, particle_stiffness, gaussian_positions, k, weight_type
        )

    # Compute squared distances
    # (N, 1, 3) - (1, P, 3) -> (N, P, 3) -> (N, P)
    diff = gaussian_positions.unsqueeze(1) - particle_positions.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=-1)  # (N, P)

    # Find k nearest neighbors
    _, knn_indices = torch.topk(dist_sq, k, dim=1, largest=False)  # (N, k)

    # Get distances and stiffness of neighbors
    knn_dist_sq = torch.gather(dist_sq, 1, knn_indices)  # (N, k)
    knn_stiffness = particle_stiffness[knn_indices]  # (N, k)

    # Compute weights
    if weight_type == 'uniform':
        weights = torch.ones_like(knn_dist_sq)
    elif weight_type == 'inverse_distance':
        # Inverse distance weighting: w = 1 / (d + eps)
        knn_dist = torch.sqrt(knn_dist_sq + 1e-8)
        weights = 1.0 / (knn_dist + 1e-6)
    elif weight_type == 'gaussian':
        # Gaussian weighting: w = exp(-d^2 / (2 * sigma^2))
        # Use median distance as sigma
        knn_dist = torch.sqrt(knn_dist_sq + 1e-8)
        sigma = torch.median(knn_dist, dim=1, keepdim=True).values + 1e-6
        weights = torch.exp(-knn_dist_sq / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    # Normalize weights
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    # Weighted average of stiffness
    gaussian_stiffness = (weights * knn_stiffness).sum(dim=1)  # (N,)

    return gaussian_stiffness


def _particle_to_gaussian_stiffness_knn_chunked(
    particle_positions: torch.Tensor,
    particle_stiffness: torch.Tensor,
    gaussian_positions: torch.Tensor,
    k: int = 4,
    weight_type: str = 'inverse_distance',
    chunk_size: int = 10000,
) -> torch.Tensor:
    """Memory-efficient chunked version of KNN interpolation."""
    device = gaussian_positions.device
    N = gaussian_positions.shape[0]

    gaussian_stiffness = torch.zeros(N, device=device)

    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk_positions = gaussian_positions[start_idx:end_idx]

        # Compute distances for this chunk
        diff = chunk_positions.unsqueeze(1) - particle_positions.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=-1)

        # Find k nearest neighbors
        _, knn_indices = torch.topk(dist_sq, k, dim=1, largest=False)
        knn_dist_sq = torch.gather(dist_sq, 1, knn_indices)
        knn_stiffness = particle_stiffness[knn_indices]

        # Compute weights
        if weight_type == 'uniform':
            weights = torch.ones_like(knn_dist_sq)
        elif weight_type == 'inverse_distance':
            knn_dist = torch.sqrt(knn_dist_sq + 1e-8)
            weights = 1.0 / (knn_dist + 1e-6)
        elif weight_type == 'gaussian':
            knn_dist = torch.sqrt(knn_dist_sq + 1e-8)
            sigma = torch.median(knn_dist, dim=1, keepdim=True).values + 1e-6
            weights = torch.exp(-knn_dist_sq / (2 * sigma ** 2))
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        gaussian_stiffness[start_idx:end_idx] = (weights * knn_stiffness).sum(dim=1)

    return gaussian_stiffness


def normalize_stiffness(
    stiffness: torch.Tensor,
    method: str = 'minmax',
    clip_percentile: Optional[float] = None,
) -> torch.Tensor:
    """
    Normalize stiffness values for stable modulation.

    Args:
        stiffness: Raw stiffness values (N,)
        method: 'minmax' (0-1), 'zscore' (zero mean, unit var), 'log_minmax'
        clip_percentile: If set, clip values outside this percentile (e.g., 1.0)

    Returns:
        Normalized stiffness (N,)
    """
    # Optionally clip outliers
    if clip_percentile is not None:
        low = torch.quantile(stiffness, clip_percentile / 100.0)
        high = torch.quantile(stiffness, 1.0 - clip_percentile / 100.0)
        stiffness = torch.clamp(stiffness, low, high)

    if method == 'minmax':
        s_min = stiffness.min()
        s_max = stiffness.max()
        if s_max - s_min < 1e-8:
            return torch.zeros_like(stiffness)
        return (stiffness - s_min) / (s_max - s_min + 1e-8)

    elif method == 'zscore':
        mean = stiffness.mean()
        std = stiffness.std() + 1e-8
        return (stiffness - mean) / std

    elif method == 'log_minmax':
        log_stiff = torch.log(stiffness + 1e-8)
        s_min = log_stiff.min()
        s_max = log_stiff.max()
        if s_max - s_min < 1e-8:
            return torch.zeros_like(stiffness)
        return (log_stiff - s_min) / (s_max - s_min + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


class StiffnessInterpolator:
    """
    Stateful class for efficient repeated stiffness interpolation.

    Precomputes KNN indices for faster updates when only stiffness values change.
    """

    def __init__(
        self,
        particle_positions: torch.Tensor,
        springs: torch.Tensor,
        gaussian_positions: torch.Tensor,
        k: int = 4,
        weight_type: str = 'inverse_distance',
        normalize_method: str = 'minmax',
    ):
        """
        Initialize interpolator with fixed geometry.

        Args:
            particle_positions: (P, 3)
            springs: (M, 2)
            gaussian_positions: (N, 3)
            k: KNN neighbors
            weight_type: Weighting scheme
            normalize_method: Stiffness normalization
        """
        self.device = gaussian_positions.device
        self.num_particles = particle_positions.shape[0]
        self.num_gaussians = gaussian_positions.shape[0]
        self.num_springs = springs.shape[0]
        self.k = min(k, self.num_particles)
        self.normalize_method = normalize_method

        # Store springs for particle stiffness computation
        self.springs = springs

        # Precompute KNN indices and weights
        self._precompute_knn(particle_positions, gaussian_positions, weight_type)

    def _precompute_knn(
        self,
        particle_positions: torch.Tensor,
        gaussian_positions: torch.Tensor,
        weight_type: str,
    ):
        """Precompute KNN indices and weights."""
        N = gaussian_positions.shape[0]
        P = particle_positions.shape[0]

        # Compute all pairwise distances (may be memory intensive for large N*P)
        # For production, use chunked or approximate KNN
        if N * P > 5e8:
            # Use chunked computation
            self._precompute_knn_chunked(
                particle_positions, gaussian_positions, weight_type
            )
            return

        diff = gaussian_positions.unsqueeze(1) - particle_positions.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=-1)  # (N, P)

        # Find k nearest neighbors
        _, self.knn_indices = torch.topk(dist_sq, self.k, dim=1, largest=False)  # (N, k)

        # Compute weights
        knn_dist_sq = torch.gather(dist_sq, 1, self.knn_indices)

        if weight_type == 'uniform':
            weights = torch.ones_like(knn_dist_sq)
        elif weight_type == 'inverse_distance':
            knn_dist = torch.sqrt(knn_dist_sq + 1e-8)
            weights = 1.0 / (knn_dist + 1e-6)
        elif weight_type == 'gaussian':
            knn_dist = torch.sqrt(knn_dist_sq + 1e-8)
            sigma = torch.median(knn_dist, dim=1, keepdim=True).values + 1e-6
            weights = torch.exp(-knn_dist_sq / (2 * sigma ** 2))
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

        # Normalize and store
        self.knn_weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    def _precompute_knn_chunked(
        self,
        particle_positions: torch.Tensor,
        gaussian_positions: torch.Tensor,
        weight_type: str,
        chunk_size: int = 10000,
    ):
        """Chunked KNN precomputation for memory efficiency."""
        N = gaussian_positions.shape[0]

        knn_indices_list = []
        knn_weights_list = []

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            chunk_positions = gaussian_positions[start_idx:end_idx]

            diff = chunk_positions.unsqueeze(1) - particle_positions.unsqueeze(0)
            dist_sq = (diff ** 2).sum(dim=-1)

            _, indices = torch.topk(dist_sq, self.k, dim=1, largest=False)
            knn_dist_sq = torch.gather(dist_sq, 1, indices)

            if weight_type == 'uniform':
                weights = torch.ones_like(knn_dist_sq)
            elif weight_type == 'inverse_distance':
                knn_dist = torch.sqrt(knn_dist_sq + 1e-8)
                weights = 1.0 / (knn_dist + 1e-6)
            elif weight_type == 'gaussian':
                knn_dist = torch.sqrt(knn_dist_sq + 1e-8)
                sigma = torch.median(knn_dist, dim=1, keepdim=True).values + 1e-6
                weights = torch.exp(-knn_dist_sq / (2 * sigma ** 2))
            else:
                raise ValueError(f"Unknown weight_type: {weight_type}")

            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

            knn_indices_list.append(indices)
            knn_weights_list.append(weights)

        self.knn_indices = torch.cat(knn_indices_list, dim=0)
        self.knn_weights = torch.cat(knn_weights_list, dim=0)

    def __call__(
        self,
        spring_Y: torch.Tensor,
        log_space: bool = True,
        normalize: bool = True,
        detach: bool = True,
    ) -> torch.Tensor:
        """
        Compute per-Gaussian stiffness from spring stiffness values.

        Args:
            spring_Y: Per-spring stiffness (M,)
            log_space: Whether spring_Y is in log space
            normalize: Whether to normalize output
            detach: Whether to detach gradients (recommended for stability)

        Returns:
            Per-Gaussian stiffness (N, 1) - ready for modulator input
        """
        # Spring -> Particle stiffness
        particle_stiffness = spring_to_particle_stiffness(
            spring_Y, self.springs, self.num_particles,
            reduction='mean', log_space=log_space
        )

        # Particle -> Gaussian using precomputed KNN
        knn_stiffness = particle_stiffness[self.knn_indices]  # (N, k)
        gaussian_stiffness = (self.knn_weights * knn_stiffness).sum(dim=1)  # (N,)

        # Normalize for stable modulation
        if normalize:
            gaussian_stiffness = normalize_stiffness(
                gaussian_stiffness, method=self.normalize_method
            )

        # Detach for clean separation of physics/appearance training
        if detach:
            gaussian_stiffness = gaussian_stiffness.detach()

        # Add channel dimension: (N,) -> (N, 1)
        return gaussian_stiffness.unsqueeze(-1)

    def update_gaussian_positions(
        self,
        new_gaussian_positions: torch.Tensor,
        particle_positions: torch.Tensor,
        weight_type: str = 'inverse_distance',
    ):
        """
        Recompute KNN for new Gaussian positions.
        Call this if Gaussians move significantly.
        """
        self.num_gaussians = new_gaussian_positions.shape[0]
        self._precompute_knn(particle_positions, new_gaussian_positions, weight_type)


def get_gaussian_stiffness(
    spring_Y: torch.Tensor,
    springs: torch.Tensor,
    particle_positions: torch.Tensor,
    gaussian_positions: torch.Tensor,
    k: int = 4,
    log_space: bool = True,
    normalize: bool = True,
    detach: bool = True,
) -> torch.Tensor:
    """
    One-shot function to compute per-Gaussian stiffness.

    Convenience function that combines all steps. For repeated calls,
    prefer using StiffnessInterpolator class.

    Args:
        spring_Y: Per-spring stiffness (M,)
        springs: Spring connectivity (M, 2)
        particle_positions: Particle positions (P, 3)
        gaussian_positions: Gaussian positions (N, 3)
        k: KNN neighbors
        log_space: Whether spring_Y is in log space
        normalize: Whether to normalize output
        detach: Whether to detach gradients

    Returns:
        Per-Gaussian stiffness (N, 1)
    """
    num_particles = particle_positions.shape[0]

    # Spring -> Particle
    particle_stiffness = spring_to_particle_stiffness(
        spring_Y, springs, num_particles,
        reduction='mean', log_space=log_space
    )

    # Particle -> Gaussian
    gaussian_stiffness = particle_to_gaussian_stiffness_knn(
        particle_positions, particle_stiffness, gaussian_positions, k=k
    )

    # Normalize
    if normalize:
        gaussian_stiffness = normalize_stiffness(gaussian_stiffness, method='minmax')

    # Detach
    if detach:
        gaussian_stiffness = gaussian_stiffness.detach()

    return gaussian_stiffness.unsqueeze(-1)
