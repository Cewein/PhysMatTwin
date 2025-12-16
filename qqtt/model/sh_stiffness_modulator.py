"""
FiLM-style SH Coefficient Modulator for Physics-to-Appearance Mapping.

This module implements a learned modulation of Spherical Harmonics (SH) coefficients
based on physical stiffness values, enabling physics-aware appearance rendering.

The FiLM modulation follows:
    SH_out = gamma(stiffness) * SH_in + beta(stiffness)

Where gamma and beta are learned functions of per-Gaussian stiffness.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SHStiffnessModulator(nn.Module):
    """
    FiLM-style modulator that scales/shifts SH coefficients based on stiffness.

    Supports two modes:
    A) Per-band modulation: gamma/beta for each SH band (l=0,1,2,3) - more stable
    B) Per-coefficient modulation: gamma/beta for each SH coeff - more expressive

    Args:
        sh_degree: Maximum SH degree (typically 3, giving 16 coefficients)
        hidden_dim: Hidden dimension for the MLP
        use_dc_context: Whether to use DC color as additional input context
        modulation_mode: 'per_band' (default) or 'per_coeff'
        init_scale: Scale for initialization (default 0.05 for stability)
    """

    # SH band structure: band l has (2l+1) coefficients
    # For degree 3: band0=1, band1=3, band2=5, band3=7 coefficients
    SH_BAND_SIZES = [1, 3, 5, 7]  # Coefficients per band for l=0,1,2,3

    def __init__(
        self,
        sh_degree: int = 3,
        hidden_dim: int = 64,
        use_dc_context: bool = True,
        modulation_mode: str = 'per_band',
        init_scale: float = 0.05,
    ):
        super().__init__()

        self.sh_degree = sh_degree
        self.num_bands = sh_degree + 1  # 4 bands for degree 3
        self.num_coeffs = (sh_degree + 1) ** 2  # 16 coefficients for degree 3
        self.modulation_mode = modulation_mode
        self.init_scale = init_scale
        self.use_dc_context = use_dc_context

        # Input dimension: stiffness (1) + optional DC color (3)
        input_dim = 1 + (3 if use_dc_context else 0)

        # Output dimension depends on mode
        if modulation_mode == 'per_band':
            output_dim = self.num_bands  # 4 gamma + 4 beta values
        else:  # per_coeff
            output_dim = self.num_coeffs  # 16 gamma + 16 beta values

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Separate heads for gamma and beta (raw outputs)
        self.gamma_head = nn.Linear(hidden_dim, output_dim)
        self.beta_head = nn.Linear(hidden_dim, output_dim)

        # Initialize for near-identity transform
        self._init_weights()

        # Build band-to-coefficient mapping for per_band mode
        self._build_band_mapping()

    def _init_weights(self):
        """Initialize weights for near-identity transform at start."""
        # Initialize feature net normally
        for m in self.feature_net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

        # Initialize gamma/beta heads to output near-zero
        # This makes gamma ~1 and beta ~0 initially
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def _build_band_mapping(self):
        """Build mapping from band indices to coefficient indices."""
        # Coefficient indices for each band
        band_starts = []
        idx = 0
        for l in range(self.num_bands):
            band_starts.append(idx)
            idx += 2 * l + 1
        self.register_buffer('band_starts', torch.tensor(band_starts, dtype=torch.long))

        # Create coefficient-to-band mapping
        coeff_to_band = []
        for l in range(self.num_bands):
            coeff_to_band.extend([l] * (2 * l + 1))
        self.register_buffer('coeff_to_band', torch.tensor(coeff_to_band, dtype=torch.long))

    def forward(
        self,
        stiffness: torch.Tensor,
        sh_features: torch.Tensor,
        dc_color: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply FiLM modulation to SH coefficients.

        Args:
            stiffness: Per-Gaussian stiffness values (N, 1)
            sh_features: SH coefficients (N, K, 3) where K = (sh_degree+1)^2
            dc_color: Optional DC color for context (N, 3)

        Returns:
            Modulated SH coefficients (N, K, 3)
        """
        N = stiffness.shape[0]

        # Build input features
        if self.use_dc_context and dc_color is not None:
            input_feats = torch.cat([stiffness, dc_color], dim=-1)  # (N, 4)
        else:
            input_feats = stiffness  # (N, 1)

        # Extract features
        features = self.feature_net(input_feats)  # (N, hidden_dim)

        # Compute raw gamma and beta
        raw_gamma = self.gamma_head(features)  # (N, num_bands) or (N, num_coeffs)
        raw_beta = self.beta_head(features)    # (N, num_bands) or (N, num_coeffs)

        # Apply stability constraints: gamma = 1 + s*tanh(raw), beta = s*tanh(raw)
        gamma = 1.0 + self.init_scale * torch.tanh(raw_gamma)
        beta = self.init_scale * torch.tanh(raw_beta)

        # Apply modulation based on mode
        if self.modulation_mode == 'per_band':
            return self._apply_band_modulation(sh_features, gamma, beta)
        else:
            return self._apply_coeff_modulation(sh_features, gamma, beta)

    def _apply_band_modulation(
        self,
        sh_features: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply per-band FiLM modulation.

        Args:
            sh_features: (N, K, 3)
            gamma: (N, num_bands)
            beta: (N, num_bands)

        Returns:
            Modulated SH (N, K, 3)
        """
        N, K, C = sh_features.shape

        # Expand gamma/beta from band to coefficient space
        # coeff_to_band: (K,) maps each coeff to its band
        gamma_expanded = gamma[:, self.coeff_to_band]  # (N, K)
        beta_expanded = beta[:, self.coeff_to_band]    # (N, K)

        # Add channel dimension for broadcasting: (N, K, 1)
        gamma_expanded = gamma_expanded.unsqueeze(-1)
        beta_expanded = beta_expanded.unsqueeze(-1)

        # FiLM modulation
        sh_out = gamma_expanded * sh_features + beta_expanded

        return sh_out

    def _apply_coeff_modulation(
        self,
        sh_features: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply per-coefficient FiLM modulation.

        Args:
            sh_features: (N, K, 3)
            gamma: (N, K)
            beta: (N, K)

        Returns:
            Modulated SH (N, K, 3)
        """
        # Add channel dimension for broadcasting: (N, K, 1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        # FiLM modulation (same gamma/beta across RGB channels)
        sh_out = gamma * sh_features + beta

        return sh_out

    def get_modulation_params(
        self,
        stiffness: torch.Tensor,
        dc_color: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get gamma and beta values without applying them.
        Useful for visualization/debugging.

        Returns:
            Tuple of (gamma, beta) tensors
        """
        if self.use_dc_context and dc_color is not None:
            input_feats = torch.cat([stiffness, dc_color], dim=-1)
        else:
            input_feats = stiffness

        features = self.feature_net(input_feats)
        raw_gamma = self.gamma_head(features)
        raw_beta = self.beta_head(features)

        gamma = 1.0 + self.init_scale * torch.tanh(raw_gamma)
        beta = self.init_scale * torch.tanh(raw_beta)

        return gamma, beta


class SHStiffnessModulatorLight(nn.Module):
    """
    Lightweight version with fewer parameters for faster training.
    Uses a single-layer MLP.
    """

    def __init__(
        self,
        sh_degree: int = 3,
        hidden_dim: int = 32,
        init_scale: float = 0.05,
    ):
        super().__init__()

        self.sh_degree = sh_degree
        self.num_bands = sh_degree + 1
        self.init_scale = init_scale

        # Minimal network: stiffness -> band gamma/beta
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.num_bands * 2),  # gamma and beta
        )

        # Initialize for near-identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Build band mapping
        coeff_to_band = []
        for l in range(self.num_bands):
            coeff_to_band.extend([l] * (2 * l + 1))
        self.register_buffer('coeff_to_band', torch.tensor(coeff_to_band, dtype=torch.long))

    def forward(
        self,
        stiffness: torch.Tensor,
        sh_features: torch.Tensor,
        dc_color: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            stiffness: (N, 1)
            sh_features: (N, K, 3)
            dc_color: Ignored in light version

        Returns:
            Modulated SH (N, K, 3)
        """
        # Get raw outputs
        out = self.net(stiffness)  # (N, num_bands * 2)
        raw_gamma = out[:, :self.num_bands]
        raw_beta = out[:, self.num_bands:]

        # Apply stability constraints
        gamma = 1.0 + self.init_scale * torch.tanh(raw_gamma)
        beta = self.init_scale * torch.tanh(raw_beta)

        # Expand to coefficient space
        gamma = gamma[:, self.coeff_to_band].unsqueeze(-1)  # (N, K, 1)
        beta = beta[:, self.coeff_to_band].unsqueeze(-1)    # (N, K, 1)

        return gamma * sh_features + beta


def create_modulator(
    sh_degree: int = 3,
    variant: str = 'standard',
    **kwargs
) -> nn.Module:
    """
    Factory function to create SH stiffness modulator.

    Args:
        sh_degree: Maximum SH degree
        variant: 'standard', 'light', 'per_coeff'
        **kwargs: Additional arguments passed to constructor

    Returns:
        SHStiffnessModulator instance
    """
    if variant == 'standard':
        return SHStiffnessModulator(
            sh_degree=sh_degree,
            modulation_mode='per_band',
            **kwargs
        )
    elif variant == 'light':
        return SHStiffnessModulatorLight(
            sh_degree=sh_degree,
            **kwargs
        )
    elif variant == 'per_coeff':
        return SHStiffnessModulator(
            sh_degree=sh_degree,
            modulation_mode='per_coeff',
            **kwargs
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
