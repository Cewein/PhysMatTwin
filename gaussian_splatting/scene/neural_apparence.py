#
# Neural Appearance Head for Physics-Stiffness-Aware Gaussian Splatting
# This module predicts per-Gaussian RGB colors using SH features, view direction, and stiffness.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_stiffness(stiffness: torch.Tensor, min_val: float = None, max_val: float = None) -> torch.Tensor:
    """
    Normalize stiffness values to [0, 1] range.

    Args:
        stiffness: Raw stiffness values
        min_val: Minimum value for normalization (uses data min if None)
        max_val: Maximum value for normalization (uses data max if None)

    Returns:
        Normalized stiffness in [0, 1]
    """
    if min_val is None:
        min_val = stiffness.min()
    if max_val is None:
        max_val = stiffness.max()
    # Avoid division by zero
    range_val = max_val - min_val
    if range_val < 1e-8:
        return torch.zeros_like(stiffness)
    normalized = (stiffness - min_val) / range_val
    return torch.clamp(normalized, 0.0, 1.0)

class NeuralAppearanceHead(nn.Module):
    """
    A small MLP that predicts per-Gaussian RGB color using:
    - Gaussian's existing SH features (flattened DC + rest)
    - Current view direction (camera center -> gaussian center)
    - A stiffness value from the physics module

    Input dimensions:
    - SH features: (N, num_sh_coeffs * 3) where num_sh_coeffs = (max_sh_degree + 1)^2
    - View direction: (N, 3) normalized
    - Stiffness: (N, 1) scalar per Gaussian

    Output:
    - RGB: (N, 3) colors in [0, 1]
    """

    def __init__(
        self,
        sh_degree: int = 3,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        use_view_dir: bool = True,
        use_stiffness: bool = True,
        view_dir_encoding_dim: int = 16,
        stiffness_encoding_dim: int = 8,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.use_view_dir = use_view_dir
        self.use_stiffness = use_stiffness

        # Calculate input dimensions
        num_sh_coeffs = (sh_degree + 1) ** 2
        sh_feature_dim = num_sh_coeffs * 3  # 3 color channels

        # Optional positional encoding for view direction
        self.view_dir_encoding_dim = view_dir_encoding_dim
        if use_view_dir:
            # Simple frequency encoding for view direction
            self.view_freqs = nn.Parameter(
                torch.randn(3, view_dir_encoding_dim // 2) * 2.0,
                requires_grad=False
            )
            view_input_dim = view_dir_encoding_dim
        else:
            view_input_dim = 0

        # Optional encoding for stiffness
        self.stiffness_encoding_dim = stiffness_encoding_dim
        if use_stiffness:
            # Learn a small embedding for stiffness
            self.stiffness_encoder = nn.Sequential(
                nn.Linear(1, stiffness_encoding_dim),
                nn.ReLU(inplace=True),
            )
            stiffness_input_dim = stiffness_encoding_dim
        else:
            stiffness_input_dim = 0

        # Total input dimension
        input_dim = sh_feature_dim + view_input_dim + stiffness_input_dim

        # Build MLP
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, 3))  # Output RGB
        layers.append(nn.Sigmoid())  # Ensure output is in [0, 1]

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Special initialization for the final layer to output ~0.5 (gray) initially
        final_layer = self.mlp[-2]  # Linear before Sigmoid
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def encode_view_direction(self, view_dirs: torch.Tensor) -> torch.Tensor:
        """
        Encode view direction using frequency encoding.

        Args:
            view_dirs: (N, 3) normalized view direction vectors

        Returns:
            (N, view_dir_encoding_dim) encoded view directions
        """
        # view_dirs: (N, 3)
        # self.view_freqs: (3, encoding_dim // 2)
        # Output: (N, encoding_dim)
        proj = view_dirs @ self.view_freqs  # (N, encoding_dim // 2)
        encoded = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, encoding_dim)
        return encoded

    def forward(
        self,
        sh_features: torch.Tensor,
        view_dirs: torch.Tensor = None,
        stiffness: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass to compute RGB colors.

        Args:
            sh_features: (N, K, 3) SH coefficients where K = (sh_degree + 1)^2
                        or (N, K*3) flattened SH features
            view_dirs: (N, 3) normalized view direction (gaussian - camera_center)
            stiffness: (N,) or (N, 1) stiffness values per Gaussian

        Returns:
            rgb: (N, 3) RGB colors in [0, 1]
        """
        N = sh_features.shape[0]
        device = sh_features.device

        # Flatten SH features if needed: (N, K, 3) -> (N, K*3)
        if sh_features.dim() == 3:
            sh_flat = sh_features.reshape(N, -1)
        else:
            sh_flat = sh_features

        # Collect all input features
        inputs = [sh_flat]

        # Add encoded view direction
        if self.use_view_dir:
            if view_dirs is None:
                # Default to forward direction if not provided
                view_dirs = torch.zeros(N, 3, device=device)
                view_dirs[:, 2] = 1.0
            view_encoded = self.encode_view_direction(view_dirs)
            inputs.append(view_encoded)

        # Add encoded stiffness
        if self.use_stiffness:
            if stiffness is None:
                # Default stiffness of 0.5 (normalized middle value)
                stiffness = torch.full((N, 1), 0.5, device=device)
            elif stiffness.dim() == 1:
                stiffness = stiffness.unsqueeze(-1)
            stiffness_encoded = self.stiffness_encoder(stiffness)
            inputs.append(stiffness_encoded)

        # Concatenate all inputs
        x = torch.cat(inputs, dim=-1)

        # Pass through MLP
        rgb = self.mlp(x)

        return rgb