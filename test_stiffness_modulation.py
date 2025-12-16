"""
Test script for stiffness-to-appearance modulation.

Validates:
1. SH modulator produces near-identity at initialization
2. Spring-to-particle stiffness aggregation
3. Particle-to-Gaussian KNN interpolation
4. Full pipeline integration

Run: python test_stiffness_modulation.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_sh_modulator():
    """Test SH stiffness modulator."""
    print("\n" + "="*50)
    print("TEST: SH Stiffness Modulator")
    print("="*50)

    from qqtt.model.sh_stiffness_modulator import (
        SHStiffnessModulator,
        SHStiffnessModulatorLight,
        create_modulator,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = 1000  # Number of Gaussians
    K = 16    # SH coefficients for degree 3
    C = 3     # RGB channels

    # Create test data
    sh_features = torch.randn(N, K, C, device=device)
    stiffness = torch.rand(N, 1, device=device)
    dc_color = torch.rand(N, 3, device=device)

    # Test standard modulator
    print("\n1. Testing SHStiffnessModulator (per-band)...")
    modulator = SHStiffnessModulator(
        sh_degree=3,
        modulation_mode='per_band',
        use_dc_context=True,
    ).to(device)

    sh_out = modulator(stiffness, sh_features, dc_color)

    assert sh_out.shape == sh_features.shape, f"Shape mismatch: {sh_out.shape} vs {sh_features.shape}"

    # Check near-identity at init (gamma~1, beta~0)
    diff = (sh_out - sh_features).abs().mean().item()
    print(f"   Mean absolute difference from identity: {diff:.6f}")
    assert diff < 0.1, f"Initial modulation should be near-identity, got diff={diff}"
    print("   PASS: Near-identity at initialization")

    # Test light modulator
    print("\n2. Testing SHStiffnessModulatorLight...")
    modulator_light = SHStiffnessModulatorLight(sh_degree=3).to(device)
    sh_out_light = modulator_light(stiffness, sh_features)
    assert sh_out_light.shape == sh_features.shape
    print("   PASS: Light modulator works")

    # Test per-coefficient modulator
    print("\n3. Testing per-coefficient modulation...")
    modulator_coeff = SHStiffnessModulator(
        sh_degree=3,
        modulation_mode='per_coeff',
    ).to(device)
    sh_out_coeff = modulator_coeff(stiffness, sh_features, dc_color)
    assert sh_out_coeff.shape == sh_features.shape
    print("   PASS: Per-coefficient modulator works")

    # Test gradient flow
    print("\n4. Testing gradient flow...")
    modulator.zero_grad()
    sh_out = modulator(stiffness, sh_features, dc_color)
    loss = sh_out.sum()
    loss.backward()

    grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in modulator.parameters())
    assert grad_exists, "Gradients should flow through modulator"
    print("   PASS: Gradients flow correctly")

    # Test factory function
    print("\n5. Testing factory function...")
    for variant in ['standard', 'light', 'per_coeff']:
        mod = create_modulator(sh_degree=3, variant=variant).to(device)
        out = mod(stiffness, sh_features, dc_color)
        assert out.shape == sh_features.shape
    print("   PASS: Factory function works for all variants")

    print("\nSH Modulator tests: ALL PASSED")


def test_stiffness_interpolation():
    """Test stiffness interpolation utilities."""
    print("\n" + "="*50)
    print("TEST: Stiffness Interpolation")
    print("="*50)

    from qqtt.model.stiffness_utils import (
        spring_to_particle_stiffness,
        particle_to_gaussian_stiffness_knn,
        normalize_stiffness,
        StiffnessInterpolator,
        get_gaussian_stiffness,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create simple test mesh (cube with springs)
    P = 100  # particles
    N = 500  # Gaussians
    M = 300  # springs

    particle_positions = torch.rand(P, 3, device=device) * 2 - 1  # [-1, 1]^3
    gaussian_positions = torch.rand(N, 3, device=device) * 2 - 1

    # Create random spring connectivity
    springs = torch.stack([
        torch.randint(0, P, (M,), device=device),
        torch.randint(0, P, (M,), device=device),
    ], dim=1)

    # Create spring stiffness with variation (log space)
    spring_Y = torch.log(torch.rand(M, device=device) * 9000 + 1000)  # [1e3, 1e4]

    # Test spring to particle
    print("\n1. Testing spring_to_particle_stiffness...")
    particle_stiff = spring_to_particle_stiffness(
        spring_Y, springs, P, reduction='mean', log_space=True
    )
    assert particle_stiff.shape == (P,), f"Wrong shape: {particle_stiff.shape}"
    print(f"   Particle stiffness range: [{particle_stiff.min():.2e}, {particle_stiff.max():.2e}]")
    print("   PASS: Spring-to-particle aggregation works")

    # Test particle to Gaussian KNN
    print("\n2. Testing particle_to_gaussian_stiffness_knn...")
    gauss_stiff = particle_to_gaussian_stiffness_knn(
        particle_positions, particle_stiff, gaussian_positions, k=4
    )
    assert gauss_stiff.shape == (N,), f"Wrong shape: {gauss_stiff.shape}"
    print(f"   Gaussian stiffness range: [{gauss_stiff.min():.2e}, {gauss_stiff.max():.2e}]")
    print("   PASS: Particle-to-Gaussian KNN works")

    # Test normalization
    print("\n3. Testing normalize_stiffness...")
    for method in ['minmax', 'zscore', 'log_minmax']:
        norm_stiff = normalize_stiffness(gauss_stiff, method=method)
        assert norm_stiff.shape == gauss_stiff.shape
    print("   PASS: Normalization methods work")

    # Test StiffnessInterpolator class
    print("\n4. Testing StiffnessInterpolator class...")
    interpolator = StiffnessInterpolator(
        particle_positions=particle_positions,
        springs=springs,
        gaussian_positions=gaussian_positions,
        k=4,
    )
    gauss_stiff_interp = interpolator(spring_Y, log_space=True, normalize=True)
    assert gauss_stiff_interp.shape == (N, 1), f"Wrong shape: {gauss_stiff_interp.shape}"

    # Check variation
    stiff_min = gauss_stiff_interp.min().item()
    stiff_max = gauss_stiff_interp.max().item()
    print(f"   Normalized stiffness range: [{stiff_min:.4f}, {stiff_max:.4f}]")
    assert stiff_min != stiff_max, "Stiffness should have variation"
    print("   PASS: StiffnessInterpolator produces varied output")

    # Test one-shot function
    print("\n5. Testing get_gaussian_stiffness convenience function...")
    gauss_stiff_oneshot = get_gaussian_stiffness(
        spring_Y, springs, particle_positions, gaussian_positions,
        k=4, log_space=True, normalize=True
    )
    assert gauss_stiff_oneshot.shape == (N, 1)
    print("   PASS: Convenience function works")

    print("\nStiffness Interpolation tests: ALL PASSED")


def test_render_integration():
    """Test that modified render function accepts sh_features."""
    print("\n" + "="*50)
    print("TEST: Render Integration")
    print("="*50)

    # Just test the import and function signature
    from gaussian_splatting.gaussian_renderer import render, render_gsplat, render_3dgs
    import inspect

    print("\n1. Checking render function signature...")
    sig = inspect.signature(render)
    params = list(sig.parameters.keys())
    assert 'sh_features' in params, "render() should have sh_features parameter"
    print(f"   Parameters: {params}")
    print("   PASS: render() has sh_features parameter")

    print("\n2. Checking render_gsplat signature...")
    sig = inspect.signature(render_gsplat)
    params = list(sig.parameters.keys())
    assert 'sh_features' in params, "render_gsplat() should have sh_features parameter"
    print("   PASS: render_gsplat() has sh_features parameter")

    print("\n3. Checking render_3dgs signature...")
    sig = inspect.signature(render_3dgs)
    params = list(sig.parameters.keys())
    assert 'sh_features' in params, "render_3dgs() should have sh_features parameter"
    print("   PASS: render_3dgs() has sh_features parameter")

    print("\nRender Integration tests: ALL PASSED")


def test_full_pipeline():
    """Test full pipeline: stiffness -> modulation -> (mock) render."""
    print("\n" + "="*50)
    print("TEST: Full Pipeline (without actual rendering)")
    print("="*50)

    from qqtt.model.sh_stiffness_modulator import create_modulator
    from qqtt.model.stiffness_utils import StiffnessInterpolator

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Simulate data sizes
    N = 1000   # Gaussians
    P = 200    # Particles
    M = 500    # Springs
    K = 16     # SH coeffs
    C = 3      # RGB

    print(f"\nSimulated setup:")
    print(f"  Gaussians: {N}")
    print(f"  Particles: {P}")
    print(f"  Springs: {M}")

    # Create mock data
    gaussian_positions = torch.rand(N, 3, device=device)
    particle_positions = torch.rand(P, 3, device=device)
    springs = torch.stack([
        torch.randint(0, P, (M,), device=device),
        torch.randint(0, P, (M,), device=device),
    ], dim=1)
    spring_Y = torch.log(torch.rand(M, device=device) * 9000 + 1000)

    # Mock SH features (what Gaussians would have)
    sh_features = torch.randn(N, K, C, device=device)

    print("\n1. Setting up StiffnessInterpolator...")
    interpolator = StiffnessInterpolator(
        particle_positions=particle_positions,
        springs=springs,
        gaussian_positions=gaussian_positions,
        k=4,
    )
    print("   PASS")

    print("\n2. Computing per-Gaussian stiffness...")
    gauss_stiff = interpolator(spring_Y, log_space=True, normalize=True, detach=True)
    print(f"   Shape: {gauss_stiff.shape}")
    print(f"   Range: [{gauss_stiff.min():.4f}, {gauss_stiff.max():.4f}]")
    print("   PASS")

    print("\n3. Creating modulator...")
    modulator = create_modulator(sh_degree=3, variant='standard').to(device)
    print(f"   Parameters: {sum(p.numel() for p in modulator.parameters())}")
    print("   PASS")

    print("\n4. Applying FiLM modulation...")
    dc_color = sh_features[:, 0, :].clone()  # Use DC as context
    sh_out = modulator(gauss_stiff, sh_features, dc_color)
    print(f"   Input shape: {sh_features.shape}")
    print(f"   Output shape: {sh_out.shape}")
    assert sh_out.shape == sh_features.shape
    print("   PASS")

    print("\n5. Verifying gradient flow...")
    modulator.zero_grad()
    loss = sh_out.sum()
    loss.backward()
    grads = [p.grad for p in modulator.parameters() if p.grad is not None]
    assert len(grads) > 0, "Should have gradients"
    total_grad = sum(g.abs().sum().item() for g in grads)
    print(f"   Total gradient magnitude: {total_grad:.4f}")
    print("   PASS")

    print("\n6. Simulating training step...")
    optimizer = torch.optim.Adam(modulator.parameters(), lr=1e-3)
    for step in range(5):
        sh_out = modulator(gauss_stiff, sh_features, dc_color)
        # Mock loss: encourage output to match some target
        target = sh_features * 1.1  # Slightly different
        loss = ((sh_out - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"   Final mock loss: {loss.item():.6f}")
    print("   PASS")

    print("\nFull Pipeline tests: ALL PASSED")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# STIFFNESS MODULATION TEST SUITE")
    print("#"*60)

    test_sh_modulator()
    test_stiffness_interpolation()
    test_render_integration()
    test_full_pipeline()

    print("\n" + "#"*60)
    print("# ALL TESTS PASSED!")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
