"""
Stiffness Modulation Ablation Evaluation Script.

Compares rendering quality across different ablation modes:
- Baseline: No modulator
- No Stiffness: Modulator with zeroed stiffness input
- With Stiffness: Full modulator with physics stiffness

Outputs:
- Side-by-side renders: GT | Baseline | No-Stiff | With-Stiff
- PSNR/SSIM metrics for each mode
- CSV with per-image metrics

Usage:
    python stiffness_ablation.py -s <source_path> \\
        --baseline_ckpt <baseline.pth> \\
        --modulator_ckpt <modulator.pth> \\
        --output_dir <output_path>
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import Scene, GaussianModel
from gaussian_splatting.utils.general_utils import safe_state
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.arguments import ModelParams, PipelineParams

from qqtt.model.sh_stiffness_modulator import create_modulator
from qqtt.model.stiffness_utils import StiffnessInterpolator


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor (C, H, W) to numpy image (H, W, C)."""
    img = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def create_side_by_side(
    gt: np.ndarray,
    baseline: np.ndarray,
    no_stiff: np.ndarray,
    with_stiff: np.ndarray,
    labels: bool = True,
) -> np.ndarray:
    """
    Create side-by-side comparison image.

    Layout: GT | Baseline | No-Stiff | With-Stiff
    """
    H, W, C = gt.shape

    # Create output image
    gap = 2  # Gap between images
    total_width = 4 * W + 3 * gap
    total_height = H + (30 if labels else 0)

    output = np.ones((total_height, total_width, C), dtype=np.uint8) * 255

    # Place images
    images = [gt, baseline, no_stiff, with_stiff]
    label_texts = ['GT', 'Baseline', 'No Stiff', 'With Stiff']

    for i, (img, label) in enumerate(zip(images, label_texts)):
        x_offset = i * (W + gap)
        output[:H, x_offset:x_offset+W] = img

    return output


def load_physics_data(physics_path: str, device: str = 'cuda'):
    """Load physics data from checkpoint or directory."""
    if physics_path and os.path.exists(physics_path):
        if os.path.isfile(physics_path):
            ckpt = torch.load(physics_path, map_location=device)
            return {
                'spring_Y': ckpt.get('spring_Y', ckpt.get('wp_spring_Y')),
                'springs': ckpt.get('springs', ckpt.get('init_springs')),
                'particle_positions': ckpt.get('vertices', ckpt.get('init_vertices')),
            }
        elif os.path.isdir(physics_path):
            physics_data = {}
            for name in ['spring_Y', 'springs', 'particle_positions']:
                fpath = os.path.join(physics_path, f'{name}.pt')
                if os.path.exists(fpath):
                    physics_data[name] = torch.load(fpath, map_location=device)
            return physics_data
    return None


def create_synthetic_physics(gaussians: GaussianModel, device: str = 'cuda'):
    """Create synthetic physics data for demo."""
    positions = gaussians.get_xyz.detach()
    N = positions.shape[0]

    max_particles = min(N, 5000)
    if N > max_particles:
        indices = torch.randperm(N)[:max_particles]
        particle_positions = positions[indices]
    else:
        indices = torch.arange(N)
        particle_positions = positions

    P = particle_positions.shape[0]
    k = min(6, P - 1)

    # KNN connectivity
    diff = particle_positions.unsqueeze(1) - particle_positions.unsqueeze(0)
    dist_sq = (diff ** 2).sum(-1)
    dist_sq.fill_diagonal_(float('inf'))
    _, knn_indices = torch.topk(dist_sq, k, dim=1, largest=False)

    springs_list = []
    for i in range(P):
        for j in range(k):
            neighbor = knn_indices[i, j].item()
            if i < neighbor:
                springs_list.append([i, neighbor])

    springs = torch.tensor(springs_list, dtype=torch.long, device=device)

    # Stiffness with spatial variation
    center = particle_positions.mean(dim=0)
    dist_from_center = (particle_positions - center).norm(dim=1)
    max_dist = dist_from_center.max() + 1e-6
    norm_dist = dist_from_center / max_dist
    particle_stiffness = 1e4 * (1.0 - 0.9 * norm_dist)

    spring_Y = torch.zeros(springs.shape[0], device=device)
    for idx, (i, j) in enumerate(springs):
        spring_Y[idx] = 0.5 * (particle_stiffness[i] + particle_stiffness[j])
    spring_Y = torch.log(spring_Y)

    return {
        'spring_Y': spring_Y,
        'springs': springs,
        'particle_positions': particle_positions,
    }


class AblationRenderer:
    """Helper class to render all ablation modes."""

    def __init__(
        self,
        gaussians: GaussianModel,
        modulator: Optional[torch.nn.Module],
        physics_data: dict,
        device: str = 'cuda',
    ):
        self.gaussians = gaussians
        self.modulator = modulator
        self.device = device

        # Setup stiffness interpolator if modulator exists
        if modulator is not None and physics_data is not None:
            self.interpolator = StiffnessInterpolator(
                particle_positions=physics_data['particle_positions'],
                springs=physics_data['springs'],
                gaussian_positions=gaussians.get_xyz.detach(),
                k=4,
                weight_type='inverse_distance',
            )
            self.spring_Y = physics_data['spring_Y']
        else:
            self.interpolator = None
            self.spring_Y = None

    def get_gaussian_stiffness(self) -> torch.Tensor:
        """Get per-Gaussian stiffness."""
        if self.interpolator is None:
            N = self.gaussians.get_xyz.shape[0]
            return torch.zeros(N, 1, device=self.device)
        return self.interpolator(self.spring_Y, log_space=True, normalize=True, detach=True)

    def render_baseline(self, cam, pipe, bg):
        """Render without modulation."""
        return render(cam, self.gaussians, pipe, bg)

    def render_no_stiffness(self, cam, pipe, bg):
        """Render with modulator but zero stiffness."""
        if self.modulator is None:
            return self.render_baseline(cam, pipe, bg)

        sh_features = self.gaussians.get_features
        N = sh_features.shape[0]
        gauss_stiff = torch.zeros(N, 1, device=self.device)
        dc_color = self.gaussians.get_features_dc.squeeze(1)

        sh_out = self.modulator(gauss_stiff, sh_features, dc_color)
        return render(cam, self.gaussians, pipe, bg, sh_features=sh_out)

    def render_with_stiffness(self, cam, pipe, bg):
        """Render with full stiffness modulation."""
        if self.modulator is None:
            return self.render_baseline(cam, pipe, bg)

        sh_features = self.gaussians.get_features
        gauss_stiff = self.get_gaussian_stiffness()
        dc_color = self.gaussians.get_features_dc.squeeze(1)

        sh_out = self.modulator(gauss_stiff, sh_features, dc_color)
        return render(cam, self.gaussians, pipe, bg, sh_features=sh_out)


def run_ablation(
    dataset,
    pipe,
    modulator_ckpt: Optional[str],
    physics_path: Optional[str],
    output_dir: str,
    num_views: int = 10,
):
    """Run ablation study and save results."""

    print(f"\n{'='*60}")
    print("Stiffness Modulation Ablation Study")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    renders_dir = os.path.join(output_dir, 'renders')
    os.makedirs(renders_dir, exist_ok=True)

    # Load Gaussian model
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    # Load modulator if checkpoint provided
    modulator = None
    if modulator_ckpt and os.path.exists(modulator_ckpt):
        print(f"Loading modulator from {modulator_ckpt}")
        ckpt = torch.load(modulator_ckpt, map_location='cuda')

        # Determine variant from checkpoint or use default
        modulator = create_modulator(
            sh_degree=dataset.sh_degree,
            variant='standard',
            use_dc_context=True,
        ).cuda()
        modulator.load_state_dict(ckpt['modulator_state_dict'])
        modulator.eval()
        print(f"  Loaded from iteration {ckpt.get('iteration', 'unknown')}")
    else:
        print("[WARNING] No modulator checkpoint provided - baseline only")

    # Load or create physics data
    physics_data = load_physics_data(physics_path)
    if physics_data is None:
        print("[INFO] Creating synthetic physics data")
        physics_data = create_synthetic_physics(gaussians)

    # Print stiffness stats
    stiff_vals = torch.exp(physics_data['spring_Y'])
    print(f"\nPhysics data:")
    print(f"  Springs: {physics_data['springs'].shape[0]}")
    print(f"  Stiffness range: [{stiff_vals.min():.2e}, {stiff_vals.max():.2e}]")

    # Create renderer
    ablation_renderer = AblationRenderer(gaussians, modulator, physics_data)

    # Verify stiffness variation
    gauss_stiff = ablation_renderer.get_gaussian_stiffness()
    stiff_min, stiff_max = gauss_stiff.min().item(), gauss_stiff.max().item()
    print(f"\nGaussian stiffness (normalized):")
    print(f"  Range: [{stiff_min:.4f}, {stiff_max:.4f}]")
    print(f"  Mean: {gauss_stiff.mean().item():.4f}")
    print(f"  Std: {gauss_stiff.std().item():.4f}")

    if stiff_min == stiff_max:
        print("[WARNING] Stiffness has no variation (min == max)!")

    # Background
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Get test cameras
    test_cameras = scene.getTestCameras()
    if not test_cameras:
        test_cameras = scene.getTrainCameras()
    test_cameras = test_cameras[:num_views]

    print(f"\nEvaluating on {len(test_cameras)} views...")

    # Metrics storage
    metrics = {
        'baseline': {'psnr': [], 'ssim': [], 'l1': []},
        'no_stiffness': {'psnr': [], 'ssim': [], 'l1': []},
        'with_stiffness': {'psnr': [], 'ssim': [], 'l1': []},
    }

    # CSV for per-image metrics
    csv_rows = []

    for idx, cam in enumerate(tqdm(test_cameras, desc="Rendering")):
        with torch.no_grad():
            gt_image = cam.original_image.cuda()
            if cam.alpha_mask is not None:
                gt_image = gt_image * cam.alpha_mask.cuda()

            # Render all modes
            baseline_pkg = ablation_renderer.render_baseline(cam, pipe, background)
            no_stiff_pkg = ablation_renderer.render_no_stiffness(cam, pipe, background)
            with_stiff_pkg = ablation_renderer.render_with_stiffness(cam, pipe, background)

            renders = {
                'baseline': baseline_pkg['render'][:3],
                'no_stiffness': no_stiff_pkg['render'][:3],
                'with_stiffness': with_stiff_pkg['render'][:3],
            }

            # Compute metrics
            row = {'image': cam.image_name if hasattr(cam, 'image_name') else f'view_{idx}'}

            for mode_name, img in renders.items():
                psnr_val = psnr(img, gt_image).mean().item()
                ssim_val = ssim(img, gt_image).item()
                l1_val = l1_loss(img, gt_image).item()

                metrics[mode_name]['psnr'].append(psnr_val)
                metrics[mode_name]['ssim'].append(ssim_val)
                metrics[mode_name]['l1'].append(l1_val)

                row[f'{mode_name}_psnr'] = psnr_val
                row[f'{mode_name}_ssim'] = ssim_val
                row[f'{mode_name}_l1'] = l1_val

            csv_rows.append(row)

            # Save side-by-side render
            gt_np = tensor_to_image(gt_image)
            baseline_np = tensor_to_image(renders['baseline'])
            no_stiff_np = tensor_to_image(renders['no_stiffness'])
            with_stiff_np = tensor_to_image(renders['with_stiffness'])

            comparison = create_side_by_side(gt_np, baseline_np, no_stiff_np, with_stiff_np)
            comp_path = os.path.join(renders_dir, f'comparison_{idx:04d}.png')
            Image.fromarray(comparison).save(comp_path)

            # Also save individual renders
            Image.fromarray(baseline_np).save(os.path.join(renders_dir, f'baseline_{idx:04d}.png'))
            Image.fromarray(no_stiff_np).save(os.path.join(renders_dir, f'no_stiff_{idx:04d}.png'))
            Image.fromarray(with_stiff_np).save(os.path.join(renders_dir, f'with_stiff_{idx:04d}.png'))

    # Compute averages
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    results = {}
    for mode_name in ['baseline', 'no_stiffness', 'with_stiffness']:
        avg_psnr = np.mean(metrics[mode_name]['psnr'])
        avg_ssim = np.mean(metrics[mode_name]['ssim'])
        avg_l1 = np.mean(metrics[mode_name]['l1'])

        results[mode_name] = {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'l1': avg_l1,
        }

        print(f"\n{mode_name.upper()}:")
        print(f"  PSNR: {avg_psnr:.4f}")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"  L1:   {avg_l1:.6f}")

    # Print improvements
    print("\n" + "-"*40)
    print("IMPROVEMENTS vs BASELINE:")
    baseline_psnr = results['baseline']['psnr']
    baseline_ssim = results['baseline']['ssim']

    for mode_name in ['no_stiffness', 'with_stiffness']:
        psnr_diff = results[mode_name]['psnr'] - baseline_psnr
        ssim_diff = results[mode_name]['ssim'] - baseline_ssim
        print(f"  {mode_name}: PSNR {psnr_diff:+.4f}, SSIM {ssim_diff:+.4f}")

    # Check acceptance criteria
    print("\n" + "-"*40)
    print("ACCEPTANCE CHECKS:")

    # Check 1: Stiffness has variation
    check1 = stiff_min != stiff_max
    print(f"  [{'PASS' if check1 else 'FAIL'}] Stiffness has variation (min != max)")

    # Check 2: With stiffness beats no stiffness
    check2 = results['with_stiffness']['psnr'] > results['no_stiffness']['psnr']
    print(f"  [{'PASS' if check2 else 'FAIL'}] With stiffness > No stiffness (PSNR)")

    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save CSV
    csv_path = os.path.join(output_dir, 'per_image_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[INFO] Results saved to {output_dir}")
    print(f"  - {results_path}")
    print(f"  - {csv_path}")
    print(f"  - {renders_dir}/")


if __name__ == "__main__":
    parser = ArgumentParser(description="Stiffness Modulation Ablation Study")

    # Standard GS args
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    # Ablation-specific args
    parser.add_argument('--modulator_ckpt', type=str, default=None,
                        help='Path to trained modulator checkpoint')
    parser.add_argument('--physics_path', type=str, default=None,
                        help='Path to physics data')
    parser.add_argument('--output_dir', type=str, default='./output/ablation',
                        help='Output directory for results')
    parser.add_argument('--num_views', type=int, default=10,
                        help='Number of test views to evaluate')
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    safe_state(args.quiet)

    run_ablation(
        dataset=lp.extract(args),
        pipe=pp.extract(args),
        modulator_ckpt=args.modulator_ckpt,
        physics_path=args.physics_path,
        output_dir=args.output_dir,
        num_views=args.num_views,
    )
