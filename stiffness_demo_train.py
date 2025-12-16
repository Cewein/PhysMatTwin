"""
Stiffness-to-Appearance Demo Training Script.

Trains a FiLM-style SH modulator that uses physics stiffness to influence
Gaussian Splatting appearance, enabling physics-aware rendering.

Usage:
    python stiffness_demo_train.py -s <source_path> -m <model_path> --physics_path <physics_ckpt>

Ablation modes:
    --ablation baseline      : No modulator (baseline rendering)
    --ablation no_stiffness  : Modulator with stiffness=0
    --ablation with_stiffness: Full modulator with stiffness (default)
"""

import os
import sys
import json
import torch
import torch.nn as nn
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import Scene, GaussianModel
from gaussian_splatting.utils.general_utils import safe_state
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams

from qqtt.model.sh_stiffness_modulator import SHStiffnessModulator, create_modulator
from qqtt.model.stiffness_utils import (
    StiffnessInterpolator,
    get_gaussian_stiffness,
    spring_to_particle_stiffness,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False


def load_physics_data(physics_path: str, device: str = 'cuda'):
    """
    Load physics simulation data (spring stiffness, connectivity, particle positions).

    Args:
        physics_path: Path to physics checkpoint or data directory
        device: Target device

    Returns:
        Dictionary with spring_Y, springs, particle_positions, etc.
    """
    if os.path.isfile(physics_path):
        # Load from checkpoint
        ckpt = torch.load(physics_path, map_location=device)
        physics_data = {
            'spring_Y': ckpt.get('spring_Y', ckpt.get('wp_spring_Y')),
            'springs': ckpt.get('springs', ckpt.get('init_springs')),
            'particle_positions': ckpt.get('vertices', ckpt.get('init_vertices')),
        }
    elif os.path.isdir(physics_path):
        # Load from directory with individual files
        physics_data = {}
        for name in ['spring_Y', 'springs', 'particle_positions']:
            fpath = os.path.join(physics_path, f'{name}.pt')
            if os.path.exists(fpath):
                physics_data[name] = torch.load(fpath, map_location=device)

    # Ensure tensors are on device
    for k, v in physics_data.items():
        if isinstance(v, torch.Tensor):
            physics_data[k] = v.to(device)

    return physics_data


def create_synthetic_physics_data(gaussians: GaussianModel, device: str = 'cuda'):
    """
    Create synthetic physics data for demo purposes when no real physics data available.

    Creates a simple spring-mass system based on Gaussian positions using KNN connectivity.

    Args:
        gaussians: GaussianModel with positions
        device: Target device

    Returns:
        Dictionary with synthetic spring_Y, springs, particle_positions
    """
    print("[INFO] Creating synthetic physics data for demo...")

    positions = gaussians.get_xyz.detach()  # (N, 3)
    N = positions.shape[0]

    # Use a subset of Gaussians as particles (for efficiency)
    max_particles = min(N, 5000)
    if N > max_particles:
        indices = torch.randperm(N)[:max_particles]
        particle_positions = positions[indices]
    else:
        indices = torch.arange(N)
        particle_positions = positions

    P = particle_positions.shape[0]

    # Create springs using KNN connectivity (k nearest neighbors)
    k = min(6, P - 1)

    # Compute pairwise distances
    diff = particle_positions.unsqueeze(1) - particle_positions.unsqueeze(0)  # (P, P, 3)
    dist_sq = (diff ** 2).sum(-1)  # (P, P)

    # Get k nearest neighbors (excluding self)
    dist_sq.fill_diagonal_(float('inf'))
    _, knn_indices = torch.topk(dist_sq, k, dim=1, largest=False)  # (P, k)

    # Create spring connectivity
    springs_list = []
    for i in range(P):
        for j in range(k):
            neighbor = knn_indices[i, j].item()
            if i < neighbor:  # Avoid duplicates
                springs_list.append([i, neighbor])

    springs = torch.tensor(springs_list, dtype=torch.long, device=device)
    M = springs.shape[0]

    # Create stiffness values with spatial variation
    # Higher stiffness in the center, lower at boundaries
    center = particle_positions.mean(dim=0)
    dist_from_center = (particle_positions - center).norm(dim=1)
    max_dist = dist_from_center.max() + 1e-6

    # Normalized distance [0, 1]
    norm_dist = dist_from_center / max_dist

    # Stiffness varies from 1e4 (center) to 1e3 (boundary)
    particle_stiffness = 1e4 * (1.0 - 0.9 * norm_dist)

    # Convert to per-spring stiffness (average of endpoints)
    spring_Y = torch.zeros(M, device=device)
    for idx, (i, j) in enumerate(springs):
        spring_Y[idx] = 0.5 * (particle_stiffness[i] + particle_stiffness[j])

    # Store in log space (as in physics simulator)
    spring_Y = torch.log(spring_Y)

    return {
        'spring_Y': spring_Y,
        'springs': springs,
        'particle_positions': particle_positions,
        'particle_indices': indices,  # For mapping back to Gaussians
    }


class StiffnessModulatedRenderer:
    """
    Wrapper that integrates stiffness interpolation and SH modulation with rendering.
    """

    def __init__(
        self,
        gaussians: GaussianModel,
        physics_data: dict,
        modulator: nn.Module,
        device: str = 'cuda',
    ):
        self.gaussians = gaussians
        self.modulator = modulator
        self.device = device

        # Extract physics data
        self.spring_Y = physics_data['spring_Y']
        self.springs = physics_data['springs']
        self.particle_positions = physics_data['particle_positions']

        # Create stiffness interpolator
        self._setup_interpolator()

    def _setup_interpolator(self):
        """Initialize the stiffness interpolator."""
        gaussian_positions = self.gaussians.get_xyz.detach()

        self.interpolator = StiffnessInterpolator(
            particle_positions=self.particle_positions,
            springs=self.springs,
            gaussian_positions=gaussian_positions,
            k=4,
            weight_type='inverse_distance',
            normalize_method='minmax',
        )

    def get_gaussian_stiffness(self, detach: bool = True):
        """Get per-Gaussian stiffness values."""
        return self.interpolator(
            self.spring_Y,
            log_space=True,
            normalize=True,
            detach=detach
        )

    def get_modulated_sh(self, zero_stiffness: bool = False):
        """
        Compute modulated SH coefficients.

        Args:
            zero_stiffness: If True, use zero stiffness (ablation)

        Returns:
            Modulated SH features (N, K, 3)
        """
        # Get SH features from Gaussians
        sh_features = self.gaussians.get_features  # (N, K, 3)

        # Get stiffness
        if zero_stiffness:
            gauss_stiff = torch.zeros(sh_features.shape[0], 1, device=self.device)
        else:
            gauss_stiff = self.get_gaussian_stiffness(detach=True)

        # Get DC color for context
        dc_color = self.gaussians.get_features_dc.squeeze(1)  # (N, 3)

        # Apply modulation
        sh_out = self.modulator(gauss_stiff, sh_features, dc_color)

        return sh_out

    def update_positions(self):
        """Recompute interpolator if Gaussian positions changed significantly."""
        self._setup_interpolator()


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint,
    debug_from,
    physics_path,
    ablation_mode,
    modulator_lr,
    modulator_variant,
    num_iterations,
):
    """Main training loop for stiffness-modulated rendering."""

    print(f"\n{'='*60}")
    print(f"Stiffness Modulation Training")
    print(f"Ablation mode: {ablation_mode}")
    print(f"{'='*60}\n")

    # Setup output
    tb_writer = prepare_output_and_logger(dataset)

    # Load Gaussian model
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    # Load checkpoint if provided
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        first_iter = 0  # Reset for modulator training
    else:
        # Setup training for Gaussians
        gaussians.training_setup(opt)
        first_iter = 0

    # Load or create physics data
    if physics_path and os.path.exists(physics_path):
        physics_data = load_physics_data(physics_path)
        print(f"[INFO] Loaded physics data from {physics_path}")
    else:
        physics_data = create_synthetic_physics_data(gaussians)
        print(f"[INFO] Using synthetic physics data")

    # Print physics data stats
    print(f"  Springs: {physics_data['springs'].shape[0]}")
    print(f"  Particles: {physics_data['particle_positions'].shape[0]}")
    stiff_vals = torch.exp(physics_data['spring_Y'])
    print(f"  Stiffness range: [{stiff_vals.min():.2e}, {stiff_vals.max():.2e}]")

    # Create modulator
    if ablation_mode == 'baseline':
        modulator = None
        print("[INFO] Baseline mode - no modulator")
    else:
        modulator = create_modulator(
            sh_degree=dataset.sh_degree,
            variant=modulator_variant,
            use_dc_context=True,
            init_scale=0.05,
        ).cuda()
        print(f"[INFO] Created modulator: {modulator_variant}")
        print(f"  Parameters: {sum(p.numel() for p in modulator.parameters())}")

    # Create renderer wrapper (if using modulator)
    if modulator is not None:
        renderer = StiffnessModulatedRenderer(
            gaussians=gaussians,
            physics_data=physics_data,
            modulator=modulator,
        )

        # Optimizer for modulator only
        modulator_optimizer = torch.optim.Adam(
            modulator.parameters(),
            lr=modulator_lr,
            betas=(0.9, 0.999),
        )
    else:
        renderer = None
        modulator_optimizer = None

    # Background
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Training cameras
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # Metrics tracking
    ema_loss = 0.0
    best_psnr = 0.0

    # Training loop
    progress_bar = tqdm(range(first_iter, num_iterations), desc="Training")

    for iteration in range(first_iter, num_iterations):
        # Pick random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        # Background
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Render
        if ablation_mode == 'baseline':
            # Baseline: no modulation
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            sh_features = None
        elif ablation_mode == 'no_stiffness':
            # Ablation: modulator with zero stiffness
            sh_features = renderer.get_modulated_sh(zero_stiffness=True)
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, sh_features=sh_features)
        else:  # with_stiffness
            # Full: modulator with actual stiffness
            sh_features = renderer.get_modulated_sh(zero_stiffness=False)
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, sh_features=sh_features)

        # Extract render outputs
        image = render_pkg["render"][:3, ...]
        gt_image = viewpoint_cam.original_image.cuda()

        # Handle masks
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            gt_image = gt_image * alpha_mask

        # Compute loss
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Backward
        loss.backward()

        # Optimizer step (modulator only)
        if modulator_optimizer is not None:
            modulator_optimizer.step()
            modulator_optimizer.zero_grad()

        # Logging
        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            current_psnr = psnr(image, gt_image).mean().item()

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss:.5f}",
                    "PSNR": f"{current_psnr:.2f}",
                })
                progress_bar.update(10)

            if tb_writer:
                tb_writer.add_scalar('train/loss', loss.item(), iteration)
                tb_writer.add_scalar('train/psnr', current_psnr, iteration)
                tb_writer.add_scalar('train/l1', Ll1.item(), iteration)

            # Save best model
            if current_psnr > best_psnr and modulator is not None:
                best_psnr = current_psnr
                save_path = os.path.join(dataset.model_path, 'modulator_best.pth')
                torch.save({
                    'modulator_state_dict': modulator.state_dict(),
                    'iteration': iteration,
                    'psnr': best_psnr,
                    'ablation_mode': ablation_mode,
                }, save_path)

            # Periodic evaluation
            if iteration in testing_iterations:
                evaluate_model(
                    scene, gaussians, renderer, pipe, background,
                    ablation_mode, tb_writer, iteration
                )

            # Save checkpoint
            if iteration in saving_iterations and modulator is not None:
                save_path = os.path.join(dataset.model_path, f'modulator_{iteration}.pth')
                torch.save({
                    'modulator_state_dict': modulator.state_dict(),
                    'iteration': iteration,
                    'ablation_mode': ablation_mode,
                }, save_path)

    progress_bar.close()

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    evaluate_model(
        scene, gaussians, renderer, pipe, background,
        ablation_mode, tb_writer, num_iterations, verbose=True
    )

    # Save final model
    if modulator is not None:
        save_path = os.path.join(dataset.model_path, 'modulator_final.pth')
        torch.save({
            'modulator_state_dict': modulator.state_dict(),
            'iteration': num_iterations,
            'ablation_mode': ablation_mode,
        }, save_path)
        print(f"[INFO] Saved final modulator to {save_path}")

    print("\nTraining complete.")


def evaluate_model(
    scene,
    gaussians,
    renderer,
    pipe,
    background,
    ablation_mode,
    tb_writer,
    iteration,
    verbose=False,
):
    """Evaluate model on test cameras."""
    test_cameras = scene.getTestCameras()
    if not test_cameras:
        test_cameras = scene.getTrainCameras()[:5]

    psnr_values = []
    ssim_values = []
    l1_values = []

    for cam in test_cameras:
        with torch.no_grad():
            # Render
            if ablation_mode == 'baseline' or renderer is None:
                render_pkg = render(cam, gaussians, pipe, background)
            elif ablation_mode == 'no_stiffness':
                sh_features = renderer.get_modulated_sh(zero_stiffness=True)
                render_pkg = render(cam, gaussians, pipe, background, sh_features=sh_features)
            else:
                sh_features = renderer.get_modulated_sh(zero_stiffness=False)
                render_pkg = render(cam, gaussians, pipe, background, sh_features=sh_features)

            image = render_pkg["render"][:3, ...]
            gt_image = cam.original_image.cuda()

            if cam.alpha_mask is not None:
                gt_image = gt_image * cam.alpha_mask.cuda()

            # Metrics
            psnr_val = psnr(image, gt_image).mean().item()
            ssim_val = ssim(image, gt_image).item()
            l1_val = l1_loss(image, gt_image).item()

            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            l1_values.append(l1_val)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_l1 = np.mean(l1_values)

    if verbose:
        print(f"  PSNR: {avg_psnr:.4f}")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"  L1: {avg_l1:.6f}")

    if tb_writer:
        tb_writer.add_scalar('eval/psnr', avg_psnr, iteration)
        tb_writer.add_scalar('eval/ssim', avg_ssim, iteration)
        tb_writer.add_scalar('eval/l1', avg_l1, iteration)

    return {'psnr': avg_psnr, 'ssim': avg_ssim, 'l1': avg_l1}


def prepare_output_and_logger(args):
    """Setup output directory and tensorboard."""
    if not args.model_path:
        import uuid
        args.model_path = os.path.join("./output/stiffness_demo/", str(uuid.uuid4())[:8])

    os.makedirs(args.model_path, exist_ok=True)
    print(f"Output folder: {args.model_path}")

    # Save config
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)

    return tb_writer


if __name__ == "__main__":
    parser = ArgumentParser(description="Stiffness Modulation Training")

    # Standard GS args
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # Stiffness-specific args
    parser.add_argument('--physics_path', type=str, default=None,
                        help='Path to physics checkpoint or data directory')
    parser.add_argument('--ablation', type=str, default='with_stiffness',
                        choices=['baseline', 'no_stiffness', 'with_stiffness'],
                        help='Ablation mode')
    parser.add_argument('--modulator_lr', type=float, default=1e-3,
                        help='Learning rate for modulator')
    parser.add_argument('--modulator_variant', type=str, default='standard',
                        choices=['standard', 'light', 'per_coeff'],
                        help='Modulator architecture variant')
    parser.add_argument('--num_iterations', type=int, default=5000,
                        help='Number of training iterations')

    # Standard args
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 3000, 5000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)

    args = parser.parse_args()

    safe_state(args.quiet)

    training(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint=args.start_checkpoint,
        debug_from=args.debug_from,
        physics_path=args.physics_path,
        ablation_mode=args.ablation,
        modulator_lr=args.modulator_lr,
        modulator_variant=args.modulator_variant,
        num_iterations=args.num_iterations,
    )
