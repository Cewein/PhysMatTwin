#!/usr/bin/env python3
"""
Demo script for visualizing physics-stiffness-driven appearance using the neural appearance head.

This script demonstrates how stiffness values from the physics module can influence
the rendered appearance of Gaussians without any ray tracing.

Usage:
    # Basic demo with synthetic stiffness
    python demo_neural_appearance.py --model_path /path/to/trained/model

    # Demo with stiffness override (0.0 = soft, 1.0 = stiff)
    python demo_neural_appearance.py --model_path /path/to/trained/model --stiffness_override 0.2

    # Generate comparison images at different stiffness levels
    python demo_neural_appearance.py --model_path /path/to/trained/model --comparison_mode

    # Interactive demo with stiffness slider (requires display)
    python demo_neural_appearance.py --model_path /path/to/trained/model --interactive
"""

import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import imageio

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaussian_splatting.scene import Scene, GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_splatting.utils.general_utils import safe_state


def load_model(args, iteration=-1):
    """Load a trained Gaussian model with optional neural appearance head."""
    # Create Gaussian model with neural appearance head enabled
    gaussians = GaussianModel(args.sh_degree, use_neural_appearance_head=True)

    # Load the scene
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

    # Try to load the appearance head if it exists
    if iteration == -1:
        # Find the latest iteration
        point_cloud_path = os.path.join(args.model_path, "point_cloud")
        if os.path.exists(point_cloud_path):
            iterations = [int(d.split("_")[1]) for d in os.listdir(point_cloud_path) if d.startswith("iteration_")]
            if iterations:
                iteration = max(iterations)

    appearance_head_path = os.path.join(
        args.model_path, "point_cloud", f"iteration_{iteration}", "appearance_head.pth"
    )
    if os.path.exists(appearance_head_path):
        print(f"Loading appearance head from {appearance_head_path}")
        gaussians.load_appearance_head(appearance_head_path)
    else:
        print(f"No pre-trained appearance head found at {appearance_head_path}")
        print("Using randomly initialized appearance head for demo.")

    return gaussians, scene


def render_with_stiffness(gaussians, viewpoint_camera, pipe, bg_color, stiffness_value):
    """Render the scene with a specific stiffness value for all Gaussians."""
    render_pkg = render(
        viewpoint_camera,
        gaussians,
        pipe,
        bg_color,
        use_neural_appearance=True,
        stiffness_override=stiffness_value
    )
    return render_pkg["render"][:3]  # RGB only, no alpha


def generate_comparison_images(gaussians, scene, pipe, bg_color, output_dir, num_stiffness_levels=5):
    """
    Generate comparison images showing the effect of different stiffness levels.

    Creates a grid of images: rows = cameras, columns = stiffness levels
    """
    os.makedirs(output_dir, exist_ok=True)

    cameras = scene.getTestCameras()
    if len(cameras) == 0:
        cameras = scene.getTrainCameras()[:5]  # Use first 5 training cameras if no test cameras

    stiffness_levels = np.linspace(0.0, 1.0, num_stiffness_levels)

    print(f"Generating comparison images for {len(cameras)} cameras and {num_stiffness_levels} stiffness levels...")

    for cam_idx, viewpoint_cam in enumerate(tqdm(cameras, desc="Cameras")):
        images = []
        for stiff_idx, stiffness in enumerate(stiffness_levels):
            with torch.no_grad():
                image = render_with_stiffness(gaussians, viewpoint_cam, pipe, bg_color, stiffness)
                image = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                images.append(image)

        # Save individual images
        for stiff_idx, (image, stiffness) in enumerate(zip(images, stiffness_levels)):
            output_path = os.path.join(output_dir, f"cam_{cam_idx:03d}_stiff_{stiffness:.2f}.png")
            imageio.imwrite(output_path, (image * 255).astype(np.uint8))

        # Create a horizontal comparison strip
        comparison = np.concatenate(images, axis=1)
        comparison_path = os.path.join(output_dir, f"cam_{cam_idx:03d}_comparison.png")
        imageio.imwrite(comparison_path, (comparison * 255).astype(np.uint8))

    print(f"Comparison images saved to {output_dir}")


def generate_stiffness_animation(gaussians, scene, pipe, bg_color, output_path, num_frames=60):
    """
    Generate an animation showing stiffness changing from low to high.
    """
    cameras = scene.getTestCameras()
    if len(cameras) == 0:
        cameras = scene.getTrainCameras()

    # Use the first camera for the animation
    viewpoint_cam = cameras[0]

    # Generate stiffness values that go from low to high and back
    t = np.linspace(0, 2 * np.pi, num_frames)
    stiffness_values = 0.5 + 0.5 * np.sin(t)

    print(f"Generating stiffness animation with {num_frames} frames...")

    frames = []
    for stiffness in tqdm(stiffness_values, desc="Frames"):
        with torch.no_grad():
            image = render_with_stiffness(gaussians, viewpoint_cam, pipe, bg_color, stiffness)
            image = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            frames.append((image * 255).astype(np.uint8))

    imageio.mimsave(output_path, frames, fps=30)
    print(f"Animation saved to {output_path}")


def demo_single_render(gaussians, scene, pipe, bg_color, stiffness_override, output_path):
    """Render a single image with specified stiffness."""
    cameras = scene.getTestCameras()
    if len(cameras) == 0:
        cameras = scene.getTrainCameras()

    viewpoint_cam = cameras[0]

    print(f"Rendering with stiffness = {stiffness_override}")

    with torch.no_grad():
        image = render_with_stiffness(gaussians, viewpoint_cam, pipe, bg_color, stiffness_override)
        image = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    imageio.imwrite(output_path, (image * 255).astype(np.uint8))
    print(f"Image saved to {output_path}")


def interactive_demo(gaussians, scene, pipe, bg_color):
    """
    Interactive demo with a stiffness slider.
    Requires matplotlib with an interactive backend.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except ImportError:
        print("Interactive demo requires matplotlib. Install with: pip install matplotlib")
        return

    cameras = scene.getTestCameras()
    if len(cameras) == 0:
        cameras = scene.getTrainCameras()

    viewpoint_cam = cameras[0]

    # Initial render
    with torch.no_grad():
        initial_image = render_with_stiffness(gaussians, viewpoint_cam, pipe, bg_color, 0.5)
        initial_image = initial_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)

    img_plot = ax.imshow(initial_image)
    ax.set_title("Neural Appearance Demo - Stiffness: 0.50")
    ax.axis('off')

    # Create slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Stiffness', 0.0, 1.0, valinit=0.5)

    def update(val):
        stiffness = slider.val
        with torch.no_grad():
            image = render_with_stiffness(gaussians, viewpoint_cam, pipe, bg_color, stiffness)
            image = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        img_plot.set_data(image)
        ax.set_title(f"Neural Appearance Demo - Stiffness: {stiffness:.2f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


def main():
    parser = ArgumentParser(description="Demo for neural appearance head with stiffness control")

    # Model parameters
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    # Demo-specific arguments
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load (default: latest)")
    parser.add_argument("--stiffness_override", type=float, default=None,
                        help="Override stiffness for all Gaussians (0.0-1.0)")
    parser.add_argument("--comparison_mode", action="store_true",
                        help="Generate comparison images at different stiffness levels")
    parser.add_argument("--animation", action="store_true",
                        help="Generate animation of stiffness changing")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive demo with stiffness slider")
    parser.add_argument("--output_dir", type=str, default="demo_output",
                        help="Output directory for generated images/videos")
    parser.add_argument("--num_stiffness_levels", type=int, default=5,
                        help="Number of stiffness levels for comparison mode")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # Initialize
    safe_state(args.quiet)

    # Set up background color
    dataset = lp.extract(args)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    pipe = pp.extract(args)

    # Load model
    print("Loading model...")
    gaussians, scene = load_model(dataset, args.iteration)

    # Create output directory
    output_dir = os.path.join(args.model_path, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Run the appropriate demo mode
    if args.interactive:
        interactive_demo(gaussians, scene, pipe, background)
    elif args.comparison_mode:
        generate_comparison_images(
            gaussians, scene, pipe, background, output_dir,
            num_stiffness_levels=args.num_stiffness_levels
        )
    elif args.animation:
        output_path = os.path.join(output_dir, "stiffness_animation.gif")
        generate_stiffness_animation(gaussians, scene, pipe, background, output_path)
    else:
        # Single render mode
        stiffness = args.stiffness_override if args.stiffness_override is not None else 0.5
        output_path = os.path.join(output_dir, f"render_stiff_{stiffness:.2f}.png")
        demo_single_render(gaussians, scene, pipe, background, stiffness, output_path)


if __name__ == "__main__":
    main()
