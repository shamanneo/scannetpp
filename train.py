from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple, List, Dict, Union, Callable
import os
import sys


from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from scene.cameras import Camera, MiniCam
from gaussian_renderer import render
from dataset import ScannetppDataset
from utils.cuda_utils import GPUCacheLoader, move_to_device
from utils.general_utils import safe_state, get_expon_lr_func
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr, tensor2image
from utils.metrics import AverageMeter


try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False


def get_dataloader(
    root_dir: str,
    scene_id: str,
    preload_images: bool = True,
    preload_device: str = "cpu",
) -> Tuple[DataLoader, ScannetppDataset, ScannetppDataset]:
    train_dataset = ScannetppDataset(
        root_dir=root_dir,
        scene_id=scene_id,
        split="train",
        preload_images=preload_images,
        # max_images=max_images,
    )
    test_dataset = ScannetppDataset(
        root_dir=root_dir,
        scene_id=scene_id,
        split="test",
        preload_images=preload_images,
    )
    if preload_images and preload_device == "cuda":
        train_loader = GPUCacheLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            device="cuda",
            collate_fn=train_dataset.collate_fn,
            verbose=True,
        )
    else:
        num_workers = 0 if preload_images else 4
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
        )
    return train_loader, train_dataset, test_dataset


def data_to_camera(data: Dict[str, torch.Tensor]) -> MiniCam:
    return MiniCam(
        width=data["image_width"],
        height=data["image_height"],
        fovx=data["fovx"],
        fovy=data["fovy"],
        znear=data["znear"],
        zfar=data["zfar"],
        world_view_transform=data["world_view_transform"],
        full_proj_transform=data["full_proj_transform"],
        image_name=data["image_name"],
    )


def training(
    data_root: str,
    scene_id: str,
    output_path: str,
    model_params: ModelParams,
    opt_params: OptimizationParams,
    pipeline_params: PipelineParams,
    test_every: int,
    save_iterations: List[int],
    checkpoint_iterations,
):
    train_loader, train_dataset, test_dataset = get_dataloader(
        data_root,
        scene_id,
        # preload_images=True,
        preload_images=False,
        # preload_device="cpu",
        # preload_device="cuda",
        preload_device=model_params.data_device,
    )

    scene_output_dir = os.path.join(output_path, scene_id)
    writer = SummaryWriter(scene_output_dir)
    train_meter = AverageMeter()
    gaussians = GaussianModel(model_params.sh_degree, opt_params.optimizer_type)
    scene = Scene(model_params, train_dataset, gaussians)
    gaussians.training_setup(opt_params)

    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    use_sparse_adam = opt_params.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE

    progress_bar = tqdm(range(1, opt_params.iterations + 1), desc="Training progress")
    train_iter = iter(train_loader)
    for iteration in range(1, opt_params.iterations + 1):
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data = next(train_iter)

        data = data[0]
        data = move_to_device(data, "cuda")

        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        bg = torch.rand((3), device="cuda") if model_params.white_background else background

        viewpoint_cam = data_to_camera(data)
        render_pkg = render(viewpoint_cam, gaussians, pipeline_params, bg, use_trained_exp=model_params.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = data["image"]
        alpha_mask = data["mask"]
        if alpha_mask is not None:
            image = image * alpha_mask
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt_params.lambda_dssim) * Ll1 + opt_params.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        with torch.no_grad():
            psnr_value = psnr(image, gt_image).mean()
            train_meter.update({
                "l1_loss": Ll1.item(),
                "ssim_loss": ssim_value.item(),
                "total_loss": loss.item(),
                "psnr": psnr_value.item(),
            })

            if iteration % 10 == 0:
                # Update the progress bar
                train_metrics = train_meter.finalize()
                progress_bar.set_postfix({
                    "Loss": f"{train_metrics['total_loss']:.4f}",
                    "PSNR": f"{train_metrics['psnr']:.4f}",
                })
                progress_bar.update(10)

            if iteration % 100 == 0:
                # Write to TensorBoard
                train_metrics = train_meter.finalize()
                writer.add_scalar("Train/Loss", train_metrics["total_loss"], iteration)
                writer.add_scalar("Train/PSNR", train_metrics["psnr"], iteration)
                writer.add_scalar("Train/SSIM_Loss", train_metrics["ssim_loss"], iteration)
                writer.add_scalar("Train/L1_Loss", train_metrics["l1_loss"], iteration)

                num_gs = gaussians.get_xyz.shape[0]
                writer.add_scalar("Train/Num_GS", num_gs, iteration)
                train_meter.reset()

            if iteration == opt_params.iterations:
                progress_bar.close()

            if iteration % test_every == 0 and not test_dataset.is_testing_scene:
                evaluate(
                    writer,
                    iteration,
                    test_dataset,
                    render,
                    gaussians,
                    (pipeline_params, background, 1., SPARSE_ADAM_AVAILABLE, None, False),
                    scene_output_dir,
                )

            if iteration in save_iterations:
                # print("\n[ITER {}] Saving Gaussians".format(iteration))
                print(f"\n[ITER {iteration}] Saving Gaussians to {scene_output_dir}")
                scene.save(iteration, scene_output_dir)

            # Densification
            if iteration < opt_params.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt_params.densify_from_iter and iteration % opt_params.densification_interval == 0:
                    size_threshold = 20 if iteration > opt_params.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt_params.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt_params.opacity_reset_interval == 0 or (model_params.white_background and iteration == opt_params.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt_params.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    os.path.join(scene_output_dir, "chkpnt" + str(iteration) + ".pth"),
                )

    with torch.no_grad():
        prepare_submission(
            test_dataset,
            render,
            gaussians,
            (pipeline_params, background, 1., SPARSE_ADAM_AVAILABLE, None, False),
            os.path.join(output_path, "submission", scene_id),
        )


def evaluate(
    writer: SummaryWriter,
    iteration: int,
    test_dataset: ScannetppDataset,
    render_func: Callable,
    gaussians: GaussianModel,
    render_params: Tuple,
    output_path: str,
):
    test_meter = AverageMeter()
    render_start = torch.cuda.Event(enable_timing=True)
    render_end = torch.cuda.Event(enable_timing=True)

    for i, data in enumerate(test_dataset):
        data = move_to_device(data, "cuda")
        gt_image = data["image"]
        alpha_mask = data["mask"]
        image_name = data["image_name"]

        viewpoint_cam = data_to_camera(data)

        # Measure the time taken for rendering
        render_start.record()

        render_pkg = render_func(viewpoint_cam, gaussians, *render_params)
        image = render_pkg["render"]
        image = torch.clamp(image, 0, 1)

        # End the timing
        render_end.record()
        torch.cuda.synchronize()
        render_time = render_start.elapsed_time(render_end)

        if alpha_mask is not None:
            Ll1 = l1_loss(image * alpha_mask, gt_image * alpha_mask)
            psnr_value = psnr(image * alpha_mask, gt_image * alpha_mask).mean()
        else:
            Ll1 = l1_loss(image, gt_image)
            psnr_value = psnr(image, gt_image).mean()

        # Save the testing image
        image = tensor2image(image, normalized=True)
        gt_image = tensor2image(gt_image, normalized=True)
        image_cat = np.concatenate((image, gt_image), axis=1)
        save_dir = os.path.join(output_path, "test")
        os.makedirs(save_dir, exist_ok=True)
        Image.fromarray(image_cat).save(os.path.join(save_dir, image_name))

        test_meter.update({
            "l1_loss": Ll1.item(),
            "psnr": psnr_value.item(),
            "render_time": render_time,
            "fps": 1000.0 / render_time,
        })

    metrics = test_meter.finalize()
    writer.add_scalar("Test/Render_Time", metrics["render_time"], iteration)
    writer.add_scalar("Test/FPS", metrics["fps"], iteration)
    writer.add_scalar("Test/Loss", metrics["l1_loss"], iteration)
    writer.add_scalar("Test/PSNR", metrics["psnr"], iteration)
    print(f"Test iteration {iteration}: PSNR: {metrics['psnr']:.4f}, FPS: {metrics['fps']:.2f} FPS, Render Time: {metrics['render_time']:.2f} ms")


def prepare_submission(
    test_dataset: ScannetppDataset,
    render_func: Callable,
    gaussians: GaussianModel,
    render_params: Tuple,
    output_path: str,
):
    """Render the test images and save them into a submission folder according to the official submission structure.
    The submission folder should be structured as follows:

    output_root
    |-- submission
        |-- scene_id1
            |-- DSC01752.JPG
            |-- DSC01753.JPG
                ⋮
        |-- scene_id2
            |-- DSC00299.JPG
            |-- DSC00143.JPG
                ⋮
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    for i, data in enumerate(test_dataset):
        data = move_to_device(data, "cuda")
        image_name = data["image_name"]
        viewpoint_cam = data_to_camera(data)
        render_pkg = render_func(viewpoint_cam, gaussians, *render_params)
        image = render_pkg["render"]
        image = torch.clamp(image, 0, 1)
        image = tensor2image(image, normalized=True)

        Image.fromarray(image).save(os.path.join(output_path, image_name))
        print(f"Saved {image_name} to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_every", type=int, default=1000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--scene_id", type=str, required=True)

    args = parser.parse_args()
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        args.data_root,
        args.scene_id,
        args.output_root,
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        # args.test_iterations,
        args.test_every,
        args.save_iterations,
        args.checkpoint_iterations,
    )
