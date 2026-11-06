import os
import numpy as np
import torch as th
import torchvision


def convert2rgb(img, bound=1.0):
    """
    Convert an image tensor from `[-bound, +bound]` into `[0, 255]`.

    Args:
        img (th.Tensor): Input image tensor.
        bound (float): Bounding value (e.g., 1.0 or 0.5).
    """
    if bound == 1.0:
        return (img + 1.0) * 127.5
    elif bound == 0.5:
        return (img + 0.5) * 255.0
    raise ValueError(f"Unsupported bound={bound}; expected 1.0 or 0.5.")


def save_video(fn, frames, fps=30):
    """
    Save video frames to `fn`.

    Args:
        fn (str): Full output file path (e.g. `.../vid/res_video.mp4`).
        frames (th.Tensor): uint8 tensor of shape `(T, H, W, C)` in `[0, 255]`.
        fps (int): Video frame rate.
    """
    os.makedirs(os.path.dirname(os.path.abspath(fn)), exist_ok=True)
    if frames.is_cuda:
        frames = frames.cpu()
    torchvision.io.write_video(video_array=frames, filename=fn, fps=fps)


def save_images(path, fn, frames):
    """
    Save individual frames as `{path}/{fn}_frame{i}.png`.

    Args:
        path (str): Output directory.
        fn (str): Filename prefix (e.g. `res`).
        frames (th.Tensor): Float tensor of shape `(T, C, H, W)` in `[0, 1]`.
    """
    os.makedirs(path, exist_ok=True)
    for i in range(frames.shape[0]):
        frame = frames[i].cpu().detach()
        torchvision.utils.save_image(tensor=frame, fp=f"{path}/{fn}_frame{i}.png")


def save_facerender_images(path, fn, frames):
    """Save individual frames as `.npy` arrays of shape `(H, W, C)`."""
    os.makedirs(path, exist_ok=True)
    for i in range(frames.shape[0]):
        frame = frames[i].detach().cpu().numpy().transpose(1, 2, 0)
        np.save(f"{path}/{fn}_frame{i}.npy", frame)


def save_images_with_fn(path, fn, frames, fn_list):
    """Save individual frames named after `fn_list` stems."""
    os.makedirs(path, exist_ok=True)
    for i in range(frames.shape[0]):
        frame = frames[i].cpu().detach()
        frame_idx = os.path.splitext(os.path.basename(fn_list[i]))[0]
        torchvision.utils.save_image(tensor=frame, fp=f"{path}/{fn}_{frame_idx}.png")
