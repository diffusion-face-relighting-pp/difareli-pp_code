import argparse
import glob
import os
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tqdm


def face_segment(segment_part, img):
    if isinstance(img, Image.Image):
        face_segment_anno = np.array(img)
    else:
        face_segment_anno = img

    if face_segment_anno.ndim == 3:
        face_segment_anno = face_segment_anno[..., 0]

    bg = (face_segment_anno == 0)
    skin = (face_segment_anno == 1)
    l_brow = (face_segment_anno == 2)
    r_brow = (face_segment_anno == 3)
    l_eye = (face_segment_anno == 4)
    r_eye = (face_segment_anno == 5)
    eye_g = (face_segment_anno == 6)
    l_ear = (face_segment_anno == 7)
    r_ear = (face_segment_anno == 8)
    ear_r = (face_segment_anno == 9)
    nose = (face_segment_anno == 10)
    mouth = (face_segment_anno == 11)
    u_lip = (face_segment_anno == 12)
    l_lip = (face_segment_anno == 13)
    neck = (face_segment_anno == 14)
    neck_l = (face_segment_anno == 15)
    cloth = (face_segment_anno == 16)
    hair = (face_segment_anno == 17)
    hat = (face_segment_anno == 18)
    face = np.logical_or.reduce(
        (skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip)
    )

    if segment_part == 'faceseg_face':
        seg_m = face
    elif segment_part == 'faceseg_head':
        seg_m = (face | neck | hair)
    elif segment_part == 'faceseg_nohead':
        seg_m = ~(face | neck | hair)
    elif segment_part == 'faceseg_face&hair':
        seg_m = ~bg
    elif segment_part == 'faceseg_bg_noface&nohair':
        seg_m = (bg | hat | neck | neck_l | cloth)
    elif segment_part == 'faceseg_bg&ears_noface&nohair':
        seg_m = (bg | hat | neck | neck_l | cloth) | (l_ear | r_ear | ear_r)
    elif segment_part == 'faceseg_bg':
        seg_m = bg
    elif segment_part == 'faceseg_bg&noface':
        seg_m = (bg | hair | hat | neck | neck_l | cloth)
    elif segment_part == 'faceseg_hair':
        seg_m = hair
    elif segment_part == 'faceseg_faceskin':
        seg_m = skin
    elif segment_part == 'faceseg_faceskin&nose':
        seg_m = (skin | nose)
    elif segment_part == 'faceseg_eyes&glasses&mouth&eyebrows':
        seg_m = (l_eye | r_eye | eye_g | l_brow | r_brow | mouth)
    elif segment_part == 'faceseg_faceskin&nose&mouth&eyebrows':
        seg_m = (skin | nose | mouth | u_lip | l_lip | l_brow | r_brow | l_eye | r_eye)
    elif segment_part == 'faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses':
        seg_m = (skin | nose | mouth | u_lip | l_lip | l_brow | r_brow | l_eye | r_eye | eye_g)
    elif segment_part == 'faceseg_face_noglasses':
        seg_m = (~eye_g & face)
    elif segment_part == 'faceseg_face_noglasses_noeyes':
        seg_m = (~(l_eye | r_eye) & ~eye_g & face)
    elif segment_part == 'faceseg_eyes&glasses':
        seg_m = (l_eye | r_eye | eye_g)
    elif segment_part == 'glasses':
        seg_m = eye_g
    elif segment_part == 'faceseg_eyes':
        seg_m = (l_eye | r_eye)
    elif segment_part in ['sobel_bg_mask', 'laplacian_bg_mask', 'sobel_bin_bg_mask']:
        seg_m = ~(face | neck | hair)
    elif segment_part in ['canny_edge_bg_mask']:
        seg_m = ~(face | neck | hair) | (l_ear | r_ear)
    else:
        raise NotImplementedError(f"Segment part: {segment_part} is not found!")

    return seg_m


def get_shadow_diff(img1, img2, c_type='L', signed=False):
    img1 = np.array(img1.convert(c_type)) / 255.0
    img2 = np.array(img2.convert(c_type)) / 255.0
    if signed:
        shadow_diff = img2 - img1
    else:
        shadow_diff = np.abs(img2 - img1)
    return shadow_diff


def create_image_grid(images, n_rows=1):
    """Creates a grid of images from a list of NumPy arrays or PIL Images."""
    images = [np.asarray(img) for img in images]
    n_images = len(images)
    n_cols = (n_images + n_rows - 1) // n_rows

    images = images + [np.zeros_like(images[0]) for _ in range(n_rows * n_cols - n_images)]

    rows = []
    for i in range(n_rows):
        row_images = images[i * n_cols:(i + 1) * n_cols]
        rows.append(np.concatenate(row_images, axis=1))

    grid_image = np.concatenate(rows, axis=0)
    return grid_image


def resolve_reshadow_frames_dir(base_dir, mode_key, ckpt, set_name, n_frames, model_filter='steps=250'):
    """
    Resolves a mapping {img_stem: frames_dir} for either:
      1) Simplified dataset structure:
         - {base_dir}/{shadow|diffuse}/{ckpt}/{img_stem}/res_frame*.png
         - {base_dir}/{shadow|diffuse}/{set_name}/{ckpt}/{img_stem}/res_frame*.png
         - {base_dir}/{img_stem}/res_frame*.png
      2) Raw reshadow.py output structure:
         - {base_dir}/.../src={img_name}/dst=*/Lerp_*/n_frames={n_frames}/res_frame*.png
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Reshadow directory not found: {base_dir}")

    # Candidate directories for simplified layout
    candidates = [
        os.path.join(base_dir, mode_key, ckpt),
        os.path.join(base_dir, mode_key, set_name, ckpt),
        os.path.join(base_dir, mode_key),
        os.path.join(base_dir, ckpt),
        base_dir,
    ]

    frame_map = {}
    for cand in candidates:
        if not os.path.isdir(cand):
            continue
        for entry in sorted(os.listdir(cand)):
            subj_dir = os.path.join(cand, entry)
            if not os.path.isdir(subj_dir):
                continue
            stem = os.path.splitext(entry)[0]
            if os.path.exists(os.path.join(subj_dir, 'res_frame0.png')):
                frame_map[stem] = subj_dir
            elif os.path.exists(os.path.join(subj_dir, 'frames', 'res_frame0.png')):
                frame_map[stem] = os.path.join(subj_dir, 'frames')
        if frame_map:
            return frame_map

    # Fallback: search raw reshadow.py output hierarchy (src=.../dst=.../Lerp_.../n_frames=...)
    legacy_mode = f"myset_{mode_key}_face"
    search_roots = [
        os.path.join(base_dir, legacy_mode),
        os.path.join(base_dir, mode_key),
        base_dir,
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        pattern = os.path.join(root, '**', f'n_frames={n_frames}')
        all_matches = sorted(glob.glob(pattern, recursive=True))
        # Prefer matches containing model_filter if provided
        filtered_matches = [m for m in all_matches if not model_filter or model_filter in m]
        for match_dir in (filtered_matches or all_matches):
            if ckpt and ckpt not in match_dir:
                continue
            if not os.path.exists(os.path.join(match_dir, 'res_frame0.png')):
                continue
            src_matches = re.findall(r'src=([^/]+)', match_dir)
            if src_matches:
                stem = os.path.splitext(src_matches[0])[0]
                frame_map[stem] = match_dir
        if frame_map:
            return frame_map

    return frame_map


def load_shadow_scores(shadow_anno_path):
    """Loads shadow scalar annotations (C) from ffhq-{set}-shadow-anno.txt into a dict keyed by image name and stem."""
    if not os.path.exists(shadow_anno_path):
        raise FileNotFoundError(f"Shadow annotation file not found: {shadow_anno_path}")
    df = pd.read_csv(shadow_anno_path, sep=r'\s+', header=None, names=['image_name', 'c_val'])
    c_dict = {}
    for _, row in df.iterrows():
        img_name = str(row['image_name'])
        c_val = float(row['c_val'])
        c_dict[img_name] = c_val
        c_dict[os.path.splitext(img_name)[0]] = c_val
    return c_dict


def process_smooth_shadow_diff(
    shadow_frames_dir,
    diffuse_frames_dir,
    face_img_path,
    mask_img_path,
    c_val,
    min_c=-4.989461058405101,
    max_c=8.481700287326827,
    n_frames_shadow=5,
    n_frames_diffuse=5,
    shadow_idx=-1,
    diffuse_idx=-1,
    thres=-0.015,
    median_blur_ksize=5,
    save_vis=False,
    save_vis_path=None,
    save_sd=False,
    save_sd_path=None,
    verbose=False,
):
    img_name = os.path.basename(face_img_path)
    img_stem = os.path.splitext(img_name)[0]

    shadow = [Image.open(os.path.join(shadow_frames_dir, f'res_frame{i}.png')) for i in range(n_frames_shadow)]
    diffuse = [Image.open(os.path.join(diffuse_frames_dir, f'res_frame{i}.png')) for i in range(n_frames_diffuse)]
    face = Image.open(face_img_path)

    # Mask
    mask_image = Image.open(mask_img_path)
    m_face = face_segment('faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses', mask_image)
    exclude = face_segment('faceseg_eyes&glasses&mouth&eyebrows', mask_image)

    # Normalized shadow scalar (C)
    c_val_norm = (c_val - min_c) / (max_c - min_c)
    if verbose:
        print(f"[#] {img_name} Shadow value: raw={c_val:.6f}, norm={c_val_norm:.6f}")

    s_grid = create_image_grid([face] + shadow, n_rows=1)
    d_grid = create_image_grid([face] + diffuse, n_rows=1)
    if s_grid.shape[1] != d_grid.shape[1]:
        to_pad = np.abs(s_grid.shape[1] - d_grid.shape[1])
        if s_grid.shape[1] > d_grid.shape[1]:
            d_grid = np.pad(d_grid, ((0, 0), (0, to_pad), (0, 0)), mode='constant', constant_values=0)
        else:
            s_grid = np.pad(s_grid, ((0, 0), (0, to_pad), (0, 0)), mode='constant', constant_values=0)

    c_type = 'L'
    shadow_diff_list = [
        get_shadow_diff(img1=shadow[i], img2=diffuse[diffuse_idx], c_type=c_type, signed=True)
        for i in range(n_frames_shadow)
    ]

    if verbose:
        plt.imshow(np.concatenate((s_grid, d_grid), axis=0))
        plt.title("Shadow and Diffuse")
        plt.show()

        plt.imshow(create_image_grid(shadow_diff_list, n_rows=1))
        plt.title("Shadow Difference (from each shadow and diffuse[-1])")
        plt.show()

    raw_masks = [(shadow_diff_list[i] < thres).astype(np.float32) for i in range(n_frames_shadow)]
    smooth_masks = [
        cv2.medianBlur((shadow_diff_list[i] < thres).astype(np.uint8), median_blur_ksize)
        for i in range(n_frames_shadow)
    ]
    out = raw_masks + smooth_masks

    out = [np.abs(1 - t) for t in out]  # Invert the mask
    out = [((t * ~exclude) + (0.0 * exclude)) for t in out]  # Exclude eyes/glasses/mouth/eyebrows
    out = [(t * m_face) for t in out]  # Mask to face region

    out_W = [t * c_val_norm for t in out]  # Weight shadow area by normalized C

    out_hl = [np.repeat((t[..., None] * 255).astype(np.uint8), 3, axis=2) for t in out.copy()]
    out_hl = [cv2.applyColorMap(t, cv2.COLORMAP_WINTER) * (t == 255) for t in out_hl]
    overlay_img = [cv2.addWeighted(np.array(face), 1, t, 0.5, 0) for t in out_hl]

    if verbose:
        plt.imshow(create_image_grid(shadow_diff_list, n_rows=1), cmap='gray')
        plt.title("Shadow Difference (from each shadow and diffuse[-1])")
        plt.show()

        shadow_mask = out + out_W
        plt.imshow(create_image_grid(shadow_mask, n_rows=4), cmap='gray')
        plt.title("Shadow Difference w/ and w/o C")
        plt.show()

        change = [face] * n_frames_shadow + overlay_img
        plt.imshow(create_image_grid(change, n_rows=3))
        plt.title("Overlay shadow area on input image. (R1=input, R2=Overlay w/o median, R3=Overlay w/ median)")
        plt.show()

    vis_out_W = (np.repeat(out_W[shadow_idx][..., None], 3, axis=2) * 255.0).astype(np.uint8)
    cv2.putText(
        vis_out_W,
        f"{img_name}, C={c_val_norm:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )

    if verbose:
        final_out = [np.array(face), vis_out_W, overlay_img[shadow_idx]]
        final_out = create_image_grid(final_out, n_rows=1)
        plt.imshow(final_out)
        plt.title("Final Output")
        plt.show()

    if save_vis:
        if save_vis_path is None:
            raise ValueError("[#] save_vis_path is not provided!")
        final_out = [np.array(face), vis_out_W, overlay_img[shadow_idx]]
        final_out = create_image_grid(final_out, n_rows=1)
        Image.fromarray(final_out).save(os.path.join(save_vis_path, f'{img_stem}.png'))

    if save_sd:
        if save_sd_path is None:
            raise ValueError("[#] save_sd_path is not provided!")
        final_out_sd_w = out_W[shadow_idx][..., None]
        assert final_out_sd_w.shape == (256, 256, 1), f"Unexpected shape: {final_out_sd_w.shape}"
        assert np.min(final_out_sd_w) >= 0.0 and np.max(final_out_sd_w) <= 1.0, (
            f"Shadow map values out of [0, 1]: min={np.min(final_out_sd_w)}, max={np.max(final_out_sd_w)}"
        )
        np.save(os.path.join(save_sd_path, f'{img_stem}.npy'), final_out_sd_w)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute smooth shadow difference maps (shadow_diff_SS_with_c_simplified) from reshadowed frames.'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Root dataset directory (e.g., /data/mint/DiFaReli++/examples/relighting_data/)',
    )
    parser.add_argument(
        '--set',
        type=str,
        default='valid',
        help='Dataset split (e.g., train, valid, test)',
    )
    # Optional path overrides (derived from dataset_dir and set by default)
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Aligned images directory (default: {dataset_dir}/images_aligned/{set})',
    )
    parser.add_argument(
        '--faceseg_dir',
        type=str,
        default=None,
        help='Face segmentation anno directory (default: {dataset_dir}/face_segment/{set}/anno)',
    )
    parser.add_argument(
        '--shadow_anno_path',
        type=str,
        default=None,
        help='Shadow scalar annotation file (default: {dataset_dir}/params/{set}/ffhq-{set}-shadow-anno.txt)',
    )
    parser.add_argument(
        '--reshadow_dir',
        type=str,
        default=None,
        help='Reshadow root directory (default: {dataset_dir}/reshadow)',
    )
    parser.add_argument(
        '--shadow_reshadow_dir',
        type=str,
        default=None,
        help='Explicit directory for shadow (to-max) reshadow frames (overrides --reshadow_dir for shadow)',
    )
    parser.add_argument(
        '--diffuse_reshadow_dir',
        type=str,
        default=None,
        help='Explicit directory for diffuse (to-min) reshadow frames (overrides --reshadow_dir for diffuse)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for .npy shadow maps (default: {dataset_dir}/shadow_diff_SS_with_c_simplified/{set})',
    )
    parser.add_argument(
        '--vis_dir',
        type=str,
        default=None,
        help='Output directory for visualizations (default: {dataset_dir}/shadow_diff_SS_with_c_simplified/vis/{set})',
    )
    # Hyperparameters
    parser.add_argument('--ckpt', type=str, default='ema_085000', help='Reshadow checkpoint folder name')
    parser.add_argument('--n_frames_shadow', type=int, default=5, help='Number of shadow reshadow frames')
    parser.add_argument('--n_frames_diffuse', type=int, default=5, help='Number of diffuse reshadow frames')
    parser.add_argument('--shadow_idx', type=int, default=-1, help='Selected shadow mask index')
    parser.add_argument('--diffuse_idx', type=int, default=-1, help='Selected diffuse frame index')
    parser.add_argument('--thres', type=float, default=-0.015, help='Shadow difference threshold')
    parser.add_argument('--median_blur_ksize', type=int, default=5, help='Median blur kernel size')
    parser.add_argument('--min_c', type=float, default=-4.989461058405101, help='Minimum C value for normalization')
    parser.add_argument('--max_c', type=float, default=8.481700287326827, help='Maximum C value for normalization')
    parser.add_argument('--no_vis', action='store_true', default=False, help='Disable saving visualization PNGs')
    parser.add_argument('--verbose', action='store_true', default=False, help='Display interactive matplotlib plots')
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    set_name = args.set

    image_dir = os.path.abspath(args.image_dir) if args.image_dir else os.path.join(dataset_dir, 'images_aligned', set_name)
    faceseg_dir = os.path.abspath(args.faceseg_dir) if args.faceseg_dir else os.path.join(dataset_dir, 'face_segment', set_name, 'anno')
    shadow_anno_path = (
        os.path.abspath(args.shadow_anno_path)
        if args.shadow_anno_path
        else os.path.join(dataset_dir, 'params', set_name, f'ffhq-{set_name}-shadow-anno.txt')
    )
    reshadow_dir = os.path.abspath(args.reshadow_dir) if args.reshadow_dir else os.path.join(dataset_dir, 'reshadow')
    shadow_reshadow_base = os.path.abspath(args.shadow_reshadow_dir) if args.shadow_reshadow_dir else reshadow_dir
    diffuse_reshadow_base = os.path.abspath(args.diffuse_reshadow_dir) if args.diffuse_reshadow_dir else reshadow_dir

    save_sd_path = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(dataset_dir, 'shadow_diff_SS_with_c_simplified', set_name)
    )
    save_vis_path = (
        os.path.abspath(args.vis_dir)
        if args.vis_dir
        else os.path.join(dataset_dir, 'shadow_diff_SS_with_c_simplified', 'vis', set_name)
    )
    save_vis = not args.no_vis

    os.makedirs(save_sd_path, exist_ok=True)
    if save_vis:
        os.makedirs(save_vis_path, exist_ok=True)

    # Resolve reshadow directories
    shadow_dict = resolve_reshadow_frames_dir(
        base_dir=shadow_reshadow_base,
        mode_key='shadow',
        ckpt=args.ckpt,
        set_name=set_name,
        n_frames=args.n_frames_shadow,
    )
    diffuse_dict = resolve_reshadow_frames_dir(
        base_dir=diffuse_reshadow_base,
        mode_key='diffuse',
        ckpt=args.ckpt,
        set_name=set_name,
        n_frames=args.n_frames_diffuse,
    )

    # Match subjects present in both shadow and diffuse reshadow outputs
    common_stems = sorted(k for k in shadow_dict.keys() if k in diffuse_dict)
    if not common_stems:
        raise RuntimeError(
            f"No matching subjects found between shadow ({shadow_reshadow_base}) and diffuse ({diffuse_reshadow_base})!"
        )

    # Map image stems to actual image filenames in image_dir
    img_files = {}
    for entry in sorted(os.listdir(image_dir)):
        full_p = os.path.join(image_dir, entry)
        if os.path.isfile(full_p):
            img_files[os.path.splitext(entry)[0]] = full_p

    c_scores = load_shadow_scores(shadow_anno_path)

    print(f"[#] Dataset Dir      : {dataset_dir}")
    print(f"[#] Split            : {set_name}")
    print(f"[#] Aligned Images   : {image_dir}")
    print(f"[#] Face Seg Masks   : {faceseg_dir}")
    print(f"[#] Shadow Anno File : {shadow_anno_path}")
    print(f"[#] Reshadow Dir     : {reshadow_dir}")
    print(f"[#] Output NPY Dir   : {save_sd_path}")
    if save_vis:
        print(f"[#] Output Vis Dir   : {save_vis_path}")
    print(f"[#] Matched Subjects : {len(common_stems)}")

    for stem in tqdm.tqdm(common_stems, desc="Computing smooth shadow maps"):
        if stem not in img_files:
            print(f"[!] Warning: Image for '{stem}' not found in {image_dir}, skipping.")
            continue

        face_img_path = img_files[stem]
        img_name = os.path.basename(face_img_path)

        # Resolve face segmentation mask path (supports anno_{stem}.png or anno_{img_name})
        mask_img_path = os.path.join(faceseg_dir, f'anno_{img_name}')
        if not os.path.exists(mask_img_path):
            mask_img_path = os.path.join(faceseg_dir, f'anno_{stem}.png')

        if stem in c_scores:
            c_val = c_scores[stem]
        elif img_name in c_scores:
            c_val = c_scores[img_name]
        else:
            raise KeyError(f"Shadow score (C) not found for '{img_name}' in {shadow_anno_path}")

        process_smooth_shadow_diff(
            shadow_frames_dir=shadow_dict[stem],
            diffuse_frames_dir=diffuse_dict[stem],
            face_img_path=face_img_path,
            mask_img_path=mask_img_path,
            c_val=c_val,
            min_c=args.min_c,
            max_c=args.max_c,
            n_frames_shadow=args.n_frames_shadow,
            n_frames_diffuse=args.n_frames_diffuse,
            shadow_idx=args.shadow_idx,
            diffuse_idx=args.diffuse_idx,
            thres=args.thres,
            median_blur_ksize=args.median_blur_ksize,
            save_vis=save_vis,
            save_vis_path=save_vis_path,
            save_sd=True,
            save_sd_path=save_sd_path,
            verbose=args.verbose,
        )


if __name__ == '__main__':
    main()
