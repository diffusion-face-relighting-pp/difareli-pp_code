# Relighting Dataset Preprocessing Guide

This directory contains the preprocessing pipeline to prepare custom face images for **DiFaReli++** relighting.

---

## Expected Output Dataset Structure

After running all preprocessing steps (for split `<set>`, e.g., `valid`), the output dataset directory (e.g., `/data/mint/DiFaReli++/examples/relighting_data/`) is organized as follows:

```text
<out_dataset_dir>/
├── images_aligned/
│   └── <set>/
│       └── <subject>.png
├── face_segment/
│   └── <set>/
│       ├── anno/
│       │   └── anno_<subject>.png
│       └── vis/
│           └── res_<subject>.png
├── face_segment_with_pupil/
│   └── <set>/
│       └── anno/
│           └── anno_<subject>.png
├── params/
│   └── <set>/
│       ├── ffhq-<set>-albedo-anno.txt
│       ├── ffhq-<set>-cam-anno.txt
│       ├── ffhq-<set>-detail-anno.txt
│       ├── ffhq-<set>-exp-anno.txt
│       ├── ffhq-<set>-faceemb-anno.txt
│       ├── ffhq-<set>-light-anno.txt
│       ├── ffhq-<set>-pose-anno.txt
│       ├── ffhq-<set>-shadow-anno.txt
│       ├── ffhq-<set>-shape-anno.txt
│       ├── ffhq-<set>-tform-anno.txt
│       ├── ffhq-<set>-trans_verts_orig-anno.npy
│       └── latent.pkl
├── rendered_images/
│   ├── deca_masked_face_images_wclip/
│   │   └── <set>/
│   │       └── <subject>.png
│   └── deca_masked_face_images_woclip/
│       └── <set>/
│           └── <subject>.npy
├── reshadow/
│   ├── diffuse/
│   │   └── ema_085000/
│   │       └── <subject>/
│   │           ├── ren_frame0.png ... ren_frame4.png
│   │           └── res_frame0.png ... res_frame4.png
│   └── shadow/
│       └── ema_085000/
│           └── <subject>/
│               ├── ren_frame0.png ... ren_frame4.png
│               └── res_frame0.png ... res_frame4.png
└── shadow_diff_SS_with_c_simplified/
    ├── <set>/
    │   └── <subject>.npy
    └── vis/
        └── <set>/
            └── <subject>.png
```

---

## Step-by-Step Preprocessing Pipeline

### Step 1: Extract Face Annotations, 3D Renderings, and Embeddings (`create_dataset.py`)

`create_dataset.py` orchestrates five preprocessing modules in sequence:

1. **FFHQ Face Alignment (`--ffhq_align`)**: Runs `./FFHQ_align/align.py` to crop and align raw input portraits to 256×256 FFHQ format (`images_aligned/<set>/`).
2. **Face Segmentation & Pupil Extraction (`--faceseg`)**:
   - Runs `./face-parsing.PyTorch/inference.py` to produce 19-class face parsing masks (`face_segment/<set>/anno/`).
   - Runs `./mediapipe/extract_face_mask_with_pupils.py` to detect iris/pupil landmarks and augment the segmentation masks with left/right pupil labels (`19` and `20`) in `face_segment_with_pupil/<set>/anno/`.
3. **3D Face Reconstruction & Shading (`--deca`)**: Runs `./DECA/script/estimate_deca_for_dpm.py` to estimate FLAME parameters (`shape`, `pose`, `exp`, `cam`, `light`, `albedo`, `detail`) in `params/<set>/` and render masked Lambertian face shadings (`rendered_images/deca_masked_face_images_{wclip,woclip}/<set>/`).
4. **Face Identity Embedding (`--arcface`)**: Runs `./Arcface/inference.py` to extract ArcFace embeddings into `params/<set>/ffhq-<set>-faceemb-anno.txt`.
5. **Shadow Scalar Estimation (`--shadow`)**: Runs `./diffae/gen_shadow.py` to project DiffAE latents onto the shadow classification boundary and save raw shadow scores $C$ in `params/<set>/ffhq-<set>-shadow-anno.txt`.

#### Usage

Run from `dataset_generation/prepare_relighting_data/`:

```bash
python create_dataset.py \
  --image_dir /path/to/raw/images \
  --out_dataset_dir /data/mint/DiFaReli++/examples/relighting_data \
  --set valid \
  --ffhq_align \
  --faceseg \
  --deca \
  --arcface \
  --shadow
```

> **Note:** If your images are already aligned in `<out_dataset_dir>/images_aligned/<set>`, omit `--ffhq_align` and pass `--image_dir /data/mint/DiFaReli++/examples/relighting_data/images_aligned/valid`.

---

### Step 2: Generate Reshadowed Frames (Shadow & Diffuse)

Before computing the spatial shadow maps, run the reshadowing model on each aligned image to synthesize two sequences of frames (default `n_frames=5`):
- **Shadow (`shadow`)**: Interpolates the shadow scalar toward maximum cast shadow ($C_{\max} \approx 8.4817$).
- **Diffuse (`diffuse`)**: Interpolates the shadow scalar toward minimum/diffuse shadow ($C_{\min} \approx -4.9895$).

Place or output the resulting frames under `<out_dataset_dir>/reshadow/`:
- `<out_dataset_dir>/reshadow/shadow/ema_085000/<subject>/res_frame{0..4}.png`
- `<out_dataset_dir>/reshadow/diffuse/ema_085000/<subject>/res_frame{0..4}.png`

*(Note: `shadow_map/gen_smooth_Sdiff.py` also supports reading directly from the nested `sample_scripts/default/reshadow.py` output directory structure without manually moving files.)*

---

### Step 3: Compute Smooth Shadow Difference Maps (`shadow_map/gen_smooth_Sdiff.py`)

Once the reshadowed frames (`shadow` and `diffuse`), face segmentation masks (`face_segment/<set>/anno/`), and shadow scalar annotations (`params/<set>/ffhq-<set>-shadow-anno.txt`) are ready, run `shadow_map/gen_smooth_Sdiff.py` to compute the smoothed spatial shadow maps:

1. Computes the signed grayscale intensity difference between the `shadow` frames and the `diffuse` frame (`diffuse[-1]`).
2. Thresholds the shadow difference (`thres = -0.015`) and applies median filtering (`kernel_size = 5`) to remove noise.
3. Masks the result to the valid facial region (`faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses`) while excluding eyes, eyeglasses, mouth, and eyebrows (`faceseg_eyes&glasses&mouth&eyebrows`).
4. Scales the binary shadow mask by the normalized shadow scalar $C_{\text{norm}} = \frac{C - C_{\min}}{C_{\max} - C_{\min}} \in [0, 1]$.
5. Saves each `(256, 256, 1)` array to `<out_dataset_dir>/shadow_diff_SS_with_c_simplified/<set>/<subject>.npy` and side-by-side diagnostic visualizations to `<out_dataset_dir>/shadow_diff_SS_with_c_simplified/vis/<set>/<subject>.png`.

#### Usage

```bash
python ./shadow_map/gen_smooth_Sdiff.py \
  --dataset_dir /data/mint/DiFaReli++/examples/relighting_data \
  --set valid
```

#### Optional Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset_dir` | *(Required)* | Root dataset directory (e.g., `/data/mint/DiFaReli++/examples/relighting_data`) |
| `--set` | `valid` | Dataset split (`train`, `valid`, `test`) |
| `--reshadow_dir` | `<dataset_dir>/reshadow` | Root directory containing `shadow` and `diffuse` reshadowed frames |
| `--shadow_reshadow_dir` | `None` | Explicit path override for `shadow` reshadowed frames |
| `--diffuse_reshadow_dir` | `None` | Explicit path override for `diffuse` reshadowed frames |
| `--image_dir` | `<dataset_dir>/images_aligned/<set>` | Aligned input images directory |
| `--faceseg_dir` | `<dataset_dir>/face_segment/<set>/anno` | Face segmentation annotations directory |
| `--shadow_anno_path` | `<dataset_dir>/params/<set>/ffhq-<set>-shadow-anno.txt` | Shadow scalar $C$ annotations file |
| `--output_dir` | `<dataset_dir>/shadow_diff_SS_with_c_simplified/<set>` | Output directory for `.npy` shadow maps |
| `--vis_dir` | `<dataset_dir>/shadow_diff_SS_with_c_simplified/vis/<set>` | Output directory for visualization `.png` grids |
| `--ckpt` | `ema_085000` | Checkpoint subdirectory name inside `reshadow/{shadow,diffuse}/` |
| `--n_frames_shadow` | `5` | Number of frames in `shadow` reshadow sequence |
| `--n_frames_diffuse` | `5` | Number of frames in `diffuse` reshadow sequence |
| `--shadow_idx` | `-1` | Frame index in `out_W` to export (`-1` selects the last median-smoothed frame) |
| `--diffuse_idx` | `-1` | Frame index in `diffuse` sequence used as reference (`-1` selects the last frame) |
| `--thres` | `-0.015` | Signed shadow difference threshold |
| `--median_blur_ksize` | `5` | Kernel size for `cv2.medianBlur` smoothing |
| `--min_c` | `-4.989461058405101` | Minimum shadow scalar $C_{\min}$ for normalization |
| `--max_c` | `8.481700287326827` | Maximum shadow scalar $C_{\max}$ for normalization |
| `--no_vis` | `False` | Skip saving visualization `.png` files |
| `--verbose` | `False` | Display interactive Matplotlib plots for debugging |
