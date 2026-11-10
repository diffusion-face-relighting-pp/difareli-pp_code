# DiFaReli++: Single-Shot Diffusion Face Relighting & Cast-Shadow Manipulation

Official repository for **DiFaReli++**, providing end-to-end preprocessing, spatial shadow map estimation, and single-shot facial relighting and cast-shadow manipulation.

---

## Table of Contents

- [1. Visual Results Gallery](#1-visual-results-gallery)
  - [1.1 Cast-Shadow Manipulation (Reshadowing)](#11-cast-shadow-manipulation-reshadowing)
  - [1.2 Rotating Spherical Harmonics Lighting (`rotateSH`)](#12-rotating-spherical-harmonics-lighting-rotatesh)
  - [1.3 Target-SH Relighting (`targetSH`)](#13-target-sh-relighting-targetsh)
- [2. Prerequisites](#2-prerequisites)
  - [2.1 Conda Environments](#21-conda-environments)
  - [2.2 Pretrained Checkpoints & Model Weights](#22-pretrained-checkpoints--model-weights)
- [3. Preparing the Data](#3-preparing-the-data)
  - [3.1 Step 1: Place Raw Input Images](#31-step-1-place-raw-input-images)
  - [3.2 Step 2: Run Dataset Preprocessing (`create_dataset.py`)](#32-step-2-run-dataset-preprocessing-create_datasetpy)
  - [3.3 Step 3: Compute Smooth Spatial Shadow Maps (`gen_smooth_Sdiff.py`)](#33-step-3-compute-smooth-spatial-shadow-maps-gen_smooth_sdiffpy)
- [4. Inference (`inference.py`)](#4-inference-inferencepy)
  - [4.1 Reshadowing (Strengthen Shadow & Diffuse Shadow)](#41-reshadowing-strengthen-shadow--diffuse-shadow)
  - [4.2 Rotate SH Lighting (`rotateSH`)](#42-rotate-sh-lighting-rotatesh)
  - [4.3 Target-SH Relighting (`targetSH`)](#43-target-sh-relighting-targetsh)
  - [4.4 Useful CLI Arguments (`inference.py`)](#44-useful-cli-arguments-inferencepy)
  - [4.5 Output Folder Structure](#45-output-folder-structure)
  - [4.6 Interactive Web Visualization (`vis.py`)](#46-interactive-web-visualization-vispy)
- [5. Training](#5-training)

---

## 1. Visual Results Gallery

### 1.1 Cast-Shadow Manipulation (Reshadowing)
Interpolating the cast-shadow scalar $C$ toward **maximum shadow** (`--shadow_diff_inc_c`) or **diffuse / shadow-free** (`--shadow_diff_dec_c`):

| Input | Strengthen (`Relit \| Shadow \| Shading`) | Diffuse (`Relit \| Shadow \| Shading`) |
| :---: | :---: | :---: |
| <img src="examples/dat/images_aligned/valid/61992.png" height="115"> | <img src="examples/res/gifs/reshadow/shadow_61992_relit_shadow_shading.gif" height="115"> | <img src="examples/res/gifs/reshadow/diffuse_61992_relit_shadow_shading.gif" height="115"> |
| <img src="examples/dat/images_aligned/valid/max.png" height="115"> | <img src="examples/res/gifs/reshadow/shadow_max_relit_shadow_shading.gif" height="115"> | <img src="examples/res/gifs/reshadow/diffuse_max_relit_shadow_shading.gif" height="115"> |

---

### 1.2 Rotating Spherical Harmonics Lighting (`rotateSH`)
Rotating the 3D Spherical Harmonics (SH) lighting around axis 2 (`--rotate_sh --rotate_sh_axis 2`) with controlled cast-shadow intensity (`--relight_with_given_c 0.9`):

| Input | Output & Conditioning (`Relit \| Shadow \| Shading`) |
| :---: | :---: |
| <img src="examples/dat/images_aligned/valid/61992.png" height="128"> | <img src="examples/res/gifs/rotateSH/rotateSH_61992_relit_shadow_shading.gif" height="128"> |
| <img src="examples/dat/images_aligned/valid/max.png" height="128"> | <img src="examples/res/gifs/rotateSH/rotateSH_max_relit_shadow_shading.gif" height="128"> |

---

### 1.3 Target-SH Relighting (`targetSH`)
Interpolating lighting and cast shadows from a source portrait (`src`) to match a target reference portrait (`dst`) (`--relight_with_dst_c`):

| Input (`Src \| Dst`) | Output & Conditioning (`Relit \| Shadow \| Shading`) |
| :---: | :---: |
| <img src="examples/dat/images_aligned/valid/61992.png" height="128"> <img src="examples/dat/images_aligned/valid/max.png" height="128"> | <img src="examples/res/gifs/targetSH/targetSH_61992_to_max_relit_shadow_shading.gif" height="128"> |
| <img src="examples/dat/images_aligned/valid/max.png" height="128"> <img src="examples/dat/images_aligned/valid/61992.png" height="128"> | <img src="examples/res/gifs/targetSH/targetSH_max_to_61992_relit_shadow_shading.gif" height="128"> |

> **Image Credits:** Example portraits are sourced from the [FFHQ Dataset (`61992.png`)](https://github.com/nvlabs/ffhq-dataset) and [The New York Times (`max.jpg`)](https://www.nytimes.com/2021/10/29/sports/autoracing/max-verstappen-f1.html).

---

## 2. Prerequisites

### 2.1 Conda Environments

Exported Conda environment files are provided under [`envs/`](envs/):

```text
envs/
├── difareli_pp_sampling.yml         # Inference, DECA 3D rendering, face segmentation, and vis.py
├── difareli_pp_create_dataset.yml   # Dataset preprocessing (ArcFace, DiffAE shadow score, MediaPipe)
├── difareli_pp_training.yml         # Model training
└── ffhq_env.yml                     # FFHQ face alignment (dlib)
```

Create the environments via:

```bash
# 1. Sampling / Inference environment (also used by DECA & face-parsing in create_dataset.py)
conda env create -f envs/difareli_pp_sampling.yml

# 2. Dataset creation environment (ArcFace, DiffAE shadow score, MediaPipe pupil extraction)
conda env create -f envs/difareli_pp_create_dataset.yml

# 3. FFHQ face alignment environment (installed locally where create_dataset.py expects it, or in conda)
conda env create -f envs/ffhq_env.yml -p ./dataset_generation/prepare_relighting_data/FFHQ_align/ffhq_env

# 4. Training environment (optional, for training from scratch)
conda env create -f envs/difareli_pp_training.yml
```

Install the local `guided_diffusion` package in editable mode inside `difareli_pp_sampling` (and `difareli_pp_training`):

```bash
conda activate difareli_pp_sampling
pip install -e .
```

---

### 2.2 Pretrained Checkpoints & Model Weights

#### A. DiFaReli++ Single-Shot Relighting Checkpoints
Place the trained **DiFaReli++** checkpoint files in your checkpoint directory (e.g., `/data/mint/DiFaReli++/ckpt/DiFaReli++/`):

```text
<log_dir>/                           # e.g., /data/mint/DiFaReli++/ckpt/DiFaReli++/
├── ImgCond_ema_0.9999_300000.pt     # Single-shot generator EMA weights
├── ImgEncoder_ema_0.9999_300000.pt  # Spatial condition encoder EMA weights
├── ImgCond_model300000.pt           # (Optional) non-EMA generator weights
└── ImgEncoder_model300000.pt        # (Optional) non-EMA condition encoder weights
```

#### B. Preprocessing & Conditioning Model Weights
Ensure the following third-party weights and 3D morphable model assets are placed at the paths below:

```text
difareli-pp_code/
├── model_3d/FLAME/data/
│   ├── generic_model.pkl
│   ├── head_template.obj
│   └── landmark_embedding.npy
│
├── sample_scripts/cond_utils/
│   ├── DECA/data/
│   │   ├── deca_model.tar
│   │   ├── generic_model.pkl
│   │   ├── head_template.obj
│   │   ├── landmark_embedding.npy
│   │   ├── fixed_displacement_256.npy
│   │   ├── texture_data_256.npy
│   │   ├── FLAME_masks.pkl
│   │   └── FLAME_masks_face-id.pkl
│   └── arcface/detector/
│       ├── mtcnn/weights/{pnet.npy, rnet.npy, onet.npy}
│       └── retinaface/weights/mobilenet0.25_Final.pth
│
└── dataset_generation/prepare_relighting_data/
    ├── face-parsing.PyTorch/res/cp/
    │   └── 79999_iter.pth
    ├── mediapipe/
    │   ├── face_landmarker.task
    │   └── face_landmarker_v2_with_blendshapes.task
    ├── DECA/data/
    │   ├── deca_model.tar
    │   ├── generic_model.pkl
    │   ├── head_template.obj
    │   ├── landmark_embedding.npy
    │   ├── fixed_displacement_256.npy
    │   ├── texture_data_256.npy
    │   ├── faces_index.pkl
    │   ├── FLAME_masks.pkl
    │   └── FLAME_masks_face-id.pkl
    ├── Arcface/
    │   ├── pretrained/BEST_checkpoint_r18.tar
    │   └── detector/
    │       ├── mtcnn/weights/{pnet.npy, rnet.npy, onet.npy}
    │       └── retinaface/weights/mobilenet0.25_Final.pth
    └── diffae/
        ├── shadow_cls_weight.pkl
        └── checkpoints/ffhq256_autoenc/last.ckpt
```

---

## 3. Preparing the Data

### 3.1 Step 1: Place Raw Input Images

Place your raw portrait images into `examples/dat/images/`:

```text
examples/dat/
└── images/
    ├── 61992.png
    └── max.jpg
```

---

### 3.2 Step 2: Run Dataset Preprocessing (`create_dataset.py`)

Run `create_dataset.py` from `dataset_generation/prepare_relighting_data/` to extract:
1. **FFHQ Face Alignment (`--ffhq_align`)**: Crops and aligns portraits to $256 \times 256$ in `images_aligned/<set>/`.
2. **Face Segmentation & Pupil Landmarks (`--faceseg`)**: Produces 19-class face parsing masks (`face_segment/<set>/anno/`) and 21-class pupil-augmented masks (`face_segment_with_pupil/<set>/anno/`).
3. **DECA 3D Reconstruction & Shading (`--deca`)**: Estimates FLAME 3D parameters (`shape`, `pose`, `exp`, `cam`, `light`, `albedo`, `detail`) in `params/<set>/` and renders masked Lambertian face shadings (`rendered_images/deca_masked_face_images_{wclip,woclip}/<set>/`).
4. **ArcFace Identity Embedding (`--arcface`)**: Extracts identity features into `params/<set>/ffhq-<set>-faceemb-anno.txt`.
5. **DiffAE Shadow Scalar Estimation (`--shadow`)**: Computes raw cast-shadow scalar scores $C$ in `params/<set>/ffhq-<set>-shadow-anno.txt`.

#### Command

```bash
cd dataset_generation/prepare_relighting_data

python create_dataset.py \
  --image_dir ../../examples/dat/images \
  --out_dataset_dir ../../examples/dat/ \
  --ffhq_align \
  --faceseg \
  --arcface \
  --shadow \
  --deca \
  --set valid
```

#### Useful Arguments (`create_dataset.py`)

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--image_dir` | *(Required)* | Directory containing raw input images (or pre-aligned images if omitting `--ffhq_align`) |
| `--out_dataset_dir` | *(Required)* | Output dataset root directory (e.g., `../../examples/dat/`) |
| `--set` | `valid` | Dataset split name (`train`, `valid`, `test`) |
| `--ffhq_align` | `False` | Run FFHQ $256 \times 256$ face alignment |
| `--faceseg` | `False` | Run `face-parsing.PyTorch` + MediaPipe pupil mask extraction |
| `--deca` | `False` | Run DECA 3D reconstruction and masked Lambertian shading rendering |
| `--arcface` | `False` | Extract ArcFace identity embeddings |
| `--shadow` | `False` | Estimate DiffAE shadow scalar $C$ |
| `--env_ffhq` | `./FFHQ_align/ffhq_env/bin/python` | Python executable for FFHQ alignment |
| `--env_create_dataset` | `.../envs/difareli_pp_create_dataset/bin/python` | Python executable for ArcFace, DiffAE, and MediaPipe |
| `--env_sampling` | `.../envs/difareli_pp_sampling/bin/python` | Python executable for FaceSeg and DECA |

#### Folder Structure After Step 2

```text
examples/dat/
├── images/
│   ├── 61992.png
│   └── max.jpg
├── images_aligned/
│   └── valid/
│       ├── 61992.png
│       └── max.png
├── face_segment/
│   └── valid/
│       ├── anno/
│       │   ├── anno_61992.png
│       │   └── anno_max.png
│       └── vis/
│           ├── res_61992.png
│           └── res_max.png
├── face_segment_with_pupil/
│   └── valid/
│       └── anno/
│           ├── anno_61992.png
│           └── anno_max.png
├── params/
│   └── valid/
│       ├── ffhq-valid-albedo-anno.txt
│       ├── ffhq-valid-cam-anno.txt
│       ├── ffhq-valid-detail-anno.txt
│       ├── ffhq-valid-exp-anno.txt
│       ├── ffhq-valid-faceemb-anno.txt
│       ├── ffhq-valid-light-anno.txt
│       ├── ffhq-valid-pose-anno.txt
│       ├── ffhq-valid-shadow-anno.txt
│       ├── ffhq-valid-shape-anno.txt
│       ├── ffhq-valid-tform-anno.txt
│       ├── ffhq-valid-trans_verts_orig-anno.npy
│       └── latent.pkl
└── rendered_images/
    ├── deca_masked_face_images_wclip/
    │   └── valid/
    │       ├── 61992.png
    │       └── max.png
    └── deca_masked_face_images_woclip/
        └── valid/
            ├── 61992.npy
            └── max.npy
```

---

### 3.3 Step 3: Compute Smooth Spatial Shadow Maps (`gen_smooth_Sdiff.py`)

#### A. Generate Reshadowed Reference Frames (via DiFaReli)
To compute the source spatial shadow map for each input portrait, first synthesize two reshadowed versions of each aligned image—one interpolated toward **strongest shadow** (`shadow`) and one toward **diffuse / shadow-free** (`diffuse`)—using the DPM-based reshadowing script from the original **DiFaReli** repository:
👉 **[https://github.com/diffusion-face-relighting/difareli_code](https://github.com/diffusion-face-relighting/difareli_code)**

Place the resulting reshadowed frames (`res_frame0.png ... res_frame4.png`, or the final frame `res_frame4.png`) under `examples/dat/reshadow/`:

```text
examples/dat/reshadow/
├── diffuse/
│   ├── 61992/
│   │   └── res_frame4.png        # (or res_frame0.png ... res_frame4.png)
│   └── max/
│       └── res_frame4.png
└── shadow/
    ├── 61992/
    │   └── res_frame4.png
    └── max/
        └── res_frame4.png
```

#### B. Run `gen_smooth_Sdiff.py`
From `dataset_generation/prepare_relighting_data/`, run:

```bash
python ./shadow_map/gen_smooth_Sdiff.py \
  --dataset_dir ../../examples/dat/ \
  --reshadow_dir ../../examples/dat/reshadow/ \
  --set valid
```

#### Useful Arguments (`gen_smooth_Sdiff.py`)

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset_dir` | *(Required)* | Root dataset directory (`../../examples/dat/`) |
| `--reshadow_dir` | `<dataset_dir>/reshadow` | Directory containing `shadow/` and `diffuse/` reshadowed frames |
| `--set` | `valid` | Dataset split (`train`, `valid`, `test`) |
| `--thres` | `-0.015` | Signed grayscale intensity difference threshold (`diffuse - shadow < thres`) |
| `--median_blur_ksize` | `5` | Kernel size for `cv2.medianBlur` smoothing |
| `--shadow_idx` | `-1` | Frame index in the shadow sequence to export (`-1` = last frame) |
| `--diffuse_idx` | `-1` | Frame index in the diffuse sequence used as reference (`-1` = last frame) |
| `--no_vis` | `False` | Disable saving side-by-side `.png` visualizations |

#### Folder Structure Added in Step 3

```text
examples/dat/
└── shadow_diff_SS_with_c_simplified/
    ├── valid/
    │   ├── 61992.npy             # (256, 256, 1) float32 shadow map weighted by normalized C in [0, 1]
    │   └── max.npy
    └── vis/
        └── valid/
            ├── 61992.png         # Diagnostic side-by-side [Input | Weighted Shadow Map | Overlay]
            └── max.png
```

---

## 4. Inference (`inference.py`)

Configure your source/target pairs in `examples/pair.json`:
```json
{
    "pair": {
        "pair1": { "src": "max.png", "dst": "61992.png" },
        "pair2": { "src": "61992.png", "dst": "max.png" }
    }
}
```

Activate `difareli_pp_sampling` and run `inference.py` from `sample_scripts/single_shot/`:

```bash
conda activate difareli_pp_sampling
cd sample_scripts/single_shot
```

---

### 4.1 Reshadowing (Strengthen Shadow & Diffuse Shadow)

- **Strengthen Shadow (`--shadow_diff_inc_c --force_src_shading_for_reshadow`)**:
  ```bash
  python inference.py \
    --dataset_root ../../examples/dat/ \
    --set valid \
    --step 300000 \
    --out_dir ../../examples/res/reshadow/shadow \
    --cfg_name DiFaReli++.yaml \
    --log_dir /data/mint/DiFaReli++/ckpt/DiFaReli++/ \
    --sample_pair_json ../../examples/pair.json \
    --itp_step 60 \
    --batch_size 1 \
    --gpu_id 2 \
    --shadow_diff_inc_c \
    --force_src_shading_for_reshadow \
    --reshadow_with_given_c 0.15 \
    --save_vid
  ```

- **Diffuse / Remove Shadow (`--shadow_diff_dec_c --force_diffuse_shading_for_reshadow --itp_diffuse_sh`)**:
  ```bash
  python inference.py \
    --dataset_root ../../examples/dat/ \
    --set valid \
    --step 300000 \
    --out_dir ../../examples/res/reshadow/diffuse \
    --cfg_name DiFaReli++.yaml \
    --log_dir /data/mint/DiFaReli++/ckpt/DiFaReli++/ \
    --sample_pair_json ../../examples/pair.json \
    --itp_step 60 \
    --batch_size 1 \
    --gpu_id 2 \
    --shadow_diff_dec_c \
    --force_diffuse_shading_for_reshadow \
    --itp_diffuse_sh \
    --save_vid
  ```

> **Reshadowing Tips:**
> - **Custom Starting Shadow Scalar (`--reshadow_with_given_c <val>`)**: By default, reshadowing starts from the input portrait's estimated shadow score $C_{\text{src}}$. To start strengthening from a lower $C$ (e.g., `0.1`) or diffusing from a higher $C$ (e.g., `0.9`) for a more pronounced transition, add `--reshadow_with_given_c 0.1` (values in `[0.0, 1.0]`).
> - **Smooth Shading Interpolation to Diffuse (`--itp_diffuse_sh`)**: Adding `--itp_diffuse_sh` alongside `--force_diffuse_shading_for_reshadow` linearly interpolates the DECA SH shading from the source directional SH to the ambient-only diffuse SH across the frames instead of switching shading in a single step.

---

### 4.2 Rotate SH Lighting (`rotateSH`)

Rotates the source SH lighting 360° around `--rotate_sh_axis` (`0`, `1`, or `2`) with ray-traced cast shadows scaled by `--relight_with_given_c`:

```bash
python inference.py \
  --dataset_root ../../examples/dat/ \
  --set valid \
  --step 300000 \
  --out_dir ../../examples/res/relight/rotateSH_a2/ \
  --cfg_name DiFaReli++.yaml \
  --log_dir /data/mint/DiFaReli++/ckpt/DiFaReli++/ \
  --sample_pair_json ../../examples/pair.json \
  --itp_step 60 \
  --batch_size 1 \
  --gpu_id 1 \
  --rotate_sh \
  --rotate_sh_axis 2 \
  --relight_with_given_c 0.9 \
  --save_vid
```

---

### 4.3 Target-SH Relighting (`targetSH`)

Relights `src` by interpolating toward `dst`'s SH lighting and applying `dst`'s shadow scalar (`--relight_with_dst_c`):

```bash
python inference.py \
  --dataset_root ../../examples/dat/ \
  --set valid \
  --step 300000 \
  --out_dir ../../examples/res/relight/targetSH/ \
  --cfg_name DiFaReli++.yaml \
  --log_dir /data/mint/DiFaReli++/ckpt/DiFaReli++/ \
  --sample_pair_json ../../examples/pair.json \
  --sample_pair_mode pair \
  --itp_step 60 \
  --batch_size 1 \
  --gpu_id 2 \
  --relight_with_dst_c \
  --save_vid
```

> **Note on Frame 0 in `rotateSH` and `targetSH`:** Frame `0` (`res_frame0.png`) uses the precomputed 2D source shadow map (`shadow_diff_SS_with_c_simplified`), whereas frames `1 .. N-1` use 3D ray-traced target shadow masks. Because there is no direct spatial blending between the 2D source shadow map and the 3D ray-traced shadow mask at frame `0`, discard frame `0` (or use the `*_no1.mp4` videos generated in `vid/`) for smooth animations.

---

### 4.4 Useful CLI Arguments (`inference.py`)

| Argument | Default | Description |
| :--- | :--- | :--- |
| **Dataset & Checkpoint** | | |
| `--dataset_root` | *(Required)* | Preprocessed dataset root (`../../examples/dat/`) |
| `--set` | *(Required)* | Dataset split (`valid`, `train`, `test`) |
| `--log_dir` | *(Required)* | Directory containing the trained checkpoint |
| `--cfg_name` | *(Required)* | Model config YAML filename (`DiFaReli++.yaml`) |
| `--step` | *(Required)* | Checkpoint training step (`300000`) |
| `--ckpt_selector` | `ema` | Weight type (`ema` or `model`) |
| **Pairs & Sampling** | | |
| `--sample_pair_json` | `None` | Path to JSON file specifying `src`/`dst` pairs |
| `--sample_pair_mode` | `pair` | JSON pairing key (`pair`, `pairwise`, `list`) |
| `--src_dst` | `[]` | Explicit CLI pair override, e.g., `--src_dst max.png 61992.png` |
| `--idx` | `[0, 99999]` | Start and end pair indices to process |
| `--itp_step` | `15` | Number of interpolation frames (`n_frames`) |
| `--batch_size` | `15` | Generator sub-batch size per forward pass |
| `--lerp` / `--slerp` | `lerp=True` | Linear (`--lerp`) or spherical (`--slerp`) interpolation |
| **Lighting & Shading** | | |
| `--rotate_sh` | `False` | Rotate source SH lighting 360° over `--itp_step` frames |
| `--rotate_sh_axis` | `0` | Rotation axis (`0`, `1`, or `2`) when `--rotate_sh` is enabled |
| `--force_src_shading_for_reshadow` | `False` | Keep source SH shading fixed across frames (used with `--shadow_diff_inc_c`) |
| `--force_diffuse_shading_for_reshadow` | `False` | Use ambient-only diffuse SH shading on target frames (used with `--shadow_diff_dec_c`) |
| `--itp_diffuse_sh` | `False` | Linearly interpolate SH from source shading to ambient diffuse shading (use with `--force_diffuse_shading_for_reshadow`) |
| `--scale_sh` | `1.0` | Multiplier on target SH coefficients |
| **Shadow Map & Weight ($C$)** | | |
| `--shadow_diff_inc_c` | `False` | **Reshadow mode**: Lock source shadow mask and linearly increase $C \to 1.0$ |
| `--shadow_diff_dec_c` | `False` | **Reshadow mode**: Lock source shadow mask and linearly decrease $C \to 0.0$ |
| `--reshadow_with_given_c` | `None` | Override starting $C \in [0, 1]$ for reshadowing (defaults to $C_{\text{src}}$) |
| `--relight_with_dst_c` | `False` | **Relight mode**: Scale ray-traced target shadow masks by destination subject's $C_{\text{dst}}$ |
| `--relight_with_given_c` | `None` | **Relight mode**: Scale ray-traced target shadow masks by a custom $C \in [0, 1]$ |
| `--relight_with_strongest_c` | `False` | **Relight mode**: Scale ray-traced target shadow masks by $C = 1.0$ |
| `--pt_round` | `1` | Number of light ray perturbations (`1` = hard ray-traced shadow) |
| **Output & Visualization** | | |
| `--out_dir` | *(Required)* | Output directory for current mode/experiment |
| `--save_vid` | `False` | Save forward (`*.mp4`), round-trip (`*_rt.mp4`), and frame-0-excluded (`*_no1.mp4`) videos |
| `--fps` | `20` | Frame rate for saved MP4 videos |
| `--gpu_id` | `"0"` | GPU device ID (`CUDA_VISIBLE_DEVICES`) |

---

### 4.5 Output Folder Structure

Single-subject modes (`reshadow/*`, `rotateSH_*`) organize outputs under `<out_dir>/<src_stem>/`, while paired target-lighting modes (`targetSH`) organize outputs under `<out_dir>/<src_stem>_to_<dst_stem>/`:

```text
examples/res/
├── reshadow/
│   ├── shadow/                              # Strengthen shadow (--shadow_diff_inc_c)
│   │   ├── 61992/
│   │   │   ├── frames/
│   │   │   │   ├── res_frame0.png ... res_frame59.png
│   │   │   │   ├── src_ren_frame0.png ...
│   │   │   │   ├── dst_ren_frame0.png ...
│   │   │   │   ├── src_shadm_shad_frame0.png ...
│   │   │   │   └── dst_shadm_shad_frame0.png ...
│   │   │   ├── vid/
│   │   │   │   ├── res.mp4, res_rt.mp4
│   │   │   │   └── out.mp4, out_rt.mp4, out_no1.mp4, out_rt_no1.mp4
│   │   │   └── res_desc.json                # Full CLI command, args, and timing metadata
│   │   └── max/
│   │       ├── frames/
│   │       ├── vid/
│   │       └── res_desc.json
│   └── diffuse/                             # Diffuse shadow (--shadow_diff_dec_c)
│       ├── 61992/
│       └── max/
│
├── relight/
│   ├── rotateSH_a2/                         # Rotate SH lighting (--rotate_sh --rotate_sh_axis 2)
│   │   ├── 61992/
│   │   │   ├── frames/
│   │   │   ├── vid/
│   │   │   └── res_desc.json
│   │   └── max/
│   └── targetSH/                            # Paired Target-SH relighting (--relight_with_dst_c)
│       ├── 61992_to_max/
│       │   ├── frames/
│       │   ├── vid/
│       │   └── res_desc.json
│       └── max_to_61992/
│           ├── frames/
│           ├── vid/
│           └── res_desc.json
│
└── gifs/                                    # High-quality showcase GIFs
    ├── reshadow/
    │   ├── shadow_61992_relit_shadow_shading.gif
    │   ├── shadow_max_relit_shadow_shading.gif
    │   ├── diffuse_61992_relit_shadow_shading.gif
    │   └── diffuse_max_relit_shadow_shading.gif
    ├── rotateSH/
    │   ├── rotateSH_61992_relit_shadow_shading.gif
    │   └── rotateSH_max_relit_shadow_shading.gif
    └── targetSH/
        ├── targetSH_61992_to_max_relit_shadow_shading.gif
        └── targetSH_max_to_61992_relit_shadow_shading.gif
```

---

### 4.6 Interactive Web Visualization (`vis.py`)

To inspect all generated frames, conditioning maps, and videos in your browser:

```bash
cd sample_scripts/single_shot
python vis.py --root_dir ../../examples/res/ --port 8888
```

---

## 5. Training

*TBD*
