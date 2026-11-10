# DiFaReli++: Single-Shot Diffusion Face Relighting & Cast-Shadow Manipulation

Official repository for **DiFaReli++**, providing end-to-end preprocessing, spatial shadow map estimation, and single-shot facial relighting and cast-shadow manipulation.

---

## Overview

- [1. Prerequisites](#1-prerequisites)
  - [1.1 Conda Environments](#11-conda-environments)
  - [1.2 Pretrained Checkpoints & Model Weights](#12-pretrained-checkpoints--model-weights)
- [2. Preparing the Data](#2-preparing-the-data)
  - [2.1 Step 1: Place Raw Input Images](#21-step-1-place-raw-input-images)
  - [2.2 Step 2: Run Dataset Preprocessing (`create_dataset.py`)](#22-step-2-run-dataset-preprocessing-create_datasetpy)
  - [2.3 Step 3: Compute Smooth Spatial Shadow Maps (`gen_smooth_Sdiff.py`)](#23-step-3-compute-smooth-spatial-shadow-maps-gen_smooth_sdiffpy)
- [3. Inference (`inference.py`)](#3-inference-inferencepy)
  - [3.1 Reshadowing (Strengthen Shadow & Diffuse Shadow)](#31-reshadowing-strengthen-shadow--diffuse-shadow)
  - [3.2 Rotate SH Lighting (`rotateSH`)](#32-rotate-sh-lighting-rotatesh)
  - [3.3 Target-SH Relighting (`targetSH`)](#33-target-sh-relighting-targetsh)
  - [3.4 CLI Arguments & Output Structure](#34-cli-arguments--output-structure)
  - [3.5 Interactive Web Visualization (`vis.py`)](#35-interactive-web-visualization-vispy)
- [4. Training](#4-training)

---

## 1. Prerequisites

### 1.1 Conda Environments

Exported Conda environment files are provided in [`envs/`](envs/):

```bash
# 1. Sampling / Inference environment (also used by DECA & face-parsing in create_dataset.py)
conda env create -f envs/difareli_pp_sampling.yml

# 2. Dataset creation environment (ArcFace, DiffAE shadow score, MediaPipe pupil extraction)
conda env create -f envs/difareli_pp_create_dataset.yml

# 3. FFHQ face alignment environment
conda env create -f envs/ffhq_env.yml -p ./dataset_generation/prepare_relighting_data/FFHQ_align/ffhq_env

# 4. Training environment (optional)
conda env create -f envs/difareli_pp_training.yml
```

Install the local `guided_diffusion` package in editable mode inside `difareli_pp_sampling` (and `difareli_pp_training`):

```bash
conda activate difareli_pp_sampling
pip install -e .
```

---

### 1.2 Pretrained Checkpoints & Model Weights

Place the **DiFaReli++** checkpoint files (`ImgCond_ema_0.9999_300000.pt`, `ImgEncoder_ema_0.9999_300000.pt`) in `<log_dir>/` (e.g., `/data/mint/DiFaReli++/ckpt/DiFaReli++/`), along with the third-party preprocessing weights.

<details>
<summary><b>📂 Click to expand required checkpoint & third-party model paths</b></summary>

```text
<log_dir>/                                  # e.g., /data/mint/DiFaReli++/ckpt/DiFaReli++/
├── ImgCond_ema_0.9999_300000.pt            # Single-shot generator EMA weights
└── ImgEncoder_ema_0.9999_300000.pt         # Spatial condition encoder EMA weights

difareli-pp_code/
├── model_3d/FLAME/data/                    # {generic_model.pkl, head_template.obj, landmark_embedding.npy}
├── sample_scripts/cond_utils/
│   ├── DECA/data/                          # {deca_model.tar, generic_model.pkl, head_template.obj, landmark_embedding.npy,
│   │                                       #  fixed_displacement_256.npy, texture_data_256.npy, FLAME_masks*.pkl}
│   └── arcface/detector/                   # mtcnn/weights/{pnet,rnet,onet}.npy, retinaface/weights/mobilenet0.25_Final.pth
└── dataset_generation/prepare_relighting_data/
    ├── face-parsing.PyTorch/res/cp/79999_iter.pth
    ├── mediapipe/                          # {face_landmarker.task, face_landmarker_v2_with_blendshapes.task}
    ├── DECA/data/                          # Same as sample_scripts/cond_utils/DECA/data/ + faces_index.pkl
    ├── Arcface/                            # pretrained/BEST_checkpoint_r18.tar + detector/{mtcnn,retinaface}
    └── diffae/                             # shadow_cls_weight.pkl + checkpoints/ffhq256_autoenc/last.ckpt
```

</details>

---

## 2. Preparing the Data

### 2.1 Step 1: Place Raw Input Images

Place raw input portraits in `examples/dat/images/` (e.g., `examples/dat/images/{61992.png, max.jpg}`).

---

### 2.2 Step 2: Run Dataset Preprocessing (`create_dataset.py`)

Run `create_dataset.py` to perform:
1. **FFHQ Alignment (`--ffhq_align`)**: $256 \times 256$ aligned faces $\rightarrow$ `images_aligned/<set>/`
2. **Face & Pupil Parsing (`--faceseg`)**: Segmentation masks $\rightarrow$ `face_segment{,_with_pupil}/<set>/anno/`
3. **DECA 3D & Lambertian Shading (`--deca`)**: FLAME parameters $\rightarrow$ `params/<set>/` & rendered shadings $\rightarrow$ `rendered_images/deca_masked_face_images_{wclip,woclip}/<set>/`
4. **ArcFace Identity (`--arcface`)** & **DiffAE Shadow Scalar (`--shadow`)**: $\rightarrow$ `params/<set>/ffhq-<set>-{faceemb,shadow}-anno.txt`

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

<details>
<summary><b>⚙️ Click to expand <code>create_dataset.py</code> arguments & output structure</b></summary>

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--image_dir` | *(Required)* | Directory containing raw input images (or pre-aligned images if omitting `--ffhq_align`) |
| `--out_dataset_dir` | *(Required)* | Output dataset root directory (`../../examples/dat/`) |
| `--set` | `valid` | Dataset split name (`train`, `valid`, `test`) |
| `--ffhq_align` | `False` | Run FFHQ $256 \times 256$ face alignment |
| `--faceseg` | `False` | Run `face-parsing.PyTorch` + MediaPipe pupil mask extraction |
| `--deca` | `False` | Run DECA 3D reconstruction and masked Lambertian shading rendering |
| `--arcface` | `False` | Extract ArcFace identity embeddings |
| `--shadow` | `False` | Estimate DiffAE shadow scalar $C$ |
| `--env_ffhq` | `./FFHQ_align/ffhq_env/bin/python` | Python executable for FFHQ alignment |
| `--env_create_dataset` | `.../envs/difareli_pp_create_dataset/bin/python` | Python executable for ArcFace, DiffAE, and MediaPipe |
| `--env_sampling` | `.../envs/difareli_pp_sampling/bin/python` | Python executable for FaceSeg and DECA |

```text
examples/dat/
├── images/                                           # {61992.png, max.jpg}
├── images_aligned/valid/                             # {61992.png, max.png}
├── face_segment/valid/{anno,vis}/                    # {anno_*, res_*}.png
├── face_segment_with_pupil/valid/anno/               # {anno_61992.png, anno_max.png}
├── params/valid/                                     # ffhq-valid-{albedo,cam,detail,exp,faceemb,light,pose,shadow,shape,tform}-anno.txt,
│                                                     # ffhq-valid-trans_verts_orig-anno.npy, latent.pkl
└── rendered_images/
    ├── deca_masked_face_images_wclip/valid/          # {61992.png, max.png}
    └── deca_masked_face_images_woclip/valid/         # {61992.npy, max.npy}
```

</details>

---

### 2.3 Step 3: Compute Smooth Spatial Shadow Maps (`gen_smooth_Sdiff.py`)

1. **Generate Reshadowed Reference Frames (via DiFaReli)**: Synthesize `shadow` (strongest shadow) and `diffuse` (shadow-free) reference frames for each aligned image using the original **[DiFaReli repository](https://github.com/diffusion-face-relighting/difareli_code)** and place them under `examples/dat/reshadow/{diffuse,shadow}/<img_stem>/res_frame4.png` (supports either a single final frame `res_frame4.png` or full sequence `res_frame0..4.png`).
2. **Compute Smooth Spatial Shadow Maps**:
   ```bash
   python ./shadow_map/gen_smooth_Sdiff.py \
     --dataset_dir ../../examples/dat/ \
     --reshadow_dir ../../examples/dat/reshadow/ \
     --set valid
   ```
   This saves the normalized $C$-weighted spatial shadow maps to `examples/dat/shadow_diff_SS_with_c_simplified/<set>/<img_stem>.npy` (and visualizations in `vis/<set>/`).

<details>
<summary><b>⚙️ Click to expand <code>gen_smooth_Sdiff.py</code> arguments & directory layout</b></summary>

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset_dir` | *(Required)* | Root dataset directory (`../../examples/dat/`) |
| `--reshadow_dir` | `<dataset_dir>/reshadow` | Directory containing `shadow/` and `diffuse/` reshadowed frames |
| `--set` | `valid` | Dataset split (`train`, `valid`, `test`) |
| `--thres` | `-0.015` | Signed grayscale intensity difference threshold (`diffuse - shadow < thres`) |
| `--median_blur_ksize` | `5` | Kernel size for `cv2.medianBlur` smoothing |
| `--shadow_idx` / `--diffuse_idx` | `-1` | Frame index in the shadow/diffuse sequence (`-1` = last frame) |

```text
examples/dat/
├── reshadow/
│   ├── diffuse/{61992,max}/res_frame4.png            # Input diffuse reference frame(s)
│   └── shadow/{61992,max}/res_frame4.png             # Input shadow reference frame(s)
└── shadow_diff_SS_with_c_simplified/
    ├── valid/{61992.npy, max.npy}                    # Output (256, 256, 1) float32 shadow maps
    └── vis/valid/{61992.png, max.png}                # Diagnostic [Input | Weighted Shadow Map | Overlay]
```

</details>

---

## 3. Inference (`inference.py`)

Configure source/target pairs in `examples/pair.json`:
```json
{
    "pair": {
        "pair1": { "src": "max.png", "dst": "61992.png" },
        "pair2": { "src": "61992.png", "dst": "max.png" }
    }
}
```

Activate `difareli_pp_sampling` and run from `sample_scripts/single_shot/`:
```bash
conda activate difareli_pp_sampling
cd sample_scripts/single_shot
```

---

### 3.1 Reshadowing (Strengthen Shadow & Diffuse Shadow)

Interpolates the cast-shadow scalar $C$ toward **maximum shadow** (`--shadow_diff_inc_c`) or **diffuse / shadow-free** (`--shadow_diff_dec_c`):

- **Strengthen Shadow**:
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

- **Diffuse / Remove Shadow**:
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
> - **Custom Starting Shadow Scalar (`--reshadow_with_given_c <val>`)**: Overrides the starting shadow scalar $C \in [0, 1]$ instead of starting from $C_{\text{src}}$ (e.g., `--reshadow_with_given_c 0.15` to start strengthening from a lighter shadow).
> - **Smooth Shading Interpolation to Diffuse (`--itp_diffuse_sh`)**: Linearly interpolates the DECA SH shading from the source directional SH to the ambient-only diffuse SH across frames when used with `--force_diffuse_shading_for_reshadow`.

#### Results (`Reshadowing`)

| Input | Strengthen (`Relit \| Shadow \| Shading`) | Diffuse (`Relit \| Shadow \| Shading`) |
| :---: | :---: | :---: |
| <img src="examples/dat/images_aligned/valid/61992.png" height="115"> | <img src="examples/res/gifs/reshadow/shadow_61992_relit_shadow_shading.gif" height="115"> | <img src="examples/res/gifs/reshadow/diffuse_61992_relit_shadow_shading.gif" height="115"> |
| <img src="examples/dat/images_aligned/valid/max.png" height="115"> | <img src="examples/res/gifs/reshadow/shadow_max_relit_shadow_shading.gif" height="115"> | <img src="examples/res/gifs/reshadow/diffuse_max_relit_shadow_shading.gif" height="115"> |

---

### 3.2 Rotate SH Lighting (`rotateSH`)

Rotates the 3D Spherical Harmonics (SH) lighting 360° around `--rotate_sh_axis` (`0`, `1`, or `2`) with controlled cast-shadow intensity (`--relight_with_given_c`):

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

#### Results (`rotateSH`)

| Input | Output & Conditioning (`Relit \| Shadow \| Shading`) |
| :---: | :---: |
| <img src="examples/dat/images_aligned/valid/61992.png" height="128"> | <img src="examples/res/gifs/rotateSH/rotateSH_61992_relit_shadow_shading.gif" height="128"> |
| <img src="examples/dat/images_aligned/valid/max.png" height="128"> | <img src="examples/res/gifs/rotateSH/rotateSH_max_relit_shadow_shading.gif" height="128"> |

---

### 3.3 Target-SH Relighting (`targetSH`)

Relights a source portrait (`src`) by interpolating toward a target portrait's (`dst`) SH lighting and shadow scalar (`--relight_with_dst_c`):

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

#### Results (`targetSH`)

| Input (`Src \| Dst`) | Output & Conditioning (`Relit \| Shadow \| Shading`) |
| :---: | :---: |
| <img src="examples/dat/images_aligned/valid/61992.png" height="128"> <img src="examples/dat/images_aligned/valid/max.png" height="128"> | <img src="examples/res/gifs/targetSH/targetSH_61992_to_max_relit_shadow_shading.gif" height="128"> |
| <img src="examples/dat/images_aligned/valid/max.png" height="128"> <img src="examples/dat/images_aligned/valid/61992.png" height="128"> | <img src="examples/res/gifs/targetSH/targetSH_max_to_61992_relit_shadow_shading.gif" height="128"> |

> **Notes:**
> - **Frame 0 in `rotateSH` / `targetSH`**: Frame `0` (`res_frame0.png`) uses the 2D source shadow map (`shadow_diff_SS_with_c_simplified`), whereas frames `1 .. N-1` use 3D ray-traced target shadow masks. Discard frame `0` (or use `vid/*_no1.mp4`) for smooth animations.
> - **Image Credits**: Example portraits are sourced from the [FFHQ Dataset (`61992.png`)](https://github.com/nvlabs/ffhq-dataset) and [The New York Times (`max.jpg`)](https://www.nytimes.com/2021/10/29/sports/autoracing/max-verstappen-f1.html).

---

### 3.4 CLI Arguments & Output Structure

Single-subject modes (`reshadow/*`, `rotateSH_*`) save outputs to `<out_dir>/<src_stem>/`, while paired target-SH modes (`targetSH`) save to `<out_dir>/<src_stem>_to_<dst_stem>/`. Each folder contains `frames/` (`res_frame*.png`, `dst_ren_frame*.png`, `dst_shadm_shad_frame*.png`), `vid/` (`*.mp4`), and `res_desc.json`.

<details>
<summary><b>⚙️ Click to expand full <code>inference.py</code> CLI arguments & output directory tree</b></summary>

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset_root` / `--set` | *(Required)* | Preprocessed dataset root (`../../examples/dat/`) and split (`valid`, `train`, `test`) |
| `--log_dir` / `--cfg_name` / `--step` | *(Required)* | Checkpoint directory, config YAML (`DiFaReli++.yaml`), and step (`300000`) |
| `--sample_pair_json` | `None` | Path to JSON file specifying `src`/`dst` pairs (`--sample_pair_mode`: `pair`, `pairwise`, `list`) |
| `--src_dst` | `[]` | Explicit CLI pair override, e.g., `--src_dst max.png 61992.png` |
| `--itp_step` / `--batch_size` | `15` / `15` | Number of interpolation frames and forward-pass sub-batch size |
| `--rotate_sh` / `--rotate_sh_axis` | `False` / `0` | Rotate source SH lighting 360° around axis `0`, `1`, or `2` |
| `--force_src_shading_for_reshadow` | `False` | Keep source SH shading fixed across frames (for `--shadow_diff_inc_c`) |
| `--force_diffuse_shading_for_reshadow` | `False` | Use ambient-only diffuse SH shading on target frames (for `--shadow_diff_dec_c`) |
| `--itp_diffuse_sh` | `False` | Linearly interpolate SH from source to diffuse ambient shading |
| `--shadow_diff_inc_c` / `--shadow_diff_dec_c` | `False` | **Reshadow mode**: Lock source shadow mask and ramp $C \to 1.0$ or $C \to 0.0$ |
| `--reshadow_with_given_c` | `None` | Override starting $C \in [0, 1]$ for reshadowing (defaults to $C_{\text{src}}$) |
| `--relight_with_dst_c` | `False` | **Relight mode**: Scale ray-traced target shadow masks by $C_{\text{dst}}$ |
| `--relight_with_given_c` | `None` | **Relight mode**: Scale ray-traced target shadow masks by custom $C \in [0, 1]$ |
| `--relight_with_strongest_c` | `False` | **Relight mode**: Scale ray-traced target shadow masks by $C = 1.0$ |
| `--out_dir` / `--save_vid` / `--fps` | *(Req)* / `False` / `20` | Output directory, MP4 video export flag, and video frame rate |

```text
examples/res/
├── reshadow/
│   ├── shadow/{61992,max}/                  # Strengthen shadow (--shadow_diff_inc_c)
│   │   ├── frames/                          # res_frame*.png, {src,dst}_ren_frame*.png, {src,dst}_shadm_shad_frame*.png
│   │   ├── vid/                             # res{,_rt}.mp4, out{,_rt,_no1,_rt_no1}.mp4
│   │   └── res_desc.json
│   └── diffuse/{61992,max}/                 # Diffuse shadow (--shadow_diff_dec_c)
├── relight/
│   ├── rotateSH_a2/{61992,max}/             # Rotate SH lighting (--rotate_sh --rotate_sh_axis 2)
│   └── targetSH/{61992_to_max,max_to_61992}/# Paired Target-SH relighting (--relight_with_dst_c)
└── gifs/{reshadow,rotateSH,targetSH}/       # Showcase GIFs
```

</details>

---

### 3.5 Interactive Web Visualization (`vis.py`)

To inspect all generated frames, conditioning maps, and videos in your browser:

```bash
cd sample_scripts/single_shot
python vis.py --root_dir ../../examples/res/ --port 8888
```

---

## 4. Training

*TBD — Training instructions will be added here.*
