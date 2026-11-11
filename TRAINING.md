# Training DiFaReli++

This guide covers the complete 4-stage pipeline for training **DiFaReli++**:
1. **[Step 1: Prepare FFHQ & Compute Shadow Maps](#step-1-prepare-ffhq--compute-shadow-maps)** — Extract face/pupil segmentation, DECA parameters/shadings, and compute spatial shadow maps (`shadow_diff_SS_with_c_simplified`) via DiFaReli reshadowing.
2. **[Step 2: Train DiFaReli++ Diffusion Model (`train_difarelipp_diffusion.py`)](#step-2-train-difareli-diffusion-model-train_difarelipp_diffusionpy)** — Train the diffusion-based relighting model conditioned on the spatial shadow maps.
3. **[Step 3: Generate Paired Relit Dataset](#step-3-generate-paired-relit-dataset)** — Use the trained DiFaReli++ diffusion model to synthesize paired relit portraits (`<src_id>_input.png` and `<src_id>_<dst_id>_relit.png`) along with their paired DECA shadings and shadow maps.
4. **[Step 4: Train DiFaReli++ Single-Shot Model (`train_difarelipp_singleshot.py`)](#step-4-train-difareli-single-shot-model-train_difarelipp_singleshotpy)** — Train the single-shot (non-DPM) generator on the paired relit dataset.

---

## Step 1: Prepare FFHQ & Compute Shadow Maps

Follow the preprocessing steps in [README.md — Section 2](README.md#2-preparing-the-data-for-inference) on the FFHQ training split (`--set train`):
1. Run [`create_dataset.py`](dataset_generation/prepare_relighting_data/create_dataset.py) (`--faceseg --deca --arcface --shadow --set train`) to generate `face_segment_with_pupil/train/anno/`, `rendered_images/deca_masked_face_images_woclip/train/`, and `params/train/`.
2. Synthesize `shadow` (strong shadow) and `diffuse` (shadow-free) reference frames for each training image using **[DiFaReli](https://github.com/diffusion-face-relighting/difareli_code)** and run [`gen_smooth_Sdiff.py`](dataset_generation/prepare_relighting_data/shadow_map/gen_smooth_Sdiff.py) (`--set train`) to save the $C$-weighted spatial shadow maps under `shadow_diff_SS_with_c_simplified/train/`.

### 📂 Diffusion Dataset Structure & Filename Convention (`ffhq_256_with_anno`)

```text
/data/mint/DPM_Dataset/ffhq_256_with_anno/
├── ffhq_256/
│   └── train/
│       ├── 0.jpg                                        # Aligned 256x256 portrait (<id>.jpg)
│       ├── 10000.jpg
│       └── ...
├── face_segment_with_pupil/
│   └── train/anno/
│       ├── anno_0.png                                   # Face + pupil parsing mask (anno_<id>.png)
│       ├── anno_10000.png
│       └── ...
├── rendered_images/
│   └── deca_masked_face_images_woclip/
│       └── train/
│           ├── 0.npy                                    # (256, 256, 3) float32 unclipped DECA Lambertian shading (<id>.npy)
│           ├── 10000.npy
│           └── ...
├── shadow_diff_SS_with_c_simplified/
│   └── train/
│       ├── 0.npy                                        # (256, 256, 1) float32 C-weighted spatial shadow map (<id>.npy)
│       ├── 10000.npy
│       └── ...
└── params/
    └── train/                                           # Space-delimited text files keyed by "<id>.jpg":
        ├── ffhq-train-shape-anno.txt                    # e.g., "0.jpg 0.52372354 0.9224942 ..." (100-D FLAME shape)
        ├── ffhq-train-pose-anno.txt                     # 6-D FLAME pose
        ├── ffhq-train-exp-anno.txt                      # 50-D FLAME expression
        ├── ffhq-train-cam-anno.txt                      # 3-D camera parameters
        ├── ffhq-train-faceemb-anno.txt                  # 512-D ArcFace identity embedding
        ├── ffhq-train-light-anno.txt                    # 27-D Spherical Harmonics (SH) lighting
        ├── ffhq-train-shadow-anno.txt                   # 1-D DiffAE shadow scalar C
        ├── ffhq-train-albedo-anno.txt                   # 100-D FLAME albedo
        ├── ffhq-train-detail-anno.txt                   # 128-D DECA detail code
        └── ffhq-train-tform-anno.txt                    # 3x3 similarity transform matrix
```

---

## Step 2: Train DiFaReli++ Diffusion Model (`train_difarelipp_diffusion.py`)

1. **Configure Dataset Paths**:
   Check the `dataset:` section in [`config/DiFaReli++/DiFaReli++_diffusion.yaml`](config/DiFaReli++/DiFaReli++_diffusion.yaml):
   ```yaml
   dataset:
     root_path: "/data/mint/DPM_Dataset/"
     training_data: "ffhq_256_with_anno"
     deca_dir: "/data/mint/DPM_Dataset/ffhq_256_with_anno/params/"
     data_dir: "/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/"
     face_segment_dir: "/data/mint/DPM_Dataset/ffhq_256_with_anno/face_segment_with_pupil/"
     deca_rendered_dir: "/data/mint/DPM_Dataset/ffhq_256_with_anno/rendered_images/"
     shadow_diff_dir: "/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/"
   ```

2. **Run Diffusion Training**:
   ```bash
   conda activate difareli_pp_training

   python ./train_scripts/train_difarelipp_diffusion.py \
     --train.log_dir /data/mint/DiFaReli++/ckpt/DiFaReli++_test/ \
     --train.save_ckpt_dir /data/mint/DiFaReli++/ckpt/DiFaReli++_test/ \
     --train.batch_size 1 \
     --train.n_gpus 4 \
     --cfg ./config/DiFaReli++/DiFaReli++_diffusion.yaml \
     --train.sampling_interval 50000 \
     --train.save_interval 100000
   ```

---

## Step 3: Generate Paired Relit Dataset

Once the DiFaReli++ diffusion model is trained, use it to generate a paired relit training dataset (`Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/`):

1. **Synthesize Relit Pairs**:
   - For each source subject `<src_id>` (e.g., `0`) and randomly paired target subject `<dst_id>` (e.g., `18029`), relight `<src_id>` toward `<dst_id>`'s target SH lighting and shadow map using the trained DiFaReli++ diffusion model.
   - Save both portraits in `images/train/`:
     - Source image: `<src_id>_input.png` (e.g., `0_input.png`)
     - Relit target image: `<src_id>_<dst_id>_relit.png` (e.g., `0_18029_relit.png`)
2. **Prepare Paired Conditionings & Parameters**:
   - **Lambertian Shadings (`rendered_images/deca_masked_face_images_woclip/train/`)**:
     - `<src_id>_input.npy`: `<src_id>`'s 3D FLAME mesh rendered under `<src_id>`'s SH lighting.
     - `<src_id>_<dst_id>_relit.npy`: `<src_id>`'s 3D FLAME mesh rendered under `<dst_id>`'s SH lighting.
   - **Spatial Shadow Maps (`shadow_diff_SS_with_c_simplified_npy/train/`)**:
     - Run reshadowing + [`gen_smooth_Sdiff.py`](dataset_generation/prepare_relighting_data/shadow_map/gen_smooth_Sdiff.py) on `<src_id>_input.png` and `<src_id>_<dst_id>_relit.png` to obtain `<src_id>_input.npy` and `<src_id>_<dst_id>_relit.npy`.
   - **Paired Parameter Files (`params_recreate/train/ffhq-train-*-anno.txt`)**:
     - Keyed by `<src_id>.png` (for `<src_id>_input.png`) and `<src_id>_<dst_id>.png` (for `<src_id>_<dst_id>_relit.png`).
     - Identity/geometry parameters (`shape`, `pose`, `exp`, `cam`, `faceemb`) for `<src_id>_<dst_id>.png` must match `<src_id>.png`, whereas `light` stores `<dst_id>.png`'s target SH coefficients.

### 📂 Single-Shot Paired Dataset Structure & Filename Convention (`Generated_Dataset_cast_shadows`)

```text
/data/mint/DPM_Dataset/Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/
├── images/
│   └── train/
│       ├── 0_input.png                                  # Source portrait (<src_id>_input.png)
│       ├── 0_18029_relit.png                            # Relit target portrait (<src_id>_<dst_id>_relit.png)
│       ├── 10000_input.png
│       ├── 10000_47549_relit.png
│       └── ...
├── rendered_images/
│   └── deca_masked_face_images_woclip/
│       └── train/
│           ├── 0_input.npy                              # (256, 256, 3) float32 source Lambertian shading
│           ├── 0_18029_relit.npy                        # (256, 256, 3) float32 target Lambertian shading (<src_id> mesh + <dst_id> SH)
│           └── ...
├── shadow_diff_SS_with_c_simplified_npy/
│   └── train/
│       ├── 0_input.npy                                  # (256, 256, 1) float32 source C-weighted shadow map
│       ├── 0_18029_relit.npy                            # (256, 256, 1) float32 target C-weighted shadow map
│       └── ...
└── params_recreate/
    └── train/                                           # Space-delimited text files keyed by "<src_id>.png" & "<src_id>_<dst_id>.png":
        ├── ffhq-train-shape-anno.txt                    # 0.png and 0_18029.png share 0.png's 100-D FLAME shape
        ├── ffhq-train-pose-anno.txt                     # 0.png and 0_18029.png share 0.png's 6-D FLAME pose
        ├── ffhq-train-exp-anno.txt                      # 0.png and 0_18029.png share 0.png's 50-D FLAME expression
        ├── ffhq-train-cam-anno.txt                      # 0.png and 0_18029.png share 0.png's 3-D camera params
        ├── ffhq-train-faceemb-anno.txt                  # 0.png and 0_18029.png share 0.png's 512-D ArcFace embedding
        ├── ffhq-train-light-anno.txt                    # 0.png has 0.png's 27-D SH; 0_18029.png has 18029.png's 27-D SH
        └── ffhq-train-{shadow,albedo,detail,tform}-anno.txt
```

---

## Step 4: Train DiFaReli++ Single-Shot Model (`train_difarelipp_singleshot.py`)

1. **Configure Dataset Paths**:
   Check the `dataset:` section in [`config/DiFaReli++/DiFaReli++.yaml`](config/DiFaReli++/DiFaReli++.yaml):
   ```yaml
   dataset:
     root_path: "/data/mint/DPM_Dataset/"
     training_data: "Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/"
     deca_dir: "/data/mint/DPM_Dataset/Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/params_recreate/"
     data_dir: "/data/mint/DPM_Dataset/Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/images/"
     face_segment_dir: "/data/mint/DPM_Dataset/Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/face_segment/"
     deca_rendered_dir: "/data/mint/DPM_Dataset/Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/rendered_images/"
     shadow_diff_dir: "/data/mint/DPM_Dataset/Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/shadow_diff_SS_with_c_simplified_npy/"
   ```

2. **Run Single-Shot Training**:
   ```bash
   conda activate difareli_pp_training

   python ./train_scripts/train_difarelipp_singleshot.py \
     --train.log_dir /data/mint/DiFaReli++/ckpt/DiFaReli++_test/ \
     --train.save_ckpt_dir /data/mint/DiFaReli++/ckpt/DiFaReli++_test/ \
     --train.batch_size 3 \
     --train.n_gpus 1 \
     --cfg ./config/DiFaReli++/DiFaReli++.yaml \
     --train.sampling_interval 50000 \
     --train.save_interval 100000
   ```

### 💾 Saved Checkpoint Files

Both training scripts save model, EMA, and optimizer checkpoints at every `--train.save_interval` steps in `--train.save_ckpt_dir`:

```text
/data/mint/DiFaReli++/ckpt/DiFaReli++_test/
├── ImgCond_model<step>.pt                        # Generator (UNet) weights at <step>
├── ImgCond_ema_0.9999_<step>.pt                  # Generator (UNet) EMA weights (used for inference)
├── ImgEncoder_model<step>.pt                     # Spatial condition encoder weights at <step>
├── ImgEncoder_ema_0.9999_<step>.pt               # Spatial condition encoder EMA weights (used for inference)
└── opt<step>.pt                                  # AdamW optimizer state at <step>
```
