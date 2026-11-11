# Training DiFaReli++

This guide covers the complete 4-stage pipeline for training **DiFaReli++**:
1. **[Step 1: Prepare FFHQ & Compute Shadow Maps](#step-1-prepare-ffhq--compute-shadow-maps)** — Extract face/pupil segmentation, DECA parameters/shadings, and compute spatial shadow maps (`shadow_diff_SS_with_c_simplified`) via DiFaReli reshadowing.
2. **[Step 2: Train DiFaReli++ Diffusion Model (`train_difarelipp_diffusion.py`)](#step-2-train-difareli-diffusion-model-train_difarelipp_diffusionpy)** — Train the diffusion-based relighting model conditioned on the new spatial shadow maps and the part-wise face segmentation mask.
3. **[Step 3: Generate Paired Relit Dataset](#step-3-generate-paired-relit-dataset)** — Use the trained DiFaReli++ diffusion model to synthesize paired relit portraits (`<src_id>_input.png` and `<src_id>_<dst_id>_relit.png`) along with their paired DECA shadings and shadow maps.
4. **[Step 4: Train DiFaReli++ Single-Shot Model (`train_difarelipp_singleshot.py`)](#step-4-train-difareli-single-shot-model-train_difarelipp_singleshotpy)** — Train the single-shot (non-DPM) generator on the paired relit dataset.

---

## Step 1: Prepare FFHQ & Compute Shadow Maps

Follow the preprocessing steps in [README.md — Section 2](README.md#2-preparing-the-data-for-inference) on the FFHQ training split (`--set train`):
1. Run [`create_dataset.py`](dataset_generation/prepare_relighting_data/create_dataset.py) (`--faceseg --deca --arcface --shadow --set train`) to generate `face_segment_with_pupil/train/anno/`, `rendered_images/deca_masked_face_images_woclip/train/`, and `params/train/`.
2. Synthesize `shadow` (strong shadow) and `diffuse` (shadow-free) reference frames for each training image using **[DiFaReli](https://github.com/diffusion-face-relighting/difareli_code)** and run [`gen_smooth_Sdiff.py`](dataset_generation/prepare_relighting_data/shadow_map/gen_smooth_Sdiff.py) (`--set train`) to save the $C$-weighted spatial shadow maps under `shadow_diff_SS_with_c_simplified/train/`.

### 📂 Diffusion Dataset Structure & Filename Convention (`ffhq_256_with_anno`)

```text
<path_to_dataset>/ffhq_256_with_anno/
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

## Step 2: Train DiFaReli++ Diffusion Model on  (`train_difarelipp_diffusion.py`)

1. **Configure Dataset Paths**:
   Update the `dataset:` section in [`config/DiFaReli++/DiFaReli++_diffusion.yaml`](config/DiFaReli++/DiFaReli++_diffusion.yaml):
   ```yaml
   dataset:
     root_path: "<path_to_dataset>/"
     training_data: "ffhq_256_with_anno"
     deca_dir: "<path_to_dataset>/ffhq_256_with_anno/params/"
     data_dir: "<path_to_dataset>/ffhq_256_with_anno/ffhq_256/"
     face_segment_dir: "<path_to_dataset>/ffhq_256_with_anno/face_segment_with_pupil/"
     deca_rendered_dir: "<path_to_dataset>/ffhq_256_with_anno/rendered_images/"
     shadow_diff_dir: "<path_to_dataset>/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/"
   ```

2. **Run Diffusion Training**:
   ```bash
   conda activate difareli_pp_training

   python ./train_scripts/train_difarelipp_diffusion.py \
     --train.log_dir <path_to_ckpt>/DiFaReli++_diffusion/ \
     --train.save_ckpt_dir <path_to_ckpt>/DiFaReli++_diffusion/ \
     --train.batch_size <batch_size> \
     --train.n_gpus <n_gpus> \
     --cfg ./config/DiFaReli++/DiFaReli++_diffusion.yaml \
     --train.sampling_interval <sampling_interval> \
     --train.save_interval <save_interval>
   ```

---

## Step 3: Generate Paired Relit Dataset

Once the DiFaReli++ diffusion model is trained, you can use it to generate a paired relit training dataset.

> **Note:** The generated dataset can be constructed in various ways (e.g., paired reshadowing, relighting toward random target lighting, or rotating the input lighting); the key requirement is to save the corresponding target/interpolated conditions and parameters (e.g., SH coefficients, shadow scalar $C$, shading references, and shadow maps) used during relighting. In this guide, we follow the paper's setup by randomly sampling target lighting from the FFHQ training set to relight each input portrait.

1. **Synthesize Relit Pairs**:
   - For each source subject `<src_id>` (e.g., `0`) and randomly paired target subject `<dst_id>` (e.g., `18029`), relight `<src_id>` toward `<dst_id>`'s target SH lighting and shadow map using the trained DiFaReli++ diffusion model.
   - Save both portraits in `images/train/`:
     - Source image: `<src_id>_input.png` (e.g., `0_input.png`)
     - Relit target image: `<src_id>_<dst_id>_relit.png` (e.g., `0_18029_relit.png`)
2. **Save the Conditionings & Parameters Used During Relighting**:
   - You **do not need to recompute** DECA, ArcFace, or reshadowing on the relit images. Simply make sure the conditionings and interpolated/target parameters constructed during relighting are saved under the matching paired keys:
   - **Shading References (`rendered_images/deca_masked_face_images_woclip/train/`)**:
     - `<src_id>_input.npy`: Source shading reference (`<src_id>`'s 3D FLAME mesh rendered under `<src_id>`'s SH lighting).
     - `<src_id>_<dst_id>_relit.npy`: Target shading reference used during relighting (`<src_id>`'s 3D FLAME mesh rendered under the target/interpolated `<dst_id>` SH lighting).
   - **Spatial Shadow Maps (`shadow_diff_SS_with_c_simplified_npy/train/`)**:
     - `<src_id>_input.npy`: Source $C$-weighted shadow map.
     - `<src_id>_<dst_id>_relit.npy`: Target $C$-weighted shadow map used during relighting.
   - **Paired Parameter Files (`params_recreate/train/ffhq-train-*-anno.txt`)**:
     - Make sure the parameters used during relighting are saved under keys `<src_id>.png` (for `<src_id>_input.png`) and `<src_id>_<dst_id>.png` (for `<src_id>_<dst_id>_relit.png`).
     - Identity and geometry parameters (`shape`, `pose`, `exp`, `cam`, `faceemb`) for `<src_id>_<dst_id>.png` are copied directly from `<src_id>.png`, whereas `light` (and `shadow`) store the target/interpolated values used during relighting.

### 📂 Single-Shot Paired Dataset Structure & Filename Convention (`Generated_Dataset_cast_shadows`)

```text
<path_to_dataset>/Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/
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
│           ├── 0_input.npy                              # (256, 256, 3) float32 source shading references
│           ├── 0_18029_relit.npy                        # (256, 256, 3) float32 target shading references (<src_id> mesh + <dst_id> SH)
│           └── ...
├── shadow_diff_SS_with_c_simplified_npy/
│   └── train/
│       ├── 0_input.npy                                  # (256, 256, 1) float32 source C-weighted shadow map
│       ├── 0_18029_relit.npy                            # (256, 256, 1) float32 target C-weighted shadow map
│       └── ...
└── params_recreate/
    └── train/                                           # Space-delimited text files keyed by "<src_id>.png" & "<src_id>_<dst_id>.png":
        ├── ffhq-train-shape-anno.txt                    
        ├── ffhq-train-pose-anno.txt                     
        ├── ffhq-train-exp-anno.txt                      
        ├── ffhq-train-cam-anno.txt                      
        ├── ffhq-train-faceemb-anno.txt                  
        ├── ffhq-train-light-anno.txt                    
        └── ffhq-train-{shadow,albedo,detail,tform}-anno.txt
```

---

## Step 4: Train DiFaReli++ Single-Shot Model (`train_difarelipp_singleshot.py`)

1. **Configure Dataset Paths**:
   Update the `dataset:` section in [`config/DiFaReli++/DiFaReli++.yaml`](config/DiFaReli++/DiFaReli++.yaml):
   ```yaml
   dataset:
     root_path: "<path_to_dataset>/"
     training_data: "Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/"
     deca_dir: "<path_to_dataset>/Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/params_recreate/"
     data_dir: "<path_to_dataset>/Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/images/"
     face_segment_dir: "<path_to_dataset>/Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/face_segment/"
     deca_rendered_dir: "<path_to_dataset>/Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/rendered_images/"
     shadow_diff_dir: "<path_to_dataset>/Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/shadow_diff_SS_with_c_simplified_npy/"
   ```

2. **Run Single-Shot Training**:
   ```bash
   conda activate difareli_pp_training

   python ./train_scripts/train_difarelipp_singleshot.py \
     --train.log_dir <path_to_ckpt>/DiFaReli++/ \
     --train.save_ckpt_dir <path_to_ckpt>/DiFaReli++/ \
     --train.batch_size <batch_size> \
     --train.n_gpus <n_gpus> \
     --cfg ./config/DiFaReli++/DiFaReli++.yaml \
     --train.sampling_interval <sampling_interval> \
     --train.save_interval <save_interval>
   ```

### 💾 Saved Checkpoint Files

Both training scripts save model, EMA, and optimizer checkpoints at every `--train.save_interval` steps in `--train.save_ckpt_dir`:

```text
<path_to_ckpt>/DiFaReli++/
├── ImgCond_model<step>.pt                        # Generator (UNet) weights at <step>
├── ImgCond_ema_0.9999_<step>.pt                  # Generator (UNet) EMA weights (used for inference)
├── ImgEncoder_model<step>.pt                     # Spatial condition encoder weights at <step>
├── ImgEncoder_ema_0.9999_<step>.pt               # Spatial condition encoder EMA weights (used for inference)
└── opt<step>.pt                                  # AdamW optimizer state at <step>
```
