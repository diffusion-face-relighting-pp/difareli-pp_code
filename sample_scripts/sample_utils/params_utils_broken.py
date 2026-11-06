from collections import defaultdict
import glob
import os
import pickle
import sys
import cv2
import numpy as np
import pandas as pd
import torch as th

DECA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../cond_utils/DECA"))


def _ensure_deca_on_path():
    """Ensure the repository's DECA package directory is on sys.path."""
    if not os.path.isdir(DECA_DIR):
        raise FileNotFoundError(f"[#] DECA directory not found at: {DECA_DIR}")
    if DECA_DIR not in sys.path:
        sys.path.insert(0, DECA_DIR)
    return DECA_DIR


def load_flame_mask(parts=['face']):
    """Load FLAME vertex masks from the repository DECA data directory."""
    f_mask_path = os.path.join(DECA_DIR, "data", "FLAME_masks_face-id.pkl")
    v_mask_path = os.path.join(DECA_DIR, "data", "FLAME_masks.pkl")
    if not os.path.exists(f_mask_path):
        raise FileNotFoundError(f"[#] FLAME_masks_face-id.pkl not found at: {f_mask_path}")
    else:
        f_mask = np.load(f_mask_path, allow_pickle=True, encoding='latin1')

    if not os.path.exists(v_mask_path):
        raise FileNotFoundError(f"[#] FLAME_masks.pkl not found at: {v_mask_path}")
    else:
        v_mask = np.load(v_mask_path, allow_pickle=True, encoding='latin1')

    print("[#] Flame masks are loaded from :", f_mask_path, "and", v_mask_path)
    print("[#] Parts to be used for mask :", parts)

    mask={
        'v_mask':sum([v_mask[part].tolist() for part in parts], []),
        'f_mask':sum([f_mask[part].tolist() for part in parts], [])
    }
    return mask        


def init_deca(useTex=False, extractTex=True, device='cuda', 
              deca_mode='only_renderer', mask=None, deca_obj=None, rasterize_type='standard'):
    
    _ensure_deca_on_path()
    from decalib import deca
    from decalib.utils.config import cfg as deca_cfg
    deca_cfg.model.use_tex = useTex
    deca_cfg.rasterizer_type = rasterize_type
    deca_cfg.model.extract_tex = extractTex
    print("[#] Rasterizer type :", rasterize_type)
    deca_obj = deca.DECA(config = deca_cfg, device=device, mode=deca_mode, mask=mask)
    return deca_obj


def gen_vis_mask(n, mask, include, exclude):
    """Compute boolean index array of included minus excluded FLAME vertices."""
    inc = []
    for m in include:
        inc.append(mask[m])
    inc = np.unique(np.concatenate(inc, axis=0))

    exc = []
    for m in exclude:
        exc.append(mask[m])
    exc = np.unique(np.concatenate(exc, axis=0))
    vis_idx = np.setdiff1d(inc, exc)

    idx = np.zeros(n, dtype=bool)
    idx[vis_idx] = True
    return idx


def sh_to_ld(sh):
    """
    Roughly convert 1st-order SH coefficients to a directional light vector.
    """
    sh = sh.reshape(-1, 9, 3)
    ld = th.mean(sh[0:1, 1:4, :], dim=2)
    return ld


def render_shadow_mask_with_smooth(
    deca_params,
    light_dir,
    idx,
    n,
    useTex=False,
    extractTex=False,
    device="cuda",
    use_detail=False,
    deca_mode="only_renderer",
    mask=None,
    repeat=True,
    deca_obj=None,
    render_same_mask=False,
    point_light_round=1,
    ext_light_dir=None,
):
    """
    Render point-light cast-shadow masks and smooth shadow maps via DECA.
    """
    if deca_obj is None:
        print("[#] No deca_obj, re-init...")
        deca_obj = init_deca(
            useTex=useTex,
            extractTex=extractTex,
            device=device,
            deca_mode=deca_mode,
            mask=mask,
        )

    _ensure_deca_on_path()
    from decalib.datasets import datasets

    testdata = datasets.TestData(
        [deca_params["raw_image_path"][0]],
        iscrop=True,
        face_detector="fan",
        sample_step=10,
    )

    if repeat:
        codedict = {
            "shape": deca_params["shape"][[idx]].repeat(n, 1).to(device).float(),
            "pose": deca_params["pose"][[idx]].repeat(n, 1).to(device).float(),
            "exp": deca_params["exp"][[idx]].repeat(n, 1).to(device).float(),
            "cam": deca_params["cam"][[idx]].repeat(n, 1).to(device).float(),
            "tform": testdata[idx]["tform"][None].repeat(n, 1, 1).to(device).float(),
            "images": testdata[idx]["image"].to(device)[None, ...].float().repeat(n, 1, 1, 1),
            "tex": deca_params["albedo"][[idx]].repeat(n, 1).to(device).float(),
            "detail": deca_params["detail"][[idx]].repeat(n, 1).to(device).float(),
        }
        original_image = (
            deca_params["raw_image"][[idx]].to(device).float().repeat(n, 1, 1, 1) / 255.0
        )
    else:
        codedict = {
            "shape": th.tensor(deca_params["shape"]).to(device).float(),
            "pose": th.tensor(deca_params["pose"]).to(device).float(),
            "exp": th.tensor(deca_params["exp"]).to(device).float(),
            "cam": th.tensor(deca_params["cam"]).to(device).float(),
            "tform": testdata[idx]["tform"][None].to(device).float(),
            "images": th.stack(
                [testdata[i]["image"] for i in range(len(deca_params["raw_image_path"]))]
            ).to(device).float(),
            "tex": th.tensor(deca_params["albedo"]).to(device).float(),
            "detail": deca_params["detail"].to(device).float(),
        }
        original_image = deca_params["raw_image"].to(device).float() / 255.0

    tform_inv = th.inverse(codedict["tform"]).transpose(1, 2)

    all_shadow_mask = []
    for _ in range(point_light_round):
        codedict["point_light"] = (
            th.tensor(ext_light_dir if ext_light_dir is not None else light_dir)
            .to(device)
            .reshape(-1, 3)
            .float()
        )
        orig_opdict, orig_visdict = deca_obj.decode(
            codedict,
            render_orig=True,
            original_image=original_image,
            tform=tform_inv,
            use_template=False,
            mean_cam=None,
            use_detail=use_detail,
            rotate_normals=False,
            render_point_light=True,
            return_normal_and_pos=render_same_mask,
        )
        orig_visdict.update(orig_opdict)
        shadow_mask = orig_visdict["point_light_images"]  # (B, 3, H, W) in [0, 1]
        all_shadow_mask.append(shadow_mask.cpu())

    shadow_mask = th.mean(th.stack(all_shadow_mask, dim=0), dim=0)

    # Smooth shadow mask using a Gaussian kernel
    shadow_mask_smooth = []
    for k in range(shadow_mask.shape[0]):
        s_k = shadow_mask[k].permute(1, 2, 0).cpu().numpy()
        s_k = cv2.GaussianBlur(
            s_k,
            ksize=(9, 9),
            sigmaX=7,
            sigmaY=7,
            borderType=cv2.BORDER_DEFAULT,
        )
        shadow_mask_smooth.append(th.from_numpy(s_k).permute(2, 0, 1))
    shadow_mask_smooth = th.stack(shadow_mask_smooth, dim=0).to(device)

    return shadow_mask, shadow_mask_smooth, orig_visdict


def render_deca(
    deca_params,
    idx,
    n,
    render_mode="shape",
    useTex=False,
    extractTex=False,
    device="cuda",
    avg_dict=None,
    rotate_normals=False,
    use_detail=False,
    deca_mode="only_renderer",
    mask=None,
    repeat=True,
    deca_obj=None,
):
    """
    Render DECA shape conditioning images (`shape_images`) and return `(rendered_image, orig_visdict)`.
    """
    if deca_obj is None:
        print("[#] No deca_obj, re-init...")
        deca_obj = init_deca(
            useTex=useTex,
            extractTex=extractTex,
            device=device,
            deca_mode=deca_mode,
            mask=mask,
        )

    _ensure_deca_on_path()
    from decalib.datasets import datasets

    testdata = datasets.TestData(
        [deca_params["raw_image_path"][0]],
        iscrop=True,
        face_detector="fan",
        sample_step=10,
    )

    if repeat:
        codedict = {
            "shape": deca_params["shape"][[idx]].repeat(n, 1).to(device).float(),
            "pose": deca_params["pose"][[idx]].repeat(n, 1).to(device).float(),
            "exp": deca_params["exp"][[idx]].repeat(n, 1).to(device).float(),
            "cam": deca_params["cam"][[idx]].repeat(n, 1).to(device).float(),
            "light": th.tensor(deca_params["light"]).to(device).reshape(-1, 9, 3).float(),
            "tform": testdata[idx]["tform"][None].repeat(n, 1, 1).to(device).float(),
            "images": testdata[idx]["image"].to(device)[None, ...].float().repeat(n, 1, 1, 1),
            "tex": deca_params["albedo"][[idx]].repeat(n, 1).to(device).float(),
            "detail": deca_params["detail"][[idx]].repeat(n, 1).to(device).float(),
        }
        original_image = (
            deca_params["raw_image"][[idx]].to(device).float().repeat(n, 1, 1, 1) / 255.0
        )
    else:
        codedict = {
            "shape": th.tensor(deca_params["shape"]).to(device).float(),
            "pose": th.tensor(deca_params["pose"]).to(device).float(),
            "exp": th.tensor(deca_params["exp"]).to(device).float(),
            "cam": th.tensor(deca_params["cam"]).to(device).float(),
            "light": th.tensor(deca_params["light"]).to(device).reshape(-1, 9, 3).float(),
            "tform": testdata[idx]["tform"][None].to(device).float(),
            "images": th.stack(
                [testdata[i]["image"] for i in range(len(deca_params["raw_image_path"]))]
            ).to(device).float(),
            "tex": th.tensor(deca_params["albedo"]).to(device).float(),
            "detail": deca_params["detail"].to(device).float(),
        }
        original_image = deca_params["raw_image"].to(device).float() / 255.0

    if rotate_normals:
        codedict["R_normals"] = th.tensor(deca_params["R_normals"]).to(device).float()

    if render_mode == "shape":
        use_template = False
        mean_cam = None
        tform_inv = th.inverse(codedict["tform"]).transpose(1, 2)
    elif render_mode == "template_shape":
        use_template = True
        mean_cam = th.tensor(avg_dict["cam"])[None, ...].repeat(n, 1).to(device).float()
        tform = (
            th.tensor(avg_dict["tform"])[None, ...]
            .repeat(n, 1)
            .to(device)
            .reshape(-1, 3, 3)
            .float()
        )
        tform_inv = th.inverse(tform).transpose(1, 2)
    else:
        raise NotImplementedError(f"Unsupported render_mode: {render_mode}")

    orig_opdict, orig_visdict = deca_obj.decode(
        codedict,
        render_orig=True,
        original_image=original_image,
        tform=tform_inv,
        use_template=use_template,
        mean_cam=mean_cam,
        use_detail=use_detail,
        rotate_normals=rotate_normals,
    )
    orig_visdict.update(orig_opdict)
    rendered_image = orig_visdict["shape_images"]
    return rendered_image, orig_visdict


# ==============================================================================
# Parameter I/O Helpers
# ==============================================================================

def read_params(path):
    params = pd.read_csv(path, header=None, sep=" ", index_col=False, lineterminator="\n")
    params.rename(columns={0: "img_name"}, inplace=True)
    return params.set_index("img_name").T.to_dict("list")


def swap_key(params):
    params_s = defaultdict(dict)
    for params_name, v in params.items():
        for img_name, params_value in v.items():
            params_s[img_name][params_name] = np.array(params_value).astype(np.float64)
    return params_s


def normalize(arr, min_val=None, max_val=None, a=-1, b=1):
    if max_val is None and min_val is None:
        max_val = np.max(arr, axis=0)
        min_val = np.min(arr, axis=0)
    arr_norm = ((b - a) * (arr - min_val) / (max_val - min_val)) + a
    return arr_norm, min_val, max_val


def denormalize(arr_norm, min_val, max_val, a=-1, b=1):
    return (((arr_norm - a) * (max_val - min_val)) / (b - a)) + min_val


def load_params(path, params_key):
    params = {}
    for k in params_key:
        for p in glob.glob(f"{path}/*{k}-anno.txt"):
            if k in p:
                print(f"Key=> {k} : Filename=>{p}")
                params[k] = read_params(path=p)

    params_s = swap_key(params)
    all_params = []
    for img_name in params_s:
        each_img = [params_s[img_name][k] for k in params_key]
        all_params.append(np.concatenate(each_img))
    all_params = np.stack(all_params, axis=0)
    return params_s, all_params


def preprocess_cond(deca_params, k, cfg):
    if k != "light":
        return deca_params
    num_SH = cfg.relighting.num_SH
    params = deca_params[k]
    params = params.reshape(params.shape[0], 9, 3)
    params = params[:, :num_SH, :]
    return params.reshape(params.shape[0], -1)


def write_params(path, params, keys):
    tmp = {}
    for k in keys:
        if th.is_tensor(params[k]):
            tmp[k] = params[k].cpu().numpy()
        else:
            tmp[k] = params[k]
    np.save(file=path, arr=tmp)
