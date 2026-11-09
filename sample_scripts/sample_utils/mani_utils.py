import itertools
import json
import os
import numpy as np
import pyshtools as pysh
import torch as th
import tqdm
from scipy.spatial.transform import Rotation as R

try:
    from sample_utils import file_utils
except ImportError:
    import file_utils


# ==============================================================================
# Interpolation Utilities
# ==============================================================================

def lerp(r, src, dst):
    """Linear interpolation between `src` and `dst` with weight `r` in [0, 1]."""
    return ((1.0 - r) * src) + (r * dst)


def slerp(r, src, dst):
    """Spherical linear interpolation between `src` and `dst` with weight `r` in [0, 1]."""
    low, high, val = src, dst, r
    low_norm = low / th.norm(low, dim=1, keepdim=True)
    high_norm = high / th.norm(high, dim=1, keepdim=True)
    omega = th.acos(th.clamp((low_norm * high_norm).sum(1), -1.0, 1.0))
    so = th.sin(omega)
    return (th.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (th.sin(val * omega) / so).unsqueeze(1) * high


def interp_cond(src_cond, dst_cond, n_step, interp_fn):
    """
    Interpolate between `src_cond` [B, C] and `dst_cond` [B, C] over `n_step` steps.
    """
    print(f"[#] Interpolate with {interp_fn}")
    if n_step <= 1:
        return src_cond

    r_interp = np.linspace(0, 1, num=n_step)
    interp = []
    for r in r_interp:
        tmp = interp_fn(r=r, src=src_cond, dst=dst_cond)
        if th.is_tensor(tmp):
            tmp = tmp.detach().cpu().numpy()
        interp.append(tmp.copy())

    return np.concatenate(interp, axis=0)


def iter_interp_cond(
    cond,
    src_idx,
    dst_idx,
    n_step,
    interp_set,
    interp_fn,
    add_remove_shadow=False,
    vary_shadow=None,
    vary_shadow_nobound=None,
    vary_shadow_range=None,
    add_shadow_to=None,
    remove_shadow_to=None,
):
    """
    Interpolate condition tensors across all keys in `interp_set`.
    """
    out_interp = {}
    for itp in interp_set:
        assert itp in cond.keys(), f"Key '{itp}' not found in cond."
        assert src_idx < len(cond[itp]) and dst_idx < len(cond[itp])
        if itp == "shadow":
            if vary_shadow:
                with open(cond["misc"]["args"].sample_pair_json, "r") as f:
                    sample_pairs = json.load(f)[cond["misc"]["args"].sample_pair_mode]
                    pair_idx = list(sample_pairs.keys())[cond["sample_idx"]]
                    assert sample_pairs[pair_idx]["src"] == cond["image_name"][src_idx]
                    min_c = sample_pairs[pair_idx]["rmv_shadow_bound"]
                    max_c = sample_pairs[pair_idx]["add_shadow_bound"]
                    n_to_min = n_step // 2 if n_step % 2 == 0 else n_step // 2 + 1
                    n_to_max = n_step // 2
                    min_c = np.linspace(min_c, cond[itp][src_idx][0], n_to_min)[..., None]
                    max_c = np.linspace(cond[itp][src_idx][0], max_c, n_to_max)[..., None]
                    interp = np.concatenate((max_c[0:1, :], min_c, max_c[1:, :]), axis=0)
            elif vary_shadow_range:
                assert len(vary_shadow_range) == 2
                min_c, max_c = float(vary_shadow_range[0]), float(vary_shadow_range[1])
                n_to_min = n_step // 2 if n_step % 2 == 0 else n_step // 2 + 1
                n_to_max = n_step // 2
                min_c = np.linspace(min_c, cond[itp][src_idx][0], n_to_min)[..., None]
                max_c = np.linspace(cond[itp][src_idx][0], max_c, n_to_max)[..., None]
                interp = np.concatenate((max_c[0:1, :], min_c, max_c[1:, :]), axis=0)
            elif vary_shadow_nobound:
                min_c, max_c = -10.0, 10.0
                if n_step % 2 == 0:
                    min_c = np.linspace(min_c, cond[itp][src_idx][0], n_step // 2)[..., None]
                    max_c = np.linspace(cond[itp][src_idx][0], max_c, (n_step // 2) + 1)[..., None]
                else:
                    min_c = np.linspace(min_c, cond[itp][src_idx][0], (n_step // 2) + 1)[..., None]
                    max_c = np.linspace(cond[itp][src_idx][0], max_c, (n_step // 2) + 1)[..., None]
                interp = np.concatenate((min_c, max_c[1:, :]), axis=0)
            elif add_remove_shadow:
                if add_shadow_to is not None:
                    interp = np.linspace(cond[itp][src_idx][0], add_shadow_to, n_step)[..., None]
                elif remove_shadow_to is not None:
                    interp = np.linspace(cond[itp][src_idx][0], remove_shadow_to, n_step)[..., None]
                else:
                    raise NotImplementedError("Specify add_shadow_to or remove_shadow_to.")
            else:
                raise NotImplementedError(
                    "With itp='shadow', specify a shadow variation mode."
                )
        else:
            if isinstance(cond[itp], list):
                interp = []
                for i in range(len(cond[itp])):
                    assert cond[itp][i][[src_idx]].shape == cond[itp][i][[dst_idx]].shape
                    interp.append(
                        interp_cond(
                            src_cond=cond[itp][i][[src_idx]],
                            dst_cond=cond[itp][i][[dst_idx]],
                            n_step=n_step,
                            interp_fn=interp_fn,
                        )
                    )
            elif th.is_tensor(cond[itp]) or isinstance(cond[itp], np.ndarray):
                interp = interp_cond(
                    src_cond=cond[itp][[src_idx]],
                    dst_cond=cond[itp][[dst_idx]],
                    n_step=n_step,
                    interp_fn=interp_fn,
                )
            else:
                raise NotImplementedError(f"Unsupported condition type for '{itp}': {type(cond[itp])}")

        out_interp[itp] = interp
    return out_interp


# ==============================================================================
# Condition Parameter Manipulation Helpers
# ==============================================================================

def without(src, rmv):
    """Remove elements in `rmv` from `src` while preserving order."""
    return [s for s in src if s not in rmv]


def repeat_cond_params(cond, base_idx, n, key):
    """Repeat condition parameters at `base_idx` `n` times along axis 0."""
    print("[#] Repeating cond : ", key)
    repeat = {}
    for p in key:
        if th.is_tensor(cond[p][[base_idx]]):
            rep = cond[p][[base_idx]].cpu().detach().numpy()
        else:
            rep = cond[p][[base_idx]]
        repeat[p] = np.repeat(rep, repeats=n, axis=0)
    return repeat


def create_cond_params(cond, key):
    """Concatenate condition parameter arrays specified by `key` into `cond['cond_params']`."""
    print("[#] Condition build from parameters in ", key)
    tmp = []
    for p in key:
        if th.is_tensor(cond[p]):
            tmp.append(cond[p].cpu().detach().numpy())
        else:
            tmp.append(cond[p])
    if tmp:
        cond_params = np.concatenate(tmp, axis=1)
        cond["cond_params"] = cond_params
    return cond


def create_cond_imgs(cond, key):
    """Concatenate condition image tensors specified by `key` into `cond['cond_img']`."""
    print("[#] Condition build from image(s) in ", key)
    tmp = [cond[p].cpu() if th.is_tensor(cond[p]) else cond[p] for p in key]
    cond_img = np.concatenate(tmp, axis=1)
    cond["cond_img"] = cond_img
    return cond


def ext_sub_step(n_step, batch_size=5):
    """Compute cumulative batch boundary indices for `n_step` frames."""
    sub_step = []
    remaining = n_step
    while remaining > 0:
        step = min(batch_size, remaining)
        sub_step.append(step)
        remaining -= step
    return np.cumsum([0] + sub_step)


# ==============================================================================
# Spherical Harmonics (SH) Utilities
# ==============================================================================

def _to_sh_coeff(c):
    t = pysh.SHCoeffs.from_zeros(2)
    t.set_coeffs(c[0], 0, 0)
    t.set_coeffs(c[1], 1, 1)
    t.set_coeffs(c[2], 1, -1)
    t.set_coeffs(c[3], 1, 0)
    t.set_coeffs(c[4], 2, -2)
    t.set_coeffs(c[5], 2, 1)
    t.set_coeffs(c[6], 2, -1)
    t.set_coeffs(c[7], 2, 2)
    t.set_coeffs(c[8], 2, 0)
    return t


def _to_rgb_coeff(c):
    return [_to_sh_coeff(c[::3]), _to_sh_coeff(c[1::3]), _to_sh_coeff(c[2::3])]


def _to_deca(c):
    a = c.coeffs
    return np.array([
        a[0, 0, 0],
        a[0, 1, 1],
        a[1, 1, 1],
        a[0, 1, 0],
        a[1, 2, 2],
        a[0, 2, 1],
        a[1, 2, 1],
        a[0, 2, 2],
        a[0, 2, 0],
    ])


def _to_rgb_deca(cc):
    return list(itertools.chain(*zip(_to_deca(cc[0]), _to_deca(cc[1]), _to_deca(cc[2]))))


def _axis_angle_to_euler(x, y, z, degree):
    xyz = np.array([x, y, z], dtype=float)
    xyz = xyz / np.linalg.norm(xyz)
    rot = R.from_mrp(xyz * np.tan(degree * np.pi / 180.0 / 4.0))
    return rot.as_euler("zyz", degrees=True)


def _rotate_sh_vec(sh_np, x, y, z, degree):
    cc = _to_rgb_coeff(sh_np)
    euler = _axis_angle_to_euler(x, y, z, degree)
    cc[0] = cc[0].rotate(*euler)
    cc[1] = cc[1].rotate(*euler)
    cc[2] = cc[2].rotate(*euler)
    return _to_rgb_deca(cc)


def _sh_to_ld_np(sh):
    sh = sh.reshape(-1, 9, 3)
    return np.mean(sh[0:1, 1:4, :], axis=2)


def diffuse_sh(cond, src_idx, n_step, interpolate=False):
    """
    Zero out 1st- and 2nd-order SH coefficients to keep only ambient (0th-order) diffuse light.
    """
    inp_sh = cond["light"][[src_idx]].flatten()  # [1, 27] -> [27,]
    diff_sh = inp_sh.clone().reshape(9, 3)
    diff_sh[1:, :] = 0
    diff_sh = diff_sh.flatten()

    if interpolate:
        print("[#] Interpolating from inp to diffuse SH...")
        r_interp = np.linspace(0, 1, num=n_step)
        interp = [lerp(r=r, src=inp_sh, dst=diff_sh) for r in r_interp]
    else:
        print("[#] Using diffuse SH...")
        interp = [diff_sh] * n_step

    out_sh = np.stack(interp, 0)  # [n_step, 27]
    return {"light": out_sh}


def rotate_sh(cond, src_idx, n_step, axis):
    """
    Rotate source SH coefficients over `n_step` steps from 0 to 360 degrees around `axis` (0, 1, or 2).
    """
    inp_sh = cond["light"][[src_idx]].flatten()  # [1, 27] -> [27,]
    out_sh = []
    for deg in np.linspace(0, 360, n_step):
        moved = _rotate_sh_vec(inp_sh, axis == 0, axis == 1, axis == 2, deg)
        out_sh.append(np.array(moved))

    out_sh = np.stack(out_sh, 0)  # [n_step, 27]
    return {"light": out_sh}


# ==============================================================================
# Dataset Pair Loading Utilities
# ==============================================================================

def _lookup_pair_indices(img_path, src_dst_pairs):
    """Dictionary lookup for `[src, dst]` basename pairs in `img_path`."""
    name_to_idx = {os.path.basename(p): i for i, p in enumerate(img_path)}
    all_img_idx = []
    for sd in src_dst_pairs:
        indices = []
        for s in sd:
            if s not in name_to_idx:
                raise ValueError(f"[#] {sd} cannot be found in image path list")
            indices.append(name_to_idx[s])
        all_img_idx.append(indices)
    return all_img_idx


def get_samples_list(sample_pair_json, sample_pair_mode, src_dst, img_path, n_subject):
    """
    Resolve sample pair indices and image names from `sample_pair_json`, explicit `src_dst`, or random sampling.
    """
    if (sample_pair_json is not None) and (sample_pair_mode is not None):
        assert os.path.isfile(sample_pair_json), f"File not found: {sample_pair_json}"
        with open(sample_pair_json, "r") as f:
            sample_pairs = json.load(f)[sample_pair_mode]

        if sample_pair_mode == "pair":
            src_dst_pairs = [
                [sample_pairs[pair_i]["src"], sample_pairs[pair_i]["dst"]]
                for pair_i in sample_pairs.keys()
            ]
            all_img_idx = _lookup_pair_indices(img_path, src_dst_pairs)
            all_img_name = [[img_path[r[0]].split("/")[-1], img_path[r[1]].split("/")[-1]] for r in all_img_idx]
            if n_subject > len(sample_pairs) or n_subject == -1:
                n_subject = len(sample_pairs)

        elif sample_pair_mode == "pairwise":
            assert len(sample_pairs["src"]) == len(sample_pairs["dst"])
            src_dst_pairs = [[s, d] for s in sample_pairs["src"] for d in sample_pairs["dst"]]
            all_img_idx = _lookup_pair_indices(img_path, src_dst_pairs)
            all_img_name = [[img_path[r[0]].split("/")[-1], img_path[r[1]].split("/")[-1]] for r in all_img_idx]
            if n_subject > len(sample_pairs["dst"]) or n_subject == -1:
                n_subject = len(sample_pairs["dst"])
            else:
                n_subject = n_subject * len(sample_pairs["dst"])

        elif sample_pair_mode == "list":
            all_img_idx = file_utils.search_index_from_listpath(list_path=img_path, search=sample_pairs)
            all_img_name = [img_path[i].split("/")[-1] for i in all_img_idx]
            n_subject = -1

        else:
            raise NotImplementedError(f"Unsupported sample_pair_mode: {sample_pair_mode}")

    elif len(src_dst) == 2:
        n_subject = 1
        all_img_idx = [_lookup_pair_indices(img_path, [src_dst])[0]]
        all_img_name = [[img_path[r[0]].split("/")[-1], img_path[r[1]].split("/")[-1]] for r in all_img_idx]
    else:
        all_img_idx = np.random.choice(a=range(len(img_path)), replace=False, size=n_subject * 2)
        all_img_idx = np.array_split(all_img_idx, n_subject)
        all_img_name = [[img_path[r[0]].split("/")[-1], img_path[r[1]].split("/")[-1]] for r in all_img_idx]

    return all_img_idx, all_img_name, n_subject


get_samples_list_fast = get_samples_list


def get_samples_pair(sample_pair_json, sample_pair_mode, src_dst, img_path, start, end, n_subject):
    """Resolve sample pairs within `[start, end)` from `sample_pair_json`."""
    if (sample_pair_json is not None) and (sample_pair_mode is not None):
        assert os.path.isfile(sample_pair_json), f"File not found: {sample_pair_json}"
        with open(sample_pair_json, "r") as f:
            sample_pairs = json.load(f)[sample_pair_mode]
        keys = list(sample_pairs.keys())
        print("[#] Sample pair length : ", len(keys))
        if sample_pair_mode == "pair":
            if start > len(keys):
                raise ValueError(f"start={start} is out of range of {len(keys)}")
            end = min(end, len(keys))
            if start > end:
                raise ValueError(f"start={start} is greater than end={end}")
            src_dst_pairs = [[sample_pairs[k]["src"], sample_pairs[k]["dst"]] for k in keys[start:end]]
            all_img_idx = _lookup_pair_indices(img_path, src_dst_pairs)
            all_img_name = [[img_path[r[0]].split("/")[-1], img_path[r[1]].split("/")[-1]] for r in all_img_idx]
            if n_subject > len(sample_pairs) or n_subject == -1:
                n_subject = len(keys)
            return all_img_idx, all_img_name, n_subject
        raise NotImplementedError(f"Unsupported sample_pair_mode: {sample_pair_mode}")

    return get_samples_list(sample_pair_json, sample_pair_mode, src_dst, img_path, n_subject)


def get_samples_pair_from_indices(sample_pair_json, sample_pair_mode, src_dst, img_path, sample_indices):
    """Resolve sample pairs at specific `sample_indices` from `sample_pair_json`."""
    if (sample_pair_json is not None) and (sample_pair_mode is not None):
        assert os.path.isfile(sample_pair_json), f"File not found: {sample_pair_json}"
        with open(sample_pair_json, "r") as f:
            sample_pairs = json.load(f)[sample_pair_mode]
        keys = list(sample_pairs.keys())
        print("[#] Sample pair length : ", len(keys))
        if sample_pair_mode == "pair":
            sample_indices_arr = np.array(sample_indices)
            if np.any(sample_indices_arr >= len(keys)) or np.any(sample_indices_arr < 0):
                raise ValueError(f"sample_indices out of range [0, {len(keys)})")
            src_dst_pairs = [[sample_pairs[keys[i]]["src"], sample_pairs[keys[i]]["dst"]] for i in sample_indices]
            all_img_idx = _lookup_pair_indices(img_path, src_dst_pairs)
            all_img_name = [[img_path[r[0]].split("/")[-1], img_path[r[1]].split("/")[-1]] for r in all_img_idx]
            return all_img_idx, all_img_name, len(keys)
    raise NotImplementedError("sample_pair_json and sample_pair_mode='pair' are required.")
