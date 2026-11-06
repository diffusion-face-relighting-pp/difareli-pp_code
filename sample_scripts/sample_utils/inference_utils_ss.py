import time
import cv2
import numpy as np
import pytorch_lightning as pl
import torch as th

try:
    from sample_utils import mani_utils
    from sample_utils import params_utils
except ImportError:
    import mani_utils
    import params_utils

# Normalization bounds for DiffAE shadow scalar (C) on FFHQ
SHADOW_MIN_C = -4.989461058405101
SHADOW_MAX_C = 8.481700287326827


class PLSampling(pl.LightningModule):
    def __init__(self, model_dict, cfg, args=None, denoised_fn=None):
        super().__init__()
        self.denoised_fn = denoised_fn
        self.model_dict = model_dict
        self.cfg = cfg
        self.args = args

    def forward_cond_network(self, model_kwargs):
        with th.no_grad():
            if self.cfg.img_cond_model.apply:
                x = model_kwargs['cond_img'].cuda().float()
                img_cond = self.model_dict[self.cfg.img_cond_model.name](x=x, emb=None)
                if self.cfg.img_cond_model.override_cond != "":
                    model_kwargs[self.cfg.img_cond_model.override_cond] = img_cond
                else:
                    raise NotImplementedError("img_cond_model.override_cond must be specified.")
        return model_kwargs

    def forward_nodpm(self, src_xstart, model_kwargs, t=None):
        model = self.model_dict[self.cfg.img_model.name]
        if model_kwargs.get('dpm_cond_img') is not None:
            inp = th.cat((src_xstart, model_kwargs['dpm_cond_img']), dim=1).float()
        else:
            inp = src_xstart.float()
        return model(inp, t, **model_kwargs)


def to_tensor(cond, key=None, device='cuda', keys=None):
    target_keys = key if key is not None else keys
    for k in target_keys:
        if k not in cond:
            continue
        val = cond[k]
        if isinstance(val, list):
            cond[k] = [th.tensor(v).to(device) for v in val]
        elif th.is_tensor(val):
            cond[k] = val.clone().to(device)
        elif isinstance(val, np.ndarray):
            cond[k] = th.tensor(val).to(device)
    return cond


def eval_mode(model_dict):
    for model in model_dict.values():
        model.eval()
    return model_dict


def normalize_shadow_scalar(c_raw, min_c=SHADOW_MIN_C, max_c=SHADOW_MAX_C):
    """Maps raw DiffAE shadow score C into [0, 1] (0 = diffuse/no shadow, 1 = strongest cast shadow)."""
    return (c_raw - min_c) / (max_c - min_c)


def prepare_cond_sampling(cond, cfg, use_render_itp=False, device='cuda'):
    """Constructs the multi-channel spatial conditioning tensor `cond_img` for the condition encoder."""
    out_cond = {'dpm_cond_img': None}

    if not cfg.img_cond_model.apply:
        out_cond['cond_img'] = None
        return out_cond

    pair_cfg = list(zip(cfg.img_cond_model.in_image, cfg.img_cond_model.noise_dpm_cond_img))
    sj_paired = cfg.img_cond_model.sj_paired
    n_frames = (
        cond['src_deca_masked_face_images_woclip'].shape[0]
        + cond['dst_deca_masked_face_images_woclip'].shape[0]
    )

    out_cond_img = []
    for i in range(n_frames):
        frame_cond = []
        for j, (k, _) in enumerate(pair_cfg):
            if use_render_itp:
                if sj_paired[j] == 'src' or (sj_paired[j] == 'dst' and i == 0):
                    tmp_img = cond[f'src_{k}'][0:1]
                elif sj_paired[j] == 'dst' and i != 0:
                    tmp_img = cond[f'dst_{k}'][[i - 1]]
                else:
                    raise NotImplementedError(f"Unsupported sj_paired entry: {sj_paired[j]}")
            else:
                tmp_img = cond[f'{k}_img']
            frame_cond.append(tmp_img.to(device))
        out_cond_img.append(th.cat(frame_cond, dim=1))

    out_cond['cond_img'] = th.cat(out_cond_img, dim=0).to(device)
    return out_cond


def _build_light_trajectory(cond, args, src_idx, dst_idx, n_step, itp_func):
    """Builds the [n_step, 27] SH lighting sequence according to CLI flags."""
    if 'render_face' not in args.interpolate:
        return mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=['light'])

    if args.rotate_sh:
        print("[#] Lighting mode: Rotate SH of src light")
        interp_cond = mani_utils.rotate_sh(cond, src_idx=src_idx, n_step=n_step, axis=args.rotate_sh_axis)
    elif args.force_diffuse_sh:
        print("[#] Lighting mode: Diffuse (matted) SH from src light")
        itp_diffuse = getattr(args, 'itp_diffuse_sh', False)
        interp_cond = mani_utils.diffuse_sh(cond, src_idx=src_idx, n_step=n_step, interpolate=itp_diffuse)
        interp_cond['light'][0:1] = cond['light'][src_idx]
    elif args.same_sh:
        print("[#] Lighting mode: Keep src SH constant")
        interp_cond = {'light': cond['light'][src_idx].repeat(n_step, 1)}
    else:
        print("[#] Lighting mode: Interpolate SH from src -> dst")
        interp_cond = mani_utils.iter_interp_cond(
            cond, interp_set=['light'], src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func
        )

    if getattr(args, 'scale_sh', 1.0) != 1.0:
        print(f"[#] Scaling target SH [1:{n_step}] by {args.scale_sh}")
        interp_cond['light'][1:] = interp_cond['light'][1:] * args.scale_sh

    return interp_cond


def _adjust_render_contrast(rendered_tmp, alpha_images, adjust_contrast):
    """Optionally compresses DECA shading contrast toward the median face color (e.g., [0, 0] -> flat matted face)."""
    down, up = adjust_contrast[0], adjust_contrast[1]
    print(f"[#] Adjusting DECA render contrast: down={down}, up={up}")
    alpha_mask = (th.cat(alpha_images).cpu().numpy() > 0.5)
    alpha_mask = np.repeat(alpha_mask, 3, axis=1)  # [B, 3, H, W]

    adjusted = []
    for i in range(rendered_tmp.shape[0]):
        r_tmp = rendered_tmp[i]
        a_tmp = alpha_mask[i]
        median = np.median(r_tmp[a_tmp], axis=0, keepdims=True)
        above = (r_tmp > median) & a_tmp
        below = (~(r_tmp > median)) & a_tmp

        new_r = r_tmp.copy()
        new_r[above] = (r_tmp[above] - median) * up + median
        new_r[below] = (r_tmp[below] - median) * down + median
        adjusted.append(np.clip(new_r, 0.0, 1.0))

    return np.stack(adjusted, axis=0)


def init_deca_modules(rasterize_type='pytorch3d', device='cuda'):
    """Initializes FLAME masks and DECA renderer instances once for reuse across subjects."""
    print("*"*100)
    print("[#] Initializing DECA modules...")
    t0_deca = time.time()
    mask = params_utils.load_flame_mask()
    deca_obj = params_utils.init_deca(mask=mask, rasterize_type=rasterize_type, device=device)
    load_deca_time = time.time() - t0_deca

    t0_shadow_deca = time.time()
    flame_face_scalp = params_utils.load_flame_mask(['face', 'scalp', 'left_eyeball', 'right_eyeball'])
    deca_obj_face_scalp = params_utils.init_deca(mask=flame_face_scalp, rasterize_type=rasterize_type, device=device)
    load_deca_for_shadow_time = time.time() - t0_shadow_deca
    print(f"[#] DECA modules initialized in {load_deca_time:.2f}s (render) + {load_deca_for_shadow_time:.2f}s (shadow)")
    print("*"*100)

    return {
        'mask': mask,
        'deca_obj': deca_obj,
        'flame_face_scalp': flame_face_scalp,
        'deca_obj_face_scalp': deca_obj_face_scalp,
        'load_deca_time': load_deca_time,
        'load_deca_for_shadow_time': load_deca_for_shadow_time,
    }


def build_condition_image(cond, misc, force_render=False, deca_modules=None):
    src_idx = misc['src_idx']
    dst_idx = misc['dst_idx']
    n_step = misc['n_step']
    render_batch_size = misc['render_batch_size']
    avg_dict = misc['avg_dict']
    args = misc['args']
    condition_img = misc['condition_img']
    img_size = misc['img_size']
    itp_func = misc['itp_func']
    clip_ren = False

    needs_render = (
        any('deca' in name for name in condition_img)
        or any('shadow_diff' in name for name in condition_img)
        or force_render
    )

    if needs_render:
        cond.update(_build_light_trajectory(cond, args, src_idx, dst_idx, n_step, itp_func))

        if deca_modules is None:
            deca_modules = misc.get('deca_modules')
        if deca_modules is None:
            deca_modules = init_deca_modules(rasterize_type=args.rasterize_type)
            load_deca_time = deca_modules['load_deca_time']
            load_deca_for_shadow_time = deca_modules['load_deca_for_shadow_time']
        else:
            load_deca_time = 0.0
            load_deca_for_shadow_time = 0.0

        mask = deca_modules['mask']
        deca_obj = deca_modules['deca_obj']
        deca_obj_face_scalp = deca_modules['deca_obj_face_scalp']

        start_t = time.time()
        print(f"[#] Rendering {n_step} frames (batch_size={render_batch_size})...")
        sub_step = mani_utils.ext_sub_step(n_step, render_batch_size)

        all_render = []
        all_alpha_images = []
        all_shadow_mask = []
        all_shadow_kk = []
        render_time = []
        pure_render_deca_time = []
        pure_render_shadow_time = []

        for i in range(len(sub_step) - 1):
            start, end = sub_step[i], sub_step[i + 1]
            print(f"[#] Sub-step rendering: {start} to {end}")
            sub_cond = cond.copy()
            sub_cond['light'] = sub_cond['light'][start:end, :]

            t0_deca = time.time()
            deca_rendered, orig_visdict = params_utils.render_deca(
                deca_params=sub_cond,
                idx=src_idx,
                n=end - start,
                avg_dict=avg_dict,
                render_mode=args.render_mode,
                rotate_normals=args.rotate_normals,
                mask=mask,
                deca_obj=deca_obj,
                repeat=True,
            )
            pure_render_deca_time.append(time.time() - t0_deca)

            t0_shadow = time.time()
            rotate_dst = getattr(args, 'rotate_sh_dst', False)
            shadow_mask, shadow_kk, _ = params_utils.render_shadow_mask_with_smooth(
                sh_light=sub_cond['light'],
                cam=sub_cond['cam'][src_idx],
                verts=orig_visdict['trans_verts_orig'],
                deca={'face_scalp': deca_obj_face_scalp},
                axis_0=(args.rotate_sh_axis == 0 and (args.rotate_sh or rotate_dst)),
                axis_1=(args.rotate_sh_axis == 1 and (args.rotate_sh or rotate_dst)),
                device='cpu',
                org_h=img_size,
                org_w=img_size,
                rt_dict={
                    'pt_round': args.pt_round,
                    'pt_radius': args.pt_radius,
                    'scale_depth': args.scale_depth,
                },
            )
            pure_render_shadow_time.append(time.time() - t0_shadow)

            all_render.append(deca_rendered)
            all_alpha_images.append(orig_visdict['alpha_images'])
            all_shadow_mask.append(shadow_mask[:, None, ...])
            all_shadow_kk.append(shadow_kk[:, None, ...])
            render_time.append(time.time() - start_t)

        cond['render_time'] = np.mean(render_time) + load_deca_time
        cond['pure_render_deca_time'] = pure_render_deca_time
        cond['pure_render_shadow_time'] = pure_render_shadow_time
        cond['load_deca_time'] = load_deca_time
        cond['load_deca_for_shadow_time'] = load_deca_for_shadow_time
        print(f"[#] Total rendering time: {time.time() - start_t:.2f}s")

        deca_rendered = th.cat(all_render, dim=0)
        if args.fixed_render and (args.shadow_diff_inc_c or args.shadow_diff_dec_c):
            fidx = int(args.shadow_diff_fidx_frac * n_step)
            print(f"[#] Fixing DECA render for reshadowing (target frames use frame {fidx})")
            ff = deca_rendered[0:1]
            rf = deca_rendered[fidx:fidx + 1].repeat_interleave(repeats=n_step - 1, dim=0)
            deca_rendered = th.cat((ff, rf), dim=0)
        elif args.fixed_render:
            print("[#] Fixing DECA render (all frames use frame 0)")
            deca_rendered = deca_rendered[0:1].repeat_interleave(repeats=n_step, dim=0)

        shadow_mask = th.cat(all_shadow_mask, dim=0)
        shadow_kk = th.cat(all_shadow_kk, dim=0)
        if args.fixed_shadow:
            print("[#] Fixing shadow mask (all frames use frame 0)")
            shadow_mask = shadow_mask[0:1].repeat_interleave(repeats=n_step, dim=0)
            shadow_kk = shadow_kk[0:1].repeat_interleave(repeats=n_step, dim=0)

    def prep_render(cond_dict, cond_img_name):
        rendered_tmp = []
        for j in range(n_step):
            r_tmp = deca_rendered[j].cpu().numpy().transpose((1, 2, 0))
            r_tmp = cv2.resize(r_tmp, (img_size, img_size), cv2.INTER_AREA)
            r_tmp = np.transpose(r_tmp, (2, 0, 1))
            rendered_tmp.append(r_tmp)
        rendered_tmp = np.stack(rendered_tmp, axis=0)

        cond_dict[cond_img_name] = th.tensor(rendered_tmp).cuda()
        cond_dict[f'src_{cond_img_name}'] = th.tensor(rendered_tmp[[0]]).cuda()
        cond_dict[f'dst_{cond_img_name}'] = th.tensor(rendered_tmp[1:]).cuda()

    def prep_shadow(cond_dict, cond_img_name):
        m_glasses_and_eyes = cond_dict[f'{cond_img_name}_meg_mask'][src_idx].cpu().numpy()
        m_face = cond_dict[f'{cond_img_name}_mface_mask'][src_idx].cpu().numpy()
        valid_face_region = m_face * (1.0 - m_glasses_and_eyes)

        shadow_diff_tmp = []
        for j in range(n_step):
            sd_tmp = shadow_mask[j]
            sdkk_tmp = shadow_kk[j]
            # Binary cast-shadow mask within valid face region
            sd_bin = (((1.0 - sd_tmp) > 0) * 1.0) * valid_face_region * ((1.0 - sdkk_tmp) > 0.0)
            shadow_diff_tmp.append(sd_bin)

        shadow_diff_tmp = np.stack(shadow_diff_tmp, axis=0)
        # Frame 0 always uses the precomputed source shadow map (shadow_diff_SS_with_c_simplified)
        shadow_diff_tmp[0] = cond_dict['shadow_diff_img'][src_idx]

        cond_dict[cond_img_name] = th.tensor(shadow_diff_tmp).cuda()
        cond_dict[f'src_{cond_img_name}'] = th.tensor(shadow_diff_tmp[[0]]).cuda()
        cond_dict[f'dst_{cond_img_name}'] = th.tensor(shadow_diff_tmp[1:]).cuda()

    for cond_img_name in condition_img:
        if ('faceseg' in cond_img_name) or ('face_structure' in cond_img_name):
            bg_tmp = [cond[f"{cond_img_name}_img"][src_idx]] * n_step
            bg_tmp = th.stack(bg_tmp, dim=0) if th.is_tensor(bg_tmp[0]) else th.tensor(np.stack(bg_tmp, axis=0))
            cond[f"src_{cond_img_name}"] = bg_tmp
        elif 'deca' in cond_img_name:
            prep_render(cond, cond_img_name)
        elif 'shadow_diff' in cond_img_name:
            prep_shadow(cond, cond_img_name)

    if force_render:
        prep_render(cond, 'deca_masked_face_images_woclip')

    return cond, clip_ren, misc


def _resolve_shadow_weights(cond, args, src_idx, dst_idx, n_step, device='cuda'):
    """
    Determines the shadow scalar weights (in [0, 1], where 0 = diffuse and 1 = max shadow)
    for the target frames (1..n_step-1) and whether the spatial shadow mask should be locked
    to a single reference frame (reshadow mode) or vary per frame (relight mode).
    """
    c_norm_src = float(normalize_shadow_scalar(cond['shadow'][src_idx]).item())
    c_norm_dst = float(normalize_shadow_scalar(cond['shadow'][dst_idx]).item())
    n_target = n_step - 1

    # --- Mode 1: Reshadowing (interpolate shadow weight C on a fixed shadow mask) ---
    if args.shadow_diff_inc_c:
        start_c = args.reshadow_with_given_c if args.reshadow_with_given_c is not None else c_norm_src
        print(f"[#] Shadow mode: Reshadow increase C ({start_c:.4f} -> 1.0 over {n_target} frames)")
        target_weights = th.linspace(start=start_c, end=1.0, steps=n_target, device=device)
        return target_weights.view(n_target, 1, 1, 1), True

    if args.shadow_diff_dec_c:
        start_c = args.reshadow_with_given_c if args.reshadow_with_given_c is not None else c_norm_src
        print(f"[#] Shadow mode: Reshadow decrease C ({start_c:.4f} -> 0.0 over {n_target} frames)")
        target_weights = th.linspace(start=start_c, end=0.0, steps=n_target, device=device)
        return target_weights.view(n_target, 1, 1, 1), True

    if getattr(args, 'reshadow_gradually_inc_c', False):
        print(f"[#] Shadow mode: Gradually increase C (0.0 -> 1.0 over {n_target} frames)")
        target_weights = th.linspace(start=0.0, end=1.0, steps=n_target, device=device)
        return target_weights.view(n_target, 1, 1, 1), True

    # --- Mode 2: Relighting (apply target shadow scalar C to ray-traced target shadow masks) ---
    if args.relight_with_given_c is not None:
        val_c = float(args.relight_with_given_c)
        print(f"[#] Shadow mode: Relight with given C = {val_c:.4f}")
    elif args.relight_with_dst_c:
        val_c = c_norm_dst
        print(f"[#] Shadow mode: Relight with dst C = {val_c:.4f}")
    elif args.relight_with_rand_max_c:
        val_c = float(np.random.uniform(0.7, 1.0))
        print(f"[#] Shadow mode: Relight with random C in [0.7, 1.0] = {val_c:.4f}")
    elif getattr(args, 'relight_with_strongest_c', False):
        val_c = 1.0
        print("[#] Shadow mode: Relight with strongest C = 1.0")
    else:
        val_c = c_norm_src
        print(f"[#] Shadow mode: Relight with src C = {val_c:.4f}")

    target_weights = th.full((n_target, 1, 1, 1), val_c, device=device, dtype=th.float32)
    return target_weights, False


def shadow_diff_with_weight_postproc(cond, misc, device='cuda'):
    """
    Post-processes the spatial shadow maps:
      - `src_shadow_diff_with_weight_simplified` (frame 0): keeps the source shadow map
        (or zeros it out if `--force_zero_src_shadow` is set).
      - `dst_shadow_diff_with_weight_simplified` (frames 1..n_step-1):
        - In reshadowing mode (`--shadow_diff_inc_c` / `--shadow_diff_dec_c`), locks the spatial
          mask to frame `fidx` (default frame 0) and scales it by the interpolated weight schedule.
        - In relighting mode, scales each frame's ray-traced binary shadow mask by the target C weight.
    """
    n_step = misc['n_step']
    src_idx = misc['src_idx']
    dst_idx = misc['dst_idx']
    args = misc['args']

    key = 'shadow_diff_with_weight_simplified'
    src_key = f'src_{key}'
    dst_key = f'dst_{key}'

    if src_key not in cond or dst_key not in cond:
        return cond, {}

    target_weights, lock_spatial_mask = _resolve_shadow_weights(
        cond=cond, args=args, src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, device=device
    )

    all_sd = th.cat((cond[src_key], cond[dst_key]), dim=0).to(device)

    # Source frame (frame 0)
    if getattr(args, 'force_zero_src_shadow', False):
        print("[#] Forcing src shadow map to 0.0")
        src_shadow = th.zeros_like(all_sd[0:1])
    else:
        src_shadow = all_sd[0:1]

    # Target frames (frames 1..n_step-1)
    if lock_spatial_mask:
        fidx = int(getattr(args, 'shadow_diff_fidx_frac', 0.0) * n_step)
        ref_mask = (all_sd[fidx:fidx + 1] > 0.0).float().repeat_interleave(repeats=n_step - 1, dim=0)
        dst_shadow = ref_mask * target_weights
    else:
        dst_mask = (all_sd[1:] > 0.0).float()
        dst_shadow = dst_mask * target_weights

    cond[src_key] = src_shadow
    cond[dst_key] = dst_shadow
    cond[key] = th.cat((src_shadow, dst_shadow), dim=0)

    print(f"[#] Shadow map values — src unique: {th.unique(src_shadow)}, dst unique: {th.unique(dst_shadow)}")
    return cond, {'target_weights': target_weights}
