import argparse
import copy
import datetime
import json
import os
import sys
import tqdm

import numpy as np
from parser import build_parser

args = build_parser().parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import time
import torch as th
import torchvision

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SAMPLE_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for p in (REPO_ROOT, SAMPLE_SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from guided_diffusion.dataloader.img_deca_datasets import load_data_img_deca
from guided_diffusion.script_util import seed_all
from guided_diffusion.tensor_util import dict_slice_se
from sample_utils import (
    ckpt_utils,
    file_utils,
    inference_utils_ss as inference_utils,
    mani_utils,
    params_utils,
    vis_utils,
)


def ext_sub_step(n_step, batch_size):
    steps = []
    remaining = n_step
    while remaining > 0:
        step = min(batch_size, remaining)
        steps.append(step)
        remaining -= step
    return np.cumsum([0] + steps)


def make_condition(cond, src_idx, dst_idx, n_step, itp_func, dataset, avg_dict, cfg, device, deca_modules=None):
    condition_img = list(filter(None, dataset.condition_image))
    args.interpolate = args.itp
    misc = {
        'condition_img': condition_img,
        'src_idx': src_idx,
        'dst_idx': dst_idx,
        'n_step': n_step,
        'avg_dict': avg_dict,
        'dataset': dataset,
        'args': args,
        'itp_func': itp_func,
        'img_size': cfg.img_model.image_size,
        'cfg': cfg,
        'batch_size': args.batch_size,
        'render_batch_size': args.render_batch_size,
    }
    cond['use_render_itp'] = (itp_func is None)

    cond, _, misc = inference_utils.build_condition_image(
        cond=cond, misc=misc, force_render=args.force_render, deca_modules=deca_modules
    )
    misc_paired = copy.deepcopy(misc)
    for i, prefix in enumerate(cfg.img_cond_model.sj_paired):
        misc_paired['condition_img'][i] = f"{prefix}_{misc_paired['condition_img'][i]}"

    cond, _ = inference_utils.shadow_diff_with_weight_postproc(cond=cond, misc=misc_paired, device=device)
    n_step = misc['n_step']
    cond.update(inference_utils.prepare_cond_sampling(cond=cond, cfg=cfg, use_render_itp=True, device=device))

    cond['cfg'] = cfg
    cond['use_cond_xt_fn'] = False

    interp_set = [k for k in args.itp if k not in ('render_face', 'light')]
    interp_cond = mani_utils.iter_interp_cond(
        cond,
        interp_set=interp_set,
        src_idx=src_idx,
        dst_idx=dst_idx,
        n_step=n_step,
        interp_fn=itp_func,
        add_remove_shadow=args.add_remove_shadow,
        add_shadow_to=args.add_shadow_to,
        remove_shadow_to=args.remove_shadow_to,
    )
    cond.update(interp_cond)

    non_spatial_keys = mani_utils.without(cfg.param_model.params_selector, args.itp + ['light', 'img_latent'])
    cond.update(mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=non_spatial_keys))

    if 'src_light' in cfg.param_model.params_selector:
        cond['src_light'] = cond['light'][0:1].repeat(repeats=n_step, axis=0)
    if 'dst_light' in cfg.param_model.params_selector:
        cond['dst_light'] = cond['light']

    active_param_keys = mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params)
    cond = mani_utils.create_cond_params(cond=cond, key=active_param_keys)

    to_tensor_keys = ['cond_params'] + cfg.param_model.params_selector
    if cfg.img_cond_model.override_cond != '':
        to_tensor_keys.append(cfg.img_cond_model.override_cond)
    cond = inference_utils.to_tensor(cond, keys=to_tensor_keys, device=device)

    return cond, misc


def run_relight(
    model_kwargs, pl_sampling, itp_func, n_step, src_idx, dst_idx, dataset, avg_dict, cfg, device, deca_modules=None
):
    cond = copy.deepcopy(model_kwargs)
    cond, misc_rvk = make_condition(
        cond=cond,
        src_idx=src_idx,
        dst_idx=dst_idx,
        n_step=n_step,
        itp_func=itp_func,
        dataset=dataset,
        avg_dict=avg_dict,
        cfg=cfg,
        device=device,
        deca_modules=deca_modules,
    )
    n_step = misc_rvk['n_step']

    print(f"[#] Running single-shot generator for {n_step} frames...")
    sub_step = ext_sub_step(n_step, args.batch_size)
    relit_out = []
    relit_time = []
    start_relight_time = time.time()

    relight_progress = tqdm.tqdm(
        range(len(sub_step) - 1),
        desc="Relighting",
        unit="sub-step",
    )
    for i in relight_progress:
        sub_t0 = time.time()
        start, end = sub_step[i], sub_step[i + 1]

        src_xstart = th.repeat_interleave(model_kwargs['image'][[0]], repeats=end - start, dim=0).cuda().float()

        cond['use_render_itp'] = True
        cond_slice = dict_slice_se(in_d=copy.deepcopy(cond), keys=cond.keys(), s=start, e=end)
        if cfg.img_cond_model.apply:
            cond_slice = pl_sampling.forward_cond_network(model_kwargs=cond_slice)

        relight_res = pl_sampling.forward_nodpm(src_xstart=src_xstart, model_kwargs=cond_slice)
        relit_out.append(relight_res['output'].detach().cpu().numpy())

        elapsed = time.time() - sub_t0
        relit_time.append(elapsed)
        relight_progress.set_description(
            f"Relighting (last: {elapsed:.2f}s, avg: {np.mean(relit_time):.2f}s, total: {time.time() - start_relight_time:.2f}s)"
        )

    relit_out = th.from_numpy(np.concatenate(relit_out, axis=0))

    out_timing = {
        'relit_time': float(np.mean(relit_time)),
        'each_relit_time': [float(t) for t in relit_time],
        'render_time': float(cond.get('render_time', 0.0)),
        'pure_render_deca_time': [float(t) for t in cond.get('pure_render_deca_time', [])],
        'pure_render_shadow_time': [float(t) for t in cond.get('pure_render_shadow_time', [])],
        'load_deca_time': float(cond.get('load_deca_time', 0.0)),
        'load_deca_for_shadow_time': float(cond.get('load_deca_for_shadow_time', 0.0)),
    }

    out_cond = cond['cond_img'] if (('render_face' in args.itp) or args.force_render) else None
    return relit_out, out_cond, out_timing


def save_video_pair(video_tensor, vid_dir, prefix, fps, exclude_first=False):
    """Writes forward (`{prefix}.mp4`) and round-trip (`{prefix}_rt.mp4`) videos."""
    vid_rt = th.cat((video_tensor, th.flip(video_tensor, dims=[0])))
    torchvision.io.write_video(video_array=video_tensor, filename=os.path.join(vid_dir, f"{prefix}.mp4"), fps=fps)
    torchvision.io.write_video(video_array=vid_rt, filename=os.path.join(vid_dir, f"{prefix}_rt.mp4"), fps=fps)
    if exclude_first and video_tensor.shape[0] > 1:
        torchvision.io.write_video(
            video_array=video_tensor[1:], filename=os.path.join(vid_dir, f"{prefix}_no1.mp4"), fps=fps
        )
        torchvision.io.write_video(
            video_array=vid_rt[1:-1], filename=os.path.join(vid_dir, f"{prefix}_rt_no1.mp4"), fps=fps
        )


def resolve_subject_dir(out_dir, src_id, dst_id):
    """
    Returns `<out_dir>/<src_stem>` by default.
    If transferring lighting between two distinct subjects (`src != dst` and not a self-manipulation
    mode like rotate_sh / force_diffuse_sh / reshadow) and `<out_dir>/<src_stem>` already exists,
    falls back to `<out_dir>/<src_stem>_to_<dst_stem>` to avoid overwriting.
    """
    src_stem = os.path.splitext(src_id)[0]
    dst_stem = os.path.splitext(dst_id)[0]
    is_self_mani = (
        src_stem == dst_stem
        or args.rotate_sh
        or args.force_diffuse_sh
        or args.same_sh
        or args.shadow_diff_inc_c
        or args.shadow_diff_dec_c
        or args.reshadow_gradually_inc_c
    )
    subj_dir = os.path.join(out_dir, src_stem)
    if not is_self_mani and os.path.exists(subj_dir):
        subj_dir = os.path.join(out_dir, f"{src_stem}_to_{dst_stem}")
    return subj_dir


if __name__ == '__main__':
    seed_all(args.seed)

    if args.cfg_name is None:
        args.cfg_name = args.log_dir + '.yaml'
    elif args.log_dir is None:
        args.log_dir = args.cfg_name.replace('.yaml', '')

    ckpt_loader = ckpt_utils.CkptLoader(log_dir=args.log_dir, cfg_name=args.cfg_name)
    cfg = ckpt_loader.cfg

    model_dict, _ = ckpt_loader.load_model(ckpt_selector=args.ckpt_selector, step=args.step)
    model_dict = inference_utils.eval_mode(model_dict)
    device = ckpt_loader.device

    # Configure dataset paths
    img_ext = '.png'
    cfg.dataset.root_path = os.path.abspath(args.dataset_root)
    cfg.dataset.data_dir = os.path.join(cfg.dataset.root_path, 'images_aligned')
    cfg.dataset.deca_dir = os.path.join(cfg.dataset.root_path, 'params')
    cfg.dataset.face_segment_dir = os.path.join(cfg.dataset.root_path, 'face_segment_with_pupil')
    cfg.dataset.deca_rendered_dir = os.path.join(cfg.dataset.root_path, 'rendered_images')
    cfg.dataset.shadow_diff_dir = (
        os.path.join(cfg.dataset.root_path, 'shadow_diff_SS_with_c_simplified')
        if args.shadow_diff_dir is None
        else os.path.abspath(args.shadow_diff_dir)
    )
    _, dataset, avg_dict = load_data_img_deca(
        data_dir=cfg.dataset.data_dir,
        deca_dir=cfg.dataset.deca_dir,
        batch_size=int(1e5),
        image_size=cfg.img_model.image_size,
        deterministic=cfg.train.deterministic,
        augment_mode=cfg.img_model.augment_mode,
        resize_mode=cfg.img_model.resize_mode,
        in_image_UNet=cfg.img_model.in_image,
        params_selector=cfg.param_model.params_selector,
        rmv_params=cfg.param_model.rmv_params,
        force_jpg_key=False,
        set_=args.set,
        cfg=cfg,
        img_ext=img_ext,
        mode='sampling',
    )

    img_paths = file_utils._list_image_files_recursively(
        os.path.join(cfg.dataset.data_dir, args.set), force_jpg_key=False
    )
    all_img_idx, all_img_name, n_subject = mani_utils.get_samples_list(
        args.sample_pair_json, args.sample_pair_mode, args.src_dst, img_paths, -1
    )

    start, end = int(args.idx[0]), min(int(args.idx[1]), n_subject)
    if start >= n_subject:
        raise ValueError(f"[#] Start index ({start}) is >= number of subjects ({n_subject})")
    print(f"[#] Processing sample indices {start} to {end}...")

    os.makedirs(args.out_dir, exist_ok=True)
    cli_command = " ".join(sys.argv)

    counter_sj = 0
    runtime_dict = {
        'relit_time': [],
        'sub_relit_time': [],
        'render_time': [],
        'sub_pure_render_deca_time': [],
        'sub_pure_render_shadow_time': [],
        'sub_load_deca_for_shadow_time': [],
        'sub_load_deca_time': [],
    }

    each_in_ch = cfg.img_cond_model.each_in_channels
    each_in_img = cfg.img_cond_model.in_image
    each_sj_anno = cfg.img_cond_model.sj_paired
    ch_bounds = [0] + list(np.cumsum(each_in_ch))
    idx_anno = [(ch_bounds[k], ch_bounds[k + 1]) for k in range(len(ch_bounds) - 1)]
    com_name = [f'{each_sj_anno[k]}_{each_in_img[k]}' for k in range(len(each_in_ch))]

    condition_img = list(filter(None, dataset.condition_image))
    needs_render = (
        any('deca' in name for name in condition_img)
        or any('shadow_diff' in name for name in condition_img)
        or args.force_render
    )
    deca_modules = (
        inference_utils.init_deca_modules(rasterize_type=args.rasterize_type, device=device)
        if needs_render
        else None
    )

    for i in range(start, end):
        img_idx = all_img_idx[i]
        src_id, dst_id = all_img_name[i][0], all_img_name[i][1]

        subset_loader = th.utils.data.DataLoader(
            th.utils.data.Subset(dataset, indices=img_idx),
            batch_size=2,
            shuffle=False,
            num_workers=4,
        )
        _, model_kwargs = next(iter(subset_loader))

        print("#" * 100)
        print(f"[#] Index = {i} | Split = {args.set} | Src = {src_id} | Dst = {dst_id}")

        pl_sampling = inference_utils.PLSampling(model_dict=model_dict, denoised_fn=None, cfg=cfg, args=args)
        model_kwargs['cfg'] = cfg
        model_kwargs['use_cond_xt_fn'] = False
        model_kwargs['use_render_itp'] = True

        itp_fn = mani_utils.slerp if args.slerp else mani_utils.lerp
        itp_fn_str = 'Slerp' if args.slerp else 'Lerp'
        itp_str = '_'.join(args.itp)

        out_relit, out_cond, time_dict = run_relight(
            model_kwargs=model_kwargs,
            pl_sampling=pl_sampling,
            itp_func=itp_fn,
            n_step=args.itp_step,
            src_idx=0,
            dst_idx=1,
            dataset=dataset,
            avg_dict=avg_dict,
            cfg=cfg,
            device=device,
            deca_modules=deca_modules,
        )

        runtime_dict['relit_time'].append(time_dict['relit_time'])
        runtime_dict['sub_relit_time'].append(time_dict['each_relit_time'])
        runtime_dict['render_time'].append(time_dict['render_time'])
        runtime_dict['sub_pure_render_deca_time'].append(time_dict['pure_render_deca_time'])
        runtime_dict['sub_pure_render_shadow_time'].append(time_dict['pure_render_shadow_time'])
        runtime_dict['sub_load_deca_for_shadow_time'].append(time_dict['load_deca_for_shadow_time'])
        runtime_dict['sub_load_deca_time'].append(time_dict['load_deca_time'])

        # Output layout: <out_dir>/<src_name>/{frames,vid,res_desc.json}
        subj_dir = resolve_subject_dir(args.out_dir, src_id, dst_id)
        frames_dir = os.path.join(subj_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        f_relit = vis_utils.convert2rgb(out_relit, cfg.img_model.input_bound) / 255.0
        vis_utils.save_images(path=frames_dir, fn="res", frames=f_relit)

        if args.eval_dir is not None:
            eval_out = os.path.join(args.eval_dir, f"{args.ckpt_selector}_{args.step}", f"SD{args.scale_depth}", "out")
            os.makedirs(eval_out, exist_ok=True)
            torchvision.utils.save_image(
                tensor=f_relit[-1], fp=os.path.join(eval_out, f"input={src_id}#pred={dst_id}.png")
            )

        is_render = out_cond is not None
        is_shadow = (
            'shadow_diff_with_weight_simplified' in cfg.img_cond_model.in_image
            or 'shadow_diff_with_weight_simplified_inverse' in cfg.img_cond_model.in_image
        )

        if is_render:
            for s in ('src', 'dst'):
                ren_key = f'{s}_deca_masked_face_images_woclip'
                if ren_key in com_name:
                    c0, c1 = idx_anno[com_name.index(ren_key)]
                    ren_frames = out_cond[:, c0:c1].mul(255).add_(0.5).clamp_(0, 255) / 255.0
                    vis_utils.save_images(path=frames_dir, fn=f"{s}_ren", frames=ren_frames)

        if is_shadow and is_render:
            for s in ('src', 'dst'):
                shad_key = f'{s}_shadow_diff_with_weight_simplified'
                if shad_key in com_name:
                    c0, c1 = idx_anno[com_name.index(shad_key)]
                    shad_frames = out_cond[:, c0:c1].mul(255).add_(0.5).clamp_(0, 255) / 255.0
                    vis_utils.save_images(path=frames_dir, fn=f"{s}_shadm_shad", frames=shad_frames)

        if args.save_vid:
            vid_dir = os.path.join(subj_dir, 'vid')
            os.makedirs(vid_dir, exist_ok=True)

            vid_relit = ((out_relit.permute(0, 2, 3, 1) + 1.0) * 127.5).clamp_(0, 255).type(th.ByteTensor)
            save_video_pair(vid_relit, vid_dir, prefix="res", fps=args.fps, exclude_first=False)

            sj_paired_vid_render = {}
            if is_render:
                for s in ('src', 'dst'):
                    ren_key = f'{s}_deca_masked_face_images_woclip'
                    if ren_key in com_name:
                        c0, c1 = idx_anno[com_name.index(ren_key)]
                        vid_ren = (
                            out_cond[:, c0:c1].permute(0, 2, 3, 1).mul(255).add_(0.5).clamp_(0, 255).type(th.ByteTensor)
                        )
                        save_video_pair(vid_ren, vid_dir, prefix=f"{s}_ren", fps=args.fps, exclude_first=True)
                        sj_paired_vid_render[s] = vid_ren

            sj_paired_vid_shadm = {}
            if is_shadow and is_render:
                for s in ('src', 'dst'):
                    shad_key = f'{s}_shadow_diff_with_weight_simplified'
                    if shad_key in com_name:
                        c0, c1 = idx_anno[com_name.index(shad_key)]
                        vid_shadm = (
                            out_cond[:, c0:c1]
                            .repeat(1, 3, 1, 1)
                            .permute(0, 2, 3, 1)
                            .mul(255)
                            .add_(0.5)
                            .clamp_(0, 255)
                            .type(th.ByteTensor)
                        )
                        save_video_pair(vid_shadm, vid_dir, prefix=f"{s}_shadm", fps=args.fps, exclude_first=True)
                        sj_paired_vid_shadm[s] = vid_shadm

            if is_render and is_shadow:
                concat_list = (
                    list(sj_paired_vid_render.values())
                    + [vid_relit]
                    + list(sj_paired_vid_shadm.values())[::-1]
                )
                all_out = th.cat(concat_list, dim=2)
                save_video_pair(all_out, vid_dir, prefix="out", fps=args.fps, exclude_first=True)
            elif is_render:
                concat_list = list(sj_paired_vid_render.values()) + [vid_relit]
                all_out = th.cat(concat_list, dim=2)
                save_video_pair(all_out, vid_dir, prefix="out", fps=args.fps, exclude_first=True)

        # Save per-subject metadata & command provenance
        desc_path = os.path.join(subj_dir, 'res_desc.json')
        with open(desc_path, 'w') as fj:
            log_dict = {
                'command': cli_command,
                'checkpoint': {
                    'log_dir': args.log_dir,
                    'cfg_name': args.cfg_name,
                    'ckpt_selector': args.ckpt_selector,
                    'step': args.step,
                },
                'dataset': {
                    'dataset_root': cfg.dataset.root_path,
                    'set': args.set,
                    'shadow_diff_dir': cfg.dataset.shadow_diff_dir,
                },
                'samples': {
                    'src_id': src_id,
                    'dst_id': dst_id,
                    'n_frames': int(out_relit.shape[0]),
                    'itp_fn': itp_fn_str,
                    'itp': itp_str,
                },
                'timing': time_dict,
                'sampling_args': vars(args),
            }
            json.dump(log_dict, fj, indent=2)

        counter_sj += 1

    # Save overall runtime summary directly in args.out_dir
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    runtime_summary_path = os.path.join(args.out_dir, f'runtime_{dt}.json')
    with open(runtime_summary_path, 'w') as fj:
        runtime_dict['command'] = cli_command
        runtime_dict['cfg_name'] = args.cfg_name
        runtime_dict['log_dir'] = args.log_dir
        runtime_dict['step'] = args.step
        runtime_dict['postfix'] = args.postfix
        runtime_dict['mean_relit_time'] = float(np.mean(runtime_dict['relit_time'])) if counter_sj > 0 else 0.0
        runtime_dict['std_relit_time'] = float(np.std(runtime_dict['relit_time'])) if counter_sj > 0 else 0.0
        runtime_dict['all_relit_time'] = np.stack(runtime_dict['sub_relit_time']).tolist() if counter_sj > 0 else []
        runtime_dict['all_render_time'] = runtime_dict['render_time']
        runtime_dict['all_pure_render_deca_time'] = (
            np.stack(runtime_dict['sub_pure_render_deca_time']).tolist() if counter_sj > 0 else []
        )
        runtime_dict['all_pure_render_shadow_time'] = (
            np.stack(runtime_dict['sub_pure_render_shadow_time']).tolist() if counter_sj > 0 else []
        )
        runtime_dict['all_load_deca_time'] = runtime_dict['sub_load_deca_time']
        runtime_dict['all_load_deca_for_shadow_time'] = runtime_dict['sub_load_deca_for_shadow_time']
        runtime_dict['set'] = args.set
        runtime_dict['n_sj'] = counter_sj
        json.dump(runtime_dict, fj, indent=2)
