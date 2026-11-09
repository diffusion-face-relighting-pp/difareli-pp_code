import argparse

def build_parser():
    parser = argparse.ArgumentParser(description="Single-shot face relighting and reshadowing (DiFaReli++).")

    # --- Dataset ---
    ds_group = parser.add_argument_group("Dataset")
    ds_group.add_argument('--dataset_root', type=str, required=True, help='Root path of preprocessed relighting dataset')
    ds_group.add_argument('--set', type=str, required=True, help='Dataset split (train, valid, test)')
    ds_group.add_argument('--shadow_diff_dir', type=str, default=None, help='Override path for shadow_diff_SS_with_c_simplified')

    # --- Model & Checkpoint ---
    model_group = parser.add_argument_group("Model & Checkpoint")
    model_group.add_argument('--step', type=str, required=True, help='Checkpoint step (e.g., 300000)')
    model_group.add_argument('--ckpt_selector', type=str, default='ema', help='Checkpoint type (ema or model)')
    model_group.add_argument('--cfg_name', type=str, required=True, help='Config YAML filename (e.g., DiFaReli++.yaml)')
    model_group.add_argument('--log_dir', type=str, required=True, help='Directory containing model checkpoint')

    # --- Sample Selection & Interpolation ---
    samp_group = parser.add_argument_group("Sample Selection & Interpolation")
    samp_group.add_argument('--idx', nargs=2, type=int, default=[0, 99999], help='Start and end sample pair indices')
    samp_group.add_argument('--sample_pair_json', type=str, default=None, help='JSON file defining src/dst pairs')
    samp_group.add_argument('--sample_pair_mode', type=str, default='pair', help='Pairing mode (e.g., pair)')
    samp_group.add_argument('--src_dst', nargs='+', default=[], help='Explicit [src, dst] image names')
    samp_group.add_argument('--itp', nargs='+', default=['render_face'], help='Interpolation targets')
    samp_group.add_argument('--itp_step', type=int, default=15, help='Number of interpolation steps (frames)')
    samp_group.add_argument('--batch_size', type=int, default=15, help='Inference sub-batch size')
    samp_group.add_argument('--render_batch_size', type=int, default=60, help='DECA & shadow ray-tracing batch size')
    samp_group.add_argument('--lerp', action='store_true', default=True, help='Use linear interpolation (default)')
    samp_group.add_argument('--slerp', action='store_true', default=False, help='Use spherical linear interpolation')

    # --- Lighting & 3D Rendering ---
    ren_group = parser.add_argument_group("Lighting & 3D Rendering")
    ren_group.add_argument('--render_mode', type=str, default='shape', help='DECA render mode (shape or template_shape)')
    ren_group.add_argument('--rasterize_type', type=str, default='pytorch3d', help='DECA rasterizer (pytorch3d or standard)')
    ren_group.add_argument('--rotate_normals', action='store_true', default=False)
    ren_group.add_argument('--rotate_sh', action='store_true', default=False, help='Rotate source SH lighting')
    ren_group.add_argument('--rotate_sh_dst', action='store_true', default=False, help='Rotate destination SH lighting')
    ren_group.add_argument('--rotate_sh_axis', type=int, default=0, help='SH rotation axis (0, 1, or 2)')
    ren_group.add_argument('--force_diffuse_shading_for_reshadow', action='store_true', default=False, help='Zero out directional SH bands for matted/diffuse DECA rendering')
    ren_group.add_argument('--force_src_shading_for_reshadow', action='store_true', default=False, help='Keep source SH constant for reshadowing')
    ren_group.add_argument('--itp_diffuse_sh', action='store_true', default=False, help='Interpolate from src SH to diffuse SH')
    ren_group.add_argument('--scale_sh', type=float, default=1.0, help='Scale factor on target SH coefficients')
    ren_group.add_argument('--fixed_render', action='store_true', default=False, help='Lock DECA rendered shading across target frames')
    ren_group.add_argument('--fixed_shadow', action='store_true', default=False, help='Lock ray-traced shadow mask across target frames')
    ren_group.add_argument('--force_render', action='store_true', default=False)

    # --- Ray-Traced Shadow & Shadow Weight (C) Control ---
    shad_group = parser.add_argument_group("Shadow Map & Weight Control")
    shad_group.add_argument('--pt_radius', type=float, default=0.2, help='Light perturbation radius for soft shadows')
    shad_group.add_argument('--pt_round', type=int, default=1, help='Number of perturbed light rays (1 = hard ray)')
    shad_group.add_argument('--scale_depth', type=float, default=256.0, help='Depth scaling factor for ray-tracing')
    shad_group.add_argument('--force_zero_src_shadow', action='store_true', default=False, help='Zero out frame-0 source shadow map')
    shad_group.add_argument('--shadow_diff_fidx_frac', type=float, default=0.0, help='Reference frame fraction when locking spatial shadow mask')
    # Relighting C modes
    shad_group.add_argument('--relight_with_given_c', type=float, default=None, help='Relight using a specific shadow scalar C in [0, 1]')
    shad_group.add_argument('--relight_with_dst_c', action='store_true', default=False, help='Relight using destination subject shadow scalar C')
    shad_group.add_argument('--relight_with_rand_max_c', action='store_true', default=False, help='Relight using random C in [0.7, 1.0]')
    shad_group.add_argument('--relight_with_strongest_c', action='store_true', default=False, help='Relight using maximum C = 1.0')
    # Reshadowing C modes
    shad_group.add_argument('--shadow_diff_inc_c', action='store_true', default=False, help='Reshadow: linearly increase C toward 1.0 (max shadow)')
    shad_group.add_argument('--shadow_diff_dec_c', action='store_true', default=False, help='Reshadow: linearly decrease C toward 0.0 (diffuse)')
    shad_group.add_argument('--reshadow_with_given_c', type=float, default=None, help='Starting C value for reshadowing (default: src C)')

    # --- Output & Misc ---
    out_group = parser.add_argument_group("Output & Misc")
    out_group.add_argument('--out_dir', type=str, required=True, help='Root output directory')
    out_group.add_argument('--eval_dir', type=str, default=None, help='Optional evaluation output directory')
    out_group.add_argument('--postfix', type=str, default='', help='Optional tag recorded in metadata')
    out_group.add_argument('--save_vid', action='store_true', default=False, help='Save MP4 videos in <src_name>/vid/')
    out_group.add_argument('--fps', nargs='?', const=20, default=20, type=int, help='Video frame rate (default: 20)')
    out_group.add_argument('--gpu_id', type=str, default='0', help='CUDA_VISIBLE_DEVICES ID')
    out_group.add_argument('--seed', type=int, default=23, help='Random seed')

    # --- Legacy / No-Op Compatibility Flags (accepted so older CLI commands do not break) ---
    legacy_group = parser.add_argument_group("Legacy Compatibility Flags (No-Op)")
    legacy_group.add_argument('--postproc_shadow_mask_smooth', action='store_true', default=True)
    legacy_group.add_argument('--inverse_with_shadow_diff', action='store_true', default=True)
    legacy_group.add_argument('--render_same_mask', action='store_true', default=False)
    legacy_group.add_argument('--add_remove_shadow', action='store_true', default=False)
    legacy_group.add_argument('--add_shadow_to', type=float, default=None)
    legacy_group.add_argument('--remove_shadow_to', type=float, default=None)

    return parser