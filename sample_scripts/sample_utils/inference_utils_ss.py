import pytorch_lightning as pl
import torch as th
import numpy as np
import blobfile as bf
import mani_utils, params_utils, sh_utils, tonemapper, hdr_utils
import cv2, PIL
import time


class PLSampling(pl.LightningModule):
    def __init__(self, 
                 model_dict, 
                 cfg, 
                 args=None, 
                 denoised_fn=None):
        
        super(PLSampling, self).__init__()
        self.denoised_fn = denoised_fn
        self.model_dict = model_dict 
        self.cfg = cfg
        self.args = args
        self.const_noise = th.randn((len(cfg.img_model.dpm_cond_img), 3, cfg.img_model.image_size, cfg.img_model.image_size)).cuda()
        
    def forward_cond_network(self, model_kwargs):
        with th.no_grad():
            if self.cfg.img_cond_model.apply:
                x = model_kwargs['cond_img'].cuda().float()
                img_cond = self.model_dict[self.cfg.img_cond_model.name](
                    x=x,
                    emb=None,
                )
                # Override the condition
                if self.cfg.img_cond_model.override_cond != "":
                    model_kwargs[self.cfg.img_cond_model.override_cond] = img_cond
                else: raise NotImplementedError
        return model_kwargs

    def forward_nodpm(self, src_xstart, model_kwargs, t=None):
        model = self.model_dict[self.cfg.img_model.name]
        if model_kwargs['dpm_cond_img'] is not None:
            relight_out = model(th.cat((src_xstart, model_kwargs['dpm_cond_img']), dim=1).float(), t, **model_kwargs)
        else:
            relight_out = model(src_xstart.float(), t, **model_kwargs)
        return relight_out

def to_tensor(cond, key, device):
    for k in key:
        if k not in cond.keys():
            continue
        else:
            if isinstance(cond[k], list):
                for i in range(len(cond[k])):
                    cond[k][i] = th.tensor(cond[k][i]).to(device)
            else:
                if th.is_tensor(cond[k]):
                    cond[k] = cond[k].clone().to(device)
                elif isinstance(cond[k], np.ndarray):
                    cond[k] = th.tensor(cond[k]).to(device)
                else:
                    continue
    return cond

def eval_mode(model_dict):
    for k, _ in model_dict.items():
        model_dict[k].eval()
    return model_dict

def prepare_cond_sampling(cond, cfg, use_render_itp=False, device='cuda'):
    """
    Prepare a condition for encoder network (e.g., adding noise, share noise with DPM)
    :param noise: noise map used in DPM
    :param t: timestep
    :param model_kwargs: model_kwargs dict
    """
    out_cond = {}
    def construct_cond_tensor(pair_cfg, sj_paired):
        """Construct a condition tensor from a list of pairs."""
        out_shape = cond['src_deca_masked_face_images_woclip'].shape[0] + cond['dst_deca_masked_face_images_woclip'].shape[0]
        out_cond_img = []
        for i in range(out_shape):
            cond_img = []
            for j, (k, p) in enumerate(pair_cfg):
                if use_render_itp: 
                    if sj_paired[j] == 'src':
                        tmp_img = cond[f'src_{k}'][0:1]
                    elif sj_paired[j] == 'dst' and i == 0:
                        tmp_img = cond[f'src_{k}'][0:1]
                    elif sj_paired[j] == 'dst' and i != 0:
                        tmp_img = cond[f'dst_{k}'][[i-1]]
                    else: raise NotImplementedError
                else: 
                    tmp_img = cond[f'{k}_img']
                cond_img.append(tmp_img.to(device))
            cond_img = th.cat((cond_img), dim=1)
            out_cond_img.append(cond_img)
        out_cond_img = th.cat((out_cond_img), dim=0)

        return out_cond_img.to(device)
    
        
    if cfg.img_cond_model.apply:
        out_cond['cond_img'] = construct_cond_tensor(pair_cfg=list(zip(cfg.img_cond_model.in_image, 
                                                                  cfg.img_cond_model.noise_dpm_cond_img)),
                                                     sj_paired = cfg.img_cond_model.sj_paired)
    else:
        out_cond['cond_img'] = None

    out_cond['dpm_cond_img'] = None
   
    return out_cond

def build_condition_image(cond, misc, force_render=False):
    src_idx = misc['src_idx']
    dst_idx = misc['dst_idx']
    n_step = misc['n_step']
    batch_size = misc['batch_size']
    render_batch_size = misc['render_batch_size']
    avg_dict = misc['avg_dict']
    dataset = misc['dataset']
    args = misc['args']
    condition_img = misc['condition_img']
    img_size = misc['img_size']
    itp_func = misc['itp_func']
    deca_obj = misc['deca_obj']
    clip_ren = None
    
    def prep_render(cond, cond_img_name, adjust_contrast=None, alpha_images=None):
        #Note: Preprocessing to separate the shading ref or shadow mask into src-dst
        rendered_tmp = []
        for j in range(n_step):
            # Resize the rendered image to the desired size
            r_tmp = deca_rendered[j].cpu().numpy().transpose((1, 2, 0))
            r_tmp = cv2.resize(r_tmp, (img_size, img_size), cv2.INTER_AREA)
            r_tmp = np.transpose(r_tmp, (2, 0, 1))
            clip_ren = False
            rendered_tmp.append(r_tmp)

        rendered_tmp = np.stack(rendered_tmp, axis=0)
        cond[cond_img_name] = th.tensor(rendered_tmp).cuda()
        cond[f'src_{cond_img_name}'] = th.tensor(rendered_tmp[[0]]).cuda()
        cond[f'dst_{cond_img_name}'] = th.tensor(rendered_tmp[1:]).cuda()
        
        return cond, clip_ren
    
    def prep_shadow(cond, cond_img_name):
        shadow_diff_tmp = []
        for j in range(n_step):
            sd_tmp = shadow_mask[j]
            sdkk_tmp = shadow_kk[j]

            m_glasses_and_eyes = cond[f'{cond_img_name}_meg_mask'][src_idx].cpu().numpy()
            m_face_parsing = cond[f'{cond_img_name}_mface_mask'][src_idx].cpu().numpy()
            m_face = m_face_parsing
            
            sd_tmp_proc = (((1 - sd_tmp) > 0) * 1.0) * (m_face * (1-m_glasses_and_eyes)) * ((1 - sdkk_tmp) > 0.0)
            sd_tmp = sd_tmp_proc

            shadow_diff_tmp.append(sd_tmp)
            
        shadow_diff_tmp = np.stack(shadow_diff_tmp, axis=0)
        cond[cond_img_name] = th.tensor(shadow_diff_tmp).cuda()
        
        # Frame-0th is always the src shadow_diff, the rest are interpolated, rotated, diffused as desired
        shadow_diff_tmp[0] = cond['shadow_diff_img'][src_idx]
        cond[f'src_{cond_img_name}'] = th.tensor(shadow_diff_tmp[[0]]).cuda()
        cond[f'dst_{cond_img_name}'] = th.tensor(shadow_diff_tmp[1:]).cuda()
        return cond
    
    # Handling the render face renderer
    if np.any(['deca' in i for i in condition_img]) or np.any(['shadow_diff' in i for i in condition_img]):
        # Render the face
        if 'render_face' in args.interpolate:
            if args.rotate_sh:
                print("[#] Rotate SH mode of src light...")
                interp_cond = mani_utils.rotate_sh(cond, src_idx=src_idx, n_step=n_step, axis=args.rotate_sh_axis)
            elif args.force_diffuse_sh:
                print("[#] Diffuse SH mode of src light...")
                interp_cond = mani_utils.diffuse_sh(cond, src_idx=src_idx, n_step=n_step, interpolate=args.itp_diffuse_sh)
                interp_cond['light'][0:1] = cond['light'][src_idx]  # Always keep the first frame as src light
            elif args.same_sh:
                print("[#] Same SH mode of src light...")
                interp_cond = {'light':cond['light'][src_idx].repeat(n_step, 1)}
            else:
                print("[#] Interpolating SH mode from src->dst light...")
                interp_cond = mani_utils.iter_interp_cond(cond, interp_set=['light'], src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
            cond.update(interp_cond)
        else:
            #NOTE: Render w/ same light
            repeated_cond = mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=['light'])
            cond.update(repeated_cond)
            
        start_t = time.time()
        if np.any(['deca_masked' in n for n in condition_img]) or force_render:
            mask = params_utils.load_flame_mask()
        else: mask=None
        
        #NOTE: Render DECA in minibatch
        print("[#] Total steps : ", n_step)
        sub_step = mani_utils.ext_sub_step(n_step, render_batch_size)
        load_deca_time = time.time() - start_t
        all_render = []
        all_alpha_images = []
        render_time = []
        all_shadow_mask = []
        all_shadow_kk = []
        pure_render_deca_time = []
        pure_render_shadow_time = []
        
        for i in range(len(sub_step)-1):
            print(f"[#] Sub step rendering : {sub_step[i]} to {sub_step[i+1]}")
            start = sub_step[i]
            end = sub_step[i+1]
            sub_cond = cond.copy()
            sub_cond['light'] = sub_cond['light'][start:end, :]
            # Deca rendered : B x 3 x H x W
            start_sub_render_deca_t = time.time()
            deca_rendered, orig_visdict = params_utils.render_deca(deca_params=sub_cond, 
                                                                idx=src_idx, n=end-start, 
                                                                avg_dict=avg_dict, 
                                                                render_mode=args.render_mode, 
                                                                rotate_normals=args.rotate_normals, 
                                                                mask=mask,
                                                                deca_obj=deca_obj,
                                                                repeat=True)

            deca_alpha_images = orig_visdict['alpha_images']  # [B, 1, H, W]
            sub_render_deca_t = time.time() - start_sub_render_deca_t
            print("[#] Rendering with the shadow mask from face + scalp of render face...")
            if i == 0:
                load_deca_for_shadow_time = time.time()
                flame_face_scalp = params_utils.load_flame_mask(['face', 'scalp', 'left_eyeball', 'right_eyeball'])
                deca_obj_face_scalp = params_utils.init_deca(mask=flame_face_scalp, rasterize_type=args.rasterize_type) # Init DECA with mask only once
                load_deca_for_shadow_time = time.time() - load_deca_for_shadow_time
            
            start_sub_render_shadow_t = time.time()
            shadow_mask, shadow_kk, render_ld = params_utils.render_shadow_mask_with_smooth(
                                            sh_light=sub_cond['light'], 
                                            cam=sub_cond['cam'][src_idx],
                                            verts=orig_visdict['trans_verts_orig'], 
                                            deca={'face_scalp':deca_obj_face_scalp}, 
                                            axis_0=args.rotate_sh_axis==0 and (args.rotate_sh or args.rotate_sh_dst),
                                            axis_1=args.rotate_sh_axis==1 and (args.rotate_sh or args.rotate_sh_dst),
                                            device='cpu',   # Prevent OOM
                                            org_h=img_size, org_w=img_size,
                                            rt_dict={'pt_round':args.pt_round, 'pt_radius':args.pt_radius, 'scale_depth':args.scale_depth}
                                        )
            sub_render_shadow_t = time.time() - start_sub_render_shadow_t
            if i == len(sub_step)-2:
                del deca_obj_face_scalp
            all_render.append(deca_rendered)
            all_alpha_images.append(deca_alpha_images)
            render_time.append(time.time() - start_t)
            pure_render_deca_time.append(sub_render_deca_t)
            pure_render_shadow_time.append(sub_render_shadow_t)
            
            all_shadow_mask.append(shadow_mask[:, None, ...])
            all_shadow_kk.append(shadow_kk[:, None, ...])
            
        
        render_time = np.mean(render_time) + load_deca_time
        cond['render_time'] = render_time
        cond['pure_render_deca_time'] = pure_render_deca_time
        cond['pure_render_shadow_time'] = pure_render_shadow_time
        cond['load_deca_time'] = load_deca_time
        cond['load_deca_for_shadow_time'] = load_deca_for_shadow_time
        print("Rendering time : ", time.time() - start_t)
        
        if args.fixed_render and (args.shadow_diff_inc_c or args.shadow_diff_dec_c):
            print("[#] Fixed the Deca renderer for Reshadowing...")
            ff = all_render[0][0:1]
            fidx = int(args.shadow_diff_fidx_frac * n_step)
            rf = all_render[0][fidx:fidx+1].repeat_interleave(repeats=n_step-1, dim=0)
            deca_rendered = th.cat((ff, rf), dim=0)
        elif args.fixed_render:
            print("[#] Fixed the Deca renderer")
            deca_rendered = all_render[0][0:1].repeat_interleave(repeats=n_step, dim=0)
        else:
            deca_rendered = th.cat(all_render, dim=0)
            
        if args.fixed_shadow:
            print("[#] Fixed the Shadow mask")
            shadow_mask = all_shadow_mask[0][0:1].repeat_interleave(repeats=n_step, dim=0)
        else:
            shadow_mask = th.cat(all_shadow_mask, dim=0)
            shadow_kk = th.cat(all_shadow_kk, dim=0)
        
        
    print("Conditoning with image : ", condition_img)
    for i, cond_img_name in enumerate(condition_img):
        if ('faceseg' in cond_img_name) or ('face_structure' in cond_img_name):
            bg_tmp = [cond[f"{cond_img_name}_img"][src_idx]] * n_step
            if th.is_tensor(cond[f"{cond_img_name}_img"][src_idx]):
                bg_tmp = th.stack(bg_tmp, dim=0)
            else:
                bg_tmp = np.stack(bg_tmp, axis=0)
            cond[f"src_{cond_img_name}"] = th.tensor(bg_tmp)
        elif ('deca' in cond_img_name):
            cond, clip_ren = prep_render(cond, cond_img_name, adjust_contrast=args.adjust_contrast, alpha_images=all_alpha_images)
        elif ('shadow_diff' in cond_img_name):
            cond = prep_shadow(cond, cond_img_name)
    
    if force_render:
        cond, clip_ren = prep_render(cond, 'deca_masked_face_images_woclip')
    
    return cond, clip_ren, misc

def shadow_diff_with_weight_postproc(cond, misc, device='cuda'):
    # This function is used to post-process the shadow_diff mask when we inject shadow weight into shadow_diff
    condition_img = misc['condition_img']
    n_step = misc['n_step']
    src_idx = misc['src_idx']
    dst_idx = misc['dst_idx']
    args = misc['args']
    
    if 'src_shadow_diff_with_weight_simplified' in cond.keys() and 'dst_shadow_diff_with_weight_simplified' in cond.keys():
        #NOTE: Concatenate the src and dst shadow_diff_with_weight_simplified to process it together
        cond['src_and_dst_shadow_diff_with_weight_simplified'] = th.cat((cond['src_shadow_diff_with_weight_simplified'], cond['dst_shadow_diff_with_weight_simplified']), dim=0)
        condition_img.append('src_and_dst_shadow_diff_with_weight_simplified')
        
    # Max-Min c-values:  -4.985533880236826 7.383497233314015 from valid set
    # Max-Min c-values: -4.989461058405101 8.481700287326827
    max_c = 8.481700287326827 # 7.383497233314015
    min_c = -4.989461058405101 # -4.985533880236826
    c_val_src = (cond['shadow'][src_idx] - min_c) / (max_c - min_c)  # Scale to 0-1
    c_val_src = 1 - c_val_src # Inverse the shadow value
    c_val_dst = (cond['shadow'][dst_idx] - min_c) / (max_c - min_c)  # Scale to 0-1
    c_val_dst = 1 - c_val_dst # Inverse the shadow value

    if args.shadow_diff_inc_c:
        print(f"[#] Increasing the shadow_diff with weight...")
        if args.reshadow_with_given_c is not None:
            print(f"[#] From the given C = {args.reshadow_with_given_c} to 1.0 for {n_step-1} steps...")
            given_c = args.reshadow_with_given_c
            weight_src = th.linspace(start=given_c, end=1.0, steps=n_step-1).to(device)
            weight_src = weight_src[..., None, None, None]
            weight_dst = th.linspace(start=given_c, end=1.0, steps=n_step-1).to(device)
            weight_dst = weight_dst[..., None, None, None]
        else:
            print(f"[#] From the default C = {c_val_src.item()}")
            weight_src = th.linspace(start=c_val_src.item(), end=1.0, steps=n_step-1).to(device)
            weight_src = weight_src[..., None, None, None]
            weight_dst = th.linspace(start=c_val_dst.item(), end=1.0, steps=n_step-1).to(device)
            weight_dst = weight_dst[..., None, None, None]
        fix_frame = True 
    elif args.shadow_diff_dec_c:
        print("[#] Decreasing the shadow_diff with weight...")
        if args.reshadow_with_given_c is not None:
            print(f"[#] From the given C = {args.reshadow_with_given_c} to 0.0 for {n_step-1} steps...")
            given_c = args.reshadow_with_given_c
            weight_src = th.linspace(start=given_c, end=0.0, steps=n_step-1).to(device)
            weight_src = weight_src[..., None, None, None]
            weight_dst = th.linspace(start=given_c, end=0.0, steps=n_step-1).to(device)
            weight_dst = weight_dst[..., None, None, None]
        else:
            print(f"[#] Default C is {c_val_src.item()}")
            weight_src = th.linspace(start=c_val_src.item(), end=0.0, steps=n_step-1).to(device)
            weight_src = weight_src[..., None, None, None]
            weight_dst = th.linspace(start=c_val_dst.item(), end=0.0, steps=n_step-1).to(device)
            weight_dst = weight_dst[..., None, None, None]
        fix_frame = True 
    elif args.postproc_shadow_mask_smooth_diffuse_to_shadow:
        print("[#] Diffuse to shadow...")
        fix_frame = True
        weight_src = c_val_src[..., None, None, None].to(device)
        weight_dst = c_val_dst[..., None, None, None].to(device)
        weight_ph = th.ones_like(c_val_src[..., None, None, None]).to(device)  # Placeholder
    else:
        print("[#] No re-weighting for shadow_diff...")
        print(f"[#] Processing with weight = {c_val_src} and relight...")
        if ('shadow_diff_with_weight_simplified' in condition_img or 
            'src_shadow_diff_with_weight_simplified' in condition_img or
            'dst_shadow_diff_with_weight_simplified' in condition_img):
            print("[#] Conditioning is shadow_diff_with_weight_simplified, set the weight to c_val...")
            weight_src = c_val_src[..., None, None, None].to(device)
            weight_dst = c_val_dst[..., None, None, None].to(device)
            if args.relight_with_strongest_c:
                print("[#] Relight with the strongest c_val, set the weight to 0.0...")
                weight_src = (c_val_src * 0.0).to(device)
                weight_dst = (c_val_dst * 0.0).to(device)
        else:
            print("[#] For supporting DiFaReli_SS (without cast shadows)...")
            weight_src = c_val_src[..., None, None, None].to(device)
            weight_dst = c_val_dst[..., None, None, None].to(device)
        fix_frame = False

    for _, cond_img_name in enumerate(condition_img):
        if ((cond_img_name == 'shadow_diff_with_weight_simplified') or 
            (cond_img_name == 'src_and_dst_shadow_diff_with_weight_simplified')
            ):
            print(f"[#] Post-processing the {cond_img_name}...")
            if fix_frame:
                fidx = int(args.shadow_diff_fidx_frac * n_step)
                tmp = th.repeat_interleave(cond[cond_img_name][fidx:fidx+1], repeats=n_step-1, dim=0)
                # Always preserve first frame, the rest is determined by the shadow_diff_fidx_frac
                sd_img = cond[cond_img_name].clone()
                sd_img = th.cat((sd_img[0:1], tmp), dim=0)
            else:
                sd_img = cond[cond_img_name].clone()
            
            if args.relight_with_shadow_diff:
                # First frame (0 is non-shadow, > 0 is shadow)   
                sd_shadow = sd_img[0:1] > 0.
                shadow_area = sd_shadow * (1-weight_src)     # Shadow area assigned weight
                shadow_ff = shadow_area
                # Rest of the frames
                if args.relight_with_dst_c:
                    shadow_rf = ((sd_img[1:] > 0.) * (1-weight_dst))
                elif args.relight_with_rand_max_c:
                    rand_c = th.tensor(np.random.uniform(0.7, 1.0, size=weight_src.shape)).type_as(weight_src)
                    print(f"[#] Relight with random c_val [0.7 to 1] => {rand_c}")
                    shadow_rf = ((sd_img[1:] > 0.) * ((weight_dst * 0.) + rand_c))
                elif args.relight_with_given_c is not None:
                    print(f"[#] Relight with the given c_val = {args.relight_with_given_c}")
                    shadow_rf = ((sd_img[1:] > 0.) * ((weight_dst * 0.) + args.relight_with_given_c))
                else:
                    shadow_rf = ((sd_img[1:] > 0.) * (1-weight_src))    # Shadow area assigned weight

                # Final frames
                shadow = th.cat((shadow_ff, shadow_rf), dim=0)
                out_sd = shadow
                cond[cond_img_name] = out_sd
            else:
                if args.postproc_shadow_mask_smooth_keep_shadow_shading:
                    #NOTE: Keep the shadow shading from perturbed light

                    # First frame (0 is non-shadow, > 0 is shadow)   
                    sd_shadow = sd_img[0:1] > 0.
                    shadow_area = sd_shadow * (1-weight_src)     # Shadow area assigned weight
                    shadow_ff = shadow_area
                    # Rest of the frames
                    if args.relight_with_dst_c:
                        shadow_rf = (sd_img[1:] * (1-weight_dst))
                    elif args.relight_with_rand_max_c:
                        rand_c = th.tensor(np.random.uniform(0.7, 1.0, size=weight_src.shape)).type_as(weight_src)
                        print(f"[#] Relight with random c_val [0.7 to 1] => {rand_c}")
                        shadow_rf = ((sd_img[1:] > 0.) * ((weight_dst * 0.) + rand_c))
                    elif args.relight_with_given_c is not None:
                        print(f"[#] Relight with the given c_val = {args.relight_with_given_c}")
                        shadow_rf = ((sd_img[1:] > 0.) * ((weight_dst * 0.) + args.relight_with_given_c))
                    else:
                        shadow_rf = (sd_img[1:] * (1-weight_src))    # Shadow area assigned weight

                    # Final frames
                    shadow = th.cat((shadow_ff, shadow_rf), dim=0)
                    out_sd = shadow
                    cond[cond_img_name] = out_sd
                    print("[#] Unique of out_sd: ", th.unique(out_sd))
                    print("[#] Unique of shadow_ff: ", th.unique(shadow_ff))
                    print("[#] Unique of shadow_rf: ", th.unique(shadow_rf))
                elif args.postproc_shadow_mask_smooth:
                    #NOTE: Do not keep the shadow shading from perturbed light
                    # First frame
                    shadow_ff = sd_img[0:1]
                    # sd_shadow = sd_img[0:1] > 0.
                    # shadow_area = sd_shadow * (1-weight_src)     # Shadow area assigned weight
                    # shadow_ff = shadow_area
                    # Rest of the frames
                    if args.relight_with_dst_c:
                        print(f"[#] Relight with the dst c_val = {weight_dst.flatten()}")
                        shadow_rf = (sd_img[1:] * (1-weight_dst))
                    elif args.relight_with_rand_max_c:
                        rand_c = th.tensor(np.random.uniform(0.7, 1.0, size=weight_src.shape)).type_as(weight_src)
                        print(f"[#] Relight with random c_val [0.7 to 1] => {rand_c}")
                        shadow_rf = ((sd_img[1:] > 0.) * ((weight_dst * 0.) + rand_c))
                    elif args.relight_with_given_c is not None:
                        print(f"[#] Relight with the given c_val = {args.relight_with_given_c}")
                        shadow_rf = ((sd_img[1:] > 0.) * ((weight_dst * 0.) + args.relight_with_given_c))
                    elif args.reshadow_with_given_c is not None:
                        print(f"[#] Reshadow with the given c_val = {args.reshadow_with_given_c}")
                        shadow_rf = ((sd_img[1:] > 0.) * ((weight_src)))
                    elif args.reshadow_gradually_inc_c:
                        print(f"[#] Gradually increase c from 0.0 to 1.0")
                        gradual_c = th.linspace(start=0.0, end=1.0, steps=sd_img[1:].shape[0]).to(device)
                        shadow_rf = ((sd_img[1:] > 0.) * (gradual_c[..., None, None, None]))
                    else:
                        print(f"[#] Relight with the src c_val = {weight_src.flatten()}")
                        shadow_rf = (sd_img[1:] * (1-weight_src))    # Shadow area assigned weight
                    # Final frames
                    shadow = th.cat((shadow_ff, shadow_rf), dim=0)
                    out_sd = shadow
                    cond[cond_img_name] = out_sd
                    print("[#] Unique of out_sd: ", th.unique(out_sd))
                    print("[#] Unique of shadow_ff: ", th.unique(shadow_ff))
                    print("[#] Unique of shadow_rf: ", th.unique(shadow_rf))
                elif args.postproc_shadow_mask_smooth_diffuse_to_shadow and fix_frame:
                    #NOTE: Same as the postproc_shadow_mask_smooth but change the weight from 0.0 -> 1.0
                    #NOTE: Working with the fixed frame and relight with given c_val
                    # First frame
                    shadow_ff = sd_img[0:1]
                    # Rest of the frames
                    # Weight use is 0.0 -> given c_val -> 1.0 with the equal step
                    given_c = args.relight_with_given_c
                    n_step_diffuse = sd_img[1:].shape[0] // 2
                    n_step_shadow = sd_img[1:].shape[0] - n_step_diffuse
                    diffuse_to_given_c = th.linspace(start=0.0, end=given_c, steps=n_step_diffuse).to(device)
                    given_c_to_shadow = th.linspace(start=given_c, end=1.0, steps=n_step_shadow).to(device)
                    weight_use = th.cat((diffuse_to_given_c, given_c_to_shadow), dim=0)
                    weight_use = weight_use[..., None, None, None]
                    # print(weight_use, weight_use.shape)
                    # weight_use = weight_ph * weight_use[..., None, None, None]
                    print(weight_ph.shape, weight_use.shape)
                    shadow_rf = ((sd_img[1:] > 0.) * (weight_use))
                    
                    # Final frames
                    shadow = th.cat((shadow_ff, shadow_rf), dim=0)
                    out_sd = shadow
                    cond[cond_img_name] = out_sd
                    print("[#] Unique of out_sd: ", th.unique(out_sd))
                    print("[#] Unique of shadow_ff: ", th.unique(shadow_ff))
                    print("[#] Unique of shadow_rf: ", th.unique(shadow_rf))
                else:
                    print("[#] Please specify either postproc_shadow method...")
                    raise NotImplementedError
                
    # Update the src_and_dst_shadow_diff_with_weight_simplified into src_shadow_diff_with_weight_simplified and dst_shadow_diff_with_weight_simplified
    if 'src_and_dst_shadow_diff_with_weight_simplified' in condition_img:
        if args.force_zero_src_shadow:
            print("[#] Force the src_shadow_diff_with_weight_simplified to 0.0...")
            cond['src_shadow_diff_with_weight_simplified'] = (cond['src_and_dst_shadow_diff_with_weight_simplified'][0:1] * 0.0)
        else:
            cond['src_shadow_diff_with_weight_simplified'] = cond['src_and_dst_shadow_diff_with_weight_simplified'][0:1]
        cond['dst_shadow_diff_with_weight_simplified'] = cond['src_and_dst_shadow_diff_with_weight_simplified'][1:]
        print("[#] Shape of src_dst_shadow_diff_with_weight_simplified: ", cond['src_and_dst_shadow_diff_with_weight_simplified'].shape)
        print("[#] Shape of src_shadow_diff_with_weight_simplified: ", cond['src_shadow_diff_with_weight_simplified'].shape)
        print("[#] Shape of dst_shadow_diff_with_weight_simplified: ", cond['dst_shadow_diff_with_weight_simplified'].shape)
    return cond, {'src':weight_src, 'dst':weight_dst}
