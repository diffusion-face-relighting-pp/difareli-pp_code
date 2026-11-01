import argparse
import os
import subprocess
import sys

def get_conda_python(env_name):
    conda_exe = os.environ.get('CONDA_EXE', os.path.expanduser('~/miniconda3/bin/conda'))
    conda_base = os.path.dirname(os.path.dirname(conda_exe))
    return os.path.join(conda_base, 'envs', env_name, 'bin', 'python')


parser = argparse.ArgumentParser(description='Create relighting dataset annotations and renderings.')
parser.add_argument('--image_dir', help='Input images directory', required=True)
parser.add_argument('--out_dataset_dir', help='Output dataset directory', required=True)
parser.add_argument('--set', help='Dataset split, e.g. [train, valid, test]', default='valid')
parser.add_argument('--ffhq_align', action='store_true', default=False)
parser.add_argument('--faceseg', action='store_true', default=False)
parser.add_argument('--deca', action='store_true', default=False)
parser.add_argument('--arcface', action='store_true', default=False)
parser.add_argument('--shadow', action='store_true', default=False)
parser.add_argument(
    '--env_ffhq',
    help='Python executable for FFHQ alignment',
    default=os.environ.get('ENV_FFHQ', './FFHQ_align/ffhq_env/bin/python'),
)
parser.add_argument(
    '--env_create_dataset',
    help='Python executable for dataset creation (Arcface, DiffAE-Shadow)',
    default=os.environ.get('ENV_CREATE_DATASET', get_conda_python('difareli_pp_create_dataset')),
)
parser.add_argument(
    '--env_sampling',
    help='Python executable for sampling/rendering (FaceSeg, DECA)',
    default=os.environ.get('ENV_SAMPLING', get_conda_python('difareli_pp_sampling')),
)
args = parser.parse_args()

# Python environments
ENV_FFHQ = args.env_ffhq
ENV_CREATE_DATASET = args.env_create_dataset
ENV_SAMPLING = args.env_sampling

SEP_MAJOR = "=" * 100
SEP_MINOR = "-" * 100

def run_step(title, input_dir, output_dir, cmd):
    print(f"\n{SEP_MAJOR}")
    print(f"[#] Step       : {title}")
    print(f"[#] Input Dir  : {input_dir}")
    print(f"[#] Output Dir : {output_dir}")
    print(SEP_MINOR)
    print(f"[>] Command    : {' '.join(cmd)}")
    print(SEP_MINOR)
    subprocess.call(cmd)
    print(SEP_MAJOR)


if __name__ == '__main__':
    image_dir = os.path.abspath(args.image_dir)
    out_dataset_dir = os.path.abspath(args.out_dataset_dir)
    dataset_set = args.set

    faceseg_out_dir = os.path.join(out_dataset_dir, 'face_segment', dataset_set)
    deca_images_out_dir = os.path.join(out_dataset_dir, 'rendered_images', 'deca_masked_face_images')
    params_out_dir = os.path.join(out_dataset_dir, 'params', dataset_set)

    enabled_steps = [
        name for name, flag in [
            ('ffhq_align', args.ffhq_align),
            ('faceseg', args.faceseg),
            ('deca', args.deca),
            ('arcface', args.arcface),
            ('shadow', args.shadow),
        ] if flag
    ]

    print(SEP_MAJOR)
    print("[#] Dataset Creation Summary")
    print(SEP_MINOR)
    print(f"    Input Image Dir    : {image_dir}")
    print(f"    Output Dataset Dir : {out_dataset_dir}")
    print(f"    Dataset Split      : {dataset_set}")
    print(f"    Enabled Steps      : {', '.join(enabled_steps) if enabled_steps else 'None'}")
    print(SEP_MAJOR)

    if args.ffhq_align:
        # NOTE: Running the FFHQ align
        '''
        python align.py -i ../../ITW/itw_images -o ../../ITW/itw_images_aligned
        '''
        save_aligned = os.path.basename(os.path.normpath(image_dir)) + "_aligned"
        aligned_out_dir = os.path.join(out_dataset_dir, save_aligned, dataset_set)
        cmd = [
            ENV_FFHQ, './FFHQ_align/align.py',
            '-i', image_dir,
            '-o', aligned_out_dir,
        ]
        run_step(
            title="FFHQ Face Alignment",
            input_dir=image_dir,
            output_dir=aligned_out_dir,
            cmd=cmd,
        )
        image_dir = os.path.abspath(aligned_out_dir)

    if args.faceseg:
        # NOTE: Running Face segment (face-parsing.PyTorch)
        '''
        python inference.py --data ../../ITW/itw_images --output_path ../../face_segment/
        '''
        cmd = [
            ENV_SAMPLING, './face-parsing.PyTorch/inference.py',
            '--data', image_dir,
            '--output_path', faceseg_out_dir,
        ]
        run_step(
            title="Face Segmentation (face-parsing.PyTorch)",
            input_dir=image_dir,
            output_dir=faceseg_out_dir,
            cmd=cmd,
        )

    if args.deca:
        # NOTE: Running DECA
        '''
        python estimate_deca_for_dpm.py --useTex True --render_orig True 
        --inputpath ../../../ITW/itw_images/  --useTemplate False --useAvgCam False 
        --useAvgTform False --save_images_folder ../../../ITW/rendered_images/decaface_images 
        --set valid --fast_save_params False --save_params_folder ../../../ITW/params/valid
        '''
        cmd = [
            ENV_SAMPLING, './DECA/script/estimate_deca_for_dpm.py',
            '--useTex', 'True',
            '--inputpath', image_dir,
            '--useTemplate', 'False',
            '--useAvgCam', 'False',
            '--useAvgTform', 'False',
            '--save_images_folder', deca_images_out_dir,
            '--set', dataset_set,
            '--params_prefix', dataset_set,
            '--save_params_folder', params_out_dir,
            '--masking_flame',
            '--fast_save_params', 'False',
            '--rasterizer_type', 'pytorch3d',
        ]
        run_step(
            title="Face Rendering & Parameters (DECA)",
            input_dir=image_dir,
            output_dir=f"{deca_images_out_dir} | {params_out_dir}",
            cmd=cmd,
        )

    if args.arcface:
        # NOTE: Running Arcface
        '''
        python inference.py --image_dir ../../ITW/itw_images --set valid --out_dir ../../ITW/params/valid/
        '''
        cmd = [
            ENV_CREATE_DATASET, './Arcface/inference.py',
            '--image_dir', image_dir,
            '--set', dataset_set,
            '--out_dir', params_out_dir,
        ]
        run_step(
            title="Face Identity Embedding (Arcface)",
            input_dir=image_dir,
            output_dir=params_out_dir,
            cmd=cmd,
        )

    if args.shadow:
        # NOTE: Running DiffAE-Shadow
        '''
        python gen_shadow.py -i test_shadow -o ./shadow_out
        '''
        cmd = [
            ENV_CREATE_DATASET, './diffae/gen_shadow.py',
            '--image_dir', image_dir,
            '--set_', dataset_set,
            '--out_dir', params_out_dir,
        ]
        run_step(
            title="Shadow Embedding (DiffAE-Shadow)",
            input_dir=image_dir,
            output_dir=params_out_dir,
            cmd=cmd,
        )

    print(f"\n[#] All requested steps completed. Output saved to: {out_dataset_dir}")
