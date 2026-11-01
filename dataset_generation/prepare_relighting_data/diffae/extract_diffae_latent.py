import sys
sys.path.append("/home/mint/Dev/DiFaReli/difareli-faster/guided_diffusion/")
from mint_logger import createLogger
from templates import *
from templates_latent import *
import argparse, pickle, os, glob
import blobfile as bf
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, default='./output_shadow_score')
parser.add_argument('--set_', type=str, default='valid')
args = parser.parse_args()

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

if __name__ == '__main__':
    logger = createLogger()
    logger.info(f"[#] Input dir: {args.image_dir}")
    logger.info(f"[#] Output dir: {args.out_dir}")
    logger.info(f"[#] Set: {args.set_}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    gpus = [0]
    conf = ffhq256_autoenc()
    conf.data_name = 'custom'
    
    for subject_folder in tqdm(os.listdir(args.image_dir)):
        # Iterate over subject folder
        image_dir = os.path.join(args.image_dir, subject_folder)
        
        conf.shadow_input_path = image_dir
        conf.latent_output_path = image_dir
        # im_name = sorted(glob.glob(os.path.join(args.image_dir, '*jpg')))
        im_name = _list_image_files_recursively(image_dir)
        logger.info(f"[#] Number of images: {len(im_name)}")
        
        logger.info(f"[#] Running inference to extract latent codes...")
        conf.eval_programs = ['infer', 'latent']
        train(conf, gpus=gpus, mode='eval')
        logger.info(f"[#] Finished inference.")
        lat = torch.load(os.path.join(image_dir, 'latent.pkl'), map_location='cpu')
        logger.info(f"[#] Latent file loaded from {os.path.join(image_dir, 'latent.pkl')}.")
        
        ndata = lat['conds'].numpy()
        
        lat['image_name'] = [im_name[i].split('/')[-1] for i in range(len(im_name))]
        # overwrite to latent.pkl
        torch.save(lat, os.path.join(image_dir, 'latent.pkl'))
        logger.info(f"[#] Latent file saved to {os.path.join(image_dir, 'latent.pkl')}.")
        