#!/bin/bash
read -p "Input the video(s) to run (e.g. Anakin_2, Sull_1, Jon_1) : " videos
read -p "Running on : " gpu

# conda activate dpm_create_dataset

for vid_name in $videos
do
    echo "[!] Running : "$vid_name
    echo -e "\n\n"
    CUDA_VISIBLE_DEVICES="$gpu" python ./create_dataset.py --image_dir /data/mint/DPM_Dataset/Videos/"$vid_name"/aligned_images/valid/ --out_dataset_dir /data/mint/DPM_Dataset/Videos/"$vid_name"/ --faceseg --deca --arcface --shadow
    echo "[!] Done : "$vid_name
done

echo "[!] Finished all : "$videos
                                                                                                                                                                                                                                                                                                                                                                                                                