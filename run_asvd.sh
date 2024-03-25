#!/bin/bash
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --constraint vram23
#SBATCH -t 3-04:00:00  # Zero day, four hour
#SBATCH -o slurm-asvd_test-%j.out
export CUDA_VISIBLE_DEVICES='0' # set environment variable

# python asvd.py --model_id="facebook/opt-125m" --act_aware --alpha 0.5 --n_calib_samples 16 --scaling_method abs_mean --ppl_target 40 --use_cache

python asvd.py --model_id="meta-llama/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache
