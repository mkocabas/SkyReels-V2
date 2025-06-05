#!/bin/bash

#SBATCH --job-name=cap_fusion    # Job name
#SBATCH --output=logs/%j.out    # Output file name (%j expands to jobID)
#SBATCH --error=logs/%j.err     # Error file name (%j expands to jobID)
#SBATCH --ntasks=1              # Number of tasks (processes)
#SBATCH --cpus-per-task=16      # Number of CPU cores per task
#SBATCH --gres=gpu:1            # Number of GPUs required
#SBATCH --time=24:00:00         # Time limit hrs:min:sec
#SBATCH --array=1-38%32         # Array job with max 4 concurrent jobs


txt_path=/home/muhammed/projects/prompt_hmr/data/bedlam_all_mp4_list.txt

increment=1000
start_idxs=($(seq 0 $increment $(($(wc -l < $txt_path) - $increment))))

PYTHON_EXE=/home/muhammed/miniconda3/envs/skycaptioner/bin/python

echo "Debug: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Debug: Accessing index ${start_idxs[$SLURM_ARRAY_TASK_ID]}"
echo "Debug: Array size=${#start_idxs[@]}"

# Adjust array index to be 0-based
adjusted_idx=$((SLURM_ARRAY_TASK_ID - 1))

$PYTHON_EXE scripts/vllm_fusion_caption.py --model_path models/Qwen2.5-32B-Instruct --input_txt $txt_path --start_idx ${start_idxs[$adjusted_idx]} --end_idx $((${start_idxs[$adjusted_idx]}+$increment)) --task t2v

$PYTHON_EXE scripts/vllm_fusion_caption.py --model_path models/Qwen2.5-32B-Instruct --input_txt $txt_path --start_idx ${start_idxs[$adjusted_idx]} --end_idx $((${start_idxs[$adjusted_idx]}+$increment)) --task i2v

echo "Done for $adjusted_idx"