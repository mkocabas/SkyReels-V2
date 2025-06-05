#!/bin/bash

#SBATCH --job-name=vid2cap    # Job name
#SBATCH --output=logs/%j.out    # Output file name (%j expands to jobID)
#SBATCH --error=logs/%j.err     # Error file name (%j expands to jobID)
#SBATCH --ntasks=1              # Number of tasks (processes)
#SBATCH --cpus-per-task=16      # Number of CPU cores per task
#SBATCH --gres=gpu:1            # Number of GPUs required
#SBATCH --time=24:00:00         # Time limit hrs:min:sec
#SBATCH --array=1-10%32         # Array job with max 4 concurrent jobs


# for bedlam2
increment=3000
start_idxs=(0 3000 6000 9000 12000 15000 18000 21000 24000 27000)

# for bedlam
# increment=2000
# start_idxs=(0 2000 4000 6000 8000 10000)

PYTHON_EXE=/home/muhammed/miniconda3/envs/skycaptioner/bin/python

echo "Debug: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Debug: Accessing index ${start_idxs[$SLURM_ARRAY_TASK_ID]}"
echo "Debug: Array size=${#start_idxs[@]}"

# Adjust array index to be 0-based
adjusted_idx=$((SLURM_ARRAY_TASK_ID - 1))

$PYTHON_EXE scripts/vllm_struct_caption.py --model_path models/skycaptioner_v1 --input_txt /home/muhammed/projects/prompt_hmr/data/bedlam2_mp4_list.txt --start_idx ${start_idxs[$adjusted_idx]} --end_idx $((${start_idxs[$adjusted_idx]}+$increment))

echo "Done for $adjusted_idx"