#!/bin/bash 
#SBATCH -c 10
#SBATCH -t 4-00:0 #Request runtime of 4 days
#SBATCH --gres=gpu:volta:1
#SBATCH --exclusive
#SBATCH -o ./logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ./logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
module load anaconda/2023a-pytorch


python llama.py meta-llama/Llama-2-7b-hf c4 ALPS 0.5 --seed 0 --nm_n 0 --nm_m 0 --model_path /home/gridsan/xmeng/Sparse_NN_shared/LLM/model --data_path /home/gridsan/xmeng/Sparse_NN_shared/LLM/data/ 