#!/bin/bash
#SBATCH --job-name=EEG
#SBATCH --output=logs/run_%A_%a.log
#SBATCH --error=logs/run_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4        
#SBATCH --mem=8G                 
#SBATCH --time=06:00:00

# Check that an argument was provided
if [ -z "$1" ]; then
  echo "ERROR: No age_group provided."
  exit 1
fi

AGE_GROUP=$1

eval "$(mamba shell hook --shell=bash)"
mamba activate Magistrska

echo "Running on host: $(hostname)"
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "Starting training with age_group=${AGE_GROUP}..."

python train_finetuned.py \
  --lr 0.00005 \
  --batch_size 16 \
  --emb_size 64 \
  --nhead 8 \
  --num_layers 2 \
  --dropout 0.4 \
  --epochs 15 \
  --age_group ${AGE_GROUP} \
  --num_channels 6

echo "Training finished."
