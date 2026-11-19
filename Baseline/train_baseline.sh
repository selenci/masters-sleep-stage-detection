#!/bin/bash
#SBATCH --job-name=EEG_Train           # Job name
#SBATCH --output=logs/train_%j.log     # Standard output and error log
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --cpus-per-task=8               # CPU cores per task
#SBATCH --mem=8G                        # Memory per node
#SBATCH --time=12:00:00                 # Max runtime (HH:MM:SS)
#SBATCH --partition=frida                  

eval "$(mamba shell hook --shell=bash)"
mamba activate Magistrska

# -------------------------------
# Optional: Print some info
# -------------------------------
echo "Running on host: $(hostname)"
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi

# -------------------------------
# Run your training script
# -------------------------------
python -c "import torch; print(torch.version.cuda)"
python train.py --lr 0.00005 --batch_size 16 --emb_size 64 --nhead 8 --num_layers 2 --dropout 0.4 --epochs 15 --num_channels 3

