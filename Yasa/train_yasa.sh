#!/bin/bash
#SBATCH --job-name=EEG_Train           # Job name
#SBATCH --output=logs/train_%j.log     # Standard output and error log
#SBATCH --error=logs/train_%j.err
#SBATCH --cpus-per-task=8               # CPU cores per task
#SBATCH --mem=32G                        # Memory per node
#SBATCH --time=01:00:00                 # Max runtime (HH:MM:SS)
#SBATCH --partition=frida                  

eval "$(mamba shell hook --shell=bash)"
mamba activate yasa-env


python yasa_eval.py

