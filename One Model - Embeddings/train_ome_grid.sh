#!/bin/bash

# 1. Define your Grid
learning_rates=(0.00005 0.0001 0.0003)
batch_sizes=(32 64)
layers=(2 3 4)
dropouts=(0.3 0.4 0.5)
configs=(
  "64 4"
  "128 4"
  "128 8"
)
num_channels=(6)
epochs_list=(15 30)


# 2. Generate the Command List File
COMMAND_FILE="jobs.txt"
> $COMMAND_FILE # Clear file

# Create the list of all python commands
for ep in "${epochs_list[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do
      for lay in "${layers[@]}"; do
        for drop in "${dropouts[@]}"; do
        for num_ch in "${num_channels[@]}"; do
          for config in "${configs[@]}"; do
              set -- $config
              EMB=$1
              HEAD=$2
              
              # Write command to file with the specific $ep variable
              echo "python train_normalize_per_group.py --lr $lr --batch_size $bs --emb_size $EMB --nhead $HEAD --num_layers $lay --dropout $drop --epochs $ep --num_channels $num_ch" >> $COMMAND_FILE
            done
          done
        done
      done
    done
  done
done

# Count jobs
TOTAL_JOBS=$(wc -l < $COMMAND_FILE)
echo "Generated $TOTAL_JOBS experiments."

sbatch --array=1-$TOTAL_JOBS%6 <<EOT
#!/bin/bash
#SBATCH --job-name=EEG_Grid
#SBATCH --output=logs/run_%A_%a.log
#SBATCH --error=logs/run_%A_%a.err
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=4        
#SBATCH --mem=8G                 
#SBATCH --time=08:00:00          

# Setup Environment
eval "\$(mamba shell hook --shell=bash)"
mamba activate Magistrska

# Pick the command corresponding to this Job ID
CMD=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" $COMMAND_FILE)

echo "Starting Array Job \$SLURM_ARRAY_TASK_ID"
echo "Command: \$CMD"

# Run it
\$CMD

EOT

echo "Submitted job array. Only 6 jobs will run at a time."