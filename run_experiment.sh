#!/bin/bash

# Directory for log files
log_dir="./log_files"
mkdir -p $log_dir  # Create the directory if it doesn't exist

# Define datasets and models
datasets=("academic4HetGNN" )  # Adjusted to your dataset
models=("HetGNN")         # Adjusted to your model

# Parameters for the task and usage of best config
task="link_prediction"
use_best_config="--use_best_config"

# Loop through datasets and models
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        job_name="${model}_${dataset}_${task}"
        echo "Submitting job: $job_name"

        # Define a unique log file for each job
        job_log_file="$log_dir/${job_name}.log"

        # Construct the command
        command="python main.py -m ${model} -d ${dataset} -t ${task} -g 0 ${use_best_config}"

        # Submit the job
        sbatch --job-name=$job_name \
               --output="${log_dir}/${job_name}_%j.out" \
               --error="${log_dir}/${job_name}_%j.err" \
               --time=03:00:00 \
               --gres=gpu:1 \
               --mem=40G \
               --cpus-per-task=1 \
               --wrap="echo \"$(date): Starting $job_name\" > $job_log_file; $command >> $job_log_file 2>&1; echo \"$(date): Finished $job_name\" >> $job_log_file"
    done
done
