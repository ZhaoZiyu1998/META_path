#!/bin/bash

# Define the base output directory for logs
output_dir="SGC_ACM"
mkdir -p "$output_dir"

# Define arrays of hyperparameters
ks=(2 4)
learning_rates=(1e-6 5e-6 1e-4 5e-4 1e-3 5e-3 1e-2)
weight_decays=(0 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3)
dropouts=(0 0.15 0.3 0.5 0.7)
# ks=(2)
# learning_rates=(1e-6 5e-2)
# weight_decays=(1e-5)
# dropouts=(0.15)
# Path to your main project directory (adjust as needed)
project_dir="/home/wxy/projects/def-caxwc/wxy/code/OpenHGNN"

# Original config file path
config_file="$project_dir/openhgnn/config.ini"

for k in "${ks[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for wd in "${weight_decays[@]}"; do
            for dropout in "${dropouts[@]}"; do
                # Unique identifier for each job
                unique_id="k${k}_lr${lr}_wd${wd}_dropout${dropout}"

                # Path for the job-specific config file
                job_config_file="${project_dir}/openhgnn/config_${unique_id}.ini"

                # Copy the original config file to a job-specific one
                cp "$config_file" "$job_config_file"

                # Modify the job-specific config file
                sed -i "
                /\\[SGC\\]/,/^$/{
                    s/^k = .*/k = $k/;
                    s/^dropout = .*/dropout = $dropout/;
                    s/^learning_rate = .*/learning_rate = $lr/;
                    s/^weight_decay = .*/weight_decay = $wd/;
                }" "$job_config_file"

                # Define the log file path
                log_file="${output_dir}/pap_sgc_${unique_id}.log"

                # Create and submit a job script
                sbatch << EOF
#!/bin/bash
#SBATCH --job-name=SGC-ACM-${unique_id}
#SBATCH --output=$log_file
#SBATCH --ntasks=1
#SBATCH --time=0:10:00
#SBATCH --mem=4G

# Execute the job
python $project_dir/main.py -m SGC -d HGBn-ACM -t node_classification -g -1 -c $job_config_file

EOF
            done
        done
    done
done
