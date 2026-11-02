#!/bin/bash

PROJECTDIR="<REPLACE_ME>"  # absolute path to this repository
CONDAENV=fedcf
ds_folder="$PROJECTDIR/datasets/fitzpatrick17k"  # folder with the Fitzpatrick17k images

export PYTHONPATH=$PROJECTDIR

for DATASET in "Fitzpatrick"; do
    for num_clients in 1 2 4 8; do
        for train_frac in 0.3; do
            for val_frac in 0.2; do
                best_param_path="$PROJECTDIR/base_fed_model/configs/base_${DATASET,,}_config.yaml"
                if [ ! -f $best_param_path ]; then
                    echo $best_param_path
                    echo "Best parameter file not found for ${DATASET} with train_frac=${train_frac} and val_frac=${val_frac}"
                    continue
                fi
                config_output_dir="$PROJECTDIR/outputs/${DATASET}/split/${train_frac}_${val_frac}_${num_clients}"
                mkdir -p ${config_output_dir}
                job_id="best_${DATASET}_split_${train_frac}_${val_frac}_${num_clients}"

sbatch <<EOT
#!/bin/bash
#SBATCH --account <REPLACE_ME>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=1-05:00:00
#SBATCH -J best_${DATASET}
#SBATCH -o ${PROJECTDIR}/logs/best/${DATASET}/best_${train_frac}_${val_frac}_%A.out
#SBATCH -e ${PROJECTDIR}/logs/best/${DATASET}/best_${train_frac}_${val_frac}_%A.err

echo Job started at `date` on `hostname`
# CONDA SETUP
source ~/.bashrc
conda deactivate
conda activate ${CONDAENV}

cd $PYTHONPATH
python base_fed_model/main.py \
--config_path=${best_param_path} \
--logging_config.use_wandb False \
--output_dir ${config_output_dir} \
--dataset_dir ${ds_folder} \
--dataset.csv_file ${PROJECTDIR}/fitzpatrick17k.csv \
--dataset_split_fractions.train ${train_frac} \
--dataset_split_fractions.valid ${val_frac} \
--job_id ${job_id} \
--epochs 20 \
--num_server_rounds 50 \
--num_clients ${num_clients} \
--fraction_fit 0.5
EOT
            done
        done
    done
done