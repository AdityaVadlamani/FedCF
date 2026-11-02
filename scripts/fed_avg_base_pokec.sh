#!/bin/bash

PROJECTDIR="<REPLACE_ME>"  # absolute path to this repository
CONDAENV=fedcf

ds_folder="$PROJECTDIR/datasets"

DATASET="Pokec"

export PYTHONPATH=$PROJECTDIR
for sens_attr_prefix in "" "_gender" "_region"; do
    sens_attr="${sens_attr_prefix//_}"
    if [ -z "${sens_attr}" ]; then
        sens_attr="region_gender"
    fi
    for train_frac in 0.3; do
        for val_frac in 0.2; do
            best_param_path="$PROJECTDIR/base_fed_model/configs/base_pokec_config.yaml"
            if [ ! -f $best_param_path ]; then
                echo "Best parameter file not found for ${DATASET} with train_frac=${train_frac} and val_frac=${val_frac}"
                continue
            fi
            config_output_dir="$PROJECTDIR/outputs/${DATASET}/split/${train_frac}_${val_frac}${sens_attr_prefix}"
            mkdir -p ${config_output_dir}
            job_id="best_${DATASET}_split_${train_frac}_${val_frac}${sens_attr_prefix}"

sbatch <<EOT
#!/bin/bash
#SBATCH --account <REPLACE_ME>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=1-05:00:00
#SBATCH -J best_${DATASET}
#SBATCH -o ${PROJECTDIR}/logs/best/${DATASET}/best_${train_frac}_${val_frac}${sens_attr_prefix}_%A.out
#SBATCH -e ${PROJECTDIR}/logs/best/${DATASET}/best_${train_frac}_${val_frac}${sens_attr_prefix}_%A.err

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
--dataset.name ${DATASET} \
--dataset_dir ${ds_folder} \
--dataset_split_fractions.train ${train_frac} \
--dataset_split_fractions.valid ${val_frac} \
--dataset.sens_attrs '["${sens_attr}"]' \
--dataset.force_reprep True \
--job_id ${job_id} \
--num_clients 2 \
--fraction_fit 0.5 \
--resource_config.cpus 8 \
--resource_config.gpus 1
EOT
        done
    done
done
