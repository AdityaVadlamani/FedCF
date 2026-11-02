#!/bin/bash

PROJECTDIR="<REPLACE_ME>"  # absolute path to this repository
CONDAENV=fedcf

DATASET="cail"

export PYTHONPATH=$PROJECTDIR
sens_attr_prefix=""
sens_attr="region_gender"

train_frac=0.3
val_frac=0.2

for num_clients in 2 4 8; do

best_param_path="$PROJECTDIR/base_fed_model/configs/base_fairlex_config.yaml"
if [ ! -f $best_param_path ]; then
    echo "Best parameter file not found for ${DATASET} with train_frac=${train_frac} and val_frac=${val_frac}"
    exit 1
fi
config_output_dir="$PROJECTDIR/outputs/${DATASET}/split/${train_frac}_${val_frac}${sens_attr_prefix}_${num_clients}"
mkdir -p ${config_output_dir}
job_id="best_${DATASET}_split_${train_frac}_${val_frac}${sens_attr_prefix}_${num_clients}"

sbatch <<EOT
#!/bin/bash
#SBATCH --account <REPLACE_ME>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH -J best_${DATASET}
#SBATCH -o ${PROJECTDIR}/logs/best/${DATASET}/best_${train_frac}_${val_frac}${sens_attr_prefix}_${num_clients}_%A.out
#SBATCH -e ${PROJECTDIR}/logs/best/${DATASET}/best_${train_frac}_${val_frac}${sens_attr_prefix}_${num_clients}_%A.err

echo Job started at `date` on `hostname`
# CONDA SETUP
source ~/.bashrc
conda deactivate
conda activate ${CONDAENV}

export TOKENIZERS_PARALLELISM=false

cd $PYTHONPATH
python base_fed_model/main.py \
--config_path=${best_param_path} \
--logging_config.use_wandb False \
--output_dir ${config_output_dir} \
--dataset.name ${DATASET} \
--dataset_split_fractions.train ${train_frac} \
--dataset_split_fractions.valid ${val_frac} \
--dataset.sens_attrs '["${sens_attr}"]' \
--dataset.force_reprep True \
--job_id ${job_id} \
--fraction_fit 0.5 \
--resource_config.cpus 8 \
--resource_config.gpus 1 \
--num_clients ${num_clients}

# Copy results to other sens attrs since the base model is the same

mkdir -p ${PROJECTDIR}/outputs/${DATASET}/split/${train_frac}_${val_frac}_${num_clients}_gender/${job_id}_gender/
cp ${config_output_dir}/${job_id}/all_prob_labels.pt ${PROJECTDIR}/outputs/${DATASET}/split/${train_frac}_${val_frac}_${num_clients}_gender/${job_id}_gender/

mkdir -p ${PROJECTDIR}/outputs/${DATASET}/split/${train_frac}_${val_frac}_${num_clients}_region/${job_id}_region/
cp ${config_output_dir}/${job_id}/all_prob_labels.pt ${PROJECTDIR}/outputs/${DATASET}/split/${train_frac}_${val_frac}_${num_clients}_region/${job_id}_region/

EOT
done