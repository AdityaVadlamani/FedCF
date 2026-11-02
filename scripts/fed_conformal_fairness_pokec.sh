#!/bin/bash

PROJECTDIR="<REPLACE_ME>"  # absolute path to this repository
CONDAENV=fedcf
TRAIN_FRAC=0.3
VAL_FRAC=0.2
ALPHA=0.1

base_output_dir="$PROJECTDIR/outputs"
ds_folder="$PROJECTDIR/datasets"
DATASET="Pokec"

client_formulation=1 # 0 for low overhead, 1 for more private

# Values to sweep; edit these to narrow a run.
# Sensitive-attribute prefix: "" = region and gender, "_gender", "_region".
SEEDS=(0 1 2)
CLOSENESS=(0.1 0.15 0.2)
SENS_ATTR_PREFIXES=("" "_gender" "_region")
METRICS=(Demographic_Parity Equal_Opportunity Predictive_Equality)
USE_MLES=(false true)
METHODS=(raps aps daps)
USE_GROUPS=(false true)

for conformal_seed in "${SEEDS[@]}"
do
    for closeness_measure in "${CLOSENESS[@]}"
    do
        for sens_attr_prefix in "${SENS_ATTR_PREFIXES[@]}"
        do
            sens_attr="${sens_attr_prefix//_}"
            if [ -z "${sens_attr}" ]; then
                sens_attr="region_gender"
            fi
                job_id="best_${DATASET}_split_${TRAIN_FRAC}_${VAL_FRAC}${sens_attr_prefix}"
                config_output_dir="$base_output_dir/${DATASET}/split/${TRAIN_FRAC}_${VAL_FRAC}${sens_attr_prefix}"
                for METRIC in "${METRICS[@]}"
                do
                    for USE_MLE in "${USE_MLES[@]}"
                    do
                        for METHOD in "${METHODS[@]}"
                        do
                            for USE_GROUP in "${USE_GROUPS[@]}"
                            do
                            config_path="$PROJECTDIR/configs/fair_fedcf_pokec_config.yaml"
                            cd ${PROJECTDIR}
sbatch <<EOT
#!/bin/bash
#SBATCH --account <REPLACE_ME>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH -J c_${DATASET}_${TRAIN_FRAC}_${VAL_FRAC}${sens_attr_prefix}_${METHOD}
#SBATCH -e logs/fedcf/${DATASET}/${TRAIN_FRAC}_${VAL_FRAC}${sens_attr_prefix}/${METRIC}/closeness_${closeness_measure}_%A.err
#SBATCH -o logs/fedcf/${DATASET}/${TRAIN_FRAC}_${VAL_FRAC}${sens_attr_prefix}/${METRIC}/closeness_${closeness_measure}_%A.out

echo Job started at `date` on `hostname`
# CONDA SETUP
source ~/.bashrc
conda deactivate
conda activate ${CONDAENV}

export DGLBACKEND=pytorch
export FEDCF_SHARED_NODE=1  # Ray uses only this job's CPUs (see run_fed_cf.py)


cd ${PROJECTDIR}
python run_fed_cf.py --config_path=${config_path} \
    --logging_config.use_wandb False \
    --dataset.name=${DATASET} \
    --dataset.sens_attrs '["${sens_attr}"]' \
    --fairness_metric=${METRIC} \
    --conformal_method=${METHOD} \
    --job_id=c_${DATASET}_${TRAIN_FRAC}_${VAL_FRAC}_${METHOD}_tdigest \
    --dataset_split_fractions.train ${TRAIN_FRAC} \
    --dataset_split_fractions.valid ${VAL_FRAC} \
    --alpha $ALPHA \
    --base_job_id=${job_id} \
    --output_dir=${config_output_dir} \
    --dataset_dir ${ds_folder} \
    --closeness_measure ${closeness_measure} \
    --quantile_method tdigest \
    --resource_config.cpus 4 \
    --resource_config.gpus 0 \
    --num_clients 2 \
    --client_formulations ${client_formulation} \
    --cf_opt.use_mle ${USE_MLE} \
    --cf_opt.use_group_information ${USE_GROUP} \
    --conformal_seed=${conformal_seed}
EOT
                        done
                    done
                done
            done
        done
    done
done
