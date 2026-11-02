#!/bin/bash

PROJECTDIR="<REPLACE_ME>"  # absolute path to this repository
CONDAENV=fedcf
TRAIN_FRAC=0.3
VAL_FRAC=0.2
ALPHA=0.1
DATASET="Fitzpatrick"
base_output_dir="$PROJECTDIR/outputs"
ds_folder="$PROJECTDIR/datasets/fitzpatrick17k"  # folder with the Fitzpatrick17k images

client_formulation=0 # 0 for low overhead, 1 for more private

# Values to sweep; edit these to narrow a run.
SEEDS=(0 1 2)
NUM_CLIENTS=(1 2 4 8)
CLOSENESS=(0.1 0.15 0.2)
METRICS=(Demographic_Parity Predictive_Equality)
METHODS=(raps aps)
USE_MLES=(false true)
USE_GROUPS=(false true)

for conformal_seed in "${SEEDS[@]}"
do
    for num_clients in "${NUM_CLIENTS[@]}"
    do
        if [ $num_clients -eq 1 ]; then
            sketch_method="mean"
        else
            sketch_method="tdigest"
        fi
        for closeness_measure in "${CLOSENESS[@]}"
        do
            config_output_dir="$PROJECTDIR/outputs/${DATASET}/split/${TRAIN_FRAC}_${VAL_FRAC}_${num_clients}"
            job_id="best_${DATASET}_split_${TRAIN_FRAC}_${VAL_FRAC}_${num_clients}_faster"

            for METRIC in "${METRICS[@]}"
            do
                for METHOD in "${METHODS[@]}"
                do
                    for USE_MLE in "${USE_MLES[@]}"
                    do
                        for USE_GROUP in "${USE_GROUPS[@]}"
                        do
                            config_path="$PROJECTDIR/configs/fair_fedcf_fitzpatrick_config.yaml"
                            cd ${PROJECTDIR}
sbatch <<EOT
#!/bin/bash
#SBATCH --account <REPLACE_ME>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00
#SBATCH -J c_${DATASET}_${TRAIN_FRAC}_${VAL_FRAC}_${METHOD}
#SBATCH -e logs/fedcf/${DATASET}/${TRAIN_FRAC}_${VAL_FRAC}/${METRIC}/closeness_${closeness_measure}_%A.err
#SBATCH -o logs/fedcf/${DATASET}/${TRAIN_FRAC}_${VAL_FRAC}/${METRIC}/closeness_${closeness_measure}_%A.out

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
    --fairness_metric=${METRIC} \
    --conformal_method=${METHOD} \
    --job_id=c_${DATASET}_${TRAIN_FRAC}_${VAL_FRAC}_${METHOD}_${sketch_method} \
    --dataset_split_fractions.train ${TRAIN_FRAC} \
    --dataset_split_fractions.valid ${VAL_FRAC} \
    --alpha $ALPHA \
    --base_job_id=${job_id} \
    --dataset.name=${DATASET} \
    --output_dir=${config_output_dir} \
    --dataset_dir ${ds_folder} \
    --dataset.csv_file ${PROJECTDIR}/fitzpatrick17k.csv \
    --closeness_measure ${closeness_measure} \
    --quantile_method ${sketch_method} \
    --num_clients ${num_clients} \
    --resource_config.cpus 10 \
    --resource_config.gpus 0 \
    --client_formulations ${client_formulation} \
    --cf_opt.use_mle ${USE_MLE} \
    --cf_opt.use_group_information ${USE_GROUP} \
    --conformal_seed ${conformal_seed} 

EOT
                        done
                    done
                done
            done
        done
    done
done