#!/bin/bash

PROJECTDIR="<REPLACE_ME>"  # absolute path to this repository
CONDAENV=fedcf
TRAIN_FRAC=0.3
VAL_FRAC=0.2
ALPHA=0.1
DATASET="Fitzpatrick"
base_output_dir="$PROJECTDIR/outputs"
ds_folder="$PROJECTDIR/datasets/fitzpatrick17k"  # folder with the Fitzpatrick17k images

# Values to sweep; edit these to narrow a run.
SEEDS=(0 1 2)
NUM_CLIENTS=(8)
CLOSENESS=(0.1 0.15 0.2)
SKETCH_METHODS=(mean tdigest ddsketch)
METRICS=(Demographic_Parity Predictive_Equality)
METHODS=(raps aps)
USE_MLES=(false true)

for conformal_seed in "${SEEDS[@]}"; do
for num_clients in "${NUM_CLIENTS[@]}"; do
# Hybrid client formulations: the first private_count clients use the
# enhanced-privacy protocol (1), the rest communication-efficient (0),
# for private_count = 0 .. num_clients.
    for ((private_count=0; private_count<=num_clients; private_count++)); do
        arr=()
        for ((i=1; i<=private_count; i++)); do arr+=($((1))); done
        for ((i=private_count+1; i<=num_clients; i++)); do arr+=($((0))); done
        client_formulation="[$(IFS=,; echo "${arr[*]}")]"
        echo "Client formulation: ${client_formulation}"
    
        for closeness_measure in "${CLOSENESS[@]}"; do
            for sketch_method in "${SKETCH_METHODS[@]}"; do
                config_output_dir="$PROJECTDIR/outputs/${DATASET}/split/${TRAIN_FRAC}_${VAL_FRAC}_${num_clients}"
                job_id="best_${DATASET}_split_${TRAIN_FRAC}_${VAL_FRAC}_${num_clients}_faster"

                for METRIC in "${METRICS[@]}"
                do
                    for METHOD in "${METHODS[@]}"
                    do
                        for USE_MLE in "${USE_MLES[@]}"
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
#SBATCH -e logs/fedcf/${DATASET}/${TRAIN_FRAC}_${VAL_FRAC}_${sketch_method}/${METRIC}/closeness_${closeness_measure}_%A.err
#SBATCH -o logs/fedcf/${DATASET}/${TRAIN_FRAC}_${VAL_FRAC}_${sketch_method}/${METRIC}/closeness_${closeness_measure}_%A.out

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
    --conformal_seed ${conformal_seed}

EOT
                            done
                        done
                    done
                done
            done
        done
    done
done