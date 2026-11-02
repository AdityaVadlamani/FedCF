#!/bin/bash

PROJECTDIR="<REPLACE_ME>"  # absolute path to this repository
CONDAENV=fedcf
TRAIN_FRAC=0.3
VAL_FRAC=0.2
ALPHA=0.1

base_output_dir="$PROJECTDIR/outputs"
ds_folder="$PROJECTDIR/datasets"

# arr=(1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0)
# client_formulation="[$(IFS=,; echo "${arr[*]}")]"
client_formulation=0 # 0 for low overhead, 1 for more private
echo "Client formulation: ${client_formulation}"

# Values to sweep; edit these to narrow a run.
SEEDS=(0 1 2)
CLOSENESS=(0.1 0.15 0.2)
PARTITIONS=(small large all) # continental_small continental_large continental_all)
DATASETS=(ACSIncome ACSEducation)
METRICS=(Demographic_Parity Equal_Opportunity Predictive_Equality)
METHODS=(raps aps)
USE_MLES=(false true)
USE_GROUPS=(false true)

for conformal_seed in "${SEEDS[@]}"
do
    for closeness_measure in "${CLOSENESS[@]}"
    do
        for folktables_partition_type in "${PARTITIONS[@]}"
        do
            for DATASET in "${DATASETS[@]}"
            do
                base_job_id="best_${DATASET}_split_${TRAIN_FRAC}_${VAL_FRAC}_${folktables_partition_type}"
                config_output_dir="$base_output_dir/${DATASET}/split/${TRAIN_FRAC}_${VAL_FRAC}_${folktables_partition_type}"

                for METRIC in "${METRICS[@]}"
                do
                    for METHOD in "${METHODS[@]}"
                    do
                        for USE_MLE in "${USE_MLES[@]}"
                        do
                            for USE_GROUP in "${USE_GROUPS[@]}"
                            do
                            config_path="$PROJECTDIR/configs/fair_fedcf_acs_config.yaml"
                            cd ${PROJECTDIR}
sbatch <<EOT
#!/bin/bash
#SBATCH --account <REPLACE_ME>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH -J c_${DATASET}_${TRAIN_FRAC}_${VAL_FRAC}_${folktables_partition_type}_${METHOD}
#SBATCH -e logs/fedcf/${DATASET}/${TRAIN_FRAC}_${VAL_FRAC}_${folktables_partition_type}/${METRIC}/closeness_${closeness_measure}_%A.err
#SBATCH -o logs/fedcf/${DATASET}/${TRAIN_FRAC}_${VAL_FRAC}_${folktables_partition_type}/${METRIC}/closeness_${closeness_measure}_%A.out

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
    --fairness_metric=${METRIC} \
    --conformal_method=${METHOD} \
    --job_id=c_${DATASET}_${TRAIN_FRAC}_${VAL_FRAC}_${METHOD}_${sketch_method} \
    --dataset_split_fractions.train ${TRAIN_FRAC} \
    --dataset_split_fractions.valid ${VAL_FRAC}  \
    --alpha $ALPHA \
    --base_job_id=${base_job_id} \
    --dataset.name=${DATASET} \
    --output_dir=$config_output_dir \
    --dataset_dir ${ds_folder} \
    --folktables_partition_type=${folktables_partition_type} \
    --closeness_measure ${closeness_measure} \
    --quantile_method tdigest \
    --resource_config.cpus 8 \
    --resource_config.gpus 0 \
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
done
