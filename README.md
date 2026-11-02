# Federated Conformal Fairness (FedCF)

A federated learning framework for fair conformal predictions using Flower (FLWR). FedCF combines conformal prediction with fairness constraints in a federated setting without sharing raw data across clients.

This repository contains the code for the NeurIPS 2026 paper "FedCF: Fair Federated Conformal Prediction" ([Paper](https://arxiv.org/pdf/2509.22907)).

FedCF extends the Conformal Fairness Framework from the ICLR 2025 paper "A Generic Framework for Conformal Fairness" ([Paper](https://openreview.net/pdf?id=xiQNfYl33p), [Code](https://github.com/AdityaVadlamani/conformal-fairness)) to the federated setting.

## Overview

The server runs a state machine (`flwr_cp_strategy.py`) and tells clients which stage to run via `configure_fit`. Clients only ever send summary statistics, never raw data.

1. **Federated conformal prediction (`fed_cp`)**: computes the federated quantile $\hat{q}(\alpha)$ that gives $1-\alpha$ marginal coverage.
2. **Population statistics (`pop_stat`)**: aggregates the per-(group, label) counts needed to bound the coverage of each group.
3. **Conformal fairness search (`cf_iter`)**: searches for the smallest per-label threshold $\lambda$ whose coverage gap between groups is at most the closeness criterion $c$. By default this uses the descent-based search from the paper (`cf_opt.use_descent_based_approach`).

Each client picks one of two protocols (`client_formulations`): `0` = communication efficient, `1` = enhanced privacy. Mixing them gives the hybrid setting.

## Environment Setup

Requires conda (or Miniconda). The environment uses Python 3.11.

```bash
conda env create -f env.yaml
conda activate fedcf
python -c "import flwr, torch, xgboost; print('ok')"
```

## Project Structure

```
FedCF/
├── run_fed_cf.py                  # Entry point for FedCF experiments
├── flwr_cp_strategy.py            # Server strategy (stages, coverage-gap aggregation, threshold search)
├── cp_server_app.py               # Flower server app
├── cp_fed_client_app.py           # Flower client app (coverage terms, DP noise)
├── fed_config.py                  # Experiment configuration (pyrallis dataclasses)
├── fed_constants.py               # Stages, protocols, ACS client partitions
├── fed_dp.py                      # Differential privacy helpers (Appendix I)
├── fed_utils.py                   # Shared utilities (including where results are written)
├── fed_data_splitter.py           # Splits data across clients
├── *_datamodule.py                # Folktables (ACS), Pokec, Fitzpatrick, FairLex data loading
├── conformal_fairness/            # Conformal Fairness core (scores, predictors, metrics)
├── base_fed_model/                # Federated training of the base models (FedAvg)
├── configs/                       # Default configs for each dataset
├── scripts/                       # SLURM scripts for the experiments
├── analysis/                      # Results, tables, figures, and analysis scripts
└── fitzpatrick17k.csv             # Fitzpatrick17k metadata (images are downloaded separately)
```

## Datasets

- **ACSIncome / ACSEducation** (Folktables): downloaded automatically to `<dataset_dir>/Folktables`. Clients are US states or regions, set with `--folktables_partition_type` (`small`, `large`, `all`, and their `continental_*` versions).
- **Pokec-{n,z}**: place the raw files under `<dataset_dir>` (same format as the [Conformal Fairness](https://github.com/AdityaVadlamani/conformal-fairness) repository).
- **Fitzpatrick17k**: the metadata is `fitzpatrick17k.csv`; download the images and pass their folder as `--dataset_dir`.

## Running Experiments

Running FedCF has two steps:

1. **Train the federated base model** (FedAvg) with `base_fed_model/main.py`, e.g. via `scripts/fed_avg_base_<dataset>.sh`. The checkpoint is saved under `--output_dir` with job id `best_<dataset>_split_<train>_<valid>_<partition>` for ACS, or `..._<num_clients>` for the other datasets.
2. **Run FedCF** with `run_fed_cf.py`, passing that job id as `--base_job_id`, e.g. via `scripts/fed_conformal_fairness_<dataset>.sh`.

The scripts in `scripts/` sweep seeds, closeness criteria, datasets, fairness metrics, and score functions, and submit one SLURM job per setting. Before running them, replace each `<REPLACE_ME>` (the repository path `PROJECTDIR` and the SLURM `--account`) and check the dataset paths and `#SBATCH` resources for your cluster.

A single run looks like:

```bash
python run_fed_cf.py --config_path=configs/fair_fedcf_acs_config.yaml \
    --dataset.name=ACSIncome \
    --folktables_partition_type=continental_large \
    --fairness_metric=Demographic_Parity \
    --conformal_method=raps \
    --closeness_measure 0.1 \
    --alpha 0.1 \
    --base_job_id=best_ACSIncome_split_0.3_0.2_continental_large \
    --output_dir=outputs/ACSIncome/split/0.3_0.2_continental_large \
    --dataset_dir ./datasets \
    --dataset_split_fractions.train 0.3 \
    --dataset_split_fractions.valid 0.2 \
    --quantile_method tdigest \
    --conformal_seed 0
```

Any field in `fed_config.py` can be overridden from the command line (e.g. `--cf_opt.num_opt_rounds 100`).

### Key Options

| Option | Values |
|---|---|
| `--fairness_metric` | `Demographic_Parity`, `Equal_Opportunity`, `Predictive_Equality`, `Equalized_Odds`, and others in `conformal_fairness/constants.py` |
| `--conformal_method` | `aps`, `raps`, `daps` (graph datasets only) |
| `--quantile_method` | `tdigest` (used in the paper), `ddsketch`, `mean` |
| `--client_formulations` | `0` (communication efficient), `1` (enhanced privacy), or a list with one value per client (hybrid) |
| `--cf_opt.use_descent_based_approach` | `True` (descent-based search) or `False` (grid search) |
| `--cf_opt.use_mle` | `True` to use MLE estimates instead of interval bounds |
| `--cf_opt.use_group_information` | `True` to use group-wise thresholds at inference time |
| `--distribution_shift_*` | Test-time perturbations for the robustness experiments (see `fed_config.py`) |
| `--add_dp_noise`, `--dp_epsilon`, `--dp_delta`, `--dp_release` | Differential privacy with truncated Laplace noise; `--dp_release` is `per_round` or `histogram` (one-shot noisy histogram, with `--dp_hist_bins`) |
| `--results_dir` | Folder under `analysis/` for the results (default `fairness_trials`) |

## Results and Analysis

Each run appends one row per setting to a CSV at
`analysis/<results_dir>/<dataset>/<clients or partition>/split_<train>_<valid>/<fairness_metric>/<score>.csv`.

- `analysis/fairness_trials*/`: the results used in the paper.
- `analysis/fedcf_analysis.ipynb`: builds the paper's tables (saved to `analysis/tables/`) and figures (saved to `analysis/figures/`).
- `analysis/plot_elbow_curve.py`: efficiency and disparity vs. closeness criterion (Figure K.8).
- `analysis/compare_to_baseline.py`: compares the DP and distribution-shift trials to the baseline.
- `analysis/plot_lambda_cg_trends.py`: plots the coverage gap over the threshold search from a run's `.err` log.

## Acknowledgements
The authors acknowledge support from National Science Foundation (NSF) grant #2112471 (AI-EDGE). Any opinions and findings are those of the author(s) and do not necessarily reflect the views of the granting agencies.

## Contact

<strong>Anutam Srinivasan:</strong> [Website](https://anutamks.github.io/) [Email](mailto:asrinivasan350@gatech.edu)

<strong>Aditya Vadlamani:</strong> [Website](https://adityavadlamani.github.io/) [Email](mailto:vadlamani.12@osu.edu)
