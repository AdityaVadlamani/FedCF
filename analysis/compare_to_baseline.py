"""
Compare mean efficiency/violation for the DP-noise and distribution-shift trials
against the baseline (no DP noise, no shift) trials, grouped by closeness (c).

Usage:
    python analysis/compare_to_baseline.py
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(
    "analysis/fairness_trials/ACSIncome/continental_large/split_0.3_0.2/Demographic_Parity/raps_rand_no_mod.csv"
)
DP_DIR = Path(
    "analysis/fairness_trials_dp/ACSIncome/continental_large/split_0.3_0.2/Demographic_Parity/raps_rand_no_mod.csv"
)
SHIFT_DIR = Path(
    "analysis/fairness_trials_shift_het/ACSIncome/continental_large/split_0.3_0.2/Demographic_Parity/raps_rand_no_mod.csv"
)

METRICS = ["efficiency", "violation"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in METRICS:
        df[col] = (
            df[col]
            .str.replace("tensor(", "", regex=False)
            .str.replace(")", "", regex=False)
            .astype(float)
        )
    return df


def shift_type(row: pd.Series) -> str:
    if row["distribution_shift_subsample"]:
        return "subsample"
    if row["distribution_shift_shuffle_group"]:
        return "shuffle_group"
    if row["distribution_shift_swap_extreme_groups"]:
        return "swap_extreme_groups"
    return "none"


def to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table without needing `tabulate`."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def cell(eff: float, viol: float) -> str:
    return f"{eff:.4f}/{viol:.4f}"


def make_table(
    baseline_stat: pd.DataFrame,
    data_stat: pd.DataFrame,
    index_vals,
    column_vals,
    column_label_fn,
) -> pd.DataFrame:
    """Build a wide-format table: rows = index_vals (c), columns = baseline +
    one per column_val, cells = "{efficiency}/{violation}" from the given stat
    (mean or std) frames.
    """
    table = pd.DataFrame(index=index_vals)
    table.index.name = "c"
    table["baseline"] = [cell(*baseline_stat.loc[c, METRICS]) for c in index_vals]
    for col_val in column_vals:
        table[column_label_fn(col_val)] = [
            cell(*data_stat.loc[(c, col_val), METRICS]) for c in index_vals
        ]
    return table.reset_index()


def main():
    baseline = load(BASE_DIR)
    baseline_mean = baseline.groupby("c")[METRICS].mean()
    baseline_std = baseline.groupby("c")[METRICS].std()

    # --- DP noise vs baseline, columns = baseline + each dp_epsilon ---
    dp = load(DP_DIR)
    dp_mean = dp.groupby(["c", "dp_epsilon"])[METRICS].mean()
    dp_std = dp.groupby(["c", "dp_epsilon"])[METRICS].std()
    dp_cs = sorted(dp["c"].unique())
    dp_epsilons = sorted(dp["dp_epsilon"].unique())
    dp_label_fn = lambda eps: f"dp_epsilon={eps:g}"

    dp_table = make_table(baseline_mean, dp_mean, dp_cs, dp_epsilons, dp_label_fn)
    dp_table_std = make_table(baseline_std, dp_std, dp_cs, dp_epsilons, dp_label_fn)

    print("## DP noise vs baseline (efficiency/violation) -- mean\n")
    print(to_markdown(dp_table))
    print()
    print("## DP noise vs baseline (efficiency/violation) -- std\n")
    print(to_markdown(dp_table_std))
    print()

    # --- Distribution shift vs baseline, columns = baseline + each shift type ---
    shift = load(SHIFT_DIR)
    shift["shift_type"] = shift.apply(shift_type, axis=1)
    shift_mean = shift.groupby(["c", "shift_type"])[METRICS].mean()
    shift_std = shift.groupby(["c", "shift_type"])[METRICS].std()
    shift_cs = sorted(shift["c"].unique())
    shift_types = sorted(shift["shift_type"].unique())
    shift_label_fn = lambda st: st

    shift_table = make_table(
        baseline_mean, shift_mean, shift_cs, shift_types, shift_label_fn
    )
    shift_table_std = make_table(
        baseline_std, shift_std, shift_cs, shift_types, shift_label_fn
    )

    print("## Distribution shift vs baseline (efficiency/violation) -- mean\n")
    print(to_markdown(shift_table))
    print()
    print("## Distribution shift vs baseline (efficiency/violation) -- std\n")
    print(to_markdown(shift_table_std))


if __name__ == "__main__":
    main()
