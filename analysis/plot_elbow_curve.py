"""
Elbow curve: closeness criterion (c) vs. prediction set size (efficiency).

Each CSV is one FedCF trial file (e.g. raps_rand_no_mod.csv). Rows are
averaged over seeds for each c. The left panel shows efficiency vs c with the
fairness-agnostic base predictor as a dashed line; the right panel shows the
measured fairness disparity vs c with the y = c target line. The elbow (the
point farthest from the chord joining the first and last c) is circled.

Usage:
    # Elbow sweep (c from 0.025 to 0.3) on ACSIncome continental large:
    python analysis/plot_elbow_curve.py \
        "Dem. Parity=analysis/fairness_trials/ACSIncome/continental_large/split_0.3_0.2/Demographic_Parity/raps_rand_no_mod.csv" \
        --out analysis/figures/elbow_acsincome_continental_large.pdf

    # One curve per DP epsilon from a single file:
    python analysis/plot_elbow_curve.py \
        analysis/fairness_trials_dp/ACSIncome/continental_large/split_0.3_0.2/Demographic_Parity/raps_rand_no_mod.csv \
        --hue dp_epsilon --out analysis/figures/elbow_dp.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["efficiency", "violation", "base_efficiency"]
# Fixed categorical order (never cycled) plus a marker per series, so curves
# stay distinguishable without color.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED = "#0b0b0b", "#52514e"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in METRICS:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"tensor\(|\)", "", regex=True), errors="coerce"
        )
    return df


def elbow_index(x: np.ndarray, y: np.ndarray) -> int:
    """Kneedle-style elbow: point with max distance from the chord (normalized axes)."""
    if len(x) < 3:
        return -1
    xn = (x - x.min()) / (np.ptp(x) or 1)
    yn = (y - y.min()) / (np.ptp(y) or 1)
    chord = np.array([xn[-1] - xn[0], yn[-1] - yn[0]])
    chord /= np.linalg.norm(chord) or 1
    pts = np.stack([xn - xn[0], yn - yn[0]], axis=1)
    dist = np.abs(pts[:, 0] * chord[1] - pts[:, 1] * chord[0])
    return int(dist.argmax())


def series(args):
    """Yield (label, DataFrame) pairs, one per plotted curve."""
    for spec in args.csvs:
        label, _, path = spec.rpartition("=")
        df = load(path)
        if args.query:
            df = df.query(args.query)
        label = label or Path(path).parent.name.replace("_", " ")
        if args.hue:
            for val, sub in df.groupby(args.hue):
                yield f"{label} ({args.hue}={val})" if len(args.csvs) > 1 else f"{args.hue} = {val}", sub
        else:
            yield label, df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+", help="CSV paths, optionally prefixed 'Label=path'")
    p.add_argument("--hue", help="extra column to split curves by (e.g. dp_epsilon, use_mle)")
    p.add_argument("--query", help="pandas query to select rows first, e.g. 'use_mle == False'")
    p.add_argument("--out", default="analysis/figures/elbow_curve.pdf")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    fig, (ax_eff, ax_viol) = plt.subplots(1, 2, figsize=(9, 3.4))
    base_effs, all_c = [], set()

    for i, (label, df) in enumerate(series(args)):
        g = df.groupby("c")[METRICS].agg(["mean", "std"]).sort_index()
        c = g.index.to_numpy(dtype=float)
        all_c.update(c)
        color, marker = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
        for ax, m in ((ax_eff, "efficiency"), (ax_viol, "violation")):
            mean, std = g[(m, "mean")].to_numpy(), g[(m, "std")].fillna(0).to_numpy()
            ax.plot(c, mean, color=color, marker=marker, lw=2, ms=7, label=label)
            ax.fill_between(c, mean - std, mean + std, color=color, alpha=0.15, lw=0)
        k = elbow_index(c, g[("efficiency", "mean")].to_numpy())
        if k >= 0:
            ax_eff.scatter(c[k], g[("efficiency", "mean")].iloc[k], s=220,
                           facecolors="none", edgecolors=INK, lw=1.2, zorder=5)
        base_effs.append(g[("base_efficiency", "mean")].mean())
        print(f"{label}: elbow at c = {c[k] if k >= 0 else 'n/a'}")
        print(g[[("efficiency", "mean"), ("violation", "mean")]].round(4).to_string(), "\n")

    ax_eff.axhline(np.mean(base_effs), color=MUTED, ls="--", lw=1.2, label="Base (fairness-agnostic)")
    cs = np.array(sorted(all_c))
    ax_viol.plot(cs, cs, color=MUTED, ls="--", lw=1.2, label="Target ($y = c$)")

    ax_eff.set_ylabel("Avg. prediction set size")
    ax_viol.set_ylabel("Worst-case fairness disparity")
    for ax in (ax_eff, ax_viol):
        ax.set_xlabel("Closeness criterion $c$")
        if len(cs) <= 5:
            ax.set_xticks(cs)
        ax.grid(alpha=0.3, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax_eff.legend(frameon=False, fontsize=8)
    ax_viol.legend(frameon=False, fontsize=8)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
