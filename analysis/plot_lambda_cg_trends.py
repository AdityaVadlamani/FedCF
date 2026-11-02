#!/usr/bin/env python3
"""
Plot cg_curr vs lambda iteration number with trendlines.
Each position in the tensor gets its own trendline.
"""

import re
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def extract_lambda_and_cg(file_path: str) -> Dict:
    """
    Extract cg_curr and q_hat from .err log file, and generate lambda values.

    Searches for:
    1. Lines with "q_hat:" to get the q_hat value
    2. Lines with "cg_curr" in the aggregate output to get cg_curr values
    3. Generates lambda values from q_hat to 2.0 over num_rounds

    Args:
        file_path: Path to the .err log file

    Returns:
        Dictionary with keys:
            - 'lambdas': List of lambda values (one per round, replicated for all positions)
            - 'cg_currents': List of cg_curr tensors (each as list of floats)
            - 'q_hat': q_hat value (float) or None if not found
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    cg_currents = []
    q_hat = None
    upper_bound = 2.0

    # Pattern to extract tensor values: tensor([val1, val2, ...])
    tensor_pattern = r"tensor\(\[([\d., ]+)\]\)"

    with open(file_path, "r") as f:
        lines = f.readlines()

    # First pass: extract q_hat value
    for line in lines:
        if "q_hat:" in line:
            try:
                q_hat_match = re.search(r"q_hat:\s*([\d.]+)", line)
                if q_hat_match:
                    q_hat = float(q_hat_match.group(1))
                    break  # q_hat should be the same throughout
            except (ValueError, IndexError):
                continue

    # Second pass: extract cg_curr values (these appear in aggregate_fit lines)
    for line in lines:
        if "cg_curr" in line and "tensor" in line and "INFO:root" in line:
            try:
                cg_match = re.search(r"'cg_curr':\s*" + tensor_pattern, line)
                if cg_match:
                    cg_str = cg_match.group(1)
                    cg_values = [
                        float(x.strip()) for x in cg_str.split(",") if x.strip()
                    ]
                    cg_currents.append(cg_values)
            except (ValueError, IndexError):
                continue

    # Generate lambda values from q_hat to upper_bound over num_rounds
    num_rounds = len(cg_currents)
    if q_hat is not None and num_rounds > 0:
        lambda_sequence = np.linspace(q_hat, upper_bound, num_rounds).tolist()
        # Convert to list of lists format (one value per round, replicated for all positions)
        num_positions = len(cg_currents[0]) if cg_currents else 0
        lambdas = [[lam] * num_positions for lam in lambda_sequence]
    else:
        lambdas = []

    return {"lambdas": lambdas, "cg_currents": cg_currents, "q_hat": q_hat}


def plot_trends(file_path: str, output_path: str = None, show_plot: bool = True):
    """
    Plot cg_curr values per lambda position with trendlines.

    Args:
        file_path: Path to the error/log file
        output_path: Optional path to save the plot image
        show_plot: Whether to display the plot
    """
    # Extract data
    results = extract_lambda_and_cg(file_path)

    lambdas = results["lambdas"]
    cg_currents = results["cg_currents"]
    q_hat = results.get("q_hat")

    if not lambdas or not cg_currents:
        print("Error: No lambda or cg_curr data found in file")
        return

    # Verify same number of entries
    if len(lambdas) != len(cg_currents):
        print(
            f"Warning: Found {len(lambdas)} lambdas but {len(cg_currents)} cg_curr values"
        )

    # Get the size of tensors (assume first one is representative)
    tensor_size = len(lambdas[0]) if lambdas else 0

    if tensor_size == 0:
        print("Error: Empty tensors")
        return

    # Verify all tensors have same size
    min_iterations = min(len(lambdas), len(cg_currents))

    # Create figure with subplots (max 3 per row)
    max_cols = 3
    num_cols = min(tensor_size, max_cols)
    num_rows = (tensor_size + max_cols - 1) // max_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Flatten axes for easier iteration
    if isinstance(axes, np.ndarray):
        if axes.ndim == 1:
            flat_axes = axes
        else:
            flat_axes = axes.flatten()
    else:
        flat_axes = [axes]

    # Plot each position
    for pos in range(tensor_size):
        ax = flat_axes[pos]

        # Collect lambda and cg_curr values for this position across iterations
        lambda_values = []
        cg_values = []

        for iter_idx in range(min_iterations):
            if pos < len(lambdas[iter_idx]) and pos < len(cg_currents[iter_idx]):
                lambda_values.append(lambdas[iter_idx][pos])
                cg_values.append(cg_currents[iter_idx][pos])

        if not cg_values:
            ax.text(
                0.5,
                0.5,
                f"Position {pos}: No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Calculate Spearman correlation (for monotonicity)
        if len(lambda_values) > 1:
            rho, p_value = spearmanr(lambda_values, cg_values)
            ax.scatter(
                lambda_values,
                cg_values,
                alpha=0.6,
                s=50,
                label=f"coverage gap values (ρ={rho:.3f})",
            )
        else:
            ax.scatter(
                lambda_values, cg_values, alpha=0.6, s=50, label="coverage gap values"
            )

        # Draw vertical line at q_hat if available
        if q_hat is not None:
            ax.axvline(
                x=q_hat,
                color="green",
                linestyle=":",
                linewidth=2,
                label=f"q_hat={q_hat:.4f}",
            )

        # Labels and title
        ax.set_xlabel(f"Lambda")
        ax.set_ylabel(f"Coverage Gap")
        ax.set_title(f"Label {pos + 1}: Coverage Gap vs Lambda")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide unused subplots
    for pos in range(tensor_size, len(flat_axes)):
        flat_axes[pos].set_visible(False)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_summary_table(file_path: str):
    """
    Print a summary table of lambda and cg_curr values.
    """
    results = extract_lambda_and_cg(file_path)

    lambdas = results["lambdas"]
    cg_currents = results["cg_currents"]

    if not lambdas or not cg_currents:
        print("No data found")
        return

    min_iterations = min(len(lambdas), len(cg_currents))
    tensor_size = len(lambdas[0]) if lambdas else 0

    print(f"\n{'='*80}")
    print(f"Summary Table: Lambda and cg_curr values")
    print(f"{'='*80}\n")

    for iter_idx in range(min_iterations):
        print(f"Iteration {iter_idx + 1}:")

        if iter_idx < len(lambdas):
            lambda_vals = lambdas[iter_idx]
            print(f"  Lambda: {[f'{v:.4f}' for v in lambda_vals]}")

        if iter_idx < len(cg_currents):
            cg_vals = cg_currents[iter_idx]
            print(f"  cg_curr: {[f'{v:.4f}' for v in cg_vals]}")

        print()


def process_directory(directory: str, pattern: str = "*.err", show_table: bool = False):
    """
    Process all matching files in a directory.

    Args:
        directory: Directory path
        pattern: File pattern to match (default: "*.err")
        show_table: Whether to show summary table instead of plot
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return

    files = sorted(dir_path.glob(pattern))
    if not files:
        print(f"No files matching '{pattern}' found in {directory}")
        return

    print(f"Found {len(files)} file(s) matching '{pattern}'")

    for file_path in files:
        print(f"\nProcessing: {file_path.name}")
        if show_table:
            plot_summary_table(str(file_path))
        else:
            # Save plots with filename-based output path
            output_name = file_path.stem + "_plot.png"
            output_path = file_path.parent / output_name
            plot_trends(str(file_path), output_path=str(output_path), show_plot=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python plot_lambda_cg_trends.py <file_path>")
        print("  python plot_lambda_cg_trends.py <file_path> --save <output.png>")
        print("  python plot_lambda_cg_trends.py <file_path> --table")
        print(
            "  python plot_lambda_cg_trends.py <directory>  (processes all .err files)"
        )
        print("  python plot_lambda_cg_trends.py <directory> --table")
        sys.exit(1)

    path = sys.argv[1]

    # Check for optional flags
    save_path = None
    show_table = False

    if len(sys.argv) > 2:
        if sys.argv[2] == "--save" and len(sys.argv) > 3:
            save_path = sys.argv[3]
        elif sys.argv[2] == "--table":
            show_table = True

    path_obj = Path(path)

    if path_obj.is_file():
        if show_table:
            plot_summary_table(path)
        else:
            plot_trends(path, output_path=save_path, show_plot=True)
    elif path_obj.is_dir():
        process_directory(path, show_table=show_table)
    else:
        print(f"Error: {path} is neither a file nor a directory")
        sys.exit(1)
