import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import (
    DEFAULT_DATA_PATH, FEATURE_NAMES, TARGET, load_concrete_data,
    split_train_test, standardize_train_test,
)
from run_q3 import train_model


def milestone_text(value):
    return f"{value:,}" if value is not None else "not reached"


def make_figure(models, results, settings, path):
    colors = {"untransformed": "#c96929", "standardized": "#2475ad", "log_transformed": "#218269"}
    labels = {"untransformed": "Untransformed", "standardized": "Standardized", "log_transformed": "log(x + 1)"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#263443", "text.color": "#263443",
        "axes.titleweight": "bold", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.2))
    fig.subplots_adjust(left=0.075, right=0.97, top=0.76, bottom=0.285, wspace=0.25)
    fig.suptitle("Q5.1. Gradient descent after log(x + 1)", x=0.075, y=0.97,
                 ha="left", fontsize=18, fontweight="bold")
    fig.text(0.075, 0.915,
             "Same 900 training observations and zero initialization. Natural log applied to all eight predictors.",
             fontsize=10, color="#596674")

    for name, model in models.items():
        updates = np.arange(len(model.loss_history_))
        label = f"{labels[name]} (α = {results[name]['learning_rate']:g})"
        axes[0].plot(updates[1:], model.loss_history_[1:], color=colors[name], lw=2.2, label=label)
        axes[1].plot(updates, model.loss_history_, color=colors[name], lw=2.2)
        for ax in axes:
            if model.n_iter_ > 0:
                ax.plot(model.n_iter_, model.loss_history_[-1], "o", color=colors[name], ms=5)

    target = settings["comparison_target_mse"]
    minimum = min(r["training"]["mse"] for r in results.values())
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(1, settings["max_iter"] * 1.05)
    axes[0].set_title("A  Full loss trajectory", loc="left", pad=14)
    axes[0].set_xlabel("Parameter updates (log scale)")
    axes[0].set_ylabel("Training MSE (MPa², log scale)")
    axes[1].set_xlim(0, settings["max_iter"] * 1.03)
    axes[1].set_ylim(minimum - 3, max(target, minimum) + 12)
    axes[1].set_title("B  Low-loss region", loc="left", pad=14)
    axes[1].set_xlabel("Parameter updates")
    axes[1].set_ylabel("Training MSE (MPa²)")
    axes[1].ticklabel_format(axis="x", style="plain", useOffset=False)
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.axhline(target, color="#7d8791", ls="--", lw=1, zorder=0)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.07, 0.875),
               ncol=3, frameon=False, fontsize=10, handlelength=2.5, columnspacing=2)

    fig.text(0.075, 0.205, f"Updates to the common training MSE target of {target:g} MPa² (dashed line):",
             fontsize=10, fontweight="bold")
    for name, x in (("untransformed", 0.075), ("standardized", 0.38), ("log_transformed", 0.68)):
        fig.text(x, 0.165,
                 f"{labels[name]}: {milestone_text(results[name]['first_update_at_target_mse'])}",
                 fontsize=11, color=colors[name], fontweight="bold")
    initial = results["untransformed"]["initial_training_mse"]
    fig.text(0.075, 0.12,
             f"Initial MSE at update 0: {initial:,.2f} (omitted in A). "
             "The log transform uses no additional standardization; y remains in MPa.",
             fontsize=9, color="#596674")
    stops = "; ".join(
        f"{labels[name]} {r['updates']:,} ({'tolerance' if r['converged'] else 'cap'})"
        for name, r in results.items()
    )
    fig.text(0.075, 0.08, "Stop: " + stops + ".", fontsize=9, color="#596674")
    fig.text(0.075, 0.04,
             "The log transform changes the model family. Lower loss alone does not establish faster convergence to its own optimum.",
             fontsize=9, color="#596674")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Q5.1: add log(x + 1) features to the custom-GD comparison.')
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "q5")
    parser.add_argument("--log-learning-rate", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=20_000)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()
    train, test = split_train_test(load_concrete_data(args.data))
    X_train, X_test = train[FEATURE_NAMES].to_numpy(float), test[FEATURE_NAMES].to_numpy(float)
    y_train, y_test = train[TARGET].to_numpy(float), test[TARGET].to_numpy(float)
    X_scaled, X_test_scaled = standardize_train_test(X_train, X_test)
    X_log, X_test_log = np.log1p(X_train), np.log1p(X_test)
    if not np.isfinite(X_log).all() or not np.isfinite(X_test_log).all():
        raise ValueError("log(x + 1) must be finite for every input feature.")
    settings = {
        "max_iter": args.max_iter, "gradient_tolerance": args.tol,
        "comparison_target_mse": 110.0, "initial_parameters": "all zeros",
        "log_transform": "natural log(1 + x) applied to all 8 predictors, without subsequent scaling",
        "target_transform": "none; original MPa units",
        "comparison_note": "Common training-error milestone, not a stopping rule or distance from each model's optimum.",
    }
    models, results = {}, {}
    for name, Xtr, Xte, alpha in (
        ("untransformed", X_train, X_test, 1e-7),
        ("standardized", X_scaled, X_test_scaled, 0.05),
        ("log_transformed", X_log, X_test_log, args.log_learning_rate),
    ):
        models[name], results[name] = train_model(
            Xtr, Xte, y_train, y_test, alpha, args.max_iter, args.tol,
            settings["comparison_target_mse"],
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = args.output_dir / "q5_1_loss_comparison.png"
    make_figure(models, results, settings, figure)
    payload = {
        "scope": "Q5.1-Q5.2: custom gradient descent; no external optimum fits",
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "train_rows": 900, "test_rows": 130, "test_sample_ids_inclusive": [501, 630],
        "settings": settings, "models": results,
    }
    (args.output_dir / "q5_gd_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nFigure: {figure}")


if __name__ == "__main__":
    main()
