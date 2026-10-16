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
from gradient_descent import LinearRegressionGD
from metrics import regression_metrics


def first_at_or_below(history, target):
    indices = np.flatnonzero(history <= target)
    return int(indices[0]) if len(indices) else None


def train_model(X_train, X_test, y_train, y_test, learning_rate, max_iter, tol, target):
    model = LinearRegressionGD(learning_rate, max_iter, tol).fit(X_train, y_train)
    train_scores = regression_metrics(y_train, model.predict(X_train))
    test_scores = regression_metrics(y_test, model.predict(X_test))
    positive = np.flatnonzero(model.loss_history_ < np.var(y_train, ddof=0))
    design = np.column_stack((np.ones(len(X_train)), X_train))
    eigenvalues = np.linalg.eigvalsh(2 * (design.T @ design) / len(X_train))
    result = {
        "learning_rate": learning_rate,
        "updates": model.n_iter_,
        "converged": model.converged_,
        "stop_reason": "gradient_tolerance" if model.converged_ else "iteration_cap",
        "final_gradient_infinity_norm": model.gradient_norm_,
        "initial_training_mse": float(model.loss_history_[0]),
        "training": train_scores,
        "testing": test_scores,
        "first_positive_training_r2_update": int(positive[0]) if len(positive) else None,
        "first_update_at_target_mse": first_at_or_below(model.loss_history_, target),
        "loss_monotonically_nonincreasing": bool(np.all(np.diff(model.loss_history_) <= 1e-10)),
        "hessian_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "stable_learning_rate_upper_bound": float(2 / eigenvalues[-1]),
    }
    if train_scores["r2"] <= 0:
        raise ValueError("Q3 requires positive training R^2 for both models; revise training settings.")
    return model, result


def make_figure(models, results, settings, output_path):
    colors = {"untransformed": "#c96929", "standardized": "#2475ad"}
    labels = {"untransformed": "Untransformed", "standardized": "Standardized"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#263443", "text.color": "#263443",
        "axes.titleweight": "bold", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.24, wspace=0.27)
    fig.suptitle("Q3. Feature scaling and gradient descent", x=0.08, y=0.97,
                 ha="left", fontsize=18, fontweight="bold")
    fig.text(0.08, 0.915, "900 training observations | Same zero initialization | Training MSE objective",
             fontsize=10, color="#596674")

    for name, model in models.items():
        history = model.loss_history_
        updates = np.arange(len(history))
        label = f"{labels[name]} (learning rate = {results[name]['learning_rate']:g})"
        axes[0].plot(updates[1:], history[1:], color=colors[name], lw=2.2, label=label)
        axes[1].plot(updates, history, color=colors[name], lw=2.2)
        for ax in axes:
            if model.n_iter_ > 0:
                ax.plot(model.n_iter_, history[-1], "o", color=colors[name], ms=5)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("A  Full loss trajectory", loc="left", pad=14)
    axes[0].set_xlabel("Parameter updates (log scale)")
    axes[0].set_ylabel("Training MSE (MPa², log scale)")
    axes[0].set_xlim(1, settings["max_iter"] * 1.05)

    target = settings["comparison_target_mse"]
    minimum = min(r["training"]["mse"] for r in results.values())
    axes[1].set_title("B  Low-loss region", loc="left", pad=14)
    axes[1].set_xlabel("Parameter updates")
    axes[1].set_ylabel("Training MSE (MPa²)")
    axes[1].set_xlim(0, settings["max_iter"] * 1.03)
    axes[1].set_ylim(minimum - 0.5, max(target, minimum) + 2.5)
    axes[1].ticklabel_format(axis="x", style="plain", useOffset=False)

    for ax in axes:
        ax.grid(axis="both", alpha=0.18, zorder=0)
        ax.axhline(target, color="#7d8791", ls="--", lw=1, zorder=0)
    for name, x_fraction in (("standardized", 0.20), ("untransformed", 0.65)):
        first = results[name]["first_update_at_target_mse"]
        if first is not None:
            axes[1].plot(first, models[name].loss_history_[first], "o", color=colors[name], ms=5)
            axes[1].annotate(
                f"{labels[name]}\n{first:,} updates to MSE ≤ {target:g}",
                xy=(first, models[name].loss_history_[first]),
                xytext=(x_fraction, 0.85), textcoords="axes fraction",
                fontsize=9, ha="center", color=colors[name],
                arrowprops={"arrowstyle": "->", "color": colors[name], "lw": 1},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2},
            )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.075, 0.88),
               ncol=2, frameon=False, fontsize=10, handlelength=2.8, columnspacing=2.5)
    raw, scaled = results["untransformed"], results["standardized"]
    initial = raw["initial_training_mse"]
    fig.text(0.08, 0.135,
             f"Dashed line: common loss target of {target:g} MPa². Initial MSE at update 0: {initial:,.2f} (omitted in panel A).",
             fontsize=9, color="#596674")
    fig.text(0.08, 0.095,
             f"Final training R²: untransformed {raw['training']['r2']:.4f}; standardized {scaled['training']['r2']:.4f}. "
             "Standardization uses training statistics only.", fontsize=9, color="#596674")
    fig.text(0.08, 0.055,
             f"Untransformed: {raw['updates']:,} updates ({raw['stop_reason'].replace('_', ' ')}). "
             f"Standardized: {scaled['updates']:,} updates ({scaled['stop_reason'].replace('_', ' ')}).",
             fontsize=9, color="#596674")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Q3: compare custom gradient descent with raw and standardized features.')
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "q3")
    parser.add_argument("--raw-learning-rate", type=float, default=1e-7)
    parser.add_argument("--scaled-learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=20_000)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    train, test = split_train_test(load_concrete_data(args.data))
    X_train, X_test = train[FEATURE_NAMES].to_numpy(float), test[FEATURE_NAMES].to_numpy(float)
    y_train, y_test = train[TARGET].to_numpy(float), test[TARGET].to_numpy(float)
    X_train_scaled, X_test_scaled = standardize_train_test(X_train, X_test)
    settings = {
        "max_iter": args.max_iter, "gradient_tolerance": args.tol,
        "comparison_target_mse": 110.0,
        "comparison_note": "Common training-loss milestone, not a stopping or tuning criterion.",
        "initial_parameters": "all zeros", "standardization_ddof": 0,
    }
    models, results = {}, {}
    for name, Xtr, Xte, alpha in (
        ("untransformed", X_train, X_test, args.raw_learning_rate),
        ("standardized", X_train_scaled, X_test_scaled, args.scaled_learning_rate),
    ):
        models[name], results[name] = train_model(
            Xtr, Xte, y_train, y_test, alpha, args.max_iter, args.tol,
            settings["comparison_target_mse"],
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = args.output_dir / "q3_loss_comparison.png"
    make_figure(models, results, settings, figure)
    payload = {
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "train_rows": 900, "test_rows": 130, "test_sample_ids_inclusive": [501, 630],
        "settings": settings, "models": results,
    }
    (args.output_dir / "q3_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nFigure: {figure}")


if __name__ == "__main__":
    main()
