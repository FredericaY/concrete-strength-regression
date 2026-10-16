import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part1"))

from data import FEATURE_NAMES, TARGET, load_concrete_data, split_train_test, standardize_train_test
from gradient_descent import LinearRegressionGD
from part2.table_style import render_table

LABELS = ["Cement", "Blast furnace slag", "Fly ash", "Water", "Superplasticizer", "Coarse aggregate", "Fine aggregate", "Age"]


def loss_figure(models, names, path):
    styles = {"untransformed": ("Raw", "0.1", "-"), "standardized": ("Standardized", "0.4", "--"), "log_transformed": ("Log", "0.65", "-.")}
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for name in names:
        label, color, style = styles[name]
        m = models[name]
        ax.plot(np.arange(1, len(m.loss_history_)), m.loss_history_[1:], label=label, color=color, linestyle=style, linewidth=1.6)
        ax.plot(m.n_iter_, m.loss_history_[-1], marker="o", markersize=3.5, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter updates")
    ax.set_ylabel("Training MSE (MPa²)")
    ax.set_xlim(1, 24000)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.7)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = ROOT / "part1/outputs"
    gd = json.loads((root / "q5/q5_gd_results.json").read_text())
    optimal = json.loads((root / "q5/q5_optimal_results.json").read_text())
    q6 = json.loads((root / "q6/q6_results.json").read_text())
    q7 = json.loads((root / "q7/q7_results.json").read_text())
    train, test = split_train_test(load_concrete_data())
    x, xt = train[FEATURE_NAMES].to_numpy(float), test[FEATURE_NAMES].to_numpy(float)
    y = train[TARGET].to_numpy(float)
    z, _ = standardize_train_test(x, xt)
    matrices = {"untransformed": x, "standardized": z, "log_transformed": np.log1p(x)}
    models = {}
    for name, matrix in matrices.items():
        saved = gd["models"][name]
        model = LinearRegressionGD(saved["learning_rate"], 20000, 1e-6).fit(matrix, y)
        assert model.n_iter_ == saved["updates"]
        np.testing.assert_allclose(model.loss_history_[-1], saved["training"]["mse"], rtol=1e-10)
        models[name] = model
    np.savez(args.output_dir / "loss_histories.npz", **{k: v.loss_history_ for k, v in models.items()})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": "black", "savefig.facecolor": "white"})
    loss_figure(models, ["untransformed", "standardized"], args.output_dir / "figure_1_scaling.png")
    loss_figure(models, list(matrices), args.output_dir / "figure_2_transformations.png")
    rows = []
    for key, label in [("untransformed", "Raw"), ("standardized", "Standardized"), ("log_transformed", "Log")]:
        r = optimal["models"][key]
        rows.append([label, f"{r['training']['mse']:.3f}", f"{r['testing']['mse']:.3f}", f"{r['training']['r2']:.4f}", f"{r['testing']['r2']:.4f}"])
    render_table(rows, ["Model", "Train MSE", "Test MSE", "Train R²", "Test R²"], [0.28, 0.19, 0.19, 0.17, 0.17], args.output_dir / "table_5_optimal_performance.png", (8, 1.45))
    rows = []
    for feature, label in zip(FEATURE_NAMES, LABELS):
        r = q6["univariate_models"][feature]
        rows.append([label, f"{r['gd']['training']['r2']:.3f}", f"{r['ols']['training']['r2']:.3f}", f"{r['ols']['slope']:.4f}", f"{r['multivariate_coefficient']:.4f}"])
    render_table(rows, ["Predictor", "GD R²", "OLS R²", "Single slope", "Multiple slope"], [0.30, 0.14, 0.14, 0.20, 0.22], args.output_dir / "table_6_coefficients.png", (8, 3.15))
    rows = []
    for feature, label in zip(FEATURE_NAMES, LABELS):
        values = [q7["models"][name]["features"][feature]["p_value"] for name in matrices]
        rows.append([label, *[f"{v:.3e}" if v < 0.001 else f"{v:.4f}" for v in values]])
    render_table(rows, ["Predictor", "Raw p", "Standardized p", "Log p"], [0.33, 0.21, 0.25, 0.21], args.output_dir / "table_7_p_values.png", (8, 3.15))


if __name__ == "__main__":
    main()
