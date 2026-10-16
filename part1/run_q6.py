import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from data import DEFAULT_DATA_PATH, FEATURE_NAMES, TARGET, load_concrete_data, split_train_test
from gradient_descent import LinearRegressionGD
from metrics import regression_metrics


LABELS = dict(zip(FEATURE_NAMES, [
    "Cement", "Blast furnace slag", "Fly ash", "Water", "Superplasticizer",
    "Coarse aggregate", "Fine aggregate", "Age",
]))


def render_table(rows, columns, widths, title, subtitle, notes, highlighted, highlight_color, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(12.8, 7.0))
    fig.text(0.055, 0.935, title, fontsize=18, fontweight="bold", color="#263443")
    fig.text(0.055, 0.875, subtitle, fontsize=10, color="#596674")
    ax = fig.add_axes([0.055, 0.265, 0.89, 0.55])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, colWidths=widths,
                     cellLoc="right", colLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.PAD = 0.055
        if row == 0:
            cell.set_facecolor("#263e53")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(highlight_color if row - 1 in highlighted else "#f4f6f8")
            cell.get_text().set_color("#263443")
        if column == 0:
            cell.get_text().set_ha("left")
    for y, note in zip((0.19, 0.13, 0.07), notes):
        fig.text(0.055, y, note, fontsize=9.5, color="#596674")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Q6: eight raw-feature GD models, optimal coefficients, and numerical diagnostics.')
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "q6")
    parser.add_argument("--max-iter", type=int, default=20_000)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()
    train, test = split_train_test(load_concrete_data(args.data))
    X, X_test = train[FEATURE_NAMES].to_numpy(float), test[FEATURE_NAMES].to_numpy(float)
    y, y_test = train[TARGET].to_numpy(float), test[TARGET].to_numpy(float)
    multivariate = LinearRegression(fit_intercept=True).fit(X, y)
    covariance = np.cov(X, rowvar=False, ddof=0)
    correlations = train[FEATURE_NAMES].corr().to_dict()
    results = {}

    for i, name in enumerate(FEATURE_NAMES):
        x = X[:, i]
        design = np.column_stack((np.ones(len(x)), x))
        eigenvalues = np.linalg.eigvalsh(2 * design.T @ design / len(x))
        learning_rate = float(1 / eigenvalues[-1])
        gd = LinearRegressionGD(learning_rate, args.max_iter, args.tol).fit(x, y)
        ols = LinearRegression(fit_intercept=True).fit(x[:, None], y)
        gd_train = regression_metrics(y, gd.predict(x))
        ols_train = regression_metrics(y, ols.predict(x[:, None]))
        positive = np.flatnonzero(gd.loss_history_ < np.var(y, ddof=0))

        centered_x = x - x.mean()
        analytic_slope = float(centered_x @ (y - y.mean()) / (centered_x @ centered_x))
        np.testing.assert_allclose(ols.coef_[0], analytic_slope, rtol=0, atol=1e-10)
        np.testing.assert_allclose(ols.intercept_, y.mean() - analytic_slope * x.mean(), rtol=0, atol=1e-9)
        contributions = {
            other: float(multivariate.coef_[j] * covariance[i, j] / covariance[i, i])
            for j, other in enumerate(FEATURE_NAMES) if j != i
        }
        contribution_sum = sum(contributions.values())
        reconstructed_slope = float(multivariate.coef_[i] + contribution_sum)
        np.testing.assert_allclose(ols.coef_[0], reconstructed_slope, rtol=0, atol=1e-10)
        assert np.all(np.diff(gd.loss_history_) <= 1e-9), "Unexpected increasing loss."
        assert gd_train["mse"] >= ols_train["mse"] - 1e-8

        results[name] = {
            "gd": {
                "learning_rate": learning_rate,
                "hessian_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
                "stable_learning_rate_upper_bound": float(2 / eigenvalues[-1]),
                "updates": gd.n_iter_, "converged": gd.converged_,
                "stop_reason": "gradient_tolerance" if gd.converged_ else "iteration_cap",
                "final_gradient_infinity_norm": gd.gradient_norm_,
                "initial_training_mse": float(gd.loss_history_[0]),
                "training": gd_train,
                "testing": regression_metrics(y_test, gd.predict(X_test[:, i])),
                "first_positive_training_r2_update": int(positive[0]) if len(positive) else None,
                "loss_monotonically_nonincreasing": True,
                "slope": float(gd.coef_[0]), "intercept": gd.intercept_,
                "training_mse_above_ols": gd_train["mse"] - ols_train["mse"],
            },
            "ols": {
                "slope": float(ols.coef_[0]), "intercept": float(ols.intercept_),
                "training": ols_train,
                "testing": regression_metrics(y_test, ols.predict(X_test[:, i, None])),
                "analytic_slope_difference": float(abs(ols.coef_[0] - analytic_slope)),
            },
            "multivariate_coefficient": float(multivariate.coef_[i]),
            "omitted_variable_decomposition": {
                "contributions": contributions, "sum": contribution_sum,
                "reconstructed_univariate_slope": reconstructed_slope,
                "absolute_error": float(abs(ols.coef_[0] - reconstructed_slope)),
            },
        }

    positive_count = sum(r["gd"]["training"]["r2"] > 0 for r in results.values())
    if positive_count < 2:
        raise ValueError("Q6 needs at least two positive training R-squared values; increase the training budget.")
    settings = {
        "max_iter": args.max_iter, "gradient_tolerance": args.tol,
        "initial_parameters": "all zeros", "feature_transform": "none",
        "learning_rate_rule": "1 / lambda_max(H), H = 2 A.T @ A / n, A = [1, raw predictor]",
        "learning_rate_selection_data": "training features only",
        "positive_r2_criterion": "training R-squared; positive R-squared is not a stopping condition",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coefficient_rows, training_rows = [], []
    sign_reversals, positive_rows = [], []
    for i, (name, r) in enumerate(results.items()):
        gd = r["gd"]
        coefficient_rows.append([LABELS[name], f"{r['ols']['slope']:.6f}", f"{r['multivariate_coefficient']:.6f}"])
        if np.sign(r["ols"]["slope"]) != np.sign(r["multivariate_coefficient"]):
            sign_reversals.append(i)
        first = gd["first_positive_training_r2_update"]
        training_rows.append([
            LABELS[name], f"{gd['learning_rate']:.3e}", f"{first:,}" if first is not None else "Not reached",
            f"{gd['training']['r2']:.6f}", f"{r['ols']['training']['r2']:.6f}",
            f"{'Tolerance' if gd['converged'] else 'Cap'} ({gd['updates']:,})",
        ])
        if gd["training"]["r2"] > 0:
            positive_rows.append(i)
    render_table(
        coefficient_rows, ["Feature", "Optimal univariate slope", "Optimal multivariate coefficient"], [0.38, 0.31, 0.31],
        "Q6.2. Univariate and multivariate coefficients",
        "Optimal ordinary least-squares fits on the same 900 training observations. All predictors remain in their original units.",
        [
            "Each univariate model uses one predictor; the multivariate model includes all eight. All models include an intercept.",
            "Units: MPa per kg/m³ for material quantities; MPa per day for Age. Amber rows indicate a coefficient sign reversal.",
            f"Multivariate intercept: {multivariate.intercept_:.6f} MPa. Individual univariate intercepts and all metrics are recorded in q6_results.json.",
        ], sign_reversals, "#fcf0d9", args.output_dir / "q6_2_coefficients.png",
    )
    render_table(
        training_rows,
        ["Feature", "Learning rate", "First update with\ntraining R² > 0", "GD training R²", "OLS training R²", "Stopping"],
        [0.23, 0.14, 0.18, 0.14, 0.14, 0.17],
        "Q6.1. Training difficulty with raw features",
        f"Zero initialization; learning rate = 1 / largest training Hessian eigenvalue; maximum {args.max_iter:,} updates.",
        [
            f"Green rows meet the positive-training-R² requirement ({positive_count} of 8 models). All eight loss curves decrease monotonically.",
            f"Tolerance means max absolute training MSE gradient ≤ {args.tol:g}. Cap means the update budget was exhausted before reaching tolerance.",
            "OLS training R² is the achievable univariate reference. Negative GD R² can reflect incomplete optimization rather than an absence of association.",
        ], positive_rows, "#edf5f1", args.output_dir / "q6_training_summary.png",
    )
    payload = {
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "train_rows": 900, "test_rows": 130, "test_sample_ids_inclusive": [501, 630],
        "settings": settings, "positive_training_r2_count": positive_count,
        "univariate_models": results, "training_feature_correlations": correlations,
        "optimal_multivariate": {
            "estimator": "sklearn.linear_model.LinearRegression(fit_intercept=True)",
            "intercept": float(multivariate.intercept_),
            "coefficients": dict(zip(FEATURE_NAMES, multivariate.coef_.tolist())),
            "training": regression_metrics(y, multivariate.predict(X)),
            "testing": regression_metrics(y_test, multivariate.predict(X_test)),
        },
    }
    (args.output_dir / "q6_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved Q6 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
