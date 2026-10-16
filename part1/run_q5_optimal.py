import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from data import DEFAULT_DATA_PATH, FEATURE_NAMES, TARGET, load_concrete_data, split_train_test
from metrics import regression_metrics


MODEL_LABELS = {
    "untransformed": "Untransformed",
    "standardized": "Standardized",
    "log_transformed": "Log-transformed: log(x + 1)",
}


def render_table(results, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(12.8, 4.8))
    fig.text(0.055, 0.91, "Q5.3. Optimal model performance", fontsize=18,
             fontweight="bold", color="#263443")
    fig.text(0.055, 0.825,
             "Ordinary least squares with an intercept. Fixed split: 900 training observations and 130 testing observations.",
             fontsize=10.5, color="#596674")
    ax = fig.add_axes([0.055, 0.31, 0.89, 0.43])
    ax.axis("off")
    rows = []
    for name, r in results.items():
        rows.append([
            MODEL_LABELS[name],
            f"{r['training']['mse']:.6f}", f"{r['testing']['mse']:.6f}",
            f"{r['training']['r2']:.6f}", f"{r['testing']['r2']:.6f}",
        ])
    table = ax.table(
        cellText=rows,
        colLabels=["Model", "Training MSE\n(MPa²)", "Testing MSE\n(MPa²)", "Training R²", "Testing R²"],
        colWidths=[0.32, 0.18, 0.18, 0.16, 0.16],
        cellLoc="right", colLoc="center", bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.PAD = 0.06
        if row == 0:
            cell.set_facecolor("#263e53")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#edf5f1" if row == 3 else ("#f1f4f7" if row == 2 else "#ffffff"))
            cell.get_text().set_color("#263443")
            if row == 3:
                cell.get_text().set_fontweight("bold")
        if column == 0:
            cell.get_text().set_ha("left")
    fig.text(0.055, 0.22,
             "All fits use scikit-learn LinearRegression. Optimal refers to the minimum training MSE; testing data are used only for evaluation.",
             fontsize=9.5, color="#596674")
    fig.text(0.055, 0.145,
             "Standardization is fitted on the training set. The log model uses natural log(1 + x) for all eight predictors, without additional scaling.",
             fontsize=9.5, color="#596674")
    fig.text(0.055, 0.07,
             "Target strength remains in MPa. Values are rounded to six decimal places; raw and standardized predictions agree to numerical precision.",
             fontsize=9.5, color="#596674")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Q5.3: compare optimal OLS fits across feature transformations.')
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "q5")
    args = parser.parse_args()
    train, test = split_train_test(load_concrete_data(args.data))
    X_train, X_test = train[FEATURE_NAMES].to_numpy(float), test[FEATURE_NAMES].to_numpy(float)
    y_train, y_test = train[TARGET].to_numpy(float), test[TARGET].to_numpy(float)
    scaler = StandardScaler().fit(X_train)
    results, predictions, checks = {}, {}, {}
    for name, Xtr, Xte in (
        ("untransformed", X_train, X_test),
        ("standardized", scaler.transform(X_train), scaler.transform(X_test)),
        ("log_transformed", np.log1p(X_train), np.log1p(X_test)),
    ):
        model = LinearRegression(fit_intercept=True).fit(Xtr, y_train)
        train_pred, test_pred = model.predict(Xtr), model.predict(Xte)
        predictions[name] = (train_pred, test_pred)
        results[name] = {
            "training": regression_metrics(y_train, train_pred),
            "testing": regression_metrics(y_test, test_pred),
            "intercept": float(model.intercept_),
            "coefficients": dict(zip(FEATURE_NAMES, model.coef_.tolist())),
            "feature_matrix_rank": int(model.rank_),
        }
        design = np.column_stack((np.ones(len(Xtr)), Xtr))
        reference, *_ = np.linalg.lstsq(design, y_train, rcond=None)
        np.testing.assert_allclose(train_pred, design @ reference, rtol=0, atol=1e-7)
        gradient = 2 * design.T @ (train_pred - y_train) / len(y_train)
        np.testing.assert_allclose(gradient, 0, rtol=0, atol=1e-7)
        checks[name] = {
            "max_absolute_prediction_difference_from_reference": float(np.max(np.abs(train_pred - design @ reference))),
            "training_mse_gradient_infinity_norm": float(np.max(np.abs(gradient))),
        }
    for raw_pred, scaled_pred in zip(predictions["untransformed"], predictions["standardized"]):
        np.testing.assert_allclose(raw_pred, scaled_pred, rtol=0, atol=1e-9)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = args.output_dir / "q5_3_optimal_model_table.png"
    render_table(results, figure)
    payload = {
        "scope": "Q5.3-Q5.4: optimal unregularized least-squares models",
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "train_rows": 900, "test_rows": 130, "test_sample_ids_inclusive": [501, 630],
        "estimator": "sklearn.linear_model.LinearRegression(fit_intercept=True)",
        "standardization_fit_on": "training set only",
        "log_transform": "natural log(1 + x) for all eight predictors, without additional scaling",
        "target_transform": "none; original MPa units",
        "models": results, "optimality_checks": checks,
    }
    (args.output_dir / "q5_optimal_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {split: r[split] for split in ("training", "testing")} for name, r in results.items()}, indent=2))
    print(f"Table: {figure}")


if __name__ == "__main__":
    main()
