import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from data import DEFAULT_DATA_PATH, FEATURE_NAMES, TARGET, load_concrete_data, split_train_test
from metrics import regression_metrics


ALPHA = 0.05
LABELS = dict(zip(FEATURE_NAMES, [
    "Cement", "Blast furnace slag", "Fly ash", "Water", "Superplasticizer",
    "Coarse aggregate", "Fine aggregate", "Age",
]))


def format_p(value):
    return f"{value:.6f}" if value >= 0.001 else f"{value:.3e}"


def coefficient_record(fit, index):
    p = float(fit.pvalues[index])
    return {
        "coefficient": float(fit.params[index]),
        "standard_error": float(fit.bse[index]),
        "t_statistic": float(fit.tvalues[index]),
        "p_value": p,
        "significant_at_0_05": p < ALPHA,
    }


def render_p_value_table(models, path):
    names = ["untransformed", "standardized", "log_transformed"]
    rows = []
    for feature in FEATURE_NAMES:
        values = [LABELS[feature]]
        for model in names:
            p = models[model]["features"][feature]["p_value"]
            values.append(format_p(p) + (" *" if p < ALPHA else ""))
        rows.append(values)
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(12.8, 7.0))
    fig.text(0.055, 0.935, "Q7.1. Predictor p-values across feature transformations",
             fontsize=18, fontweight="bold", color="#263443")
    fig.text(0.055, 0.875,
             "OLS with an intercept, fitted on the same 900 training observations. The response remains in MPa.",
             fontsize=10.5, color="#596674")
    ax = fig.add_axes([0.055, 0.265, 0.89, 0.55])
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Feature", "Untransformed\np-value", "Standardized\np-value", "Log-transformed\np-value"],
        colWidths=[0.31, 0.23, 0.23, 0.23],
        cellLoc="right", colLoc="center", bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.PAD = 0.055
        if row == 0:
            cell.set_facecolor("#263e53")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            significant = column > 0 and models[names[column - 1]]["features"][FEATURE_NAMES[row - 1]]["significant_at_0_05"]
            cell.set_facecolor("#eaf4ef" if significant else "#f4f6f8")
            cell.get_text().set_color("#263443")
        if column == 0:
            cell.get_text().set_ha("left")
    df = models["untransformed"]["residual_degrees_of_freedom"]
    fig.text(0.055, 0.19,
             f"Two-sided tests of H₀: βⱼ = 0. Conventional OLS standard errors (nonrobust); residual degrees of freedom = {df}.",
             fontsize=9.5, color="#596674")
    fig.text(0.055, 0.13,
             "* p < 0.05 (unadjusted); green cells mark significant associations. Every fit includes an intercept, omitted from this table.",
             fontsize=9.5, color="#596674")
    fig.text(0.055, 0.07,
             "Standardization uses training means and standard deviations. The log model applies natural log(1 + x) to all eight predictors.",
             fontsize=9.5, color="#596674")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Q7: OLS feature p-values, their invariance to scaling, and Q6 comparison.')
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "q7")
    parser.add_argument("--q6-results", type=Path, default=root / "outputs" / "q6" / "q6_results.json")
    args = parser.parse_args()
    if not args.q6_results.is_file():
        raise FileNotFoundError("The Q7 numerical comparison requires the Q6 results. Run python run_q6.py first, or specify --q6-results.")
    q6 = json.loads(args.q6_results.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if q6["dataset_sha256"] != dataset_hash or q6["test_sample_ids_inclusive"] != [501, 630]:
        raise ValueError("Q6 and Q7 must use the same original dataset and prescribed split.")
    if q6["settings"]["feature_transform"] != "none":
        raise ValueError("The numerical comparison expects Q6 models trained on untransformed predictors.")
    train, test = split_train_test(load_concrete_data(args.data))
    X, X_test = train[FEATURE_NAMES].to_numpy(float), test[FEATURE_NAMES].to_numpy(float)
    y, y_test = train[TARGET].to_numpy(float), test[TARGET].to_numpy(float)
    scaler = StandardScaler().fit(X)
    fits, models, predictions = {}, {}, {}
    for name, Xtr, Xte in (
        ("untransformed", X, X_test),
        ("standardized", scaler.transform(X), scaler.transform(X_test)),
        ("log_transformed", np.log1p(X), np.log1p(X_test)),
    ):
        design = sm.add_constant(Xtr, has_constant="add")
        fit = sm.OLS(y, design, missing="raise").fit(cov_type="nonrobust", use_t=True)
        fits[name] = fit
        train_pred = fit.predict(design)
        test_pred = fit.predict(sm.add_constant(Xte, has_constant="add"))
        predictions[name] = (train_pred, test_pred)
        np.testing.assert_allclose(fit.tvalues, fit.params / fit.bse, rtol=1e-12, atol=0)
        expected_p = 2 * student_t.sf(np.abs(fit.tvalues), df=fit.df_resid)
        np.testing.assert_allclose(fit.pvalues, expected_p, rtol=1e-12, atol=0)
        assert fit.df_resid == len(y) - len(FEATURE_NAMES) - 1
        models[name] = {
            "training": regression_metrics(y, train_pred),
            "testing": regression_metrics(y_test, test_pred),
            "n_observations": int(fit.nobs),
            "residual_degrees_of_freedom": int(fit.df_resid),
            "residual_variance_estimate": float(fit.scale),
            "intercept": coefficient_record(fit, 0),
            "features": {feature: coefficient_record(fit, i + 1) for i, feature in enumerate(FEATURE_NAMES)},
        }
    raw, scaled = fits["untransformed"], fits["standardized"]
    np.testing.assert_allclose(scaled.params[1:], raw.params[1:] * scaler.scale_, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(scaled.bse[1:], raw.bse[1:] * scaler.scale_, rtol=1e-9, atol=0)
    np.testing.assert_allclose(raw.tvalues[1:], scaled.tvalues[1:], rtol=1e-9, atol=0)
    np.testing.assert_allclose(raw.pvalues[1:], scaled.pvalues[1:], rtol=1e-9, atol=0)
    for left, right in zip(predictions["untransformed"], predictions["standardized"]):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-8)

    comparison = {}
    for feature in FEATURE_NAMES:
        gd = q6["univariate_models"][feature]["gd"]
        comparison[feature] = {
            "q6_gd_training_r2": gd["training"]["r2"],
            "q6_first_positive_training_r2_update": gd["first_positive_training_r2_update"],
            "q6_converged": gd["converged"],
            "raw_multivariate_p_value": models["untransformed"]["features"][feature]["p_value"],
            "log_multivariate_p_value": models["log_transformed"]["features"][feature]["p_value"],
        }
    significant = {
        name: [feature for feature, record in model["features"].items() if record["significant_at_0_05"]]
        for name, model in models.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "q7_1_p_values.png"
    render_p_value_table(models, table_path)
    payload = {
        "dataset_sha256": dataset_hash,
        "q6_results_sha256": hashlib.sha256(args.q6_results.read_bytes()).hexdigest(),
        "train_rows": 900, "test_rows": 130, "test_sample_ids_inclusive": [501, 630],
        "inference": {
            "package": f"statsmodels {version('statsmodels')}",
            "estimator": "OLS with intercept, fitted only on training observations",
            "covariance_type": "nonrobust", "use_t": True,
            "null_hypothesis": "individual predictor coefficient equals zero, two-sided",
            "significance_level": ALPHA, "multiple_testing_adjustment": "none",
            "assumptions": "Conventional OLS inference: correctly specified conditional mean, independent homoskedastic errors; exact finite-sample t tests additionally assume normal errors.",
        },
        "feature_transformations": {"untransformed": "none", "standardized": "training-set z scores", "log_transformed": "natural log(1 + x), all eight predictors, no additional scaling"},
        "models": models, "significant_features": significant,
        "q6_comparison": comparison,
        "checks": {
            "max_absolute_raw_standardized_feature_t_difference": float(np.max(np.abs(raw.tvalues[1:] - scaled.tvalues[1:]))),
            "max_relative_raw_standardized_feature_p_difference": float(np.max(np.abs(raw.pvalues[1:] - scaled.pvalues[1:]) / raw.pvalues[1:])),
            "intercept_excluded_from_invariance_check": True,
        },
    }
    (args.output_dir / "q7_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Significant features (p < 0.05): " + json.dumps(significant))
    print(f"Table: {table_path}")


if __name__ == "__main__":
    main()
