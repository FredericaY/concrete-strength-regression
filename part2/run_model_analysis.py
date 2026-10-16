import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from part1.data import DEFAULT_DATA_PATH, FEATURE_NAMES, TARGET, load_concrete_data, split_train_test
from part1.metrics import regression_metrics
from table_style import render_table


SCENARIOS = [
    ("cement", "Cement (kg/m³)", 25.0),
    ("water", "Water (kg/m³)", -10.0),
    ("superplasticizer", "Superplasticizer (kg/m³)", 2.0),
    ("age", "Curing age (days)", 28.0),
]


def predictions(features, model, log_features=False):
    values = features[FEATURE_NAMES].to_numpy(dtype=float)
    if log_features:
        values = np.log1p(values)
    weights = np.array([model["coefficients"][name] for name in FEATURE_NAMES])
    return model["intercept"] + values @ weights


def select_baseline(train):
    features = train[FEATURE_NAMES]
    candidates = features.loc[features["age"] == 28]
    if candidates.empty:
        raise ValueError("A 28-day training observation is required for the baseline.")
    median = features.median()
    iqr = (features.quantile(0.75) - features.quantile(0.25)).replace(0, 1.0)
    distance = ((candidates - median) / iqr).pow(2).sum(axis=1)
    sample_id = int(distance.idxmin())
    return sample_id, train.loc[[sample_id], FEATURE_NAMES].copy(), float(distance.loc[sample_id])


def main():
    parser = argparse.ArgumentParser(description="Part 2: model performance and single-input prediction scenarios.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models", type=Path, default=PROJECT_ROOT / "part1/outputs/q5/q5_optimal_results.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    model_results = json.loads(args.models.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if model_results["dataset_sha256"] != dataset_hash:
        raise ValueError("The saved models and current dataset do not match.")
    train, test = split_train_test(load_concrete_data(args.data))
    assert model_results["train_rows"] == len(train) == 900
    assert model_results["test_rows"] == len(test) == 130
    assert model_results["test_sample_ids_inclusive"] == [501, 630]
    models = model_results["models"]
    performance, performance_rows = {}, []
    for name, label, use_log in (
        ("untransformed", "Original features", False),
        ("log_transformed", "log(x + 1) features", True),
    ):
        scores = {}
        for split, frame in (("training", train), ("testing", test)):
            metrics = regression_metrics(frame[TARGET].to_numpy(float), predictions(frame, models[name], use_log))
            np.testing.assert_allclose(
                [metrics["mse"], metrics["r2"]],
                [models[name][split]["mse"], models[name][split]["r2"]], rtol=0, atol=1e-9,
            )
            scores[split] = {**metrics, "rmse": float(np.sqrt(metrics["mse"]))}
        performance[name] = scores
        performance_rows.append([
            label, f"{scores['training']['r2']:.4f}", f"{scores['testing']['r2']:.4f}",
            f"{scores['training']['rmse']:.2f}", f"{scores['testing']['rmse']:.2f}",
        ])

    sample_id, baseline, distance = select_baseline(train)
    model = models["log_transformed"]
    baseline_prediction = float(predictions(baseline, model, True)[0])
    scenario_rows = [[f"Baseline (sample {sample_id})", "n/a", "n/a", f"{baseline_prediction:.2f}", "0.00"]]
    scenarios = []
    minimum, maximum = train[FEATURE_NAMES].min(), train[FEATURE_NAMES].max()
    for feature, label, change in SCENARIOS:
        changed = baseline.copy()
        before = float(baseline.at[sample_id, feature])
        after = before + change
        changed.at[sample_id, feature] = after
        remaining = [name for name in FEATURE_NAMES if name != feature]
        np.testing.assert_array_equal(changed[remaining], baseline[remaining])
        assert ((changed.iloc[0] >= minimum) & (changed.iloc[0] <= maximum)).all()
        prediction = float(predictions(changed, model, True)[0])
        delta = prediction - baseline_prediction
        expected_delta = model["coefficients"][feature] * (np.log1p(after) - np.log1p(before))
        np.testing.assert_allclose(delta, expected_delta, rtol=0, atol=1e-10)
        format_input = ".0f" if feature == "age" else ".2f"
        scenario_rows.append([
            label, format(before, format_input), format(after, format_input),
            f"{prediction:.2f}", f"{delta:+.2f}",
        ])
        scenarios.append({
            "changed_feature": feature, "before": before, "after": after,
            "inputs": {name: float(changed.at[sample_id, name]) for name in FEATURE_NAMES},
            "predicted_strength_mpa": prediction, "change_from_baseline_mpa": delta,
            "within_training_marginal_ranges": True,
        })

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": "black", "savefig.facecolor": "white",
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_table(
        performance_rows,
        ["Model", "Training R²", "Testing R²", "Training RMSE\n(MPa)", "Testing RMSE\n(MPa)"],
        [0.28, 0.17, 0.17, 0.19, 0.19],
        args.output_dir / "table_2_model_performance.png", (8.8, 1.5),
    )
    render_table(
        scenario_rows,
        ["Scenario / input", "Before", "After", "Predicted strength\n(MPa)", "Δ vs baseline\n(MPa)"],
        [0.30, 0.12, 0.12, 0.24, 0.22],
        args.output_dir / "table_3_scenario_analysis.png", (9, 2.7),
    )
    payload = {
        "dataset_sha256": dataset_hash,
        "model_results_sha256": hashlib.sha256(args.models.read_bytes()).hexdigest(),
        "training_samples": len(train), "testing_samples": len(test),
        "test_sample_ids_inclusive": [501, 630],
        "performance": performance,
        "scenario_model": "log_transformed",
        "baseline": {
            "training_sample_id": sample_id,
            "selection": "Among 28-day training observations, minimum squared distance from training-feature medians after scaling by training IQRs; predictors only.",
            "selection_distance": distance,
            "inputs": {name: float(baseline.at[sample_id, name]) for name in FEATURE_NAMES},
            "observed_strength_mpa": float(train.at[sample_id, TARGET]),
            "predicted_strength_mpa": baseline_prediction,
        },
        "scenario_interpretation": "One input changed at a time with all other predictors fixed; model point predictions, without causal identification or mix-feasibility validation.",
        "scenarios": scenarios,
    }
    (args.output_dir / "model_analysis.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"performance": performance, "baseline_sample_id": sample_id,
                      "baseline_prediction_mpa": baseline_prediction,
                      "scenario_changes_mpa": {r['changed_feature']: r['change_from_baseline_mpa'] for r in scenarios}}, indent=2))
    print(f"Saved model performance and scenario tables to {args.output_dir}")


if __name__ == "__main__":
    main()
