import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from data import DEFAULT_DATA_PATH, FEATURE_NAMES, TARGET, load_concrete_data, split_train_test
from metrics import regression_metrics


def main():
    parser = argparse.ArgumentParser(description='Q4: compare optimal ordinary least-squares fits before and after scaling.')
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "q4")
    args = parser.parse_args()

    train, test = split_train_test(load_concrete_data(args.data))
    X_train, X_test = train[FEATURE_NAMES].to_numpy(float), test[FEATURE_NAMES].to_numpy(float)
    y_train, y_test = train[TARGET].to_numpy(float), test[TARGET].to_numpy(float)
    scaler = StandardScaler().fit(X_train)
    Z_train, Z_test = scaler.transform(X_train), scaler.transform(X_test)
    raw = LinearRegression(fit_intercept=True).fit(X_train, y_train)
    scaled = LinearRegression(fit_intercept=True).fit(Z_train, y_train)

    results, predictions = {}, {}
    for name, model, Xtr, Xte in (
        ("untransformed", raw, X_train, X_test),
        ("standardized", scaled, Z_train, Z_test),
    ):
        predictions[name] = (model.predict(Xtr), model.predict(Xte))
        results[name] = {
            "training": regression_metrics(y_train, predictions[name][0]),
            "testing": regression_metrics(y_test, predictions[name][1]),
            "intercept": float(model.intercept_),
            "coefficients": dict(zip(FEATURE_NAMES, model.coef_.tolist())),
            "feature_matrix_rank": int(model.rank_),
        }

    mapped_coef = scaled.coef_ / scaler.scale_
    mapped_intercept = float(scaled.intercept_ - scaler.mean_ @ mapped_coef)
    differences = {
        "max_absolute_training_prediction_difference": float(np.max(np.abs(predictions["untransformed"][0] - predictions["standardized"][0]))),
        "max_absolute_testing_prediction_difference": float(np.max(np.abs(predictions["untransformed"][1] - predictions["standardized"][1]))),
        "max_absolute_mapped_coefficient_difference": float(np.max(np.abs(raw.coef_ - mapped_coef))),
        "absolute_mapped_intercept_difference": float(abs(raw.intercept_ - mapped_intercept)),
    }
    np.testing.assert_allclose(raw.coef_, mapped_coef, rtol=0, atol=1e-9)
    np.testing.assert_allclose(raw.intercept_, mapped_intercept, rtol=0, atol=1e-9)
    for raw_pred, scaled_pred in zip(predictions["untransformed"], predictions["standardized"]):
        np.testing.assert_allclose(raw_pred, scaled_pred, rtol=0, atol=1e-9)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "train_rows": 900, "test_rows": 130, "test_sample_ids_inclusive": [501, 630],
        "estimator": "sklearn.linear_model.LinearRegression(fit_intercept=True)",
        "standardization_fit_on": "training set only",
        "models": results, "equivalence_checks": differences,
    }
    (args.output_dir / "q4_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(differences, indent=2))
    print(f"\nSaved Q4 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
