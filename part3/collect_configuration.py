import argparse
import hashlib
from importlib.metadata import distributions, version
import json
import platform
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from part1.data import DEFAULT_DATA_PATH, FEATURE_NAMES, load_concrete_data, split_train_test
from part2.table_style import render_table


LABELS = ["Cement", "Blast furnace slag", "Fly ash", "Water", "Superplasticizer", "Coarse aggregate", "Fine aggregate", "Age"]


def main():
    parser = argparse.ArgumentParser(description="Collect Part 1 experiment settings and recorded stopping outcomes.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    part1 = PROJECT_ROOT / "part1"
    paths = {
        "q3": part1 / "outputs/q3/q3_results.json",
        "q5_gd": part1 / "outputs/q5/q5_gd_results.json",
        "q6": part1 / "outputs/q6/q6_results.json",
        "q7": part1 / "outputs/q7/q7_results.json",
    }
    records = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    dataset_hash = hashlib.sha256(DEFAULT_DATA_PATH.read_bytes()).hexdigest()
    assert all(r["dataset_sha256"] == dataset_hash for r in records.values())
    train, test = split_train_test(load_concrete_data())
    assert len(train) == 900 and len(test) == 130
    assert test.index.tolist() == list(range(501, 631))
    for record in (records["q3"], records["q5_gd"], records["q6"]):
        assert record["settings"]["max_iter"] == 20000
        assert record["settings"]["gradient_tolerance"] == 1e-6
        assert record["settings"]["initial_parameters"] == "all zeros"
    multivariate = []
    multi_rows = []
    for key, label, questions in [
        ("untransformed", "Raw (Q3, Q5)", ["Q3", "Q5"]),
        ("standardized", "Standardized (Q3, Q5)", ["Q3", "Q5"]),
        ("log_transformed", "Log (Q5)", ["Q5"]),
    ]:
        run = records["q5_gd"]["models"][key]
        if key in records["q3"]["models"]:
            original = records["q3"]["models"][key]
            assert all(run[k] == original[k] for k in ["learning_rate", "updates", "converged", "stop_reason"])
        item = {"features": key, "questions": questions, "learning_rate": run["learning_rate"],
                "max_updates": 20000, "gradient_tolerance": 1e-6, "actual_updates": run["updates"],
                "stop_reason": run["stop_reason"], "final_gradient_infinity_norm": run["final_gradient_infinity_norm"]}
        assert item["stop_reason"] == ("gradient_tolerance" if item["final_gradient_infinity_norm"] <= 1e-6 else "iteration_cap")
        multivariate.append(item)
        multi_rows.append([label, f"{run['learning_rate']:g}", "20,000", "1e-6", f"{run['updates']:,}",
                           "Tolerance" if run["converged"] else "Cap"])
    univariate, uni_rows = [], []
    for feature, label in zip(FEATURE_NAMES, LABELS):
        run = records["q6"]["univariate_models"][feature]["gd"]
        x = train[feature].to_numpy(float)
        design = np.column_stack((np.ones(len(x)), x))
        maximum_eigenvalue = float(np.linalg.eigvalsh(2 * design.T @ design / len(x))[-1])
        np.testing.assert_allclose(run["learning_rate"], 1 / maximum_eigenvalue, rtol=1e-12, atol=0)
        item = {"feature": feature, "learning_rate": run["learning_rate"], "max_updates": 20000,
                "gradient_tolerance": 1e-6, "actual_updates": run["updates"], "stop_reason": run["stop_reason"],
                "final_gradient_infinity_norm": run["final_gradient_infinity_norm"]}
        assert item["stop_reason"] == ("gradient_tolerance" if item["final_gradient_infinity_norm"] <= 1e-6 else "iteration_cap")
        univariate.append(item)
        uni_rows.append([label, f"{run['learning_rate']:.6e}", f"{run['updates']:,}", "Tolerance" if run["converged"] else "Cap"])
    references = [
        {"questions": "Q4", "estimator": "LinearRegression", "inputs": "Raw; standardized", "settings": "Intercept; no penalty"},
        {"questions": "Q5.3", "estimator": "LinearRegression", "inputs": "Raw; standard; log", "settings": "Intercept; no penalty"},
        {"questions": "Q6 reference", "estimator": "LinearRegression", "inputs": "8 single; 1 multi", "settings": "Raw; intercept"},
        {"questions": "Q7", "estimator": "statsmodels OLS", "inputs": "Raw; standard; log", "settings": "t test; df = 891"},
    ]
    direct_versions = {name: version(name) for name in ["numpy", "pandas", "xlrd", "matplotlib", "scikit-learn", "statsmodels", "scipy"]}
    environment_rows = [["Python", platform.python_version()], ["Operating system", platform.system()], *[list(item) for item in direct_versions.items()]]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": "black", "savefig.facecolor": "white"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_table(multi_rows, ["Model", "Learning\nrate", "Max\nupdates", "Tolerance", "Updates", "Stop"],
                 [0.29, 0.17, 0.14, 0.12, 0.14, 0.14], args.output_dir / "table_1_multivariate_gd.png", (8, 1.5))
    render_table(uni_rows, ["Predictor", "Learning rate", "Updates", "Stop"],
                 [0.36, 0.26, 0.18, 0.20], args.output_dir / "table_2_univariate_gd.png", (8, 3.2))
    render_table([[r[k] for k in ["questions", "estimator", "inputs", "settings"]] for r in references],
                 ["Experiment", "Estimator", "Feature sets", "Settings"], [0.19, 0.27, 0.27, 0.27],
                 args.output_dir / "table_3_reference_estimators.png", (8, 1.75))
    render_table(environment_rows, ["Component", "Version"], [0.6, 0.4],
                 args.output_dir / "table_4_environment.png", (5.2, 3.3))
    script_names = ["gradient_descent.py", "data.py", "metrics.py", "run_q2.py", "run_q3.py", "run_q4.py", "run_q5.py", "run_q5_optimal.py", "run_q6.py", "run_q7.py"]
    payload = {
        "dataset_sha256": dataset_hash,
        "common": {"training_samples": 900, "testing_samples": 130, "test_sample_ids_inclusive": [501, 630],
                   "batch_size": 900, "initialization": "all parameters zero", "fit_intercept": True, "regularization": "none",
                   "max_updates": 20000, "gradient_infinity_norm_tolerance": 1e-6, "target_transform": "none",
                   "scaling_ddof": 0, "seed": None, "row_shuffle": False,
                   "mse_110_role": "comparison milestone only", "positive_r2_role": "post-fit assignment check only"},
        "multivariate_gd": multivariate, "univariate_gd": univariate, "reference_estimators": references,
        "inference": records["q7"]["inference"],
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "direct_and_scipy": direct_versions,
                        "installed_packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions()))},
        "source_results_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "source_code_sha256": {name: hashlib.sha256((part1 / name).read_bytes()).hexdigest() for name in script_names},
    }
    (args.output_dir / "experiment_configuration.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"multivariate_models": len(multivariate), "univariate_models": len(univariate),
                      "python": platform.python_version(), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
