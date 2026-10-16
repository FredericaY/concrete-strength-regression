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
from table_style import render_table


VARIABLES = {
    "cement": ("Cement", "kg/m³"),
    "slag": ("Blast furnace slag", "kg/m³"),
    "fly_ash": ("Fly ash", "kg/m³"),
    "water": ("Water", "kg/m³"),
    "superplasticizer": ("Superplasticizer", "kg/m³"),
    "coarse_aggregate": ("Coarse aggregate", "kg/m³"),
    "fine_aggregate": ("Fine aggregate", "kg/m³"),
    "age": ("Curing age", "days"),
    "strength": ("Compressive strength", "MPa"),
}


def render_statistics(statistics, path):
    rows = []
    for feature, row in statistics.iterrows():
        label, unit = VARIABLES[feature]
        rows.append([label, unit, *(f"{row[name]:,.2f}" for name in ("mean", "std", "min", "max"))])
    render_table(
        rows, ["Variable", "Unit", "Mean", "Std. dev.", "Minimum", "Maximum"],
        [0.29, 0.11, 0.15, 0.15, 0.15, 0.15], path, (9, 3.5), center_columns=(1,),
    )


def plot_relationship(train, feature, path):
    descriptions = {
        "age": ("Curing age (days)", (0, 380), [0, 60, 120, 180, 240, 300, 365]),
        "cement": ("Cement content (kg/m³)", (90, 555), [100, 200, 300, 400, 500]),
    }
    xlabel, limits, ticks = descriptions[feature]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.14)
    points = ax.scatter(train[feature], train[TARGET], s=18, alpha=0.4,
                        color="#333333", edgecolors="none", zorder=3)
    offsets = np.asarray(points.get_offsets())
    expected = train[[feature, TARGET]].to_numpy(dtype=float)
    np.testing.assert_array_equal(offsets, expected)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel("Compressive strength (MPa)", labelpad=8)
    ax.set_xlim(*limits)
    ax.set_xticks(ticks)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.grid(False)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return {"x": feature, "y": TARGET, "observations_plotted": len(offsets), "x_scale": "linear", "y_scale": "linear"}


def main():
    parser = argparse.ArgumentParser(description="Part 2: training-set descriptive statistics and relationship plots.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    train, test = split_train_test(load_concrete_data(args.data))
    expected_ids = list(range(1, 501)) + list(range(631, 1031))
    assert train.index.tolist() == expected_ids
    assert test.index.tolist() == list(range(501, 631))
    assert len(train) == 900 and not train.isna().any().any()
    statistics = train[FEATURE_NAMES + [TARGET]].agg(["mean", "std", "min", "max"]).T
    values = train[FEATURE_NAMES + [TARGET]].to_numpy(dtype=float)
    np.testing.assert_allclose(statistics["mean"], values.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(statistics["std"], values.std(axis=0, ddof=1), rtol=1e-12)
    np.testing.assert_array_equal(statistics["min"], values.min(axis=0))
    np.testing.assert_array_equal(statistics["max"], values.max(axis=0))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "black", "savefig.facecolor": "white",
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_statistics(statistics, args.output_dir / "table_1_training_statistics.png")
    plots = {}
    for number, feature in enumerate(("age", "cement"), start=1):
        filename = f"figure_{number}_{feature}_vs_strength.png"
        plots[filename] = plot_relationship(train, feature, args.output_dir / filename)
    records = {}
    for feature, row in statistics.iterrows():
        records[feature] = {
            "label": VARIABLES[feature][0], "unit": VARIABLES[feature][1], "count": int(train[feature].count()),
            "mean": float(row["mean"]), "sample_std": float(row["std"]),
            "min": float(row["min"]), "max": float(row["max"]),
        }
    payload = {
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "worksheet": "Sheet1", "training_samples": len(train), "excluded_test_samples": len(test),
        "training_sample_id_ranges_inclusive": [[1, 500], [631, 1030]],
        "excluded_test_sample_ids_inclusive": [501, 630],
        "sample_id_convention": "1-based observations, excluding the Excel header",
        "standard_deviation_ddof": 1, "display_decimal_places": 2,
        "statistics": records, "plots": plots,
    }
    (args.output_dir / "training_statistics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(statistics.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"\nTraining observations: {len(train)}. Saved table and two relationship plots to {args.output_dir}")


if __name__ == "__main__":
    main()
