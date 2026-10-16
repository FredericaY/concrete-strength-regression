from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "cement", "slag", "fly_ash", "water", "superplasticizer",
    "coarse_aggregate", "fine_aggregate", "age",
]
TARGET = "strength"
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "concrete+compressive+strength" / "Concrete_Data.xls"
)


def load_concrete_data(path=DEFAULT_DATA_PATH):
    data = pd.read_excel(path, sheet_name="Sheet1", engine="xlrd")
    if data.shape != (1030, 9):
        raise ValueError(f"Expected the original 1030 x 9 dataset; got {data.shape}.")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("The dataset must contain finite numerical values only.")
    data.columns = FEATURE_NAMES + [TARGET]
    data.index = pd.RangeIndex(1, 1031, name="sample_id")
    return data


def split_train_test(data):
    if len(data) != 1030:
        raise ValueError("The prescribed split requires all 1030 original rows.")
    test = data.iloc[500:630].copy()
    train = pd.concat((data.iloc[:500], data.iloc[630:])).copy()
    return train, test


def standardize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0, ddof=0)
    scale = np.where(scale == 0, 1.0, scale)
    return (X_train - mean) / scale, (X_test - mean) / scale
