import numpy as np


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Targets and predictions must be matching nonempty vectors.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Targets and predictions must be finite.")
    mse = float(np.mean((y_true - y_pred) ** 2))
    variance = float(np.var(y_true, ddof=0))
    return {"mse": mse, "r2": 1.0 - mse / variance if variance > 0 else float("nan")}
