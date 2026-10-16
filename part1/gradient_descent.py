import numpy as np


def _feature_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty, finite vector or matrix.")
    return X


def _loss_and_gradient(design, y, theta):
    with np.errstate(over="raise", invalid="raise"):
        residual = design @ theta - y
        loss = float(np.mean(residual ** 2))
        gradient = (2.0 / len(y)) * (design.T @ residual)
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("Non-finite training loss or gradient.")
    return loss, gradient


class LinearRegressionGD:

    def __init__(self, learning_rate=0.05, max_iter=20_000, tol=1e-6):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be finite and nonnegative.")
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = _feature_matrix(X)
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or len(y) != len(X) or not np.isfinite(y).all():
            raise ValueError("y must be a finite vector with one value per row of X.")

        design = np.column_stack((np.ones(len(X)), X))
        theta = np.zeros(design.shape[1])
        history = []
        try:
            loss, gradient = _loss_and_gradient(design, y, theta)
            history.append(loss)
            for _ in range(self.max_iter):
                if np.max(np.abs(gradient)) <= self.tol:
                    break
                with np.errstate(over="raise", invalid="raise"):
                    theta = theta - self.learning_rate * gradient
                loss, gradient = _loss_and_gradient(design, y, theta)
                history.append(loss)
        except FloatingPointError as error:
            raise FloatingPointError(
                "Gradient descent became non-finite. Reduce the learning rate "
                "or scale the features."
            ) from error

        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:].copy()
        self.loss_history_ = np.asarray(history)
        self.n_iter_ = len(history) - 1
        self.gradient_norm_ = float(np.max(np.abs(gradient)))
        self.converged_ = self.gradient_norm_ <= self.tol
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit() before predict().")
        X = _feature_matrix(X)
        if X.shape[1] != len(self.coef_):
            raise ValueError("Prediction features must match the fitted feature count.")
        return self.intercept_ + X @ self.coef_
