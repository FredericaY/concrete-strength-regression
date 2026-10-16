import argparse
from pathlib import Path

from data import (
    DEFAULT_DATA_PATH, FEATURE_NAMES, TARGET, load_concrete_data,
    split_train_test, standardize_train_test,
)
from gradient_descent import LinearRegressionGD
from metrics import regression_metrics


def main():
    parser = argparse.ArgumentParser(description='Q2 entry point: train the custom optimizer and evaluate the fixed split.')
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--features", nargs="+", choices=FEATURE_NAMES, default=FEATURE_NAMES)
    parser.add_argument("--scaling", choices=("none", "standard"), default="standard")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=20_000)
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Stop when max(abs(training MSE gradient)) <= tol.")
    args = parser.parse_args()

    train, test = split_train_test(load_concrete_data(args.data))
    X_train = train[args.features].to_numpy(dtype=float)
    X_test = test[args.features].to_numpy(dtype=float)
    y_train = train[TARGET].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)
    if args.scaling == "standard":
        X_train, X_test = standardize_train_test(X_train, X_test)

    model = LinearRegressionGD(args.learning_rate, args.max_iter, args.tol)
    model.fit(X_train, y_train)
    train_metrics = regression_metrics(y_train, model.predict(X_train))
    test_metrics = regression_metrics(y_test, model.predict(X_test))

    print("Q2: custom full-batch gradient descent")
    print(f"Data: {args.data}")
    print(f"Split: {len(train)} train / {len(test)} test (sample IDs 501-630)")
    print(f"Features: {', '.join(args.features)}")
    print(f"Scaling: {args.scaling}; learning rate: {args.learning_rate:g}")
    print(f"Updates: {model.n_iter_}; converged: {model.converged_}")
    print(f"Final gradient infinity norm: {model.gradient_norm_:.6g}")
    if not model.converged_:
        print("Stopped at max_iter; the gradient tolerance was not reached.")
    print(f"Training MSE: {model.loss_history_[0]:.6f} -> {model.loss_history_[-1]:.6f}")
    print(f"\n{'Set':<10} {'MSE (MPa^2)':>15} {'R^2':>12}")
    for name, scores in (("Training", train_metrics), ("Testing", test_metrics)):
        print(f"{name:<10} {scores['mse']:>15.6f} {scores['r2']:>12.6f}")
    print(f"\nIntercept: {model.intercept_:.6f}")
    print(f"Coefficients (feature space: {args.scaling}):")
    for name, coef in zip(args.features, model.coef_):
        print(f"  {name:<20} {coef:>12.6f}")


if __name__ == "__main__":
    main()
