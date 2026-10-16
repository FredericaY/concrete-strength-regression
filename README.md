# Concrete strength regression

Concrete compressive strength prediction with custom gradient descent, regression analysis, and reproducible experiments.

This project implements batch gradient descent from scratch, compares raw, standardized and logarithmic inputs, and examines univariate coefficients and OLS p-values. It also includes descriptive statistics and the scenario analyses used in the client report.

## Repository contents

- part1: gradient descent, data preparation, metrics, Q2-Q7 runners, nine unit tests and recorded results.
- part2: descriptive statistics, client-model comparisons, scenario analysis and the shared table helper.
- part3: the reproduction driver, experiment configuration collector, technical-appendix figures and dependency lock.
- concrete+compressive+strength: the original Concrete_Data.xls dataset.

Report prose is maintained separately. The scripts produce calculations, figures and structured results.

## Setup and reproduce

The recorded environment is Windows 11 with Python 3.13.5. From the repository root in PowerShell:

~~~powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r part3\requirements-lock.txt
.\.venv\Scripts\python.exe part3\reproduce.py
~~~

The driver uses the active interpreter for every subprocess. It runs nine unit tests, Q2, Q3, Q4, Q5 gradient descent, Q5 optimal fits, Q6 and Q7, then rebuilds the Part 3 configuration tables and analytical figures. Q7 reads Q6 results, so that order matters. Generated outputs are replaced inside the repository.

To reproduce the client-report statistics and scenarios after the main driver:

~~~powershell
.\.venv\Scripts\python.exe part2\run_descriptive_analysis.py
.\.venv\Scripts\python.exe part2\run_model_analysis.py
~~~

For individual experiments, enter part1:

~~~powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\.venv\Scripts\python.exe run_q2.py
..\.venv\Scripts\python.exe run_q3.py --raw-learning-rate 1e-7 --scaled-learning-rate 0.05 --max-iter 20000 --tol 1e-6
..\.venv\Scripts\python.exe run_q4.py
..\.venv\Scripts\python.exe run_q5.py --log-learning-rate 0.001 --max-iter 20000 --tol 1e-6
..\.venv\Scripts\python.exe run_q5_optimal.py
..\.venv\Scripts\python.exe run_q6.py --max-iter 20000 --tol 1e-6
..\.venv\Scripts\python.exe run_q7.py
~~~

From part3, regenerate only the technical-appendix assets:

~~~powershell
..\.venv\Scripts\python.exe collect_configuration.py
..\.venv\Scripts\python.exe build_analysis_assets.py
~~~

Dataset paths resolve relative to the source files. The Q2-Q7 runners accept --data PATH. Q7 also accepts --q6-results PATH when Q6 results are stored elsewhere. The Part 2 model analysis reads part1/outputs/q5/q5_optimal_results.json by default.

## Data and experimental settings

Data: Yeh, I. (1998), [Concrete Compressive Strength, UCI Machine Learning Repository](https://doi.org/10.24432/C5PK67).

The workbook is included without modification under the dataset's [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).

The original workbook has 1,030 observations and nine columns. Its SHA-256 is:

710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4

Test sample IDs are 501 through 630, excluding the Excel header. The other 900 observations form the training set. Row order and duplicates are preserved. Standardization uses training means and population standard deviations. The log model applies natural log(1 + x) to all eight inputs without additional scaling. Strength remains in MPa.

All GD parameters start at zero. Training uses full batches and stops at a gradient infinity norm of 1e-6 or after 20,000 updates. Raw, standardized and logarithmic multivariate learning rates are 1e-7, 0.05 and 0.001. Q6 uses the reciprocal of each training Hessian's largest eigenvalue. Positive training R² and MSE 110 are reporting checks.

## Results and validation

Part 1 outputs contain full-precision JSON results and question figures. Part 2 outputs contain training statistics, model comparisons and single-input scenarios. Part 3 outputs contain minimalist figures and tables, loss_histories.npz, and experiment_configuration.json with package versions and data/source hashes.

The implementation was rerun in the recorded environment. Nine unit tests passed. Six Part 1 result files and 661 numerical values matched the saved records with relative tolerance 1e-10 and absolute tolerance 1e-12. GD update counts were unchanged. A fresh dependency installation was not tested; small floating-point differences may occur across platforms.
