from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
PART1 = ROOT / "part1"
PYTHON = sys.executable

subprocess.run([PYTHON, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=PART1, check=True)
commands = [
    ["run_q2.py"],
    ["run_q3.py", "--raw-learning-rate", "1e-7", "--scaled-learning-rate", "0.05", "--max-iter", "20000", "--tol", "1e-6"],
    ["run_q4.py"],
    ["run_q5.py", "--log-learning-rate", "0.001", "--max-iter", "20000", "--tol", "1e-6"],
    ["run_q5_optimal.py"],
    ["run_q6.py", "--max-iter", "20000", "--tol", "1e-6"],
    ["run_q7.py"],
]
for command in commands:
    subprocess.run([PYTHON, *command], cwd=PART1, check=True)
for script in ["collect_configuration.py", "build_analysis_assets.py"]:
    subprocess.run([PYTHON, str(ROOT / "part3" / script)], cwd=ROOT / "part3", check=True)
