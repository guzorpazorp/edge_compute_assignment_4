"""Q2 steps 1-6: hardware-aware architecture search (Optuna TPE) on CIFAR-10.

-> results/nas/{history.csv, history.json, best_arch.json, study.db}
Re-running resumes the study until n_trials is reached.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.engine import add_common_args, get_device, load_config  # noqa: E402
from src.nas.search import run_search  # noqa: E402

if __name__ == "__main__":
    args = add_common_args(argparse.ArgumentParser()).parse_args()
    run_search(load_config("nas", args.set), get_device())
