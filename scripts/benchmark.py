"""Automated benchmarking of every available model on this machine's CPU.

Measures accuracy, parameters, FLOPs, checkpoint/ONNX size, batch-1 latency (PyTorch and
ONNX Runtime) and peak inference memory.  -> results/benchmark.csv (+ benchmark_machine.json)
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deploy.runtime import Predictor  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.nas.hw_profile import machine_info, profile_model  # noqa: E402
from src.paths import ALL_STUDENT_RUNS, NAS_RUN, ONNX_DIR, RESULTS_DIR, TEACHER_RUN, best_ckpt  # noqa: E402


def ort_latency(pred, res, warmup=10, runs=100):
    x = np.random.randn(1, 3, res, res).astype(np.float32)
    for _ in range(warmup):
        pred(x)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        pred(x)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs", type=int, default=100, help="timed iterations per latency measurement")
    ap.add_argument("--limit", type=int, default=None, help="max test images for accuracy")
    ap.add_argument("--no-teacher", action="store_true")
    args = ap.parse_args()

    runs = ([] if args.no_teacher else [TEACHER_RUN]) + ALL_STUDENT_RUNS + [NAS_RUN]
    rows = []
    for run in runs:
        if not best_ckpt(run).exists():
            print(f"[{run}] skipped (no checkpoint)")
            continue
        model, meta = load_checkpoint(best_ckpt(run))
        res = meta["resolution"]
        hw = profile_model(model, res, args.threads, args.runs)
        hw["torch_latency_ms"] = hw.pop("latency_ms")
        row = {"run": run, "arch": meta["arch"], "mode": meta.get("mode", ""), "dataset": meta["dataset"],
               "resolution": res, "train_best_acc": meta.get("acc"), **hw}

        onnx_path = ONNX_DIR / f"{run}.onnx"
        if onnx_path.exists():
            pred = Predictor(run, "onnx", args.threads)
            row["onnx_size_mb"] = round(onnx_path.stat().st_size / 2**20, 4)
            row["ort_latency_ms"] = round(ort_latency(pred, res, runs=args.runs), 4)
            ev = pred.evaluate_testset(limit=args.limit)
        else:
            ev = Predictor(run, "torch", args.threads).evaluate_testset(limit=args.limit)
        row.update({"test_top1": round(ev["top1"], 2), "test_top5": round(ev["top5"], 2), "test_n": ev["n"],
                    "eval_backend": "onnx" if onnx_path.exists() else "torch"})
        rows.append(row)
        print({k: row[k] for k in ("run", "test_top1", "params", "size_mb", "torch_latency_ms")}
              | ({"ort_latency_ms": row["ort_latency_ms"]} if "ort_latency_ms" in row else {}))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "benchmark.csv", index=False)
    with open(RESULTS_DIR / "benchmark_machine.json", "w") as f:
        json.dump({**machine_info(), "threads": args.threads, "timed_runs": args.runs}, f, indent=2)
    print(f"\nSaved {len(df)} rows to {RESULTS_DIR / 'benchmark.csv'}")
    if len(df):
        print(df[[c for c in ("run", "test_top1", "params", "size_mb", "torch_latency_ms", "ort_latency_ms",
                              "peak_mem_mb") if c in df]].to_string(index=False))
