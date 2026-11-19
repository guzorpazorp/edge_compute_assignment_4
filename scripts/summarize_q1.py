"""Summarise Q1 training runs (teacher, baselines, KD students) from their logs and checkpoints.

  python scripts/summarize_q1.py   -> results/q1_summary.csv, results/q1_summary.md
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.engine import count_params  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.paths import ALL_STUDENT_RUNS, RESULTS_DIR, TEACHER_RUN, best_ckpt  # noqa: E402


def summarize(run):
    log = pd.read_csv(RESULTS_DIR / "logs" / f"{run}.csv")
    best = log.loc[log["test_acc"].idxmax()]
    model, meta = load_checkpoint(best_ckpt(run))
    return {
        "run": run, "model": meta["arch"], "mode": meta.get("mode", "teacher"),
        "params_M": round(count_params(model) / 1e6, 3),
        "top1": float(best["test_acc"]), "top5": float(best["test_top5"]),
        "best_epoch": int(best["epoch"]), "epochs": len(log),
        "train_min": round(float(log["time_s"].iloc[-1]) / 60, 1),
    }


if __name__ == "__main__":
    runs = [r for r in [TEACHER_RUN, *ALL_STUDENT_RUNS] if best_ckpt(r).exists()]
    df = pd.DataFrame([summarize(r) for r in runs])
    base = {row.model: row.top1 for row in df.itertuples() if row.mode == "baseline"}
    df["delta_vs_baseline"] = [round(r.top1 - base[r.model], 2) if r.mode == "kd" and r.model in base
                               else None for r in df.itertuples()]
    df.to_csv(RESULTS_DIR / "q1_summary.csv", index=False)
    cells = df.astype(object).where(df.notna(), "")
    md = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in cells.itertuples(index=False)]
    (RESULTS_DIR / "q1_summary.md").write_text("\n".join(md) + "\n")
    print(df.to_string(index=False))
