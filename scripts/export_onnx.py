"""Export trained models to ONNX (+ parity check against ONNX Runtime).

  python scripts/export_onnx.py                 # every trained student + NAS model found
  python scripts/export_onnx.py --runs kd_mbv2  # specific runs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deploy.export import export_run  # noqa: E402
from src.paths import ALL_STUDENT_RUNS, NAS_RUN, TEACHER_RUN, best_ckpt  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None)
    ap.add_argument("--include-teacher", action="store_true")
    args = ap.parse_args()
    runs = args.runs or ALL_STUDENT_RUNS + [NAS_RUN] + ([TEACHER_RUN] if args.include_teacher else [])
    for run in runs:
        if best_ckpt(run).exists():
            export_run(run)
        else:
            print(f"[{run}] skipped (no checkpoint at {best_ckpt(run)})")
