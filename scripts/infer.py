"""Unified inference over any trained model, with the PyTorch or ONNX Runtime CPU backend.

  python scripts/infer.py --run kd_mbv2 --backend onnx --image path/to/img.png
  python scripts/infer.py --run all --backend both --testset --limit 2000
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deploy.runtime import Predictor  # noqa: E402
from src.paths import ALL_STUDENT_RUNS, NAS_RUN, ONNX_DIR, best_ckpt  # noqa: E402


def available(run, backend):
    return best_ckpt(run).exists() if backend == "torch" else (ONNX_DIR / f"{run}.onnx").exists()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", default=["all"], help="run names or 'all'")
    ap.add_argument("--backend", choices=["torch", "onnx", "both"], default="onnx")
    ap.add_argument("--image", type=str, help="classify a single image")
    ap.add_argument("--testset", action="store_true", help="evaluate on the test split")
    ap.add_argument("--limit", type=int, default=None, help="max test images")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if not (args.image or args.testset):
        ap.error("give --image PATH and/or --testset")

    runs = ALL_STUDENT_RUNS + [NAS_RUN] if args.run == ["all"] else args.run
    backends = ["torch", "onnx"] if args.backend == "both" else [args.backend]
    for run in runs:
        for be in backends:
            if not available(run, be):
                print(f"{run:22s} {be:5s}  -- not available")
                continue
            p = Predictor(run, be, args.threads)
            if args.image:
                preds = ", ".join(f"{c} ({pr:.1%})" for c, pr in p.predict_image(args.image))
                print(f"{run:22s} {be:5s}  {preds}")
            if args.testset:
                t = time.perf_counter()
                r = p.evaluate_testset(limit=args.limit)
                print(f"{run:22s} {be:5s}  top1={r['top1']:.2f}%  top5={r['top5']:.2f}%  "
                      f"n={r['n']}  ({time.perf_counter() - t:.1f}s)")
