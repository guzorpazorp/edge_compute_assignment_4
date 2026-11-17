"""Central path resolution.

Every script reads/writes through these paths so that the same code runs locally,
on the GTX 1060 machine and on Colab (where CKPT_DIR / RESULTS_DIR point to /content/out).

Checkpoint convention:  CKPT_DIR/<run_name>/last.pt  (full state, for resuming)
                        CKPT_DIR/<run_name>/best.pt  (weights + metadata, used downstream)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
CKPT_DIR = Path(os.environ.get("CKPT_DIR", ROOT / "checkpoints"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
ONNX_DIR = Path(os.environ.get("ONNX_DIR", ROOT / "onnx"))
CONFIG_DIR = ROOT / "configs"

# Canonical run names shared by training, export, inference and benchmarking.
STUDENTS = ["mbv2", "squeezenet", "efflite0"]
TEACHER_RUN = "teacher_resnet50"
NAS_RUN = "nas_best"
ALL_STUDENT_RUNS = [f"{mode}_{s}" for mode in ("baseline", "kd") for s in STUDENTS]


def run_dir(run_name: str) -> Path:
    d = CKPT_DIR / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def best_ckpt(run_name: str) -> Path:
    return CKPT_DIR / run_name / "best.pt"
