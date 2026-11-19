# edge_compute_assignment_4: Edge KD + NAS (Assignment 4: Edge Computing & Intelligent Systems)

- **Q1:** a knowledge-distillation framework that distils a ResNet-50 teacher into MobileNetV2,
  SqueezeNet and EfficientNet-Lite0 students on CIFAR-100, then exports them to ONNX.
- **Q2:** a hardware-aware neural architecture search (Optuna TPE) on CIFAR-10 with measured CPU
  latency, size and memory constraints, followed by retraining and ONNX deployment of the best
  architecture.

Design decisions and their rationale are in [`docs/notes.md`](docs/notes.md).

## Repository layout
```
configs/        teacher.yaml · student.yaml · nas.yaml   (all hyper-parameters; override with --set k=v)
src/
  paths.py      data / checkpoint / results locations (env-overridable) + run names
  data.py       CIFAR loaders, GPU-resident loader for NAS
  engine.py     train/eval loops, schedules, checkpoint + resume + CSV logging
  models/       resnet.py · mobilenetv2.py · squeezenet.py · efficientnet_lite.py
  kd/distill.py soft-target (T) + hard-label + feature (FitNets) distillation, joint student training
  nas/          space.py (search space) · builder.py (arch generator) · hw_profile.py · search.py (Optuna)
  deploy/       export.py (ONNX + parity check) · runtime.py (unified torch / ONNX Runtime predictor)
scripts/        one entry point per pipeline step (below)
notebooks/      Colab launchers for the GPU-heavy steps
checkpoints/    <run>/best.pt   (weights + metadata; last.pt = resume state, not committed)
onnx/           <run>.onnx + <run>.json (metadata for PyTorch-free inference)
results/        logs/<run>.csv · q1_summary.{csv,md} · nas/{history.csv,history.json,best_arch.json} · benchmark.csv
docs/           notes.md · report/ (PDF) · screenshots/
```
Run names: `teacher_resnet50`, `baseline_{mbv2,squeezenet,efflite0}`, `kd_{mbv2,squeezenet,efflite0}`, `nas_best`.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
# PyTorch: CPU-only build for the laptop, CUDA build for a GPU machine (see pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/download_data.py --dataset cifar100 cifar10     # ~350 MB, via Hugging Face mirror
```

## Pipeline: where each step runs

| # | Step | Command | Machine | Time |
|---|---|---|---|---|
| 1 | Teacher ResNet-50 | `notebooks/colab_teacher.ipynb` → `scripts/train_teacher.py` | Colab T4 | ~45–60 min |
| 2 | Student baselines (no KD) | `notebooks/colab_students.ipynb` → `train_student.py --mode baseline` | Colab T4 | ~20–30 min (20 epochs) |
| 3 | Distilled students | `notebooks/colab_students.ipynb` → `train_student.py --mode kd` | Colab T4 | ~30–40 min (20 epochs) |
| 4 | NAS search | `python scripts/nas_search.py` | GTX 1060 | ~20–30 min |
| 5 | Retrain best NAS arch from scratch | `python scripts/nas_retrain.py` | GTX 1060 | ~15–25 min |
| 6 | ONNX export (+ parity check) | `python scripts/export_onnx.py` | laptop CPU | ~1 min |
| 7 | Inference (torch / ONNX Runtime) | `python scripts/infer.py ...` | laptop CPU | seconds |
| 8 | Benchmark → `results/benchmark.csv` | `python scripts/benchmark.py` | laptop CPU | ~15–30 min |

Every training script resumes automatically from `CKPT_DIR/<run>/last.pt` as long as that file
still exists (on the GTX 1060 machine and locally, but not after a Colab runtime reset).

### Steps 1–3 on Colab (manual upload/download, no Google Drive)
1. Make the dataset archive locally, once:
   `tar -czf cifar100.tar.gz -C data cifar-100-python` (~160 MB).
2. Open `notebooks/colab_teacher.ipynb` in Colab and set the runtime to **T4 GPU**.
3. Drag `cifar100.tar.gz` into the Colab **Files** pane. If it's missing, the notebook opens a
   file picker.
4. Run all cells. The notebook clones the repo, extracts the dataset and trains. Outputs go to
   `/content/out`, which mirrors the repo layout. When training finishes it downloads
   `teacher_outputs.tar.gz`.
5. Open `colab_students.ipynb` and upload `cifar100.tar.gz` and `teacher_outputs.tar.gz`. It
   downloads `baseline_outputs.tar.gz` after the baselines and `kd_outputs.tar.gz` after KD.
6. Extract each bundle into this repo: `tar -xzf <bundle>.tar.gz -C .` (this fills
   `checkpoints/<run>/best.pt` and `results/logs/<run>.csv`), then run
   `python scripts/summarize_q1.py` to write `results/q1_summary.{csv,md}`.

Colab storage is wiped when the runtime disconnects, so keep the tab active and download each
bundle as soon as its stage ends.

### Steps 4–5 on the GTX 1060 machine
```bash
python scripts/download_data.py --dataset cifar10
python scripts/nas_search.py            # resumable; writes results/nas/*
python scripts/nas_retrain.py           # reads results/nas/best_arch.json -> checkpoints/nas_best/best.pt
```
Then commit and push `results/nas/` and `checkpoints/nas_best/best.pt`.

### Steps 6–8 locally (CPU)
```bash
python scripts/export_onnx.py                                   # all students + NAS model -> onnx/
python scripts/infer.py --run kd_mbv2 --backend onnx --image some_image.png
python scripts/infer.py --run all --backend both --testset      # accuracy, PyTorch vs ONNX Runtime
python scripts/benchmark.py --threads 1                         # -> results/benchmark.csv
```

### Smoke test (any machine, a few minutes on CPU)
```bash
export CKPT_DIR=/tmp/smoke/ckpt RESULTS_DIR=/tmp/smoke/results ONNX_DIR=/tmp/smoke/onnx
S="epochs=1 limit_batches=2 batch_size=16 num_workers=0"
python scripts/train_teacher.py --set pretrained=false $S
python scripts/train_student.py --mode baseline --set $S
python scripts/train_student.py --mode kd --set $S
python scripts/nas_search.py --set n_trials=4 n_startup_trials=2 proxy_train_size=2000 proxy_train.epochs=1 limit_batches=3
python scripts/nas_retrain.py --set retrain.epochs=1 retrain.limit_batches=3
python scripts/export_onnx.py && python scripts/benchmark.py --runs 10 --limit 256
```

## Deliverables map
| # | Deliverable | Where |
|---|---|---|
| 1 | Teacher model | `checkpoints/teacher_resnet50/best.pt`, `scripts/train_teacher.py` |
| 2 | Student architectures | `src/models/{mobilenetv2,squeezenet,efficientnet_lite}.py` |
| 3 | Baseline students | `checkpoints/baseline_*/best.pt` |
| 4 | Knowledge distillation | `src/kd/distill.py`, `scripts/train_student.py`, `checkpoints/kd_*/best.pt` |
| 5 | NAS search space | `src/nas/space.py`, `src/nas/builder.py` |
| 6 | NAS algorithm | `src/nas/search.py`, `scripts/nas_search.py` |
| 7 | NAS results | `results/nas/best_arch.json`, `results/nas/history.{csv,json}` |
| 8 | Optimized architecture | `checkpoints/nas_best/best.pt` |
| 9 | Inference pipeline | `src/deploy/runtime.py`, `scripts/infer.py` |
| 10 | ONNX deployment | `onnx/*.onnx`, `src/deploy/export.py`, `scripts/export_onnx.py` |
| 11 | Benchmarking | `scripts/benchmark.py`, `results/benchmark.csv` |
| 12 | Technical report | `docs/report/` |
| 13 | README | this file |
