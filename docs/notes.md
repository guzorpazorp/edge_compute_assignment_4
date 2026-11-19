# Design notes

Decisions that are not obvious from the code, with the reasoning behind them.

## Q1: Knowledge distillation

### Teacher: ImageNet-initialised ResNet-50, fine-tuned on CIFAR-100
The teacher is our own ResNet-50 implementation (`src/models/resnet.py`), trained on CIFAR-100.
Its parameter names mirror torchvision's, so its weights are **initialised from ImageNet-1k**
(`IMAGENET1K_V2`) before fine-tuning. Only the 100-class `fc` layer starts from random weights.

Why fine-tune instead of training from scratch:
- **Better teacher, so more to distil.** A ResNet-50 trained from scratch on 50k 32×32 images
  usually reaches ~76–78% top-1 and needs ~200 epochs to get there. Starting from ImageNet
  features gives roughly 80%+ in ~30 epochs. KD gains grow with the quality of the teacher's
  soft targets ("dark knowledge"), so a stronger teacher directly benefits the students.
- **Compute budget.** About 30 epochs (~45–60 min on a Colab T4) instead of ~200 epochs (~5 h).
  That fits the 3-day deadline and Colab's session limits.
- **Realistic deployment practice.** In edge pipelines the large teacher runs server-side and
  is almost always pretrained. Only the small students are deployed.
- **Students are not pretrained.** All three students train from random initialisation, both
  as baselines and with KD. The baseline-vs-KD comparison therefore isolates the effect of
  distillation.

Stem adaptation: the pretrained 7×7 `conv1` is kept so its weights still apply. Its stride
changes from 2 to 1 and the max-pool is removed, so a 32×32 image is not reduced to 8×8
before `layer1`. The backbone uses 0.2× the head learning rate to preserve pretrained features.

Training from scratch is still supported: `--set pretrained=false epochs=200 lr=0.1`.

### Students: CIFAR-adapted, written from scratch
MobileNetV2, SqueezeNet 1.1 and EfficientNet-Lite0 follow their papers. They are adapted to
32×32 inputs by removing early downsampling (stride-1 stem, plus stride 1 in the first stride-2
stage), so the final feature map is 4×4 instead of 1×1.
- **SqueezeNet** also gets BatchNorm inside its Fire modules (the original is hard to train at
  CIFAR learning rates) and no ReLU on the final logits, because non-negative logits distort
  softmax/KD.
- **EfficientNet-Lite0** = B0 without squeeze-excitation, with ReLU6, and with a fixed
  32-channel stem and 1280-channel head (the "Lite" changes for mobile accelerators).

### KD loss
`L = α·T²·KL(σ(z_s/T) ‖ σ(z_t/T)) + (1−α)·CE(z_s, y) + β·‖n(A(f_s)) − n(f_t)‖²`
- T² scaling keeps the soft-target gradients on the same scale as the CE term (Hinton et al.).
- **Feature KD (FitNets):** the teacher's `layer3` output (1024 ch, 8×8) is the hint. Each
  student's 8×8 stage output is mapped by a 1×1 conv + BN adapter `A`. Both maps are
  L2-normalised per sample (`n`), so the term lies in [0, 4] whatever the teacher's activation
  scale. Without normalisation it measured ~90 at initialisation and swamped the other terms.
- Defaults: T = 4, α = 0.9, β = 5. Each term can be switched off in `configs/student.yaml` for
  ablations.
- **Joint training:** all three students train in lock-step on the same batches, sharing one
  fp16 teacher forward pass. The teacher is the most expensive network, so this saves ~2/3 of
  the teacher compute compared with three separate KD runs.

### Q1 results (students: 20 epochs; teacher: 30 epochs)
Best test top-1 on CIFAR-100, from `results/q1_summary.md` (regenerate with `scripts/summarize_q1.py`):

| Student | Params | Baseline | KD | Δ |
|---|---|---|---|---|
| MobileNetV2 | 2.35 M | 69.22 | 69.62 | +0.40 |
| SqueezeNet | 0.78 M | 58.75 | 53.21 | −5.54 |
| EfficientNet-Lite0 | 3.50 M | 70.33 | 71.35 | +1.02 |

Teacher ResNet-50 (23.7 M params): 86.74%.

- **The two larger students gain a little from KD.** Their top-5 improves too: +0.8 for
  MobileNetV2 and +1.1 for EfficientNet-Lite0. KD students also show a smaller train/test gap
  (train accuracy about 80–85% vs about 85–90% for the baselines), which fits soft targets acting
  as a regulariser. Gains are modest because the budget is only 20 epochs. KD typically pays off
  more over longer schedules.
- **SqueezeNet gets worse with KD and underfits.** Its final train accuracy is 56% with KD vs
  65% without, so this is not overfitting. Likely causes:
  - *Capacity gap.* The student has 30× fewer parameters than the teacher, and the teacher's
    86.7% distribution is hard to match.
  - *Competing objectives.* With α = 0.9, only 10% of the loss is the hard-label CE. The feature
    hint also forces SqueezeNet's 256-channel stage to mimic ResNet's 1024-channel layer3, which
    takes capacity away from classification.
  - *Short schedule.* SqueezeNet's KD loss is still falling at epoch 20.

  The standard fixes are a lower α, no feature term, or a teacher assistant. These were not run
  within the time budget, and the result is reported as is.
- The kd_mbv2 test accuracy oscillates in epochs 4–8 (22.6 → 42.0 → 27.3). This is most likely noise from
  BN statistics at the peak learning rate, and training is stable after the LR decays.

## Q2: Hardware-aware NAS

- **Strategy:** Optuna TPE (Bayesian optimisation). The first `n_startup_trials` are random.
  A MedianPruner stops clearly poor candidates during proxy training.
- **Constraint handling:** each sampled architecture is built and profiled on the CPU *before*
  any training: median batch-1 latency with 1 thread, fp32 `state_dict` size, and peak inference
  memory from the PyTorch profiler's allocation events plus the weights. Violators are recorded
  with accuracy 0 and the reason, so TPE learns to avoid that region and no GPU time is wasted.
- **Duplicates:** TPE can re-propose an architecture it has already evaluated. Those trials reuse
  the earlier result instead of re-profiling, which matters because latency noise could flip the
  verdict.
- **Constraint values** (3 ms / 1 MB / 12 MB) were calibrated by profiling 40 random
  architectures on the laptop (i7-8665U). About 70% pass, so the constraints are binding but not
  degenerate. Latency depends on the hardware, so `best_arch.json` records the CPU it was
  measured on, and `benchmark.py` re-measures the final model on the deployment laptop.
- **Proxy training:** 4 epochs on a 20k-image subset, with the dataset kept on the GPU and
  augmented on-device. Tiny CNNs are otherwise bottlenecked by CPU data loading. Validation uses
  a 5k split held out from the training set; the test set is never used during search.
- **fp32 on the GTX 1060:** Pascal GPUs have very low fp16 throughput, so AMP is disabled for
  the NAS configs.

## Data
`scripts/download_data.py` defaults to the Hugging Face parquet mirror
(`uoft-cs/cifar10`, `uoft-cs/cifar100`) and converts it to the exact on-disk format torchvision
expects. The original cs.toronto.edu server measured ~4 KB/s from our network. `src/data.py`
therefore checks that the files exist instead of matching the original tarball md5s.

## Storage
Only weights-only `best.pt` files are committed; `last.pt`, which holds the full resume state,
is ignored. The teacher's `best.pt` is stored in fp16 (47.6 MB) to stay under GitHub's 100 MB
file limit. It is converted back to fp32 on load.
