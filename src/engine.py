"""Shared training utilities: config loading, seeding, optimisers, schedules,
train/eval loops and checkpoint/resume/logging (RunTracker)."""
import csv
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.paths import CONFIG_DIR, RESULTS_DIR, run_dir


# ----------------------------------------------------------------------------- config
def load_config(name, overrides=None):
    """Load configs/<name>.yaml and apply `key=value` overrides (values parsed as YAML)."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, val = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(val)
    return cfg


def add_common_args(parser):
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                        help="override config values, e.g. --set epochs=1 limit_batches=5")
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- optim
def make_optimizer(param_groups, cfg):
    return torch.optim.SGD(param_groups, lr=cfg["lr"], momentum=0.9, nesterov=True,
                           weight_decay=cfg["weight_decay"])


def make_scheduler(optimizer, cfg, steps_per_epoch):
    """Linear warm-up followed by cosine decay, stepped every iteration."""
    total = cfg["epochs"] * steps_per_epoch
    warm = cfg.get("warmup_epochs", 1) * steps_per_epoch

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def make_scaler(device, enabled):
    return torch.amp.GradScaler(device.type, enabled=enabled and device.type == "cuda")


def autocast(device, enabled):
    return torch.autocast(device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


# ----------------------------------------------------------------------------- loops
def train_epoch(model, loader, optimizer, scheduler, scaler, device, amp=True, limit_batches=None):
    """Plain cross-entropy epoch (teacher, NAS). Returns (loss, acc%)."""
    model.train()
    tot_loss = correct = seen = 0
    for k, (x, y) in enumerate(loader):
        if limit_batches and k >= limit_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, amp):
            out = model(x)
            loss = F.cross_entropy(out, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()
        tot_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
    return tot_loss / seen, 100.0 * correct / seen


@torch.no_grad()
def evaluate(model, loader, device, amp=True, limit_batches=None):
    """Returns (loss, top1%, top5%)."""
    model.eval()
    tot_loss = c1 = c5 = seen = 0
    for k, (x, y) in enumerate(loader):
        if limit_batches and k >= limit_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, amp):
            out = model(x)
        out = out.float()
        tot_loss += F.cross_entropy(out, y, reduction="sum").item()
        top5 = out.topk(min(5, out.size(1)), 1).indices
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y[:, None]).any(1).sum().item()
        seen += y.size(0)
    return tot_loss / seen, 100.0 * c1 / seen, 100.0 * c5 / seen


# ----------------------------------------------------------------------------- checkpoints
def save_best(path, model, meta, half=False):
    """Weights-only checkpoint + metadata consumed by inference / export / benchmark."""
    sd = {k: (v.half() if half and v.is_floating_point() else v).cpu()
          for k, v in model.state_dict().items()}
    torch.save({**meta, "state_dict": sd}, path)


class RunTracker:
    """Per-run resume (last.pt), best-model saving (best.pt) and CSV logging."""

    def __init__(self, run_name, meta, half_best=False):
        self.name, self.meta, self.half_best = run_name, meta, half_best
        self.dir = run_dir(run_name)
        self.log_path = RESULTS_DIR / "logs" / f"{run_name}.csv"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_acc = -1.0
        self.t0 = time.time()

    def resume(self, objects):
        """objects: dict name -> object with state_dict/load_state_dict. Returns start epoch."""
        last = self.dir / "last.pt"
        if not last.exists():
            if self.log_path.exists():
                self.log_path.unlink()
            return 0
        state = torch.load(last, map_location="cpu", weights_only=False)
        for k, obj in objects.items():
            obj.load_state_dict(state[k])
        self.best_acc = state["best_acc"]
        print(f"[{self.name}] resumed from epoch {state['epoch'] + 1}, best acc {self.best_acc:.2f}")
        return state["epoch"] + 1

    def end_epoch(self, epoch, objects, model, metrics):
        metrics = {"epoch": epoch + 1, **{k: round(v, 4) for k, v in metrics.items()},
                   "time_s": round(time.time() - self.t0, 1)}
        new_file = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics))
            if new_file:
                w.writeheader()
            w.writerow(metrics)

        acc = metrics["test_acc"]
        if acc > self.best_acc:
            self.best_acc = acc
            save_best(self.dir / "best.pt", model, {**self.meta, "acc": acc, "epoch": epoch + 1},
                      half=self.half_best)
        torch.save({**{k: o.state_dict() for k, o in objects.items()},
                    "epoch": epoch, "best_acc": self.best_acc}, self.dir / "last.pt")
        print(f"[{self.name}] " + "  ".join(f"{k}={v}" for k, v in metrics.items())
              + f"  best={self.best_acc:.2f}")


def fit(run_name, model, train_loader, test_loader, cfg, device, meta, param_groups=None, half_best=False):
    """Standard supervised training run with resume, logging and best-checkpoint saving."""
    optimizer = make_optimizer(param_groups or model.parameters(), cfg)
    scheduler = make_scheduler(optimizer, cfg, len(train_loader))
    scaler = make_scaler(device, cfg["amp"])
    tracker = RunTracker(run_name, meta, half_best=half_best)
    objects = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "scaler": scaler}
    lb = cfg.get("limit_batches")
    for epoch in range(tracker.resume(objects), cfg["epochs"]):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg["amp"], lb)
        te_loss, te_acc, te_top5 = evaluate(model, test_loader, device, cfg["amp"], lb)
        tracker.end_epoch(epoch, objects, model, {"train_loss": tr_loss, "train_acc": tr_acc,
                                                  "test_loss": te_loss, "test_acc": te_acc, "test_top5": te_top5})
    return tracker.best_acc


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters())
