"""Knowledge-distillation framework.

Loss (each term can be toggled in configs/student.yaml):
    L = alpha * T^2 * KL( softmax(z_s/T) || softmax(z_t/T) )      soft targets, temperature T
      + (1 - alpha) * CE(z_s, y)                                   hard labels
      + beta * || n(adapter(f_s)) - n(f_t) ||^2                   feature distillation (FitNets), n = L2-normalise

The adapter is a 1x1 conv + BN mapping the student's 8x8 feature map to the teacher's
layer3 channel count (1024); adaptive pooling aligns spatial size if they ever differ.

`train_epoch_multi` trains several students in lock-step on the same batches. With a teacher
it performs ONE teacher forward pass per batch shared by all students (KD mode); with
teacher=None it is plain cross-entropy training (baseline mode).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engine import autocast


class FeatureAdapter(nn.Module):
    def __init__(self, s_ch, t_ch):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(s_ch, t_ch, 1, bias=False), nn.BatchNorm2d(t_ch))

    def forward(self, f_s, size):
        f = self.proj(f_s)
        return f if f.shape[-2:] == size else F.adaptive_avg_pool2d(f, size)


def soft_target_loss(s_logits, t_logits, T):
    """Hinton KD term. Scaled by T^2 so gradients keep the same magnitude as the CE term."""
    return F.kl_div(F.log_softmax(s_logits / T, 1), F.softmax(t_logits / T, 1),
                    reduction="batchmean") * T * T


def feature_loss(f_s, f_t):
    """FitNets-style hint loss on L2-normalised feature maps.

    Each sample's (C,H,W) map is flattened and normalised to unit length before the squared
    distance, so the term lies in [0, 4] regardless of the teacher's activation scale.
    """
    f_s, f_t = F.normalize(f_s.flatten(1), dim=1), F.normalize(f_t.flatten(1), dim=1)
    return (f_s - f_t).pow(2).sum(1).mean()


def kd_loss(s_logits, y, kd_cfg, t_logits=None, s_feat=None, t_feat=None, adapter=None):
    """Returns (total loss, dict of individual terms)."""
    hard = F.cross_entropy(s_logits, y)
    parts = {"hard": hard}
    if t_logits is None:                              # baseline: hard labels only
        return hard, parts

    use_soft, use_hard = kd_cfg["use_soft"], kd_cfg["use_hard"]
    alpha = kd_cfg["alpha"] if (use_soft and use_hard) else float(use_soft)
    total = (1 - alpha) * hard if use_hard else 0.0
    if use_soft:
        parts["soft"] = soft_target_loss(s_logits, t_logits, kd_cfg["temperature"])
        total = total + alpha * parts["soft"]
    if kd_cfg["use_feat"]:
        parts["feat"] = feature_loss(adapter(s_feat, t_feat.shape[-2:]), t_feat)
        total = total + kd_cfg["beta"] * parts["feat"]
    return total, parts


@dataclass
class StudentState:
    name: str
    model: nn.Module
    adapter: nn.Module | None
    optimizer: torch.optim.Optimizer
    scheduler: object
    scaler: torch.amp.GradScaler
    tracker: object


def train_epoch_multi(students, loader, device, kd_cfg, amp=True, teacher=None, limit_batches=None):
    """One epoch for every StudentState in `students`. Returns {name: {loss, acc, ...}}."""
    for s in students:
        s.model.train()
        if s.adapter is not None:
            s.adapter.train()
    stats = {s.name: {"train_loss": 0.0, "correct": 0} for s in students}
    seen = 0
    use_feat = teacher is not None and kd_cfg["use_feat"]

    for k, (x, y) in enumerate(loader):
        if limit_batches and k >= limit_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        t_logits = t_feat = None
        if teacher is not None:
            with torch.no_grad(), autocast(device, amp):
                t_logits, t_feat = teacher(x, return_feats=True)
            t_logits, t_feat = t_logits.float(), t_feat.float()

        for s in students:
            with autocast(device, amp):
                s_logits, s_feat = s.model(x, return_feats=True)
            # losses in fp32 for numerical stability of the softmax/KL terms
            loss, _ = kd_loss(s_logits.float(), y, kd_cfg, t_logits,
                              s_feat.float() if use_feat else None, t_feat, s.adapter)
            s.optimizer.zero_grad(set_to_none=True)
            s.scaler.scale(loss).backward()
            s.scaler.step(s.optimizer)
            s.scaler.update()
            s.scheduler.step()
            st = stats[s.name]
            st["train_loss"] += loss.item() * y.size(0)
            st["correct"] += (s_logits.argmax(1) == y).sum().item()
        seen += y.size(0)

    return {n: {"train_loss": st["train_loss"] / seen, "train_acc": 100.0 * st["correct"] / seen}
            for n, st in stats.items()}
