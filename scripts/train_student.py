"""Q1 steps 4-5: train the three students, either as baselines or with knowledge distillation.

  python scripts/train_student.py --mode baseline     -> CKPT_DIR/baseline_<student>/best.pt
  python scripts/train_student.py --mode kd           -> CKPT_DIR/kd_<student>/best.pt

All selected students are trained in lock-step on the same batches; in KD mode the frozen
teacher runs ONE forward pass per batch that is shared by every student.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import NUM_CLASSES, get_loaders  # noqa: E402
from src.engine import (RunTracker, add_common_args, evaluate, get_device, load_config,  # noqa: E402
                        make_optimizer, make_scaler, make_scheduler, set_seed)
from src.kd.distill import FeatureAdapter, StudentState, train_epoch_multi  # noqa: E402
from src.models import get_model, load_checkpoint  # noqa: E402
from src.paths import best_ckpt  # noqa: E402

if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--mode", choices=["baseline", "kd"], required=True)
    ap.add_argument("--students", nargs="+", default=None, help="subset of: mbv2 squeezenet efflite0")
    args = ap.parse_args()
    cfg = load_config("student", args.set)
    set_seed(cfg["seed"])
    device = get_device()
    nc = NUM_CLASSES[cfg["dataset"]]
    kd_cfg, lb = cfg["kd"], cfg.get("limit_batches")
    train_loader, test_loader = get_loaders(cfg["dataset"], cfg["batch_size"], cfg["num_workers"])

    teacher = None
    if args.mode == "kd":
        teacher, tmeta = load_checkpoint(best_ckpt(kd_cfg["teacher_run"]), device)
        teacher.requires_grad_(False)
        print(f"Teacher {kd_cfg['teacher_run']} loaded (test top-1 {tmeta['acc']:.2f}%)")

    states, objects, starts = [], {}, []
    for name in args.students or cfg["students"]:
        model = get_model(name, nc).to(device)
        adapter = None
        if teacher is not None and kd_cfg["use_feat"]:
            adapter = FeatureAdapter(model.feat_channels, teacher.feat_channels).to(device)
        params = list(model.parameters()) + (list(adapter.parameters()) if adapter else [])
        opt = make_optimizer(params, cfg)
        sched = make_scheduler(opt, cfg, len(train_loader))
        scaler = make_scaler(device, cfg["amp"])
        run = f"{args.mode}_{name}"
        meta = {"run": run, "arch": name, "mode": args.mode, "dataset": cfg["dataset"], "num_classes": nc,
                "resolution": 32, "classes": train_loader.dataset.classes, "config": cfg}
        tracker = RunTracker(run, meta)
        objs = {"model": model, "optimizer": opt, "scheduler": sched, "scaler": scaler}
        if adapter is not None:
            objs["adapter"] = adapter
        starts.append(tracker.resume(objs))
        objects[name] = objs
        states.append(StudentState(name, model, adapter, opt, sched, scaler, tracker))

    for epoch in range(min(starts), cfg["epochs"]):
        active = [s for s, st in zip(states, starts) if st <= epoch]
        train_stats = train_epoch_multi(active, train_loader, device, kd_cfg, cfg["amp"], teacher, lb)
        for s in active:
            te_loss, te_acc, te_top5 = evaluate(s.model, test_loader, device, cfg["amp"], lb)
            s.tracker.end_epoch(epoch, objects[s.name], s.model,
                                {**train_stats[s.name], "test_loss": te_loss, "test_acc": te_acc,
                                 "test_top5": te_top5})

    for s in states:
        print(f"{s.tracker.name}: best test top-1 {s.tracker.best_acc:.2f}%")
