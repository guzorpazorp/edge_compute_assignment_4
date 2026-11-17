"""Hardware-aware NAS with Optuna (TPE / Bayesian optimisation).

Per trial:
  1. sample an architecture from the search space (space.sample_arch)
  2. build it (builder.build_model)
  3. profile it on the CPU: latency, storage size, peak memory  (hw_profile)
  4. REJECT it if it violates a deployment constraint -> recorded with accuracy 0 and the
     reason, without spending any GPU time (TPE learns to avoid that region)
  5. otherwise proxy-train for a few epochs on a CIFAR-10 subset and return validation
     accuracy; a MedianPruner stops clearly bad candidates early.

The study is stored in SQLite (results/nas/study.db) so an interrupted search resumes.
Outputs: results/nas/history.csv, results/nas/history.json, results/nas/best_arch.json
"""
import json

import optuna
import torch

from src.data import GPUTensorLoader, load_tensors, split_indices
from src.engine import evaluate, make_optimizer, make_scaler, make_scheduler, set_seed, train_epoch
from src.nas.builder import build_model
from src.nas.hw_profile import check_constraints, machine_info, profile_model
from src.nas.space import describe, sample_arch
from src.paths import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "nas"


def make_data(cfg, device):
    x, y = load_tensors("cifar10", train=True)
    tr_idx, va_idx = split_indices(len(y), cfg["val_size"], cfg["seed"])
    tr_idx = tr_idx[:cfg["proxy_train_size"]]
    return (x[tr_idx], y[tr_idx]), (x[va_idx], y[va_idx])


def make_objective(cfg, device, data):
    (xtr, ytr), (xva, yva) = data
    tcfg = cfg["proxy_train"]

    def objective(trial):
        arch = sample_arch(trial, cfg["space"])
        desc = describe(arch)
        # TPE can re-propose an already evaluated architecture: reuse that result instead of
        # re-profiling (noisy) and re-training (expensive).
        for prev in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if prev.user_attrs.get("desc") == desc:
                for k, v in prev.user_attrs.items():
                    trial.set_user_attr(k, v)
                trial.set_user_attr("status", "duplicate")
                trial.set_user_attr("reason", f"same architecture as trial {prev.number}")
                print(f"[trial {trial.number}] duplicate of trial {prev.number}")
                return prev.value
        model = build_model(arch, 10)
        hw = profile_model(model, arch["resolution"], cfg["latency_threads"], cfg["latency_runs"])
        violations = check_constraints(hw, cfg["constraints"])
        trial.set_user_attr("arch", arch)
        trial.set_user_attr("desc", desc)
        for k, v in hw.items():
            trial.set_user_attr(k, v)
        if violations:
            trial.set_user_attr("status", "rejected")
            trial.set_user_attr("reason", "; ".join(violations))
            print(f"[trial {trial.number}] REJECTED {desc} :: {'; '.join(violations)}")
            return 0.0

        set_seed(cfg["seed"] + trial.number)
        model = model.to(device)
        train_loader = GPUTensorLoader(xtr, ytr, "cifar10", tcfg["batch_size"], device, True, arch["resolution"])
        val_loader = GPUTensorLoader(xva, yva, "cifar10", 512, device, False, arch["resolution"])
        opt = make_optimizer(model.parameters(), tcfg)
        sched = make_scheduler(opt, tcfg, len(train_loader))
        scaler = make_scaler(device, tcfg["amp"])
        acc = 0.0
        for epoch in range(tcfg["epochs"]):
            train_epoch(model, train_loader, opt, sched, scaler, device, tcfg["amp"],
                        cfg.get("limit_batches"))
            _, acc, _ = evaluate(model, val_loader, device, tcfg["amp"], cfg.get("limit_batches"))
            trial.report(acc, epoch)
            if trial.should_prune():
                trial.set_user_attr("status", "pruned")
                trial.set_user_attr("reason", f"median-pruned at epoch {epoch + 1} (val_acc={acc:.2f})")
                raise optuna.TrialPruned()
        trial.set_user_attr("status", "accepted")
        trial.set_user_attr("reason", "")
        print(f"[trial {trial.number}] val_acc={acc:.2f}  {desc}  "
              f"lat={hw['latency_ms']:.2f}ms size={hw['size_mb']:.3f}MB mem={hw['peak_mem_mb']}MB")
        return acc

    return objective


def export_results(study, cfg, verbose=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in study.trials:
        ua = t.user_attrs
        rows.append({
            "trial": t.number, "state": t.state.name, "status": ua.get("status", t.state.name.lower()),
            "val_acc": t.value if t.value is not None else (max(t.intermediate_values.values())
                                                             if t.intermediate_values else None),
            **{k: ua.get(k) for k in ("params", "mflops", "size_mb", "latency_ms", "peak_mem_mb")},
            "reason": ua.get("reason", ""), "desc": ua.get("desc", ""),
            "arch": json.dumps(ua.get("arch")),
            "duration_s": t.duration.total_seconds() if t.duration else None,
        })
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT_DIR / "history.csv", index=False)
    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(rows, f, indent=1)

    accepted = [t for t in study.trials if t.user_attrs.get("status") == "accepted"]
    if not accepted:
        if verbose:
            print("No architecture satisfied the constraints.")
        return None
    best = max(accepted, key=lambda t: t.value)
    result = {
        "trial": best.number, "val_acc_proxy": best.value, "arch": best.user_attrs["arch"],
        "desc": best.user_attrs["desc"],
        "hw": {k: best.user_attrs[k] for k in ("params", "mflops", "size_mb", "latency_ms", "peak_mem_mb")},
        "constraints": cfg["constraints"], "latency_threads": cfg["latency_threads"],
        "machine": machine_info(),
        "n_trials": len(study.trials),
        "n_rejected": sum(t.user_attrs.get("status") == "rejected" for t in study.trials),
        "n_pruned": sum(t.user_attrs.get("status") == "pruned" for t in study.trials),
        "n_duplicate": sum(t.user_attrs.get("status") == "duplicate" for t in study.trials),
    }
    with open(OUT_DIR / "best_arch.json", "w") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"Best: trial {best.number} val_acc={best.value:.2f}  {result['desc']}")
    return result


def run_search(cfg, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = make_data(cfg, device)
    study = optuna.create_study(
        study_name=cfg["study_name"], direction="maximize",
        storage=f"sqlite:///{OUT_DIR / 'study.db'}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"], n_startup_trials=cfg["n_startup_trials"]),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=cfg["n_startup_trials"], n_warmup_steps=1),
    )
    remaining = max(0, cfg["n_trials"] - len(study.trials))
    print(f"Running {remaining} trials (already done: {len(study.trials)}) on {device}; "
          f"CPU profiling on: {machine_info()['cpu']}")
    study.optimize(make_objective(cfg, device, data), n_trials=remaining,
                   callbacks=[lambda s, t: export_results(s, cfg, verbose=False)])
    return export_results(study, cfg)
