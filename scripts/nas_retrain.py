"""Q2 step 7: train the best discovered architecture from scratch on full CIFAR-10.

reads results/nas/best_arch.json  ->  CKPT_DIR/nas_best/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import GPUTensorLoader, class_names, load_tensors  # noqa: E402
from src.engine import add_common_args, fit, get_device, load_config, set_seed  # noqa: E402
from src.nas.builder import build_model  # noqa: E402
from src.nas.space import describe  # noqa: E402
from src.paths import NAS_RUN, RESULTS_DIR  # noqa: E402

if __name__ == "__main__":
    args = add_common_args(argparse.ArgumentParser()).parse_args()
    cfg = load_config("nas", args.set)
    rc = cfg["retrain"]
    set_seed(cfg["seed"])
    device = get_device()

    with open(RESULTS_DIR / "nas" / "best_arch.json") as f:
        best = json.load(f)
    arch = best["arch"]
    print(f"Retraining trial {best['trial']} from scratch: {describe(arch)}")

    res = arch["resolution"]
    train_loader = GPUTensorLoader(*load_tensors("cifar10", True), "cifar10", rc["batch_size"], device, True, res)
    test_loader = GPUTensorLoader(*load_tensors("cifar10", False), "cifar10", 512, device, False, res)
    model = build_model(arch, 10).to(device)

    meta = {"run": NAS_RUN, "arch": "nas", "arch_config": arch, "dataset": "cifar10", "num_classes": 10,
            "resolution": res, "classes": class_names("cifar10"), "search": best, "config": rc}
    acc = fit(NAS_RUN, model, train_loader, test_loader, rc, device, meta)
    print(f"NAS model done. Best test top-1: {acc:.2f}%")
