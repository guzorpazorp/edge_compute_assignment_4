"""Q1 step 1: train the ResNet-50 teacher on CIFAR-100.  -> CKPT_DIR/teacher_resnet50/best.pt"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import NUM_CLASSES, get_loaders  # noqa: E402
from src.engine import add_common_args, fit, get_device, load_config, set_seed  # noqa: E402
from src.models.resnet import ResNet50, load_imagenet_weights  # noqa: E402
from src.paths import TEACHER_RUN  # noqa: E402

if __name__ == "__main__":
    args = add_common_args(argparse.ArgumentParser()).parse_args()
    cfg = load_config("teacher", args.set)
    set_seed(cfg["seed"])
    device = get_device()

    train_loader, test_loader = get_loaders(cfg["dataset"], cfg["batch_size"], cfg["num_workers"])
    model = ResNet50(NUM_CLASSES[cfg["dataset"]])
    if cfg["pretrained"]:
        load_imagenet_weights(model)
    model.to(device)

    head = list(model.fc.parameters())
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    mult = cfg["backbone_lr_mult"] if cfg["pretrained"] else 1.0
    groups = [{"params": backbone, "lr": cfg["lr"] * mult}, {"params": head, "lr": cfg["lr"]}]

    meta = {"run": TEACHER_RUN, "arch": "resnet50", "dataset": cfg["dataset"],
            "num_classes": NUM_CLASSES[cfg["dataset"]], "resolution": 32,
            "classes": train_loader.dataset.classes, "config": cfg}
    best = fit(TEACHER_RUN, model, train_loader, test_loader, cfg, device, meta, groups, cfg["save_half"])
    print(f"Teacher done. Best test top-1: {best:.2f}%")
