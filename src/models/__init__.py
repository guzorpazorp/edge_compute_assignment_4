"""Model registry + checkpoint loading used by every downstream step."""
import torch

from src.models.efficientnet_lite import EfficientNetLite0
from src.models.mobilenetv2 import MobileNetV2
from src.models.resnet import ResNet50
from src.models.squeezenet import SqueezeNet

REGISTRY = {
    "resnet50": ResNet50,
    "mbv2": MobileNetV2,
    "squeezenet": SqueezeNet,
    "efflite0": EfficientNetLite0,
}


def get_model(name, num_classes=100):
    return REGISTRY[name](num_classes=num_classes)


def build_from_meta(meta):
    """Rebuild an (untrained) model from the metadata stored in a best.pt checkpoint."""
    if meta["arch"] == "nas":
        from src.nas.builder import build_model
        return build_model(meta["arch_config"], meta["num_classes"])
    return get_model(meta["arch"], meta["num_classes"])


def load_checkpoint(path, device="cpu"):
    """Returns (model in eval mode, fp32, on device; metadata dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_from_meta(ckpt)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v
                           for k, v in ckpt["state_dict"].items()})
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model.to(device).eval(), meta
