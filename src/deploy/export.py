"""PyTorch -> ONNX export with a numerical parity check against ONNX Runtime.

For every run, writes onnx/<run>.onnx plus onnx/<run>.json (metadata needed for inference
without PyTorch: dataset, input resolution, normalisation stats, class names).
"""
import json
import warnings

import numpy as np
import onnxruntime as ort
import torch

from src.data import STATS
from src.models import load_checkpoint
from src.paths import ONNX_DIR, best_ckpt


def export_run(run, opset=17):
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    model, meta = load_checkpoint(best_ckpt(run))
    res = meta["resolution"]
    dummy = torch.randn(1, 3, res, res)
    out_path = ONNX_DIR / f"{run}.onnx"

    with warnings.catch_warnings():  # tracer warns about the (unused) return_feats branch
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (dummy,), str(out_path), input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=opset, dynamo=False)

    # parity check on a random batch
    x = torch.randn(8, 3, res, res)
    with torch.inference_mode():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())

    sidecar = {
        "run": run, "arch": meta["arch"], "arch_config": meta.get("arch_config"),
        "dataset": meta["dataset"], "num_classes": meta["num_classes"], "resolution": res,
        "mean": STATS[meta["dataset"]][0], "std": STATS[meta["dataset"]][1],
        "classes": meta["classes"], "torch_acc": meta.get("acc"), "parity_max_abs_diff": max_diff,
    }
    with open(ONNX_DIR / f"{run}.json", "w") as f:
        json.dump(sidecar, f, indent=1)
    status = "OK" if max_diff < 1e-3 else "MISMATCH"
    print(f"[{run}] -> {out_path.name}  ({out_path.stat().st_size / 2**20:.2f} MB)  "
          f"parity max|diff|={max_diff:.2e} {status}")
    return out_path, max_diff
