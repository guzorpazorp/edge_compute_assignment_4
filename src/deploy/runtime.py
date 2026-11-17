"""Unified inference pipeline.

    p = Predictor("kd_mbv2", backend="onnx")    # or backend="torch"; any run name works
    p.predict_image("cat.png")                   # -> [(class, prob), ...]
    p.evaluate_testset()                         # -> {"top1": .., "top5": .., ...}

Both backends share the same preprocessing and return logits as numpy arrays, so the
teacher, all six student checkpoints and the NAS model are used through one interface.
"""
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data import STATS, build_transforms, get_dataset
from src.models import load_checkpoint
from src.paths import ONNX_DIR, best_ckpt


class Predictor:
    def __init__(self, run, backend="torch", threads=None):
        self.run, self.backend = run, backend
        if backend == "torch":
            if threads:
                torch.set_num_threads(threads)
            self.model, self.meta = load_checkpoint(best_ckpt(run))
        elif backend == "onnx":
            import onnxruntime as ort
            with open(ONNX_DIR / f"{run}.json") as f:
                self.meta = json.load(f)
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            self.session = ort.InferenceSession(str(ONNX_DIR / f"{run}.onnx"), so,
                                                providers=["CPUExecutionProvider"])
        else:
            raise ValueError(f"unknown backend {backend}")
        self.dataset, self.res = self.meta["dataset"], self.meta["resolution"]
        self.classes = self.meta["classes"]

    def __call__(self, x):
        """x: float32 array/tensor (N,3,H,W), already normalised. Returns logits (N,C) numpy."""
        if self.backend == "torch":
            with torch.inference_mode():
                return self.model(torch.as_tensor(x)).numpy()
        return self.session.run(None, {"input": np.asarray(x, dtype=np.float32)})[0]

    def preprocess(self, image: Image.Image):
        image = image.convert("RGB").resize((self.res, self.res), Image.BILINEAR)
        mean, std = STATS[self.dataset]
        x = (np.asarray(image, dtype=np.float32) / 255.0 - mean) / std
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def predict_image(self, path, topk=5):
        logits = self(self.preprocess(Image.open(path)))[0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        idx = probs.argsort()[::-1][:topk]
        return [(self.classes[i], float(probs[i])) for i in idx]

    def evaluate_testset(self, batch_size=256, limit=None):
        ds = get_dataset(self.dataset, train=False, transform=build_transforms(self.dataset, False, self.res))
        c1 = c5 = n = 0
        for x, y in DataLoader(ds, batch_size, num_workers=2):
            if limit:
                x, y = x[:limit - n], y[:limit - n]
            logits = self(x.numpy())
            top5 = np.argsort(-logits, 1)[:, :5]
            y = y.numpy()
            c1 += (top5[:, 0] == y).sum()
            c5 += (top5 == y[:, None]).any(1).sum()
            n += len(y)
            if limit and n >= limit:
                break
        return {"top1": float(100.0 * c1 / n), "top5": float(100.0 * c5 / n), "n": int(n)}
