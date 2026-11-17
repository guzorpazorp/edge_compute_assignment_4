"""Download CIFAR datasets into DATA_DIR (default: ./data).

  --source hf       (default) Hugging Face parquet mirror, converted to the exact on-disk
                    layout torchvision expects (cifar-100-python/, cifar-10-batches-py/).
                    Use this when the original Toronto server is slow.
  --source toronto  torchvision's built-in download from cs.toronto.edu.
"""
import argparse
import io
import json
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import get_dataset  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402

HF = {  # dataset -> (HF repo, parquet config, label column, torchvision folder)
    "cifar10": ("uoft-cs/cifar10", "plain_text", "label", "cifar-10-batches-py"),
    "cifar100": ("uoft-cs/cifar100", "cifar100", "fine_label", "cifar-100-python"),
}


def fetch_parquet(repo, config, split, dest):
    url = f"https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}/0.parquet"
    if not dest.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return pq.read_table(dest)


def to_numpy(table, label_col):
    imgs = [np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB"), dtype=np.uint8)
            for d in table.column("img" if "img" in table.column_names else "image").to_pylist()]
    data = np.stack(imgs).transpose(0, 3, 1, 2).reshape(len(imgs), -1)  # N x 3072, CHW row-major
    return data, table.column(label_col).to_pylist()


def download_hf(name):
    repo, config, label_col, folder = HF[name]
    out = DATA_DIR / folder
    if (out / ("test" if name == "cifar100" else "test_batch")).exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    tmp = DATA_DIR / "_hf"
    tmp.mkdir(parents=True, exist_ok=True)
    splits = {s: fetch_parquet(repo, config, s, tmp / f"{name}_{s}.parquet") for s in ("train", "test")}
    names = json.loads(splits["train"].schema.metadata[b"huggingface"])["info"]["features"][label_col]["names"]

    def dump(obj, fname):
        with open(out / fname, "wb") as f:
            pickle.dump(obj, f)

    if name == "cifar100":
        for s in ("train", "test"):
            data, labels = to_numpy(splits[s], label_col)
            dump({"data": data, "fine_labels": labels,
                  "coarse_labels": splits[s].column("coarse_label").to_pylist()}, s)
        dump({"fine_label_names": names}, "meta")
    else:
        data, labels = to_numpy(splits["train"], label_col)
        for i, chunk in enumerate(np.array_split(np.arange(len(labels)), 5), 1):
            dump({"data": data[chunk], "labels": [labels[j] for j in chunk]}, f"data_batch_{i}")
        data, labels = to_numpy(splits["test"], label_col)
        dump({"data": data, "labels": labels}, "test_batch")
        dump({"label_names": names}, "batches.meta")
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["cifar100"], choices=["cifar10", "cifar100"])
    ap.add_argument("--source", choices=["hf", "toronto"], default="hf")
    args = ap.parse_args()
    for name in args.dataset:
        if args.source == "hf":
            download_hf(name)
        tr, te = get_dataset(name, True), get_dataset(name, False)
        print(f"{name}: {len(tr)} train / {len(te)} test images, {len(tr.classes)} classes in {DATA_DIR}")
