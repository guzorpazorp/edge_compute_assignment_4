"""CIFAR-10 / CIFAR-100 data pipelines.

* get_loaders()     : standard torchvision DataLoaders (Q1 teacher/students, final NAS retrain)
* GPUTensorLoader   : whole dataset kept on the GPU with on-device augmentation.
                      Used for the NAS proxy training where tiny CNNs would otherwise be
                      bottlenecked by CPU-side data loading.
"""
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.paths import DATA_DIR

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def _files_present(self):
    # Accept both the original tarball contents and the HF-converted files written by
    # scripts/download_data.py (same layout/format, different md5).
    root = Path(self.root) / self.base_folder
    return all((root / f).exists() for f, _ in self.train_list + self.test_list)


def _load_meta_no_md5(self):
    with open(Path(self.root) / self.base_folder / self.meta["filename"], "rb") as f:
        self.classes = pickle.load(f, encoding="latin1")[self.meta["key"]]
    self.class_to_idx = {c: i for i, c in enumerate(self.classes)}


class CIFAR10(datasets.CIFAR10):
    _check_integrity = _files_present
    _load_meta = _load_meta_no_md5


class CIFAR100(datasets.CIFAR100):
    _check_integrity = _files_present
    _load_meta = _load_meta_no_md5


_DATASETS = {"cifar10": CIFAR10, "cifar100": CIFAR100}


def get_dataset(name, train, transform=None, download=True):
    return _DATASETS[name](DATA_DIR, train=train, transform=transform, download=download)


def class_names(name):
    return get_dataset(name, train=False).classes


def build_transforms(name, train, resolution=32):
    mean, std = STATS[name]
    tf = []
    if train:
        tf += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if resolution != 32:
        tf.append(transforms.Resize(resolution, antialias=True))
    tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(tf)


def split_indices(n, val_size, seed=0):
    perm = np.random.RandomState(seed).permutation(n)
    return perm[val_size:].tolist(), perm[:val_size].tolist()


def get_loaders(name, batch_size=128, num_workers=2, resolution=32, val_size=0, seed=0):
    """Returns (train_loader, eval_loader).

    val_size == 0 -> eval on the official test split (Q1 convention).
    val_size  > 0 -> hold out `val_size` training images as a validation split.
    """
    train_set = get_dataset(name, True, build_transforms(name, True, resolution))
    if val_size > 0:
        eval_base = get_dataset(name, True, build_transforms(name, False, resolution))
        tr_idx, va_idx = split_indices(len(train_set), val_size, seed)
        train_set, eval_set = Subset(train_set, tr_idx), Subset(eval_base, va_idx)
    else:
        eval_set = get_dataset(name, False, build_transforms(name, False, resolution))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin, drop_last=True, persistent_workers=num_workers > 0)
    eval_loader = DataLoader(eval_set, batch_size * 2, shuffle=False, num_workers=num_workers,
                             pin_memory=pin)
    return train_loader, eval_loader


class GPUTensorLoader:
    """Iterates over (x, y) batches of a dataset held entirely on `device`.

    Augmentation (random crop with 4px reflect padding + horizontal flip) and resizing to
    the candidate's input resolution are done on-device, per sample.
    """

    def __init__(self, images_u8, labels, dataset, batch_size, device, train, resolution=32):
        self.x = images_u8.to(device)  # N,3,32,32 uint8
        self.y = labels.to(device)
        mean, std = STATS[dataset]
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.bs, self.train, self.res, self.device = batch_size, train, resolution, device

    def __len__(self):
        n = len(self.y)
        return n // self.bs if self.train else (n + self.bs - 1) // self.bs

    def _augment(self, x):
        b = x.shape[0]
        x = F.pad(x, (4, 4, 4, 4), mode="reflect")
        i = torch.randint(0, 9, (b,), device=x.device)
        j = torch.randint(0, 9, (b,), device=x.device)
        ar = torch.arange(32, device=x.device)
        rows = (i[:, None] + ar)[:, None, :, None]
        cols = (j[:, None] + ar)[:, None, None, :]
        x = x[torch.arange(b, device=x.device)[:, None, None, None],
              torch.arange(3, device=x.device)[None, :, None, None], rows, cols]
        flip = torch.rand(b, device=x.device) < 0.5
        return torch.where(flip[:, None, None, None], x.flip(3), x)

    def __iter__(self):
        n = len(self.y)
        order = torch.randperm(n, device=self.device) if self.train else torch.arange(n, device=self.device)
        for k in range(len(self)):
            idx = order[k * self.bs:(k + 1) * self.bs]
            x = self.x[idx].float().div_(255)
            if self.train:
                x = self._augment(x)
            if self.res != 32:
                x = F.interpolate(x, size=self.res, mode="bilinear", align_corners=False, antialias=True)
            yield (x - self.mean) / self.std, self.y[idx]


def load_tensors(name, train):
    ds = get_dataset(name, train)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # N,3,32,32 uint8
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x, y
