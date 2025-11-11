#!/usr/bin/env python
import os
import csv
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


def _paired_lists_from_dirs(root_npz: str, root_labels: str) -> List[Tuple[str, str]]:
    npz_names = {os.path.splitext(f)[0]: os.path.join(root_npz, f)
                 for f in os.listdir(root_npz) if f.endswith(".npz")}
    pairs = []
    for fname in os.listdir(root_labels):
        if not fname.endswith(".png"):
            continue
        stem = os.path.splitext(fname)[0]
        lbl_path = os.path.join(root_labels, fname)
        npz_path = npz_names.get(stem)
        if npz_path and os.path.exists(npz_path):
            pairs.append((npz_path, lbl_path))
    pairs.sort()
    return pairs


class SegPatchesDataset(Dataset):
    """
    Dataset para parches .npz (X: (H,W,12) con 11 num + 1 cat) y label .png (H,W).
    Puede leer desde un CSV (npz,label) o parear por nombre desde carpetas.

    Args:
      patch_index_csv: CSV con columnas npz,label (rutas absolutas o relativas).
      root_npz, root_labels: si no hay CSV, pareamos por nombre en estas carpetas.
      limit: limita la cantidad de muestras (debug).
      verify_exists: valida existencia de archivos.
      augment: activa flips aleatorios horizontales/verticales.
    """
    def __init__(
        self,
        patch_index_csv: Optional[str] = None,
        root_npz: Optional[str] = None,
        root_labels: Optional[str] = None,
        limit: Optional[int] = None,
        verify_exists: bool = True,
        augment: bool = False,
    ):
        assert (patch_index_csv is not None) or (root_npz and root_labels), \
            "Debes pasar patch_index_csv o root_npz y root_labels"

        self.samples: List[Tuple[str, str]] = []
        if patch_index_csv:
            with open(patch_index_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    npz_path = r["npz"].strip()
                    lbl_path = r["label"].strip()
                    if not os.path.isabs(npz_path):
                        if root_npz: npz_path = os.path.join(root_npz, npz_path)
                        else: npz_path = os.path.abspath(npz_path)
                    if not os.path.isabs(lbl_path):
                        if root_labels: lbl_path = os.path.join(root_labels, lbl_path)
                        else: lbl_path = os.path.abspath(lbl_path)
                    self.samples.append((npz_path, lbl_path))
        else:
            self.samples = _paired_lists_from_dirs(root_npz, root_labels)

        if limit is not None:
            self.samples = self.samples[:limit]

        if verify_exists:
            for npz_path, lbl_path in self.samples:
                if not os.path.exists(npz_path):
                    raise FileNotFoundError(f"No existe npz: {npz_path}")
                if not os.path.exists(lbl_path):
                    raise FileNotFoundError(f"No existe label: {lbl_path}")

        self.augment = augment

    def __len__(self): return len(self.samples)

    def _maybe_augment(self, numeric, cat, label):
        # numeric: (11,H,W) float32; cat/label: (H,W) int64
        if random.random() < 0.5:
            numeric = torch.flip(numeric, dims=[2])  # horiz
            cat = torch.flip(cat, dims=[1])
            label = torch.flip(label, dims=[1])
        if random.random() < 0.5:
            numeric = torch.flip(numeric, dims=[1])  # vert
            cat = torch.flip(cat, dims=[0])
            label = torch.flip(label, dims=[0])
        return numeric, cat, label

    def __getitem__(self, idx: int):
        npz_path, lbl_path = self.samples[idx]
        arr = np.load(npz_path)
        X = arr["X"]  # (H,W,12)

        numeric = X[..., :11].transpose(2, 0, 1).astype(np.float32)  # (11,H,W)
        cat = X[..., 11].astype(np.int64)                             # (H,W)
        label = np.array(Image.open(lbl_path), dtype=np.int64)        # (H,W)

        numeric = torch.from_numpy(numeric)
        cat = torch.from_numpy(cat)
        label = torch.from_numpy(label)

        if self.augment:
            numeric, cat, label = self._maybe_augment(numeric, cat, label)

        return numeric, cat, label, npz_path