#!/usr/bin/env python
import os
import csv
import json
import argparse
from typing import Optional, Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------- TU DATASET -------
from utils.dataset import SegPatchesDataset

# ------- TU LOADER DE MODELO -------
# Debes tener en tu proyecto:
# from model_loader import SFANet  (si lo usas directo)
# o bien:
from models.model_loader import load_model   # como tú mostraste

# ==========================
# Utilidades de métricas
# ==========================

@torch.no_grad()
def batch_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    pred, target: (B,H,W) int64
    Devuelve matriz (C,C) en el mismo device que pred/target.
    """
    valid = (target >= 0) & (target < num_classes)
    if not valid.any():
        return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=pred.device)
    t = target[valid].to(torch.int64)
    p = pred[valid].to(torch.int64)
    cm = torch.bincount(
        num_classes * t + p,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return cm

def _safe_div(n, d):
    d = np.maximum(d, 1)
    return n / d

def metrics_from_confusion(cm: np.ndarray, ignore_classes: Optional[list] = None) -> Dict[str, object]:
    """
    cm: (C,C) numpy int64
    ignore_classes: lista de ids a excluir de métricas (p.ej. [0] para ignorar background)
    Retorna:
      - pixel_accuracy
      - per_class_precision/recall/f1/iou/support
      - macro_f1, weighted_f1
      - mIoU, weighted_IoU
    """
    cm = cm.astype(np.int64, copy=False)
    C = cm.shape[0]
    all_ids = np.arange(C)

    if ignore_classes:
        keep = np.array([i for i in all_ids if i not in set(ignore_classes)])
        cm_eval = cm[np.ix_(keep, keep)]
        ids_eval = keep
    else:
        cm_eval = cm
        ids_eval = all_ids

    C_eff = cm_eval.shape[0]
    tp = np.diag(cm_eval)
    fp = cm_eval.sum(0) - tp
    fn = cm_eval.sum(1) - tp

    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)

    f1 = np.zeros(C_eff, dtype=np.float64)
    denom = (precision + recall)
    mask = denom > 0
    f1[mask] = 2 * precision[mask] * recall[mask] / denom[mask]

    support = cm_eval.sum(1)  # verdaderos por clase

    weighted_f1 = _safe_div((f1 * support).sum(), support.sum())

    iou = _safe_div(tp, tp + fp + fn)
    mIoU = iou.mean()
    weighted_IoU = _safe_div((iou * support).sum(), support.sum())

    pixel_acc = _safe_div(tp.sum(), cm_eval.sum())

    # mapear de vuelta a ids globales
    per_class_precision = np.zeros(C, dtype=np.float64)
    per_class_recall    = np.zeros(C, dtype=np.float64)
    per_class_f1        = np.zeros(C, dtype=np.float64)
    per_class_iou       = np.zeros(C, dtype=np.float64)
    per_class_support   = np.zeros(C, dtype=np.int64)

    for j, cid in enumerate(ids_eval):
        per_class_precision[cid] = precision[j]
        per_class_recall[cid]    = recall[j]
        per_class_f1[cid]        = f1[j]
        per_class_iou[cid]       = iou[j]
        per_class_support[cid]   = support[j]

    return {
        "pixel_accuracy": float(pixel_acc),
        "macro_f1": float(f1.mean()) if C_eff > 0 else 0.0,
        "weighted_f1": float(weighted_f1),
        "mIoU": float(mIoU),
        "weighted_IoU": float(weighted_IoU),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_iou": per_class_iou.tolist(),
        "support": per_class_support.tolist(),
    }

# ==========================
# Loop de entrenamiento / eval
# ==========================

def compute_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -1, class_weights: Optional[torch.Tensor] = None):
    """
    CrossEntropy 2D clásica.
    - logits: (B,C,H,W)
    - target: (B,H,W) con ints en [0..C-1]
    """
    return nn.functional.cross_entropy(logits, target, weight=class_weights, ignore_index=ignore_index)

def train_one_epoch(model, loader, optimizer, device, num_classes, amp=True, class_weights=None):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        numeric, cat, label, _ = batch
        numeric = numeric.to(device, non_blocking=True)
        cat     = cat.to(device, non_blocking=True)
        label   = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            logits, _ = model(numeric, cat)          # (B,C,H,W)
            loss = compute_loss(logits, label, ignore_index=-1, class_weights=class_weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * numeric.size(0)

        del numeric, cat, label, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return running_loss / max(1, len(loader.dataset))

@torch.no_grad()
def evaluate(model, loader, device, num_classes, amp=True) -> Tuple[float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    cm_total = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")

    for batch in tqdm(loader, desc="Eval", leave=False):
        numeric, cat, label, _ = batch
        numeric = numeric.to(device, non_blocking=True)
        cat     = cat.to(device, non_blocking=True)
        label   = label.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            logits, _ = model(numeric, cat)
            loss = compute_loss(logits, label, ignore_index=-1)

        pred = torch.argmax(logits, dim=1).to(torch.int64)
        cm_b = batch_confusion_matrix(pred, label, num_classes)
        cm_total += cm_b.detach().cpu()

        total_loss += loss.item() * numeric.size(0)

        del numeric, cat, label, logits, pred, cm_b, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, cm_total.numpy()

def save_metrics(outdir: str, cm: np.ndarray, metrics: Dict[str, object], class_names: Optional[list] = None):
    os.makedirs(outdir, exist_ok=True)

    # summary.csv
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["pixel_accuracy", metrics["pixel_accuracy"]])
        w.writerow(["macro_f1", metrics["macro_f1"]])
        w.writerow(["weighted_f1", metrics["weighted_f1"]])
        w.writerow(["mIoU", metrics["mIoU"]])
        w.writerow(["weighted_IoU", metrics["weighted_IoU"]])

    # per_class_metrics.csv
    with open(os.path.join(outdir, "per_class_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "precision", "recall", "f1", "iou", "support", "name"])
        C = cm.shape[0]
        for c in range(C):
            w.writerow([
                c,
                metrics["per_class_precision"][c],
                metrics["per_class_recall"][c],
                metrics["per_class_f1"][c],
                metrics["per_class_iou"][c],
                int(metrics["support"][c]),
                (class_names[c] if class_names and c < len(class_names) else "")
            ])

    # confusion_matrix.csv
    with open(os.path.join(outdir, "confusion_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["true\\pred"] + [f"{c}" for c in range(cm.shape[1])]
        w.writerow(header)
        for r in range(cm.shape[0]):
            w.writerow([f"{r}"] + [int(v) for v in cm[r, :]])

    # json
    with open(os.path.join(outdir, "final_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

# ==========================
# Main
# ==========================

# def main():
#     ap = argparse.ArgumentParser(description="Train + Val + Test for SFANet on SegPatchesDataset")
#     # datos
#     ap.add_argument("--train_csv", required=True, help="CSV train con columnas npz,label")
#     ap.add_argument("--val_csv",   required=True, help="CSV val con columnas npz,label")
#     ap.add_argument("--test_csv",  required=True, help="CSV test con columnas npz,label")
#     ap.add_argument("--root_npz",  default=None, help="Prefijo opcional si las rutas en CSV son relativas")
#     ap.add_argument("--root_labels", default=None, help="Prefijo opcional si las rutas en CSV son relativas")

#     # modelo
#     ap.add_argument("--weights", default=None, help="Ruta a pesos .pth para iniciar (opcional)")
#     ap.add_argument("--num_classes", type=int, default=12)
#     ap.add_argument("--cat_num_categories", type=int, default=15)
#     ap.add_argument("--cat_emb_dim", type=int, default=4)

#     # entrenamiento
#     ap.add_argument("--device", default="cuda:0")
#     ap.add_argument("--epochs", type=int, default=20)
#     ap.add_argument("--batch_size", type=int, default=4)
#     ap.add_argument("--num_workers", type=int, default=4)
#     ap.add_argument("--lr", type=float, default=1e-4)
#     ap.add_argument("--weight_decay", type=float, default=1e-4)
#     ap.add_argument("--amp", action="store_true", help="AMP en GPU")
#     ap.add_argument("--ignore_bg_in_metrics", action="store_true", help="Ignorar clase 0 en métricas")

#     # salida
#     ap.add_argument("--outdir", default="runs/exp1")
#     args = ap.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")

#     # ===== datasets & loaders =====
#     ds_train = SegPatchesDataset(
#         patch_index_csv=args.train_csv,
#         root_npz=args.root_npz, root_labels=args.root_labels,
#         verify_exists=True, augment=True
#     )
#     ds_val = SegPatchesDataset(
#         patch_index_csv=args.val_csv,
#         root_npz=args.root_npz, root_labels=args.root_labels,
#         verify_exists=True, augment=False
#     )
#     ds_test = SegPatchesDataset(
#         patch_index_csv=args.test_csv,
#         root_npz=args.root_npz, root_labels=args.root_labels,
#         verify_exists=True, augment=False
#     )

#     dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
#                           num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))
#     dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
#                           num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))
#     dl_test  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
#                           num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))

#     # ===== modelo =====
#     # usando tu load_model para crear la arquitectura; si no pasas pesos -> arranca aleatorio
#     model = load_model(
#         weights_path=args.weights,
#         device=device,
#         num_classes=args.num_classes,
#         in_channels=11,
#         cat_num_categories=args.cat_num_categories,
#         cat_emb_dim=args.cat_emb_dim
#     )
#     model.to(device)
#     model.train()  # aseguramos modo train

#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     best_val_miou = -1.0
#     best_ckpt = os.path.join(args.outdir, "best.pth")

#     # ===== loop =====
#     for epoch in range(1, args.epochs + 1):
#         print(f"\n=== Epoch {epoch}/{args.epochs} ===")
#         tr_loss = train_one_epoch(model, dl_train, optimizer, device, args.num_classes, amp=args.amp)
#         print(f"train_loss: {tr_loss:.6f}")

#         val_loss, cm_val = evaluate(model, dl_val, device, args.num_classes, amp=args.amp)
#         mets_val = metrics_from_confusion(cm_val, ignore_classes=([0] if args.ignore_bg_in_metrics else None))
#         print(f"val_loss: {val_loss:.6f} | pixel_acc={mets_val['pixel_accuracy']:.4f} | mIoU={mets_val['mIoU']:.4f} | macro_f1={mets_val['macro_f1']:.4f}")

#         # guardar mejor por mIoU
#         if mets_val["mIoU"] > best_val_miou:
#             best_val_miou = mets_val["mIoU"]
#             torch.save(model.state_dict(), best_ckpt)
#             print(f"✓ Nuevo mejor (mIoU={best_val_miou:.4f}). Guardado: {best_ckpt}")

#         # guardado por época (opcional)
#         torch.save(model.state_dict(), os.path.join(args.outdir, f"epoch_{epoch:03d}.pth"))

#     # ===== eval en TEST con el MEJOR checkpoint =====
#     if os.path.exists(best_ckpt):
#         model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
#         model.to(device)
#         model.eval()

#     test_loss, cm_test = evaluate(model, dl_test, device, args.num_classes, amp=args.amp)
#     mets_test = metrics_from_confusion(cm_test, ignore_classes=([0] if args.ignore_bg_in_metrics else None))

#     print("\n=== TEST (mejor checkpoint) ===")
#     print(f"test_loss: {test_loss:.6f}")
#     print(f"pixel_accuracy: {mets_test['pixel_accuracy']:.6f}")
#     print(f"macro_f1:       {mets_test['macro_f1']:.6f}")
#     print(f"weighted_f1:    {mets_test['weighted_f1']:.6f}")
#     print(f"mIoU:           {mets_test['mIoU']:.6f}")
#     print(f"weighted_IoU:   {mets_test['weighted_IoU']:.6f}")

#     # guardar métricas/CM de test
#     save_metrics(os.path.join(args.outdir, "test"), cm_test, mets_test, class_names=None)

# if __name__ == "__main__":
#     main()
import yaml

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Ruta al archivo YAML con parámetros")
    args = ap.parse_args()

    cfg = load_config(args.config)

    paths     = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    outdir    = cfg.get("outdir", "runs/exp1")
    ignore_bg = cfg.get("ignore_bg_in_metrics", False)

    os.makedirs(outdir, exist_ok=True)

    # ===== datasets & loaders =====
    ds_train = SegPatchesDataset(
        patch_index_csv=paths["train_csv"],
        root_npz=paths.get("root_npz"),
        root_labels=paths.get("root_labels"),
        verify_exists=True, augment=True
    )
    ds_val = SegPatchesDataset(
        patch_index_csv=paths["val_csv"],
        root_npz=paths.get("root_npz"),
        root_labels=paths.get("root_labels"),
        verify_exists=True, augment=False
    )
    ds_test = SegPatchesDataset(
        patch_index_csv=paths["test_csv"],
        root_npz=paths.get("root_npz"),
        root_labels=paths.get("root_labels"),
        verify_exists=True, augment=False
    )

    print(f"[DATA] train={len(ds_train)} | val={len(ds_val)} | test={len(ds_test)}")

    dl_train = DataLoader(
        ds_train, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], pin_memory=True,
        persistent_workers=(train_cfg["num_workers"] > 0)
    )
    dl_val = DataLoader(
        ds_val, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"], pin_memory=True,
        persistent_workers=(train_cfg["num_workers"] > 0)
    )
    dl_test = DataLoader(
        ds_test, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"], pin_memory=True,
        persistent_workers=(train_cfg["num_workers"] > 0)
    )

    # ===== modelo =====
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = load_model(
        weights_path=model_cfg.get("weights"),  # puede ser None
        device=device,
        num_classes=model_cfg["num_classes"],
        in_channels=11,
        cat_num_categories=model_cfg["cat_num_categories"],
        cat_emb_dim=model_cfg["cat_emb_dim"]
    )
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(),
                            lr=train_cfg["lr"],
                            weight_decay=train_cfg["weight_decay"])

    best_val_miou = -1.0
    best_ckpt = os.path.join(outdir, "best.pth")

    # ===== loop de entrenamiento =====
    for epoch in range(1, train_cfg["epochs"] + 1):
        print(f"\n=== Epoch {epoch}/{train_cfg['epochs']} ===")
        tr_loss = train_one_epoch(
            model, dl_train, optimizer, device,
            num_classes=model_cfg["num_classes"], amp=train_cfg.get("amp", False)
        )
        print(f"train_loss: {tr_loss:.6f}")

        val_loss, cm_val = evaluate(
            model, dl_val, device,
            num_classes=model_cfg["num_classes"], amp=train_cfg.get("amp", False)
        )
        mets_val = metrics_from_confusion(
            cm_val, ignore_classes=([0] if ignore_bg else None)
        )
        print(f"val_loss: {val_loss:.6f} | pixel_acc={mets_val['pixel_accuracy']:.4f} "
              f"| mIoU={mets_val['mIoU']:.4f} | macro_f1={mets_val['macro_f1']:.4f}")

        # guarda mejor por mIoU
        if mets_val["mIoU"] > best_val_miou:
            best_val_miou = mets_val["mIoU"]
            torch.save(model.state_dict(), best_ckpt)
            print(f"✓ Nuevo mejor (mIoU={best_val_miou:.4f}). Guardado: {best_ckpt}")

        # opcional: checkpoint por época
        torch.save(model.state_dict(), os.path.join(outdir, f"epoch_{epoch:03d}.pth"))

    # ===== eval en TEST con el mejor checkpoint =====
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
        model.to(device)
        model.eval()

    test_loss, cm_test = evaluate(
        model, dl_test, device,
        num_classes=model_cfg["num_classes"], amp=train_cfg.get("amp", False)
    )
    mets_test = metrics_from_confusion(
        cm_test, ignore_classes=([0] if ignore_bg else None)
    )

    print("\n=== TEST (mejor checkpoint) ===")
    print(f"test_loss:      {test_loss:.6f}")
    print(f"pixel_accuracy: {mets_test['pixel_accuracy']:.6f}")
    print(f"macro_f1:       {mets_test['macro_f1']:.6f}")
    print(f"weighted_f1:    {mets_test['weighted_f1']:.6f}")
    print(f"mIoU:           {mets_test['mIoU']:.6f}")
    print(f"weighted_IoU:   {mets_test['weighted_IoU']:.6f}")

    save_metrics(os.path.join(outdir, "test"), cm_test, mets_test, class_names=None)

if __name__ == "__main__":
    main()