#!/usr/bin/env python
import os, csv, json, argparse
from typing import Optional, Dict, Tuple
import numpy as np
from PIL import Image
from contextlib import nullcontext

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.dataset import SegPatchesDataset
from models.model_loader import load_model



def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return device, local_rank

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

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
def evaluate(model, loader, device, num_classes, amp=True, use_ddp=False) -> Tuple[float, np.ndarray]:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_items = torch.tensor(0.0, device=device)

    cm_total = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    autocast_ctx = torch.amp.autocast("cuda") if (amp and device.type == "cuda") else nullcontext()

    for batch in tqdm(loader, desc="Eval", leave=False):
        numeric, cat, label, _ = batch
        numeric = numeric.to(device, non_blocking=True)
        cat     = cat.to(device, non_blocking=True)
        label   = label.to(device, non_blocking=True)

        with autocast_ctx:
            logits, _ = model(numeric, cat)
            loss = compute_loss(logits, label, ignore_index=-1)

        pred = torch.argmax(logits, dim=1).to(torch.int64)
        cm_b = batch_confusion_matrix(pred, label, num_classes).to(device)

        cm_total += cm_b
        total_loss += loss.detach() * numeric.size(0)
        total_items += numeric.size(0)

        del numeric, cat, label, logits, pred, cm_b, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # all-reduce entre procesos
    if use_ddp and dist.is_initialized():
        dist.all_reduce(cm_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_items, op=dist.ReduceOp.SUM)

    avg_loss = (total_loss / torch.clamp(total_items, min=1)).item()
    return avg_loss, cm_total.detach().cpu().numpy()

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




def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Ruta al archivo YAML con parámetros")
    args = ap.parse_args()

    cfg = load_config(args.config)



    # Secciones esperadas
    paths     = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    outdir    = cfg.get("outdir", "runs/exp1")
    ignore_bg = cfg.get("ignore_bg_in_metrics", False)

    # Casteo robusto de tipos numéricos (evita errores cuando vienen como string)
    lr           = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    batch_size   = int(train_cfg["batch_size"])
    num_workers  = int(train_cfg["num_workers"])
    epochs       = int(train_cfg["epochs"])
    use_amp      = bool(train_cfg.get("amp", False))

    use_ddp = bool(train_cfg.get("ddp", False))
    if use_ddp:
        device, local_rank = setup_ddp()
    else:
        device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")

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
    print(f"[CFG] lr={lr} | weight_decay={weight_decay} | batch_size={batch_size} | num_workers={num_workers} | amp={use_amp}")

    if use_ddp:
        train_sampler = DistributedSampler(ds_train, shuffle=True)
        val_sampler   = DistributedSampler(ds_val, shuffle=False)
        test_sampler  = DistributedSampler(ds_test, shuffle=False)
    else:
        train_sampler = val_sampler = test_sampler = None

    dl_train = DataLoader(ds_train, batch_size=train_cfg["batch_size"],
                        shuffle=(train_sampler is None),
                        sampler=train_sampler,
                        num_workers=train_cfg["num_workers"],
                        pin_memory=bool(train_cfg.get("pin_memory", True)),
                        persistent_workers=bool(train_cfg.get("persistent_workers", False)))

    dl_val = DataLoader(ds_val, batch_size=train_cfg["batch_size"],
                        shuffle=False, sampler=val_sampler,
                        num_workers=train_cfg["num_workers"],
                        pin_memory=bool(train_cfg.get("pin_memory", True)),
                        persistent_workers=bool(train_cfg.get("persistent_workers", False)))

    dl_test = DataLoader(ds_test, batch_size=train_cfg["batch_size"],
                        shuffle=False, sampler=test_sampler,
                        num_workers=train_cfg["num_workers"],
                        pin_memory=bool(train_cfg.get("pin_memory", True)),
                        persistent_workers=bool(train_cfg.get("persistent_workers", False)))

    # ===== modelo =====
    # device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = load_model(
        weights_path=model_cfg.get("weights"),  # puede ser None
        device=device,
        num_classes=int(model_cfg["num_classes"]),
        in_channels=11,
        cat_num_categories=int(model_cfg["cat_num_categories"]),
        cat_emb_dim=int(model_cfg["cat_emb_dim"])
    )

    model.to(device)

    # BatchNorm con batch pequeño: usa SyncBN o congélalo
    if train_cfg.get("sync_bn", False):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False
        )
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_miou = -1.0
    best_ckpt = os.path.join(outdir, "best.pth")

    # ===== loop de entrenamiento =====
    for epoch in range(1, epochs + 1):
        if use_ddp:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if isinstance(val_sampler, DistributedSampler):
                val_sampler.set_epoch(epoch)
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        tr_loss = train_one_epoch(
            model, dl_train, optimizer, device,
            num_classes=int(model_cfg["num_classes"]), amp=use_amp
        )
        if is_main_process():
            print(f"train_loss: {tr_loss:.6f}")

        val_loss, cm_val = evaluate(model, dl_val, device, num_classes=int(model_cfg["num_classes"]),
                            amp=use_amp, use_ddp=use_ddp)
        mets_val = metrics_from_confusion(
            cm_val, ignore_classes=([0] if ignore_bg else None)
        )
        if is_main_process():
            print(f"val_loss: {val_loss:.6f} | pixel_acc={mets_val['pixel_accuracy']:.4f} "
                f"| mIoU={mets_val['mIoU']:.4f} | macro_f1={mets_val['macro_f1']:.4f}")

        # guarda mejor por mIoU
        if is_main_process() and mets_val["mIoU"] > best_val_miou:
            best_val_miou = mets_val["mIoU"]
            sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(sd, best_ckpt)
            print(f"✓ Nuevo mejor (mIoU={best_val_miou:.4f}). Guardado: {best_ckpt}")

        # opcional: checkpoint por época
        if is_main_process() and epoch % 5 == 0:
            sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(sd, os.path.join(outdir, f"epoch_{epoch:03d}.pth"))

    # ===== eval en TEST con el mejor checkpoint =====
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location="cpu")
        (model.module if hasattr(model, "module") else model).load_state_dict(state)
        model.to(device)
        model.eval()

    test_loss, cm_test = evaluate(model, dl_test, device, num_classes=int(model_cfg["num_classes"]),
                              amp=use_amp, use_ddp=use_ddp)
    mets_test = metrics_from_confusion(
        cm_test, ignore_classes=([0] if ignore_bg else None)
    )
    if is_main_process():
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
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()