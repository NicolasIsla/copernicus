import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import gc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import timm
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from einops import rearrange
from models.arq_sfanet_2_e5 import SFANet
import torch.backends.cudnn as cudnn
import psutil
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

nvmlInit()
_gpu_handle = nvmlDeviceGetHandleByIndex(0)  


def get_system_stats():
    """
    Devuelve estadísticas de CPU, RAM, GPU y disco en un dict.
    Pensado para ser llamado 1–2 veces por época (inicio/fin train).
    """
    cpu_percent = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    gpu_mem = nvmlDeviceGetMemoryInfo(_gpu_handle)
    disk_io = psutil.disk_io_counters()

    if not hasattr(get_system_stats, "last_disk_io"):
        get_system_stats.last_disk_io = disk_io
        get_system_stats.last_time = time.time()
        read_speed = write_speed = 0.0
    else:
        now = time.time()
        time_diff = max(now - get_system_stats.last_time, 1e-6)
        read_speed = (disk_io.read_bytes - get_system_stats.last_disk_io.read_bytes) / (1024 ** 2 * time_diff)
        write_speed = (disk_io.write_bytes - get_system_stats.last_disk_io.write_bytes) / (1024 ** 2 * time_diff)
        get_system_stats.last_disk_io = disk_io
        get_system_stats.last_time = now

    return {
        "cpu": cpu_percent,
        "ram_used": ram.used / (1024 ** 3),   # GB
        "ram_total": ram.total / (1024 ** 3),
        "gpu_used": gpu_mem.used / (1024 ** 2),   # MB
        "gpu_total": gpu_mem.total / (1024 ** 2),
        "disk_read": disk_io.read_bytes / (1024 ** 2),   # MB totales
        "disk_write": disk_io.write_bytes / (1024 ** 2),
        "disk_read_speed": read_speed,    # MB/s
        "disk_write_speed": write_speed,  # MB/s
    }


def plot_performance(history, system_stats, filename="performance_metrics.png"):
    """
    Grafica pérdidas train/val + uso RAM/GPU/CPU + velocidad de disco.
    history: dict del trainer
    system_stats: lista de dicts con estadísticas de hardware
    """
    if not system_stats:
        print("⚠️ No hay datos de estadísticas del sistema para graficar.")
        return

    fig, axs = plt.subplots(4, 2, figsize=(18, 20))

    # --- Pérdidas ---
    if "train_loss" in history and history["train_loss"]:
        axs[0, 0].plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        axs[0, 0].plot(history["val_loss"], label="Val Loss")
    axs[0, 0].set_title("Loss")
    axs[0, 0].legend()

    # --- IoU / F1 ---
    if "train_iou" in history and history["train_iou"]:
        axs[0, 1].plot(history["train_iou"], label="Train mIoU")
    if "val_iou" in history and history["val_iou"]:
        axs[0, 1].plot(history["val_iou"], label="Val mIoU")
    axs[0, 1].set_title("mIoU")
    axs[0, 1].legend()

    # --- RAM ---
    ram_used = [s.get("ram_used", 0) for s in system_stats]
    axs[1, 0].plot(ram_used, label="RAM Used (GB)")
    if "ram_total" in system_stats[0]:
        axs[1, 0].axhline(y=system_stats[0]["ram_total"], linestyle="--", label="RAM Total")
    axs[1, 0].set_title("Uso RAM")
    axs[1, 0].legend()

    # --- GPU ---
    gpu_used = [s.get("gpu_used", 0) for s in system_stats]
    axs[1, 1].plot(gpu_used, label="GPU Mem (MB)")
    if "gpu_total" in system_stats[0]:
        axs[1, 1].axhline(y=system_stats[0]["gpu_total"], linestyle="--", label="GPU Total")
    axs[1, 1].set_title("Uso GPU")
    axs[1, 1].legend()

    # --- CPU ---
    cpu_used = [s.get("cpu", 0) for s in system_stats]
    axs[2, 0].plot(cpu_used, label="CPU (%)")
    axs[2, 0].set_title("Uso CPU")
    axs[2, 0].legend()

    # --- Velocidad disco ---
    disk_read_speed = [s.get("disk_read_speed", 0) for s in system_stats]
    disk_write_speed = [s.get("disk_write_speed", 0) for s in system_stats]
    axs[3, 0].plot(disk_read_speed, label="Read MB/s")
    axs[3, 0].set_title("Disk Read Speed")
    axs[3, 0].legend()
    axs[3, 1].plot(disk_write_speed, label="Write MB/s")
    axs[3, 1].set_title("Disk Write Speed")
    axs[3, 1].legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"📈 Gráfico de rendimiento guardado en {filename}")


# ============================================================
# 1) Modelo SFANet con carga de pesos opcional
# ============================================================
class SFANetPretrained(SFANet):
    def __init__(
        self, 
        weights_path=None, 
        in_channels=11, 
        num_classes=12, 
        cat_num_categories=15, 
        cat_emb_dim=4, 
        device='cuda', 
        backbone_name='efficientnet_b5'
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            cat_num_categories=cat_num_categories,
            cat_emb_dim=cat_emb_dim,
            backbone_name=backbone_name,
            pretrained=False 
        )
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        if weights_path:
            self.load_pretrained_weights(weights_path)

    def load_pretrained_weights(self, path):
        try:
            pretrained_dict = torch.load(path, map_location=self.device)
            # Some checkpoints are saved as {'model': state_dict}
            if isinstance(pretrained_dict, dict) and 'model' in pretrained_dict:
                pretrained_dict = pretrained_dict['model']
            model_dict = self.state_dict()
            # 1. Keep only keys that exist and match shape
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            # 2. Update model
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"✅ Successfully loaded pretrained weights from {path}")
        except Exception as e:
            print(f"❌ Error loading pretrained weights: {e}")

    def forward(self, x_num, x_cat):
        return super().forward(x_num, x_cat)

# ============================================================
# 2) Dataset lazy (npz + label png)
# ============================================================
def load_npz_file_lazy(file_path):
    """Load only needed bands from NPZ."""
    try:
        with np.load(file_path) as data:
            img_10m = data["gsd_10"].astype(np.float32)
            img_20m = data["gsd_20"].astype(np.float32)
            return img_10m, img_20m
    except Exception as e:
        print(f"Error loading NPZ file {file_path}: {e}")
        return None, None

def load_label_lazy(file_path):
    """Load preprocessed label PNG (values 0..C)."""
    try:
        with Image.open(file_path) as img:
            arr = np.array(img, dtype=np.uint8)
            return arr
    except Exception as e:
        print(f"Error processing label {file_path}: {e}")
        return None

class LazySatelliteDataset(Dataset):
    def __init__(self, npz_paths, label_paths):
        self.npz_paths = npz_paths
        self.label_paths = label_paths
        self.num_classes = 24 

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx):
        with np.load(self.npz_paths[idx]) as data:
            patch = data['X'].astype(np.float32)  # (H, W, 12): 11 numeric + 1 categorical
        patch_numeric = patch[..., :-1]                      # (H, W, 11)
        patch_categorical = patch[..., -1].astype(np.int64)  # (H, W)

        patch_tensor = torch.from_numpy(patch_numeric).permute(2, 0, 1)  # (11, H, W)
        categorical_tensor = torch.from_numpy(patch_categorical)         # (H, W)

        label = np.array(Image.open(self.label_paths[idx]))
        label_tensor = torch.from_numpy(label).long()                    # (H, W)

        return patch_tensor, categorical_tensor, label_tensor

# ============================================================
# 3) BalancedTrainingManager SIN MONITOREO
# ============================================================
class BalancedTrainingManager:
    def __init__(
        self, 
        model, 
        device, 
        num_classes, 
        batch_metricas, 
        train_dataset=None, 
        alpha=0.5,
        label_smoothing=0.10, 
        focal_gamma=2.0, 
        use_confusion_loss=False, 
        confusion_loss_weight=0.1
    ):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.batch_metricas = batch_metricas
        self.alpha = alpha
        self.class_names = [str(i) for i in range(num_classes)]
        self.system_stats = []

        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.use_confusion_loss = use_confusion_loss
        self.confusion_loss_weight = confusion_loss_weight
        self.patience = 30
        self.best_confusion_matrix = None


        self._initialize_training_components(train_dataset)

    # ----------------------------
    # Pesos híbridos por frecuencia
    # ----------------------------
    def _initialize_training_components(self, train_dataset):
        self.hybrid_weights = self._calculate_hybrid_weights(train_dataset)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=4e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
        )
        self.scaler = torch.amp.GradScaler()
        self.best_metrics = {
            'val_iou': 0.0,
            'val_f1': 0.0,
            'val_loss': float('inf'),
            'val_ious': np.zeros(self.num_classes),
            'val_f1s': np.zeros(self.num_classes),
            'epoch': -1
        }

    def _calculate_hybrid_weights(self, dataset, epsilon=1e-6):
        if dataset is None:
            return torch.ones(self.num_classes).to(self.device)
            
        print("\nCalculando pesos híbridos para balance de clases...")
        class_counts = torch.zeros(self.num_classes)
        sample_size = min(2000, len(dataset))
        indices = torch.randperm(len(dataset))[:sample_size]
        
        # Count class frequencies (you can ignore specific classes if needed)
        for idx in indices:
            _, _, label = dataset[idx]
            unique, counts = torch.unique(label, return_counts=True)
            for u, c in zip(unique, counts):
                if u < self.num_classes:
                    class_counts[u] += c
        
        # Example: ignore a given class (7) in frequency computation
        valid_classes = [i for i in range(self.num_classes) if i != 7]
        valid_counts = class_counts[valid_classes]
        
        freq_weights = torch.ones(self.num_classes)
        freq_weights[valid_classes] = 1.0 / (valid_counts + epsilon)
        
        # Normalize valid classes
        sum_valid_weights = freq_weights[valid_classes].sum()
        freq_weights[valid_classes] = freq_weights[valid_classes] / sum_valid_weights * len(valid_classes)

        # Manually adjust a class if desired (example: class 7)
        avg_weight = freq_weights[valid_classes].mean()
        freq_weights[7] = avg_weight * 0 + 4  # tune if needed
        
        # Clip max
        freq_weights = torch.clamp(freq_weights, max=4.0)
        
        # Equal weights
        equal_weights = torch.ones(self.num_classes) * freq_weights.mean()
        equal_weights = equal_weights / equal_weights.sum() * self.num_classes

        # Hybrid combination
        hybrid_weights = self.alpha * freq_weights + (1 - self.alpha) * equal_weights
        hybrid_weights = hybrid_weights.to(self.device)
        
        self._print_weight_summary(class_counts, freq_weights, equal_weights, hybrid_weights)
        
        return hybrid_weights

    def _print_weight_summary(self, counts, freq_w, equal_w, hybrid_w):
        total = counts.sum()
        print(f"\n{'Clase':<10} {'Frecuencia':<12} {'Peso Frec.':<12} "
              f"{'Peso Igual':<12} {'Peso Final':<12}")
        for i in range(self.num_classes):
            freq = counts[i] / total if total > 0 else 0
            print(f"{i:<10} {freq:<12.4f} {freq_w[i]:<12.4f} "
                  f"{equal_w[i]:<12.4f} {hybrid_w[i]:<12.4f}")

    # ----------------------------
    # Loss híbrida (CE + focal + confusion-aware)
    # ----------------------------


    def _hybrid_loss(self, inputs, targets, confusion_matrix=None, ignore_index=None):
        # inputs: [B, C, H, W], targets: [B, H, W]

        # --- CrossEntropy base (con smoothing si corresponde) ---
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.hybrid_weights,
            reduction="none",
            label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0,
        )  # (B,H,W)

        # --- Focal opcional sobre CE ---
        if self.focal_gamma > 0:
            logpt = -ce_loss
            pt = torch.exp(logpt)
            focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
            base_loss = focal_loss
        else:
            base_loss = ce_loss   # (B,H,W)

        # --- Máscara de píxeles válidos ---
        if ignore_index is not None:
            valid_mask = (targets != ignore_index)
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        base_flat = base_loss[valid_mask]        # (N,)
        t_flat    = targets[valid_mask].view(-1) # (N,)
        preds     = torch.argmax(inputs, dim=1)  # (B,H,W)
        p_flat    = preds[valid_mask].view(-1)   # (N,)

        # Si no usamos penalización por confusión, es solo la loss media
        if not (self.use_confusion_loss and confusion_matrix is not None):
            return base_flat.mean()

        # --- Construir matriz de penalización [C,C] en Torch ---
        # confusion_matrix suele venir como numpy; lo pasamos a tensor
        if isinstance(confusion_matrix, np.ndarray):
            cm = torch.from_numpy(confusion_matrix).float().to(inputs.device)
        else:
            cm = confusion_matrix.float().to(inputs.device)

        # Normalizar por filas: prob de predecir p dado true t
        row_sums = cm.sum(dim=1, keepdim=True) + 1e-6
        cm_norm = cm / row_sums

        # Penalización por píxel: penalty[t,p]
        penalties = cm_norm[t_flat, p_flat]  # (N,)

        # No penalizamos aciertos (t == p)
        same = (t_flat == p_flat)
        penalties = penalties.clone()
        penalties[same] = 0.0

        # Loss final: base_loss * (1 + λ * penalty)
        total = base_flat * (1.0 + self.confusion_loss_weight * penalties)

        return total.mean()
    
    def _log_system_stats(self, phase):
        stats = get_system_stats()
        stats['phase'] = phase
        stats['timestamp'] = time.time()
        self.system_stats.append(stats)

    # ----------------------------
    # Métricas desde matriz de confusión
    # ----------------------------
    def _calculate_metrics_from_confusion(self, confusion):
        ious, f1_scores = [], []
        for cls in range(self.num_classes):
            tp = confusion[cls, cls]
            fp = confusion[:, cls].sum() - tp
            fn = confusion[cls, :].sum() - tp

            iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

            ious.append(iou)
            f1_scores.append(f1)

        class_support = confusion.sum(axis=1)
        if class_support.sum() > 0:
            weighted_f1 = np.average(f1_scores, weights=class_support)
        else:
            weighted_f1 = 0.0

        return {
            'mean_iou': np.mean(ious),
            'weighted_f1': weighted_f1,
            'ious': ious,
            'f1_scores': f1_scores,
            'confusion': confusion
        }

    def _print_class_report(self, metrics, phase):
        print(f"\n{'='*10} Métricas por Clase ({phase}) {'='*10}")
        print(f"{'Clase':<10} {'IOU':<10} {'F1':<10}")
        for i, (iou, f1) in enumerate(zip(metrics['ious'], metrics['f1_scores'])):
            print(f"{i:<10} {iou:.4f}    {f1:.4f}")
        print("="*40)



    def train_epoch(self, train_loader, epoch_idx):
        self.model.train()
        total_loss = 0.0
        # ahora como tensor en GPU
        confusion_torch = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=self.device
        )

        for batch_idx, (x, cat, y) in enumerate(train_loader):
            x = x.to(self.device, dtype=torch.float32, non_blocking=True)
            cat = cat.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                outputs = self.model(x, cat)
                confusion_matrix = self.best_confusion_matrix if self.use_confusion_loss else None
                loss = self._hybrid_loss(outputs[0], y, confusion_matrix=confusion_matrix)

                

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # Confusion matrix en Torch
            preds   = torch.argmax(outputs[0], dim=1)  # (B,H,W)
            targets = y

            mask = (targets >= 0) & (targets < self.num_classes) & \
                (preds   >= 0) & (preds   < self.num_classes)

            if mask.any():
                t = targets[mask].to(torch.int64)
                p = preds[mask].to(torch.int64)
                idx = t * self.num_classes + p
                cm_batch = torch.bincount(
                    idx,
                    minlength=self.num_classes * self.num_classes
                ).reshape(self.num_classes, self.num_classes)
                confusion_torch += cm_batch

            # opcional: liberar referencias grandes
            del x, cat, y, outputs, preds, targets

        # pasar a numpy solo 1 vez
        confusion = confusion_torch.cpu().numpy()
        self.best_confusion_matrix = confusion.copy()
        avg_loss = total_loss / len(train_loader)
        final_metrics = self._calculate_metrics_from_confusion(confusion)
        return avg_loss, final_metrics

    # ----------------------------
    # Validation
    # ----------------------------
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0

        # 👉 confusión como tensor en GPU
        confusion_torch = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=self.device
        )

        with torch.no_grad():
            for batch_idx, (x, cat, y) in enumerate(val_loader):
                x = x.to(self.device, dtype=torch.float32, non_blocking=True)
                cat = cat.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                    outputs = self.model(x, cat)
                    confusion_matrix = self.best_confusion_matrix if self.use_confusion_loss else None
                    loss = self._hybrid_loss(outputs[0], y, confusion_matrix=confusion_matrix)

                total_loss += loss.item()

                preds   = torch.argmax(outputs[0], dim=1)  # (B,H,W)
                targets = y

                mask = (targets >= 0) & (targets < self.num_classes) & \
                       (preds   >= 0) & (preds   < self.num_classes)

                if mask.any():
                    t = targets[mask].to(torch.int64)
                    p = preds[mask].to(torch.int64)
                    idx = t * self.num_classes + p
                    cm_batch = torch.bincount(
                        idx,
                        minlength=self.num_classes * self.num_classes
                    ).reshape(self.num_classes, self.num_classes)
                    confusion_torch += cm_batch

                del x, cat, y, outputs, preds, targets

        # 👉 bajar a numpy una sola vez
        confusion = confusion_torch.cpu().numpy()
        avg_loss = total_loss / len(val_loader)
        final_metrics = self._calculate_metrics_from_confusion(confusion)
        return avg_loss, final_metrics

    # ----------------------------
    # Training loop (sin monitoreo de hardware)
    # ----------------------------
    def train(self, train_loader, val_loader, epochs=50, patience=None):
        if patience is not None:
            self.patience = patience

        history = {
            "train_loss": [],
            "train_iou": [],
            "train_f1": [],
            "train_ious": [],
            "train_f1s": [],
            "val_loss": [],
            "val_iou": [],
            "val_f1": [],
            "val_ious": [],
            "val_f1s": [],
            "epoch_time": [],
        }

        for epoch in range(epochs):
            torch.cuda.empty_cache()
            gc.collect()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            self._log_system_stats('start_train')
            start_time.record()
            
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            self._log_system_stats('start_train')
            val_loss, val_metrics = self.validate(val_loader)
            
            end_time.record()
            torch.cuda.synchronize()
            epoch_time_ms = start_time.elapsed_time(end_time)
            epoch_time = epoch_time_ms / 1000.0  # seconds

            # # 👉 actualizar penalización de confusión a partir de la matriz de valid
            # if self.use_confusion_loss and "confusion" in val_metrics:
            #     cm = val_metrics["confusion"].astype(np.float32)  # (C,C)
            #     row_sums = cm.sum(axis=1, keepdims=True) + 1e-6
            #     cm_norm = cm / row_sums
            #     # no penalizar aciertos
            #     np.fill_diagonal(cm_norm, 0.0)
            #     # guardar como tensor en GPU para la próxima época
            #     self.confusion_penalty = torch.from_numpy(cm_norm).to(self.device)

            self.scheduler.step(val_loss)
            self._update_history(
                history, train_loss, train_metrics, val_loss, val_metrics, epoch_time
            )
            self._print_epoch_progress(
                epoch, epochs, epoch_time, train_loss, train_metrics, val_loss, val_metrics
            )
            if val_loss < self.best_metrics['val_loss']:
                self._update_best_metrics(val_loss, val_metrics, epoch, epoch_time)
                torch.save(self.model.state_dict(), 'best_model_test.pth')
                print(f"✅ Nuevo mejor modelo guardado "
                      f"(IoU: {val_metrics['mean_iou']:.4f}, F1: {val_metrics['weighted_f1']:.4f})")

            if (epoch - self.best_metrics['epoch']) >= self.patience:
                print(f"⏹ Detención temprana en época {epoch+1}")
                break

        self._finalize_training(history)
        return history

    def _update_history(self, history, train_loss, train_metrics, val_loss, val_metrics, epoch_time):
        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_metrics["mean_iou"])
        history["train_f1"].append(train_metrics["weighted_f1"])
        history["train_ious"].append(train_metrics["ious"])
        history["train_f1s"].append(train_metrics["f1_scores"])

        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_metrics["mean_iou"])
        history["val_f1"].append(val_metrics["weighted_f1"])
        history["val_ious"].append(val_metrics["ious"])
        history["val_f1s"].append(val_metrics["f1_scores"])

        history["epoch_time"].append(epoch_time)

    def _print_epoch_progress(self, epoch, epochs, epoch_time, train_loss, train_metrics, val_loss, val_metrics):
        mins, secs = divmod(epoch_time, 60)
        time_str = f"{int(mins):02d}:{int(secs):02d}"
        print(f"\nÉpoca {epoch+1}/{epochs} | Tiempo: {time_str}")
        print(f"Entrenamiento - Pérdida: {train_loss:.4f} | IoU: {train_metrics['mean_iou']:.4f} "
              f"| F1: {train_metrics['weighted_f1']:.4f}")
        print(f"Validación    - Pérdida: {val_loss:.4f} | IoU: {val_metrics['mean_iou']:.4f} "
              f"| F1: {val_metrics['weighted_f1']:.4f}")
        print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            self._print_class_report(val_metrics, 'Validación')

    def _update_best_metrics(self, val_loss, val_metrics, epoch, epoch_time):
        self.best_metrics = {
            'val_loss': val_loss,
            'val_iou': val_metrics['mean_iou'],
            'val_f1': val_metrics['weighted_f1'],
            'val_ious': val_metrics['ious'],
            'val_f1s': val_metrics['f1_scores'],
            'epoch': epoch,
            'epoch_time': epoch_time,
            'confusion': val_metrics['confusion']
        }

    def _finalize_training(self, history):
        plot_performance(history, self.system_stats)
        pd.DataFrame(self.system_stats).to_csv('system_stats.csv', index=False)

        self._plot_confusion_matrix(self.best_metrics['confusion'])
        self._save_final_report()

        try:
            with open("training_history.json", "w") as f:
                json.dump(history, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else x)
            print("💾 Historial de entrenamiento guardado en training_history.json")
        except Exception as e:
            print(f"⚠️ No se pudo guardar training_history.json: {e}")

        self.model.load_state_dict(torch.load('best_model.pth'))
        print(f"\n🏆 Mejor modelo - IoU: {self.best_metrics['val_iou']:.4f}, "
              f"F1: {self.best_metrics['val_f1']:.4f}, "
              f"Pérdida: {self.best_metrics['val_loss']:.4f}")

    def _plot_confusion_matrix(self, confusion, filename='confusion_matrix.png'):
        plt.figure(figsize=(12, 10))
        norm_confusion = confusion.astype('float') / (confusion.sum(axis=1)[:, np.newaxis] + 1e-6)
        plt.imshow(norm_confusion, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Matriz de Confusión Normalizada")
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)

        thresh = norm_confusion.max() / 2.
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(
                    j, i, f"{norm_confusion[i, j]:.2f}",
                    horizontalalignment="center",
                    color="white" if norm_confusion[i, j] > thresh else "black"
                )
        plt.tight_layout()
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        plt.savefig(filename)
        plt.close()

    def _save_final_report(self):
        final_metrics = pd.DataFrame({
            'Clase': self.class_names,
            'IOU': self.best_metrics['val_ious'],
            'F1': self.best_metrics['val_f1s'],
            'Peso': self.hybrid_weights.cpu().numpy()
        })
        final_metrics.to_csv('final_class_metrics.csv', index=False)
        confusion = self.best_metrics['confusion']
        pred_dist = confusion.sum(axis=0) / confusion.sum()
        true_dist = confusion.sum(axis=1) / confusion.sum()
        print("\n📊 Reporte Final:")
        print(final_metrics.to_string(index=False))
        print("\nDistribución de clases:")
        print(f"{'Clase':<10} {'Real':<12} {'Predicho':<12} {'Diferencia':<12}")
        for i in range(self.num_classes):
            diff = pred_dist[i] - true_dist[i]
            print(f"{i:<10} {true_dist[i]:<12.4f} {pred_dist[i]:<12.4f} {diff:<12.4f}")



if __name__ == "__main__":
    # ================== HIPERPARÁMETROS (según documento) ==================
    # DataLoader
    BATCH_SIZE        = 8      # 6 para EfficientNet-B5
    PREFETCH_FACTOR   = 4      # prefetch de 4 batches
    WORKERS_TRAIN     = 12
    WORKERS_VAL       = 4

    # Muestreo de patches (puedes poner None para usar todos)
    MAX_FILES         = 92000   # pon 92000 o None cuando quieras full dataset

    # Entrenamiento
    EPOCHS            = 80
    EARLY_STOP        = 20
    BATCH_METRICAS    = 4000   # tamaño lógico para imprimir métricas (interno del trainer)

    # Pérdida / balance de clases
    ALPHA                 = 0.4    # pesos híbridos (40% freq, 60% iguales)
    LABEL_SMOOTHING       = 0.20   # 20%
    FOCAL_GAMMA           = 2.0
    USE_CONFUSION_LOSS    = True
    CONFUSION_LOSS_WEIGHT = 0.1

    NUM_CLASSES = 12

    # ================== DEVICE ==================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")

    # ================== CARGAR CSV CON PATHS ==================
    # index_path = "out/patch_index.csv"   # cámbialo si lo tienes en otra ruta

    BASE_DIR   = "/data/nisla/copernicus2"
    index_path = os.path.join(BASE_DIR, "out", "patch_index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No se encontró {index_path}")

    df_temp = pd.read_csv(index_path)

    # print(f"Total rows en patch_index.csv: {len(df_temp)}")

    # 🔧 Normalizar rutas a absolutas (si son relativas)
    def make_abs(p):
        if os.path.isabs(p):
            return p
        return os.path.join(BASE_DIR, p)

    df_temp["npz"]   = df_temp["npz"].apply(make_abs)
    df_temp["label"] = df_temp["label"].apply(make_abs)

    print(f"Total rows en patch_index.csv: {len(df_temp)}")

    # Opcional: limitar número de patches
    if MAX_FILES is not None and MAX_FILES < len(df_temp):
        df_mapping = df_temp.sample(n=MAX_FILES, random_state=42).reset_index(drop=True)
        print(f"Usando una muestra de {len(df_mapping)} patches (MAX_FILES={MAX_FILES})")
    else:
        df_mapping = df_temp

    # ================== SPLIT TRAIN / VAL ==================
    train_df, val_df = train_test_split(
        df_mapping,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    print(f"Filas train: {len(train_df)}")
    print(f"Filas val  : {len(val_df)}")

    # ================== DATASETS LAZY ==================
    train_dataset = LazySatelliteDataset(
        train_df["npz"].tolist(),
        train_df["label"].tolist()
    )
    val_dataset = LazySatelliteDataset(
        val_df["npz"].tolist(),
        val_df["label"].tolist()
    )

    # (opcional) alinear num_classes del dataset con el modelo
    train_dataset.num_classes = NUM_CLASSES
    val_dataset.num_classes   = NUM_CLASSES

    # ================== DATALOADERS ==================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS_TRAIN,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS_VAL,
        pin_memory=True,
        persistent_workers=True,
    )

    # ================== MODELO SFANet ==================
    # model = SFANet(in_channels=11, num_classes=NUM_CLASSES)
    # Si quieres partir de un checkpoint:
    model = SFANetPretrained(
        in_channels=11,
        num_classes=NUM_CLASSES,
        cat_num_categories=15,
        cat_emb_dim=4,
        backbone_name="efficientnet_b5",
        weights_path="/home/nisla/copernicus/best_model.pth",
    )

    # ================== TRAINER CON PESOS HÍBRIDOS ==================
    trainer = BalancedTrainingManager(
        model=model,
        device=device,
        num_classes=NUM_CLASSES,
        batch_metricas=BATCH_METRICAS,
        train_dataset=train_dataset,
        alpha=ALPHA,
        label_smoothing=LABEL_SMOOTHING,
        focal_gamma=FOCAL_GAMMA,
        use_confusion_loss=USE_CONFUSION_LOSS,
        confusion_loss_weight=CONFUSION_LOSS_WEIGHT,
    )

    # ================== ENTRENAMIENTO ==================
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        patience=EARLY_STOP,
    )

    # Guardar modelo final
    torch.save(model.state_dict(), "fmodel_val.pth")
    print("✅ Modelo final guardado en fmodel_val.pth")

    # ================== REPORTE FINAL ==================
    print("\n📊 Reporte Final:")
    final_metrics = pd.DataFrame({
        'Clase': trainer.class_names,
        'IOU': trainer.best_metrics['val_ious'],
        'F1': trainer.best_metrics['val_f1s'],
        'Peso Final': trainer.hybrid_weights.cpu().numpy()
    })
    print(final_metrics.to_string(index=False))

    confusion = trainer.best_metrics['confusion']
    pred_dist = confusion.sum(axis=0) / confusion.sum()
    true_dist = confusion.sum(axis=1) / confusion.sum()

    print("\nDistribución de clases (val):")
    print(f"{'Clase':<10} {'Real':<12} {'Predicho':<12} {'Diferencia':<12}")
    for i in range(trainer.num_classes):
        diff = pred_dist[i] - true_dist[i]
        print(f"{i:<10} {true_dist[i]:<12.4f} {pred_dist[i]:<12.4f} {diff:<12.4f}")