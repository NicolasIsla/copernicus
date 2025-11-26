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
import time
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import timm
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from einops import rearrange
from models.arq_sfanet_2_e5 import SFANet
import torch.backends.cudnn as cudnn

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
            pretrained=False # importante, no cargar imagenet!
        )
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        if weights_path:
            self.load_pretrained_weights(weights_path)

    def load_pretrained_weights(self, path):
        try:
            pretrained_dict = torch.load(path, map_location=self.device)
            # Si el modelo fue guardado como {'model': state_dict}
            if isinstance(pretrained_dict, dict) and 'model' in pretrained_dict:
                pretrained_dict = pretrained_dict['model']
            model_dict = self.state_dict()
            # 1. Filtrar pesos que existen en el modelo actual y que coincidan en forma
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                               if k in model_dict and v.shape == model_dict[k].shape}
            # 2. Sobreescribir parámetros del modelo
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"✅ Successfully loaded pretrained weights from {path}")
        except Exception as e:
            print(f"❌ Error loading pretrained weights: {e}")

    def forward(self, x_num, x_cat):
        return super().forward(x_num, x_cat)

# ==================== MONITOREO DE HARDWARE ====================
# Inicialización NVML para monitoreo GPU
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # Asume GPU 0

def get_system_stats():
    """Obtiene estadísticas del sistema"""
    cpu_percent = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    gpu_mem = nvmlDeviceGetMemoryInfo(handle)
    disk_io = psutil.disk_io_counters()
    
    # Calcular diferencia desde última medición
    if not hasattr(get_system_stats, 'last_disk_io'):
        get_system_stats.last_disk_io = disk_io
        get_system_stats.last_time = time.time()
        read_speed = write_speed = 0.0
    else:
        time_diff = time.time() - get_system_stats.last_time
        read_speed = (disk_io.read_bytes - get_system_stats.last_disk_io.read_bytes) / (1024**2 * time_diff)
        write_speed = (disk_io.write_bytes - get_system_stats.last_disk_io.write_bytes) / (1024**2 * time_diff)
        get_system_stats.last_disk_io = disk_io
        get_system_stats.last_time = time.time()    
    return {
        'cpu': cpu_percent,
        'ram_used': ram.used / (1024**3),  # GB
        'ram_total': ram.total / (1024**3),
        'gpu_used': gpu_mem.used / (1024**2),  # MB
        'gpu_total': gpu_mem.total / (1024**2),
        'disk_read': disk_io.read_bytes / (1024**2),  # MB totales leídos
        'disk_write': disk_io.write_bytes / (1024**2),  # MB totales escritos
        'disk_read_speed': read_speed,  # MB/s
        'disk_write_speed': write_speed  # MB/s
    }

def plot_performance(history, system_stats):
    """Visualiza las métricas de rendimiento con manejo de casos vacíos"""
    if not system_stats:  # Si system_stats está vacío
        print("⚠️ No hay datos de estadísticas del sistema para graficar")
        return
    
    fig, axs = plt.subplots(4, 2, figsize=(18, 20))
    
    # Métricas de entrenamiento
    if 'train_loss' in history:
        axs[0,0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axs[0,0].plot(history['val_loss'], label='Val Loss')
    axs[0,0].legend()
    
    # Uso de RAM
    if 'ram_used' in system_stats[0]:
        ram_used = [s.get('ram_used', 0) for s in system_stats]
        axs[1,0].plot(ram_used, label='RAM Used (GB)')
        
        if 'ram_total' in system_stats[0]:
            axs[1,0].axhline(y=system_stats[0]['ram_total'], color='r', linestyle='--', label='Total RAM')
        axs[1,0].legend()
    
    # Uso de GPU
    if 'gpu_used' in system_stats[0]:
        gpu_used = [s.get('gpu_used', 0) for s in system_stats]
        axs[1,1].plot(gpu_used, label='GPU Memory Used (MB)')
        
        if 'gpu_total' in system_stats[0]:
            axs[1,1].axhline(y=system_stats[0]['gpu_total'], color='r', linestyle='--', label='Total GPU Memory')
        axs[1,1].legend()
    
    # Uso de CPU
    if 'cpu' in system_stats[0]:
        cpu_used = [s.get('cpu', 0) for s in system_stats]
        axs[2,0].plot(cpu_used, label='CPU Usage (%)')
        axs[2,0].legend()
    
    # Velocidad de disco
    if system_stats and 'disk_read_speed' in system_stats[0]:
        disk_read_speed = [s.get('disk_read_speed', 0) for s in system_stats]
        disk_write_speed = [s.get('disk_write_speed', 0) for s in system_stats]
        
        axs[3,0].plot(disk_read_speed, label='Read Speed (MB/s)')
        axs[3,0].set_title('Disk Read Speed')
        axs[3,0].legend()
        
        axs[3,1].plot(disk_write_speed, label='Write Speed (MB/s)')
        axs[3,1].set_title('Disk Write Speed')
        axs[3,1].legend()
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png')
    plt.close()
    
class TimedDataLoader:
    """DataLoader con temporización para medir transferencias"""
    def __init__(self, loader):
        self.loader = loader
        self.batch_times = []
        self.data_transfer_times = []
        
    def __iter__(self):
        for batch in self.loader:
            start_time = time.time()
            yield batch
            self.batch_times.append(time.time() - start_time)
            
    def log_transfer(self, device, start_time):
        self.data_transfer_times.append(time.time() - start_time)

# ==================== FUNCIONES DE CARGA ====================
def load_npz_file_lazy(file_path):
    """Carga solo las bandas necesarias del archivo NPZ."""
    try:
        with np.load(file_path) as data:
            # Cargamos directamente como float16 para ahorrar memoria
            img_10m = data["gsd_10"].astype(np.float16)
            img_20m = data["gsd_20"].astype(np.float16)
            return img_10m, img_20m
    except Exception as e:
        print(f"Error loading NPZ file {file_path}: {e}")
        return None, None

def load_label_lazy(file_path):
    """Carga etiquetas PNG ya preprocesadas (valores de 0 a 12)"""
    try:
        with Image.open(file_path) as img:
            arr = np.array(img, dtype=np.uint8)
            return arr
    except Exception as e:
        print(f"Error processing label {file_path}: {e}")
        return None

# ==================== DATASET CON ETIQUETAS SPARSE ====================
class LazySatelliteDataset(Dataset):
    def __init__(self, npz_paths, label_paths):
        self.npz_paths = npz_paths
        self.label_paths = label_paths
        self.num_classes = 24

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx):
        with np.load(self.npz_paths[idx]) as data:
            patch = data['X'].astype(np.float32)  # (192,192, 12) suponiendo 11 bandas + 1 categórica
        patch_numeric = patch[..., :-1]                     # (H, W, 11) - bandas numéricas
        patch_categorical = patch[..., -1].astype(np.int64) # (H, W) - banda categórica como sparse
        patch_tensor = torch.from_numpy(patch_numeric).permute(2, 0, 1)  # (11, H, W)
        categorical_tensor = torch.from_numpy(patch_categorical)         # (H, W)
        label = np.array(Image.open(self.label_paths[idx]))
        label_tensor = torch.from_numpy(label).long()                    # (H, W)
        return patch_tensor, categorical_tensor, label_tensor
        
# ==================== MANEJADOR DE ENTRENAMIENTO OPTIMIZADO ====================
class BalancedTrainingManager:
    def __init__(
        self, model, device, num_classes, batch_metricas, train_dataset=None, alpha=0.5,
        label_smoothing=0.10, focal_gamma=2.0, use_confusion_loss=False, confusion_loss_weight=0.1
    ):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.batch_metricas = batch_metricas
        self.system_stats = []
        self.batch_times = {'train': [], 'val': []}
        self.transfer_times = {'train': [], 'val': []}
        self.disk_speeds = []
        self.load_times = []
        self.alpha = alpha
        self.class_names = [str(i) for i in range(num_classes)]

        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.use_confusion_loss = use_confusion_loss
        self.confusion_loss_weight = confusion_loss_weight
        self.patience = 30
        self.best_confusion_matrix = None

        self._initialize_training_components(train_dataset)

    def _initialize_training_components(self, train_dataset):
        self.hybrid_weights = self._calculate_hybrid_weights(train_dataset)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=4e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6)
        self.scaler = torch.amp.GradScaler()
        self.best_metrics = {
            'val_iou': 0.0, 'val_f1': 0.0, 'val_loss': float('inf'),
            'val_ious': np.zeros(self.num_classes),
            'val_f1s': np.zeros(self.num_classes),
            'epoch': -1}

    def _calculate_hybrid_weights(self, dataset, epsilon=1e-6):
        if dataset is None:
            return torch.ones(self.num_classes).to(self.device)
            
        print("\nCalculando pesos híbridos para balance de clases...")
        class_counts = torch.zeros(self.num_classes)
        sample_size = min(2000, len(dataset))
        indices = torch.randperm(len(dataset))[:sample_size]
        
        # Contar frecuencias de clases (excluyendo la clase 11)
        for idx in indices:
            _, _, label = dataset[idx]
            unique, counts = torch.unique(label, return_counts=True)
            for u, c in zip(unique, counts):
                if u < self.num_classes:
                    class_counts[u] += c
        
        # Calcular pesos por frecuencia excluyendo la clase 7
        valid_classes = [i for i in range(self.num_classes) if i != 7]
        valid_counts = class_counts[valid_classes]
        
        freq_weights = torch.ones(self.num_classes)
        freq_weights[valid_classes] = 1.0 / (valid_counts + epsilon)
        
        # Normalizar los pesos de las clases válidas
        sum_valid_weights = freq_weights[valid_classes].sum()
        freq_weights[valid_classes] = freq_weights[valid_classes] / sum_valid_weights * len(valid_classes)

        # Calcular el promedio de los pesos válidos para la clase 7
        avg_weight = freq_weights[valid_classes].mean()
        freq_weights[7] = avg_weight*0 + 4
        
        # Limitar el valor máximo a 4
        freq_weights = torch.clamp(freq_weights, max=4.0)
        
        # Pesos iguales
        equal_weights = torch.ones(self.num_classes)*freq_weights.mean()
        equal_weights = equal_weights / equal_weights.sum() * self.num_classes

        # Combinación híbrida
        hybrid_weights = self.alpha * freq_weights + (1 - self.alpha) * equal_weights
        hybrid_weights = hybrid_weights.to(self.device)
        
        # Mostrar resumen
        self._print_weight_summary(class_counts, freq_weights, equal_weights, hybrid_weights)
        
        return hybrid_weights

    def _print_weight_summary(self, counts, freq_w, equal_w, hybrid_w):
        total = counts.sum()
        print(f"\n{'Clase':<10} {'Frecuencia':<12} {'Peso Frec.':<12} {'Peso Igual':<12} {'Peso Final':<12}")
        for i in range(self.num_classes):
            freq = counts[i]/total if total > 0 else 0
            print(f"{i:<10} {freq:<12.4f} {freq_w[i]:<12.4f} {equal_w[i]:<12.4f} {hybrid_w[i]:<12.4f}")

    def _hybrid_loss(self, inputs, targets, confusion_matrix=None, ignore_index=None):
        # Inputs: [B, C, H, W], targets: [B, H, W]
        if self.label_smoothing > 0:
            ce_loss = F.cross_entropy(
                inputs, targets, weight=self.hybrid_weights, reduction='none', label_smoothing=self.label_smoothing
            )  # [B, H, W]
        else:
            ce_loss = F.cross_entropy(
                inputs, targets, weight=self.hybrid_weights, reduction='none'
            )  # [B, H, W]
    
        total_loss = ce_loss
    
        if self.focal_gamma > 0:
            logpt = -ce_loss
            pt = torch.exp(logpt)
            focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
            total_loss = focal_loss
    
        # Flatten everything for per-pixel loss
        ce_flat = ce_loss.flatten()                # [N]
        if ignore_index is not None:
            mask = (targets.flatten() != ignore_index)
            t_flat = targets.flatten()[mask]
            p_flat = torch.argmax(inputs, dim=1).flatten()[mask]
            ce_flat = ce_flat[mask]
        else:
            t_flat = targets.flatten()
            p_flat = torch.argmax(inputs, dim=1).flatten()
    
        # Confusion-aware loss
        if self.use_confusion_loss and confusion_matrix is not None:
            with torch.no_grad():
                confusion_norm = confusion_matrix / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-6)
                penalties = torch.zeros_like(ce_flat)
                for idx in range(len(t_flat)):
                    t = int(t_flat[idx])
                    p = int(p_flat[idx])
                    if t != p and t < self.num_classes and p < self.num_classes:
                        penalties[idx] = confusion_norm[t, p]
                confusion_loss = penalties * ce_flat
            total_loss = total_loss.flatten()
            total_loss = total_loss + self.confusion_loss_weight * confusion_loss
    
        return total_loss.mean()

    def _log_system_stats(self, phase):
        stats = get_system_stats()
        stats['phase'] = phase
        stats['timestamp'] = time.time()
        self.system_stats.append(stats)

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
        # Ponderar F1 por la frecuencia real de cada clase
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

    # ==================== ENTRENAMIENTO POR ÉPOCA (MATRIZ GLOBAL + VECTORIZADA) ====================
    def train_epoch(self, train_loader, epoch_idx):
        self.model.train()
        total_loss = 0.0

        # Matriz de confusión global para TODA la época
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)

        for batch_idx, (x, cat, y) in enumerate(train_loader):
            x = x.to(self.device, dtype=torch.float32, non_blocking=True)
            cat = cat.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(x, cat)
                confusion_matrix = self.best_confusion_matrix if self.use_confusion_loss else None
                loss = self._hybrid_loss(outputs[0], y, confusion_matrix=confusion_matrix)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # ===== MATRIZ DE CONFUSIÓN VECTORIAL =====
            preds = torch.argmax(outputs[0], dim=1).view(-1)
            targets = y.view(-1)

            mask = (targets >= 0) & (targets < self.num_classes) & \
                   (preds   >= 0) & (preds   < self.num_classes)

            if mask.any():
                targets_np = targets[mask].to('cpu', dtype=torch.int64).numpy()
                preds_np   = preds[mask].to('cpu', dtype=torch.int64).numpy()
                idx = targets_np * self.num_classes + preds_np
                cm_batch = np.bincount(
                    idx,
                    minlength=self.num_classes * self.num_classes
                ).reshape(self.num_classes, self.num_classes)
                confusion += cm_batch
            # =========================================

        # guardar esta confusión para la confusion_loss, si la usas
        self.best_confusion_matrix = confusion.copy()

        avg_loss = total_loss / len(train_loader)
        final_metrics = self._calculate_metrics_from_confusion(confusion)
        return avg_loss, final_metrics

    # ==================== VALIDACIÓN (MATRIZ GLOBAL + VECTORIZADA) ====================
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)

        with torch.no_grad():
            for batch_idx, (x, cat, y) in enumerate(val_loader):
                x = x.to(self.device, dtype=torch.float32, non_blocking=True)
                cat = cat.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(x, cat)
                    confusion_matrix = self.best_confusion_matrix if self.use_confusion_loss else None
                    loss = self._hybrid_loss(outputs[0], y, confusion_matrix=confusion_matrix)

                total_loss += loss.item()

                # ===== MATRIZ DE CONFUSIÓN VECTORIAL =====
                preds = torch.argmax(outputs[0], dim=1).view(-1)
                targets = y.view(-1)

                mask = (targets >= 0) & (targets < self.num_classes) & \
                       (preds   >= 0) & (preds   < self.num_classes)

                if mask.any():
                    targets_np = targets[mask].to('cpu', dtype=torch.int64).numpy()
                    preds_np   = preds[mask].to('cpu', dtype=torch.int64).numpy()
                    idx = targets_np * self.num_classes + preds_np
                    cm_batch = np.bincount(
                        idx,
                        minlength=self.num_classes * self.num_classes
                    ).reshape(self.num_classes, self.num_classes)
                    confusion += cm_batch
                # =========================================

        avg_loss = total_loss / len(val_loader)
        final_metrics = self._calculate_metrics_from_confusion(confusion)
        return avg_loss, final_metrics

    # ==================== LOOP DE ENTRENAMIENTO ====================
    def train(self, train_loader, val_loader, epochs=50, patience=None):
        if patience is not None:
            self.patience = patience

        history = {
            'train_loss': [], 'train_iou': [], 'train_f1': [], 'train_ious': [], 'train_f1s': [],
            'val_loss': [], 'val_iou': [], 'val_f1': [], 'val_ious': [], 'val_f1s': [], 'epoch_time': []
        }

        for epoch in range(epochs):
            torch.cuda.empty_cache()
            gc.collect()
            start_time = time.time()
            self._log_system_stats('start_train')  
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            self._log_system_stats('end_train') 
            val_loss, val_metrics = self.validate(val_loader)
            epoch_time = time.time() - start_time
            self.scheduler.step(val_loss)
            self._update_history(history, train_loss, train_metrics, val_loss, val_metrics, epoch_time)
            self._print_epoch_progress(epoch, epochs, epoch_time, train_loss, train_metrics, val_loss, val_metrics)
            if val_loss < self.best_metrics['val_loss']:
                self._update_best_metrics(val_loss, val_metrics, epoch, epoch_time)
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"✅ Nuevo mejor modelo guardado (IoU: {val_metrics['mean_iou']:.4f}, F1: {val_metrics['weighted_f1']:.4f})")
            if (epoch - self.best_metrics['epoch']) >= self.patience:
                print(f"⏹ Detención temprana en época {epoch+1}")
                break
        self._finalize_training(history)
        return history

    def _update_history(self, history, train_loss, train_metrics, val_loss, val_metrics, epoch_time):
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_metrics['mean_iou'])
        history['train_f1'].append(train_metrics['weighted_f1'])
        history['train_ious'].append(train_metrics['ious'])
        history['train_f1s'].append(train_metrics['f1_scores'])
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_metrics['mean_iou'])
        history['val_f1'].append(val_metrics['weighted_f1'])
        history['val_ious'].append(val_metrics['ious'])
        history['val_f1s'].append(val_metrics['f1_scores'])
        history['epoch_time'].append(epoch_time)

    def _print_epoch_progress(self, epoch, epochs, epoch_time, train_loss, train_metrics, val_loss, val_metrics):
        mins, secs = divmod(epoch_time, 60)
        time_str = f"{int(mins):02d}:{int(secs):02d}"
        print(f"\nÉpoca {epoch+1}/{epochs} | Tiempo: {time_str}")
        print(f"Entrenamiento - Pérdida: {train_loss:.4f} | IoU: {train_metrics['mean_iou']:.4f} | F1: {train_metrics['weighted_f1']:.4f}")
        print(f"Validación    - Pérdida: {val_loss:.4f} | IoU: {val_metrics['mean_iou']:.4f} | F1: {val_metrics['weighted_f1']:.4f}")
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
        self.model.load_state_dict(torch.load('best_model.pth'))
        print(f"\n🏆 Mejor modelo - IoU: {self.best_metrics['val_iou']:.4f}, " +
              f"F1: {self.best_metrics['val_f1']:.4f}, Pérdida: {self.best_metrics['val_loss']:.4f}")

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
                plt.text(j, i, f"{norm_confusion[i, j]:.2f}",
                        horizontalalignment="center",
                        color="white" if norm_confusion[i, j] > thresh else "black")
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

# ==================== EJECUCIÓN PRINCIPAL CON MONITOREO ====================
if __name__ == "__main__":
    # Configuración
    BATCH_SIZE = 6 #10 con modelo eB3, 6 con eB5
    PREFETCH = 4
    MAX_FILES = 1000
    WORKERS_TRAIN = 12
    WORKERS_TEST = 4
    BATCH_METRICAS = 4000
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== Configurar dispositivo y cuDNN =====
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cudnn.benchmark = True  # para convoluciones más rápidas
        ngpus = torch.cuda.device_count()
        print(f"Usando CUDA con {ngpus} GPU(s)")
    else:
        device = torch.device("cpu")
        print("⚠ Entrenando en CPU")
    
    # Cargar datos (solo paths)
    BASE_DIR = "/data/nisla/copernicus2"
    # Normalizar rutas de npz y labels
    df_temp = pd.read_csv("/data/nisla/copernicus2/out/patch_index.csv")

    df_temp["npz"] = df_temp["npz"].apply(
        lambda p: p if os.path.isabs(p) else os.path.join(BASE_DIR, p)
    )
    df_temp["label"] = df_temp["label"].apply(
        lambda p: p if os.path.isabs(p) else os.path.join(BASE_DIR, p)
    )
    df_mapping = df_temp.sample(n=min(MAX_FILES, len(df_temp)), random_state=42)
    train_df, val_df = train_test_split(df_mapping, test_size=0.20, random_state=42)
    
    # Crear datasets lazy
    train_dataset = LazySatelliteDataset(
        train_df["npz"].tolist(), 
        train_df["label"].tolist()
    )
    val_dataset = LazySatelliteDataset(
        val_df["npz"].tolist(), 
        val_df["label"].tolist()
    )
    
    # Configurar DataLoaders optimizados
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS_TRAIN,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS_TEST,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Inicializar modelo y trainer
    # model = SFANet(in_channels=11, num_classes=12)
    # model = SFANetPretrained(in_channels=11, num_classes=12, weights_path='best_model.pth')

    base_model = SFANet(in_channels=11, num_classes=12)
    # base_model = SFANetPretrained(in_channels=11, num_classes=12, weights_path='best_model.pth')

    # Si hay más de 1 GPU, usar DataParallel
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"✅ Activando DataParallel en {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    # Usar el nuevo trainer con pesos híbridos
    trainer = BalancedTrainingManager(
        model, 
        device, 
        num_classes=12,
        batch_metricas=BATCH_METRICAS,
        train_dataset=train_dataset,
        alpha=0.4
    )

    
    # Entrenamiento con monitoreo
    history = trainer.train(train_loader, val_loader, epochs=10, patience=20)
    # Guardar modelo final
    torch.save(model.state_dict(), '/home/nisla/copernicus/runs/exp0/fmodel_val.pth')
    
    # Generar reporte de rendimiento
    print("\n📊 Performance Report:")
    print("\n3. Tiempos promedio:")
    print(f"   Batch train: {np.mean(trainer.batch_times['train']) if trainer.batch_times['train'] else 0:.4f}s")
    print(f"   Transfer train: {np.mean(trainer.transfer_times['train']) if trainer.transfer_times['train'] else 0:.4f}s")
    print(f"   Batch val: {np.mean(trainer.batch_times['val']) if trainer.batch_times['val'] else 0:.4f}s")
    print(f"   Transfer val: {np.mean(trainer.transfer_times['val']) if trainer.transfer_times['val'] else 0:.4f}s")
    
    print("\n📊 Reporte Final Mejorado:")
    final_metrics = pd.DataFrame({
        'Clase': trainer.class_names,
        'IOU': trainer.best_metrics['val_ious'],
        'F1': trainer.best_metrics['val_f1s'],
        'Peso Final': trainer.hybrid_weights.cpu().numpy()
    })
    print(final_metrics.to_string(index=False))
    
    # Mostrar distribución de predicciones vs reales
    confusion = trainer.best_metrics['confusion']
    pred_dist = confusion.sum(axis=0) / confusion.sum()
    true_dist = confusion.sum(axis=1) / confusion.sum()
    
    print("\nDistribución de clases:")
    print(f"{'Clase':<10} {'Real':<12} {'Predicho':<12} {'Diferencia':<12}")
    for i in range(trainer.num_classes):
        diff = pred_dist[i] - true_dist[i]
        print(f"{i:<10} {true_dist[i]:<12.4f} {pred_dist[i]:<12.4f} {diff:<12.4f}")
        
    print("\nLos gráficos de rendimiento se han guardado en 'performance_metrics.png'")
    print("La matriz de confusión se ha guardado en 'confusion_matrix.png'")
    print("Los datos detallados se han guardado en 'system_stats.csv' y 'timing_metrics.csv'")
