import os
import time
import inspect
from itertools import cycle

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

import albumentations as A

from scipy.stats import entropy
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)

# Okabe-Ito colour-blind-safe palette + distinct line styles (readable in grayscale print)
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000']
LINESTYLES = ['-', '--', '-.', ':']


def set_journal_style():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.linewidth': 1.0,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.fontsize': 9.5,
        'legend.frameon': True,
        'legend.edgecolor': '0.3',
        'legend.fancybox': False,
        'figure.dpi': 120,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,   # editable/embedded fonts for journal submission
        'ps.fonttype': 42,
    })


def _save(fig, save_dir, name, dpi):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'{name}.png'), dpi=dpi)
    fig.savefig(os.path.join(save_dir, f'{name}.pdf'))   # vector version
    fig.savefig(os.path.join(save_dir, f'{name}.svg'))


def _style_curve_axes(ax, xlabel, ylabel, title):
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.03)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def plot_confusion_matrix(cm, class_names, save_dir=None, dpi=600, show_percent=True):
    cm_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100  # row-wise (recall) %
    if show_percent:
        annot = np.array([[f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)' for j in range(cm.shape[1])]
                          for i in range(cm.shape[0])])
    else:
        annot = cm

    size = max(4.5, 1.1 * len(class_names) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    cmap = 'YlGn'  # light yellow -> deep green
    sns.heatmap(cm_pct, annot=annot, fmt='' if show_percent else 'd', cmap=cmap,
                vmin=0, vmax=100, square=True, linewidths=1.2, linecolor='white',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                cbar_kws={'shrink': 0.8, 'label': 'Row-normalised (%)'}, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix', fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45 if len(class_names) > 4 else 0, ha='right' if len(class_names) > 4 else 'center')
    plt.setp(ax.get_yticklabels(), rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    fig.tight_layout()
    _save(fig, save_dir, 'confusion_matrix', dpi)
    plt.show()


def plot_roc(ax, y_true, probs, class_names):
    n_classes = len(class_names)

    if n_classes == 2:
        # One curve for the positive class (the other is its mirror image)
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        ax.plot(fpr, tpr, color=COLORS[0], lw=2.2, label=f'{class_names[1]} (AUC = {auc(fpr, tpr):.4f})')
    else:
        y_bin = label_binarize(y_true, classes=range(n_classes))
        for i, (color, ls) in zip(range(n_classes), zip(cycle(COLORS), cycle(LINESTYLES))):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            ax.plot(fpr, tpr, color=color, ls=ls, lw=1.8, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.4f})')

    _style_curve_axes(ax, 'False Positive Rate', 'True Positive Rate', 'ROC Curve')
    ax.legend(loc='lower right')


def plot_pr(ax, y_true, probs, class_names):
    n_classes = len(class_names)

    if n_classes == 2:
        p, r, _ = precision_recall_curve(y_true, probs[:, 1])
        ap = average_precision_score(y_true, probs[:, 1])
        ax.plot(r, p, color=COLORS[1], lw=2.2, label=f'{class_names[1]} (AP = {ap:.4f})')
    else:
        y_bin = label_binarize(y_true, classes=range(n_classes))
        for i, (color, ls) in zip(range(n_classes), zip(cycle(COLORS), cycle(LINESTYLES))):
            p, r, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
            ap = average_precision_score(y_bin[:, i], probs[:, i])
            ax.plot(r, p, color=color, ls=ls, lw=1.8, label=f'{class_names[i]} (AP = {ap:.4f})')

    _style_curve_axes(ax, 'Recall', 'Precision', 'Precision-Recall Curve')
    ax.legend(loc='lower left')


def plot_roc_pr(y_true, probs, class_names, save_dir=None, dpi=600):
    # ROC and PR side by side in one row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    plot_roc(ax1, y_true, probs, class_names)
    plot_pr(ax2, y_true, probs, class_names)
    fig.tight_layout()
    _save(fig, save_dir, 'roc_pr_curves', dpi)
    plt.show()


def evaluate_model(model, dataloader, label_mapping, device=None, save_dir='figures', dpi=600):
    """
    label_mapping: dict {class_name: class_index}
    save_dir: folder for PNG (600 dpi) + PDF/SVG (vector) copies; None to skip saving
    """
    set_journal_style()
    if device is None:
        device = next(model.parameters()).device

    class_names = sorted(label_mapping, key=label_mapping.get)  # ordered by class index
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    probs = np.array(all_probs)

    print(f'Accuracy: {(y_true == y_pred).mean():.4f}')
    print(classification_report(y_true, y_pred, labels=range(len(class_names)),
                                target_names=class_names, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plot_confusion_matrix(cm, class_names, save_dir, dpi)
    plot_roc_pr(y_true, probs, class_names, save_dir, dpi)



def evaluate_model_with_metrics(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            probs = torch.softmax(outputs, dim=1)  # for PR AUC

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Base metrics
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    # PR AUC (weighted average)
    try:
        pr_auc = average_precision_score(all_labels, all_probs, average="weighted")
    except ValueError:
        pr_auc = float('nan')

    # Cohen's Kappa
    kappa = cohen_kappa_score(all_labels, all_preds)

    # Specificity (macro average)
    cm = confusion_matrix(all_labels, all_preds)
    TN = []
    FP = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        TN.append(tn)
        FP.append(fp)
    specificity_per_class = [TN[i] / (TN[i] + FP[i]) if (TN[i] + FP[i]) > 0 else 0 for i in range(len(cm))]
    specificity = np.average(specificity_per_class)  # macro average

    # Print metrics
    print(f"Accuracy:    {accuracy:.5f}")
    print(f"Recall:      {recall:.5f}")
    print(f"Precision:   {precision:.5f}")
    print(f"F1 Score:    {f1:.5f}")
    print(f"MCC:         {mcc:.5f}")
    print(f"PR AUC:      {pr_auc:.5f}")
    print(f"Kappa:       {kappa:.5f}")
    print(f"Specificity: {specificity:.5f}")


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, patience=5, min_delta=0.001):
    model.to(device)
    scaler = GradScaler()  # AMP gradient scaler

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    start_time = time.time()  # Start timing
    total_inference_time = 0.0  # <-- Track inference time

    for epoch in range(num_epochs):
        model.train()
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop with AMP
        for inputs, labels in tqdm(train_loader, desc='Training', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():  # Forward pass in mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation loop with AMP (inference only)
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        val_start_time = time.time()  # <-- start inference timing
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_end_time = time.time()
        total_inference_time += (val_end_time - val_start_time)  # <-- accumulate inference time

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
        print(f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')

        # Early Stopping
        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience Counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

        scheduler.step()

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Inference time breakdown
    inf_hours, inf_rem = divmod(total_inference_time, 3600)
    inf_minutes, inf_seconds = divmod(inf_rem, 60)

    total_epochs_run = epoch + 1
    epochs_range = range(1, total_epochs_run + 1)

    # Plotting curves
    plt.figure(figsize=(20, 8))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, marker='o', label='Training Loss')
    plt.plot(epochs_range, val_losses, marker='o', label='Validation Loss')
    plt.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, marker='o', label='Validation Accuracy')
    plt.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Reports
    print("\nTraining Report:")
    print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"Total epochs run: {total_epochs_run}")
    print(f"Best Training Loss: {train_losses[best_epoch-1]:.4f} at epoch {best_epoch}")
    print(f"Best Training Accuracy: {train_accuracies[best_epoch-1]:.4f}")

    print("\nValidation Report:")
    print(f"Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Best Validation Accuracy: {val_accuracies[best_epoch-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}")

    print("\nInference Report:")
    print(f"Total Inference Time (Validation only): {int(inf_hours):02d}:{int(inf_minutes):02d}:{int(inf_seconds):02d}")


# ============================================================
# CLINICALLY-REALISTIC CORRUPTION FUNCTIONS (4 severity levels)
# ============================================================

def gaussian_noise(image, severity):
    sigma = [0.03, 0.06, 0.09, 0.12][severity - 1] * 255
    noisy = image.astype(np.float32) + np.random.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def gaussian_blur(image, severity):
    """Sigma-controlled blur for a gentler severity progression than kernel-size scaling.
    A fixed, modest kernel size (7x7) is used throughout; only the blur intensity (sigma)
    increases with severity, avoiding the steep jump caused by growing kernel size."""
    sigma = [0.5, 0.8, 1.0, 1.2][severity - 1]
    return cv2.GaussianBlur(image, (7, 7), sigmaX=sigma, sigmaY=sigma)

def brightness_shift(image, severity):
    factor = [0.8, 0.9, 1.2, 1.4][severity - 1]
    shifted = image.astype(np.float32) * factor
    return np.clip(shifted, 0, 255).astype(np.uint8)

def _build_jpeg_compression(quality, p=1.0):
    """Version-safe across albumentations releases
    (API renamed quality_lower/upper -> quality_range in newer versions)."""
    params = inspect.signature(A.ImageCompression.__init__).parameters
    if "quality_range" in params:
        return A.ImageCompression(quality_range=(quality, quality), p=p)
    else:
        return A.ImageCompression(quality_lower=quality, quality_upper=quality, p=p)

def jpeg_compression(image, severity):
    """image: raw uint8 HWC RGB numpy array (matches what CustomImageDataset hands to transform)."""
    quality = [75, 55, 35, 15][severity - 1]
    transform = _build_jpeg_compression(quality)
    corrupted = transform(image=image)["image"]
    return corrupted

CORRUPTION_FNS = {
    'Gaussian Noise': gaussian_noise,
    'Gaussian Blur': gaussian_blur,
    'Brightness Shift': brightness_shift,
    'JPEG Compression': jpeg_compression,
}

from PIL import Image
from torchvision import transforms

def make_corruption_transform(corruption_fn, severity, mean, std, img_size=224):
    """
    Handles PIL Image input (my_dataset1.py) and numpy array input
    (cv2-based CustomImageDataset) uniformly. Resizes to img_size BEFORE
    corruption, matching what your normal data_transform does internally
    for clean evaluation — otherwise batches contain mismatched image sizes.
    """
    resize = transforms.Resize((img_size, img_size))

    def _transform(image):
        # Normalize to PIL first, resize, then to numpy for corruption
        if isinstance(image, Image.Image):
            pil_img = image
        else:
            pil_img = Image.fromarray(image)

        pil_img = resize(pil_img)             # <-- the missing step
        image_np = np.array(pil_img)          # PIL -> numpy uint8 HWC

        corrupted = corruption_fn(image_np, severity)

        pil_corrupted = transforms.ToPILImage()(corrupted)
        tensor = transforms.ToTensor()(pil_corrupted)
        tensor = transforms.Normalize(mean, std)(tensor)
        return tensor
    return _transform


def _evaluate_loader(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.numpy() if torch.is_tensor(labels) else labels)
            all_preds.extend(preds.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    return acc, f1, mcc


# ============================================================
# CORRUPTION ROBUSTNESS EVALUATION — aggregated (mean ± drop)
# ============================================================

def evaluate_corruption_robustness(model, dataset, device, mean, std,
                                    severities=(1, 2, 3, 4),
                                    batch_size=32, num_workers=0):
    """
    Returns:
        summary_df      : one row per corruption type — mean accuracy/F1/MCC
                           across all severities, plus drop from clean baseline
        per_severity_df : full per-severity results (used only for the figure)
        clean_metrics   : (acc, f1, mcc) on uncorrupted test data
    """
    model.eval()
    original_transform = dataset.transform

    # --- Clean baseline first ---
    clean_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    clean_acc, clean_f1, clean_mcc = _evaluate_loader(model, clean_loader, device)
    print(f"{'Clean':20s} | Acc={clean_acc:.4f} | F1={clean_f1:.4f} | MCC={clean_mcc:.4f}")

    per_severity_rows = []

    try:
        for corruption_name, corruption_fn in CORRUPTION_FNS.items():
            for severity in severities:
                dataset.transform = make_corruption_transform(corruption_fn, severity, mean, std)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                acc, f1, mcc = _evaluate_loader(model, loader, device)
                per_severity_rows.append({
                    'Perturbation': corruption_name, 'Severity': severity,
                    'Accuracy': acc, 'Macro F1': f1, 'MCC': mcc
                })
                print(f"{corruption_name:20s} | sev={severity} | Acc={acc:.4f} | F1={f1:.4f} | MCC={mcc:.4f}")
    finally:
        dataset.transform = original_transform  # always restore, even on error

    per_severity_df = pd.DataFrame(per_severity_rows)

    # --- Aggregate: mean across severities per corruption type, + drop from clean ---
    summary_rows = []
    for corruption_name in CORRUPTION_FNS.keys():
        sub = per_severity_df[per_severity_df['Perturbation'] == corruption_name]
        mean_acc = sub['Accuracy'].mean()
        mean_f1 = sub['Macro F1'].mean()
        mean_mcc = sub['MCC'].mean()
        summary_rows.append({
            'Perturbation': corruption_name,
            'Mean Accuracy': mean_acc,
            'Mean Macro F1': mean_f1,
            'Mean MCC': mean_mcc,
            'Accuracy Drop': clean_acc - mean_acc,
            'F1 Drop': clean_f1 - mean_f1,
            'MCC Drop': clean_mcc - mean_mcc,
        })

    summary_df = pd.DataFrame(summary_rows)

    # --- Overall row: mean across all four corruption types ---
    overall_row = {
        'Perturbation': 'Overall (mean across corruptions)',
        'Mean Accuracy': summary_df['Mean Accuracy'].mean(),
        'Mean Macro F1': summary_df['Mean Macro F1'].mean(),
        'Mean MCC': summary_df['Mean MCC'].mean(),
        'Accuracy Drop': summary_df['Accuracy Drop'].mean(),
        'F1 Drop': summary_df['F1 Drop'].mean(),
        'MCC Drop': summary_df['MCC Drop'].mean(),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)

    # --- Clean baseline row on top, for reference in the same table ---
    clean_row = pd.DataFrame([{
        'Perturbation': 'Clean (no corruption)',
        'Mean Accuracy': clean_acc, 'Mean Macro F1': clean_f1, 'Mean MCC': clean_mcc,
        'Accuracy Drop': 0.0, 'F1 Drop': 0.0, 'MCC Drop': 0.0,
    }])
    summary_df = pd.concat([clean_row, summary_df], ignore_index=True)

    return summary_df, per_severity_df, (clean_acc, clean_f1, clean_mcc)


# ============================================================
# FIGURE — per-severity trend curves (table stays aggregated)
# ============================================================

def _plot_corruption_figure(per_severity_df, clean_metrics, save_prefix='corruption_robustness'):
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    sns.set_style('whitegrid', {'grid.linestyle': '--'})

    clean_acc, _, _ = clean_metrics
    fig, ax = plt.subplots(figsize=(8, 6))

    palette = sns.color_palette('deep', n_colors=per_severity_df['Perturbation'].nunique())
    for color, corruption in zip(palette, per_severity_df['Perturbation'].unique()):
        sub = per_severity_df[per_severity_df['Perturbation'] == corruption].sort_values('Severity')
        ax.plot(sub['Severity'], sub['Accuracy'] * 100, marker='o',
                color=color, linewidth=2, markersize=7, label=corruption)

    ax.axhline(clean_acc * 100, linestyle='--', color='black', alpha=0.6, label='Clean accuracy')
    ax.set_xlabel('Corruption Severity Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Corruption Robustness Across Severity Levels', fontsize=14, fontweight='bold')
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(frameon=False, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(f'{save_prefix}_figure.png', dpi=600, bbox_inches='tight')
    fig.savefig(f'{save_prefix}_figure.pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved figure: {save_prefix}_figure.png / .pdf")


# ============================================================
# FULL PIPELINE
# ============================================================

def run_corruption_analysis(model, dataset, device, mean, std, save_prefix='corruption_robustness'):
    print("=" * 60)
    print("CLINICALLY-REALISTIC CORRUPTION ROBUSTNESS")
    print("=" * 60)

    summary_df, per_severity_df, clean_metrics = evaluate_corruption_robustness(
        model, dataset, device, mean, std, severities=(1, 2, 3, 4)
    )

    summary_df.to_csv(f'{save_prefix}_summary.csv', index=False)
    print(f"\nSaved aggregated table: {save_prefix}_summary.csv")
    print("\n" + summary_df.to_string(index=False))

    _plot_corruption_figure(per_severity_df, clean_metrics, save_prefix=save_prefix)

    return summary_df, per_severity_df


# ============================================================
# 1. DIAGNOSTIC — confirm dropout layers exist and toggle correctly
# ============================================================

def diagnose_dropout(model):
    found = False
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if cls == 'Dropout' or 'drop' in name.lower():
            p = getattr(module, 'p', getattr(module, 'drop_prob', 'N/A'))
            print(f"{name:35s} | {cls:12s} | p={p}")
            found = True
    if not found:
        print("No Dropout/DropPath modules found.")


def enable_mc_dropout(model):
    """Only flips real nn.Dropout layers to train mode — leaves BatchNorm/LayerNorm
    and DropPath (stochastic depth) untouched, since standard MC Dropout targets
    nn.Dropout specifically."""
    for module in model.modules():
        if module.__class__.__name__ == 'Dropout':
            module.train()


# ============================================================
# 2. MC DROPOUT
# ============================================================

def mc_dropout_predict(model, dataloader, device, n_samples=30):
    model.eval()
    enable_mc_dropout(model)

    all_labels = []
    all_mean_probs = []
    all_pred_entropy = []
    all_aleatoric = []
    all_disagreement = []  # fraction of MC passes disagreeing with the majority vote, per sample

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            batch_probs = []

            for _ in range(n_samples):
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())

            batch_probs = np.stack(batch_probs, axis=0)   # (S, B, C)
            mean_probs = batch_probs.mean(axis=0)          # (B, C)

            pred_entropy = entropy(mean_probs.T + 1e-12)

            per_sample_entropy = np.array([
                entropy(batch_probs[s].T + 1e-12) for s in range(n_samples)
            ])
            aleatoric = per_sample_entropy.mean(axis=0)

            # Disagreement diagnostic: per-sample, per-pass argmax vs majority vote
            pass_preds = np.argmax(batch_probs, axis=2)              # (S, B)
            majority = np.array([np.bincount(pass_preds[:, b]).argmax()
                                  for b in range(pass_preds.shape[1])])
            disagreement = (pass_preds != majority[None, :]).mean(axis=0)  # (B,)

            all_labels.extend(labels.numpy())
            all_mean_probs.append(mean_probs)
            all_pred_entropy.extend(pred_entropy)
            all_aleatoric.extend(aleatoric)
            all_disagreement.extend(disagreement)

    all_labels = np.array(all_labels)
    all_mean_probs = np.concatenate(all_mean_probs, axis=0)
    all_pred_entropy = np.array(all_pred_entropy)
    all_aleatoric = np.array(all_aleatoric)
    all_disagreement = np.array(all_disagreement)
    epistemic = np.clip(all_pred_entropy - all_aleatoric, a_min=0, a_max=None)

    print(f"[Diagnostic] Mean per-sample MC-pass disagreement rate: {all_disagreement.mean()*100:.2f}%")
    print(f"[Diagnostic] Samples with ANY disagreement across {n_samples} passes: "
          f"{(all_disagreement > 0).mean()*100:.2f}%")

    return all_mean_probs, all_labels, all_pred_entropy, all_aleatoric, epistemic


# ============================================================
# 3. TEST-TIME AUGMENTATION (TTA) — matched to your CustomImageDataset
# ============================================================

def get_tta_transforms(mean, std):
    """
    Your dataset already resizes to 224x224 with cv2 before calling self.transform,
    and passes a uint8 HWC RGB numpy array. So we start with ToPILImage()
    (NOT Resize — that's already done) then augment, ToTensor, Normalize.

    IMPORTANT: replace `mean`/`std` below with the exact values used in your
    existing `data_transform` — mismatched normalization will silently distort results.
    """
    return [
        transforms.Compose([  # 1. original, no aug
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([  # 2. horizontal flip
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([  # 3. vertical flip
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([  # 4. small rotation
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([  # 5. brightness/contrast jitter
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    ]


def tta_predict(model, dataset, device, mean, std, batch_size=32, num_workers=0):
    """
    dataset: the CustomImageDataset instance (e.g. `test_dataset`), NOT a DataLoader.
    We swap `.transform` on it before each pass and build a fresh DataLoader each time.

    num_workers=0 by default: on Windows, DataLoader workers use `spawn`, which
    re-pickles the dataset per worker — safer to keep 0 here to avoid stale-transform
    or pickling issues when swapping `.transform` between passes.
    """
    model.eval()
    tta_transforms = get_tta_transforms(mean, std)
    n_aug = len(tta_transforms)

    all_labels = None
    all_aug_probs = []

    for t_idx, tform in enumerate(tta_transforms):
        dataset.transform = tform
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        aug_probs = []
        aug_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                aug_probs.append(probs.cpu().numpy())
                aug_labels.extend(labels.numpy() if torch.is_tensor(labels) else labels)

        aug_probs = np.concatenate(aug_probs, axis=0)
        all_aug_probs.append(aug_probs)

        if t_idx == 0:
            all_labels = np.array(aug_labels)

    all_aug_probs = np.stack(all_aug_probs, axis=0)   # (n_aug, N, C)
    mean_probs = all_aug_probs.mean(axis=0)

    pred_entropy = entropy(mean_probs.T + 1e-12)
    per_aug_entropy = np.array([entropy(all_aug_probs[a].T + 1e-12) for a in range(n_aug)])
    aleatoric = per_aug_entropy.mean(axis=0)
    epistemic = np.clip(pred_entropy - aleatoric, a_min=0, a_max=None)

    return mean_probs, all_labels, pred_entropy, aleatoric, epistemic

# ============================================================
# CALIBRATION METRICS — ECE, MCE, Brier Score, NLL
# ============================================================

def calibration_metrics(y_true, mean_probs, n_bins=10, eps=1e-12):
    """
    Computes ECE, MCE, Brier Score, and NLL from true labels and predicted
    probability vectors. Returns a dict of scalar metrics plus the per-bin
    DataFrame (for reliability diagrams).
    """
    y_true = np.array(y_true)
    mean_probs = np.array(mean_probs)
    num_classes = mean_probs.shape[1]

    confidences = np.max(mean_probs, axis=1)
    predictions = np.argmax(mean_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    # --- ECE + MCE (bin-based) ---
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            gap = abs(bin_acc - bin_conf)
            ece += bin_weight * gap
            mce = max(mce, gap)  # MCE = worst single-bin gap, unweighted
            bin_data.append({'bin_lower': bins[i], 'bin_upper': bins[i + 1],
                              'accuracy': bin_acc, 'confidence': bin_conf, 'count': mask.sum()})
    bin_df = pd.DataFrame(bin_data)

    # --- Brier Score (multi-class): mean squared error vs. one-hot labels ---
    one_hot = np.zeros((len(y_true), num_classes))
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    brier = np.mean(np.sum((mean_probs - one_hot) ** 2, axis=1))

    # --- NLL (categorical cross-entropy on the true class) ---
    true_class_probs = mean_probs[np.arange(len(y_true)), y_true]
    nll = -np.mean(np.log(true_class_probs + eps))

    return {
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'nll': nll,
    }, bin_df

# ============================================================
# 5. FULL PIPELINE
# ============================================================

def run_uncertainty_analysis(model, dataloader, tta_dataset, device,
                              class_names, mean, std,
                              mc_samples=30, save_prefix='uncertainty'):
    print("=" * 60); print("MC DROPOUT UNCERTAINTY ANALYSIS"); print("=" * 60)
    mc_probs, mc_labels, mc_pred_ent, mc_aleatoric, mc_epistemic = mc_dropout_predict(
        model, dataloader, device, n_samples=mc_samples
    )
    mc_preds = np.argmax(mc_probs, axis=1)
    mc_acc = accuracy_score(mc_labels, mc_preds)
    mc_f1 = f1_score(mc_labels, mc_preds, average='macro', zero_division=0)
    mc_cal, mc_bins = calibration_metrics(mc_labels, mc_probs)

    print(f"Accuracy: {mc_acc:.5f} | Macro F1: {mc_f1:.5f}")
    print(f"ECE: {mc_cal['ece']:.5f} | MCE: {mc_cal['mce']:.5f} | "
          f"Brier: {mc_cal['brier']:.5f} | NLL: {mc_cal['nll']:.5f}")
    print(f"Mean predictive entropy: {mc_pred_ent.mean():.5f} | "
          f"Aleatoric: {mc_aleatoric.mean():.5f} | Epistemic: {mc_epistemic.mean():.5f}")

    print("\n" + "=" * 60); print("TTA UNCERTAINTY ANALYSIS"); print("=" * 60)
    tta_probs, tta_labels, tta_pred_ent, tta_aleatoric, tta_epistemic = tta_predict(
        model, tta_dataset, device, mean=mean, std=std
    )
    tta_preds = np.argmax(tta_probs, axis=1)
    tta_acc = accuracy_score(tta_labels, tta_preds)
    tta_f1 = f1_score(tta_labels, tta_preds, average='macro', zero_division=0)
    tta_cal, tta_bins = calibration_metrics(tta_labels, tta_probs)

    print(f"Accuracy: {tta_acc:.5f} | Macro F1: {tta_f1:.5f}")
    print(f"ECE: {tta_cal['ece']:.5f} | MCE: {tta_cal['mce']:.5f} | "
          f"Brier: {tta_cal['brier']:.5f} | NLL: {tta_cal['nll']:.5f}")
    print(f"Mean predictive entropy: {tta_pred_ent.mean():.5f} | "
          f"Aleatoric: {tta_aleatoric.mean():.5f} | Epistemic: {tta_epistemic.mean():.5f}")

    summary = pd.DataFrame({
        'Method': ['MC Dropout', 'TTA'],
        'Accuracy': [mc_acc, tta_acc],
        'Macro F1': [mc_f1, tta_f1],
        'ECE': [mc_cal['ece'], tta_cal['ece']],
        'MCE': [mc_cal['mce'], tta_cal['mce']],
        'Brier Score': [mc_cal['brier'], tta_cal['brier']],
        'NLL': [mc_cal['nll'], tta_cal['nll']],
        'Mean Predictive Entropy': [mc_pred_ent.mean(), tta_pred_ent.mean()],
        'Mean Aleatoric': [mc_aleatoric.mean(), tta_aleatoric.mean()],
        'Mean Epistemic': [mc_epistemic.mean(), tta_epistemic.mean()],
    })
    summary.to_csv(f'{save_prefix}_summary.csv', index=False)
    print(f"\nSaved: {save_prefix}_summary.csv")

    _plot_uncertainty_figures(mc_labels, mc_preds, mc_pred_ent, mc_epistemic, mc_bins,
                               tta_labels, tta_preds, tta_pred_ent, tta_epistemic, tta_bins,
                               save_prefix=save_prefix)

    return {
        'mc_dropout': dict(probs=mc_probs, labels=mc_labels, preds=mc_preds,
                            pred_entropy=mc_pred_ent, aleatoric=mc_aleatoric,
                            epistemic=mc_epistemic, accuracy=mc_acc, f1=mc_f1, **mc_cal),
        'tta': dict(probs=tta_probs, labels=tta_labels, preds=tta_preds,
                    pred_entropy=tta_pred_ent, aleatoric=tta_aleatoric,
                    epistemic=tta_epistemic, accuracy=tta_acc, f1=tta_f1, **tta_cal),
        'summary_table': summary,
    }


def _plot_uncertainty_figures(mc_labels, mc_preds, mc_pred_ent, mc_epistemic, mc_bins,
                               tta_labels, tta_preds, tta_pred_ent, tta_epistemic, tta_bins,
                               save_prefix='uncertainty'):
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    sns.set_style('whitegrid', {'grid.linestyle': '--'})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, bins_df, title in [(axes[0, 0], mc_bins, 'MC Dropout Reliability Diagram'),
                                (axes[1, 0], tta_bins, 'TTA Reliability Diagram')]:
        centers = (bins_df['bin_lower'] + bins_df['bin_upper']) / 2
        ax.bar(centers, bins_df['accuracy'], width=0.08, edgecolor='black', color='#4C72B0', alpha=0.85)
        ax.plot([0, 1], [0, 1], '--', color='black', label='Perfect calibration')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy'); ax.set_title(title, fontweight='bold')
        ax.legend()

    for ax, labels, preds, pred_ent, title in [
        (axes[0, 1], mc_labels, mc_preds, mc_pred_ent, 'MC Dropout: Entropy by Correctness'),
        (axes[1, 1], tta_labels, tta_preds, tta_pred_ent, 'TTA: Entropy by Correctness')]:
        correct = (labels == preds)
        sns.kdeplot(pred_ent[correct], ax=ax, label='Correct', fill=True, color='#55A868', alpha=0.4)
        sns.kdeplot(pred_ent[~correct], ax=ax, label='Incorrect', fill=True, color='#C44E52', alpha=0.4)
        ax.set_xlabel('Predictive Entropy'); ax.set_ylabel('Density'); ax.set_title(title, fontweight='bold')
        ax.legend()

    for ax, epi, title in [(axes[0, 2], mc_epistemic, 'MC Dropout: Epistemic Uncertainty'),
                            (axes[1, 2], tta_epistemic, 'TTA: Epistemic Uncertainty')]:
        sns.histplot(epi, bins=30, ax=ax, color='#8172B2', edgecolor='black', alpha=0.85)
        ax.set_xlabel('Epistemic Uncertainty (MI)'); ax.set_ylabel('Count'); ax.set_title(title, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f'{save_prefix}_figures.png', dpi=600, bbox_inches='tight')
    fig.savefig(f'{save_prefix}_figures.pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_prefix}_figures.png / .pdf")
