"""
Evaluation utilities for segmentation and classification:
threshold sweep, metrics, visualisation, PR curves, confusion matrices.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
)

from utils import denorm_batch, IMAGENET_MEAN, IMAGENET_STD


# ============================================================
# Segmentation evaluation
# ============================================================

@torch.no_grad()
def evaluate_on_loader_fast(model, loader, device, threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    total_tp = total_fp = total_fn = total_tn = 0.0
    eps = 1e-8
    for imgs, masks in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()
        preds = (torch.sigmoid(model(imgs)) > threshold).float()
        pf = preds.view(preds.size(0), -1)
        mf = masks.view(masks.size(0), -1)
        total_tp += (pf * mf).sum(dim=1).sum().item()
        total_fp += (pf * (1 - mf)).sum(dim=1).sum().item()
        total_fn += ((1 - pf) * mf).sum(dim=1).sum().item()
        total_tn += ((1 - pf) * (1 - mf)).sum(dim=1).sum().item()
    return {
        "IoU":       total_tp / (total_tp + total_fp + total_fn + eps),
        "Dice":      2 * total_tp / (2 * total_tp + total_fp + total_fn + eps),
        "Precision": total_tp / (total_tp + total_fp + eps),
        "Recall":    total_tp / (total_tp + total_fn + eps),
        "Acc":       (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps),
    }


@torch.no_grad()
def best_threshold_on_val(model, val_loader, device,
                          t_min: float = 0.1, t_max: float = 0.9,
                          steps: int = 17) -> Tuple[float, float]:
    """Coarse sweep used during training (accumulates per-batch IoU)."""
    model.eval()
    t_candidates = torch.linspace(t_min, t_max, steps, device=device)
    sums  = torch.zeros_like(t_candidates)
    count = 0
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(device), masks.to(device).float()
        prob = torch.sigmoid(model(imgs))
        for i, t in enumerate(t_candidates):
            pred = (prob > t).float()
            tp = (pred * masks).sum(); fp = (pred * (1 - masks)).sum(); fn = ((1 - pred) * masks).sum()
            sums[i] += tp / (tp + fp + fn + 1e-8)
        count += 1
    idx = int(torch.argmax(sums))
    return float(t_candidates[idx].item()), float((sums[idx] / max(1, count)).item())


@torch.no_grad()
def best_threshold_sweep_fast(model, val_loader, device,
                              t_min: float = 0.1, t_max: float = 0.95,
                              steps: int = 18) -> Tuple[float, float]:
    """Memory-efficient sweep: collects all probs/masks then sweeps thresholds at once."""
    model.eval()
    all_probs, all_masks = [], []
    for imgs, masks in val_loader:
        all_probs.append(torch.sigmoid(model(imgs.to(device, non_blocking=True))).cpu())
        all_masks.append(masks)
    all_probs = torch.cat(all_probs, 0).float()
    all_masks = torch.cat(all_masks, 0).float()
    best_iou, best_t = 0.0, 0.5
    eps = 1e-8
    for t in np.linspace(t_min, t_max, steps):
        preds = (all_probs > t).float()
        pf = preds.view(preds.size(0), -1); mf = all_masks.view(all_masks.size(0), -1)
        tp = (pf * mf).sum(); fp = (pf * (1 - mf)).sum(); fn = ((1 - pf) * mf).sum()
        iou = (tp / (tp + fp + fn + eps)).item()
        if iou > best_iou:
            best_iou, best_t = iou, float(t)
    return best_t, best_iou


@torch.no_grad()
def sweep_curve(model, val_loader, device, t_min=0.1, t_max=0.9, steps=41):
    model.eval()
    thresholds = torch.linspace(t_min, t_max, steps, device=device)
    sums = torch.zeros_like(thresholds)
    count = 0
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(device), masks.to(device).float()
        prob = torch.sigmoid(model(imgs))
        for i, t in enumerate(thresholds):
            pred = (prob > t).float()
            tp = (pred * masks).sum(); fp = (pred * (1 - masks)).sum(); fn = ((1 - pred) * masks).sum()
            sums[i] += tp / (tp + fp + fn + 1e-8)
        count += 1
    return thresholds.cpu().numpy(), (sums / max(1, count)).cpu().numpy()


@torch.no_grad()
def gather_probs_targets(model, loader, device):
    model.eval()
    allp, ally = [], []
    for imgs, masks in loader:
        imgs = imgs.to(device)
        p = torch.sigmoid(model(imgs)).detach().cpu().numpy().ravel()
        y = masks.detach().cpu().numpy().ravel()
        allp.append(p); ally.append(y)
    return np.concatenate(allp), np.concatenate(ally)


def pixel_pr_curve_from_arrays(p, y, steps=101):
    ts = np.linspace(0, 1, steps)
    P, R = [], []
    for t in ts:
        pred = (p > t).astype(np.float32)
        tp = (pred * y).sum(); fp = (pred * (1 - y)).sum(); fn = ((1 - pred) * y).sum()
        P.append(tp / (tp + fp + 1e-8)); R.append(tp / (tp + fn + 1e-8))
    order = np.argsort(R)
    AP = np.trapz(np.array(P)[order], np.array(R)[order])
    return np.array(R), np.array(P), AP


def save_overlay_grid(model, loader, device, n=12, save_path="qual_val.png", t=0.5):
    model.eval()
    imgs_list, preds_list, gts_list = [], [], []
    import cv2
    with torch.no_grad():
        for imgs, masks in loader:
            imgs_gpu = imgs.to(device)
            probs = torch.sigmoid(model(imgs_gpu))
            preds = (probs > t).float()
            imgs_list.append(imgs); preds_list.append(preds.cpu()); gts_list.append(masks)
            if sum(x.size(0) for x in imgs_list) >= n:
                break
    X = torch.cat(imgs_list)[:n]; P = torch.cat(preds_list)[:n]; Y = torch.cat(gts_list)[:n]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    X = torch.clamp(X * std + mean, 0, 1).mul(255).byte().permute(0, 2, 3, 1).numpy()
    P = P.squeeze(1).byte().numpy(); Y = Y.squeeze(1).byte().numpy()
    grid = []
    for i in range(X.shape[0]):
        img = X[i].copy()
        img[cv2.Canny((Y[i] * 255), 50, 150) > 0] = [255, 0, 0]
        img[cv2.Canny((P[i] * 255), 50, 150) > 0] = [0, 255, 0]
        grid.append(img)
    rows, cols = math.ceil(len(grid) / 4), 4
    H, W = grid[0].shape[:2]
    canvas = np.ones((rows * H, cols * W, 3), dtype=np.uint8) * 255
    for idx, im in enumerate(grid):
        r, c = divmod(idx, cols); canvas[r*H:(r+1)*H, c*W:(c+1)*W] = im
    import cv2
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return save_path


@torch.no_grad()
def visualize_predictions_batch(model, loader, device, threshold: float,
                                num_samples: int = 6, save_dir: Optional[str] = None):
    model.eval()
    shown = 0
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for bidx, (imgs, masks) in enumerate(loader):
        if shown >= num_samples: break
        take = min(imgs.size(0), num_samples - shown)
        probs = torch.sigmoid(model(imgs[:take].to(device))).cpu()[:, 0].numpy()
        preds = (probs > threshold).astype(np.uint8)
        imgs_denorm = denorm_batch(imgs[:take])
        masks_np    = masks[:take, 0].numpy()
        fig, axes = plt.subplots(take, 3, figsize=(12, 4 * take))
        if take == 1: axes = np.expand_dims(axes, 0)
        for i in range(take):
            axes[i, 0].imshow(imgs_denorm[i]); axes[i, 0].set_title("Image"); axes[i, 0].axis("off")
            axes[i, 1].imshow(masks_np[i], cmap="gray"); axes[i, 1].set_title("GT"); axes[i, 1].axis("off")
            axes[i, 2].imshow(preds[i], cmap="gray"); axes[i, 2].set_title(f"Pred t={threshold:.2f}"); axes[i, 2].axis("off")
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"batch_{bidx}.png"), dpi=100, bbox_inches="tight")
        plt.show()
        shown += take


def run_test_block_optimized(model, device, val_loader, test_loader=None,
                             sweep_on_val: bool = True, n_vis: int = 6,
                             save_dir: Optional[str] = None):
    if sweep_on_val:
        best_t, best_iou = best_threshold_sweep_fast(model, val_loader, device)
        print(f"[Threshold] Best: t={best_t:.2f} (IoU={best_iou:.4f})")
    else:
        best_t = 0.5; print("[Threshold] Using fixed t=0.50")

    loader = test_loader if test_loader is not None else val_loader
    split_name = "test" if test_loader is not None else "val"
    metrics = evaluate_on_loader_fast(model, loader, device, threshold=best_t)
    print(f"[{split_name}] IoU: {metrics['IoU']:.4f} | Dice: {metrics['Dice']:.4f} | "
          f"Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f} | Acc: {metrics['Acc']:.4f}")
    if n_vis > 0:
        visualize_predictions_batch(model, loader, device, threshold=best_t,
                                    num_samples=n_vis, save_dir=save_dir)
    return metrics, best_t


def run_test_block_with_loading(ckpt_path, model_class, model_kwargs, device,
                                val_loader, test_loader=None, sweep_on_val=True,
                                n_vis=6, save_dir=None):
    model = model_class(**model_kwargs).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state); model.eval()
    return run_test_block_optimized(model, device, val_loader, test_loader, sweep_on_val, n_vis, save_dir)


# ============================================================
# Classification evaluation
# ============================================================

CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
NUM_CLASSES  = len(CLASS_NAMES)


@torch.no_grad()
def _gather_cls(model, loader, device, tta_fn=None, bias=None):
    model.eval()
    probs_list, y_list = [], []
    for pre, post, y in loader:
        pre, post = pre.to(device), post.to(device)
        logits = tta_fn(model, pre, post) if tta_fn else model(pre, post)
        if bias is not None: logits = logits + bias.to(logits.device)
        probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        y_list.append(y.numpy())
    probs  = np.vstack(probs_list)
    y_true = np.concatenate(y_list)
    return y_true, probs.argmax(1), probs


@torch.no_grad()
def fit_bias_on_val(model, loader, device,
                    grid2=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                    grid3=(-0.6, -0.4, -0.2, 0.0),
                    major_weight: float = 0.6):
    model.eval()
    Ls, Ys = [], []
    for pre, post, y in loader:
        Ls.append(model(pre.to(device), post.to(device)).cpu()); Ys.append(y)
    L = torch.cat(Ls); Y = torch.cat(Ys).numpy()
    base = torch.zeros(L.size(1)); best_score, best_b = -1.0, None
    for b2 in grid2:
        for b3 in grid3:
            b = base.clone(); b[2] += b2; b[3] += b3
            pred  = (L + b).argmax(1).numpy()
            macro = f1_score(Y, pred, average="macro")
            rep   = classification_report(Y, pred, output_dict=True, zero_division=0)
            score = major_weight * rep["2"]["f1-score"] + (1 - major_weight) * macro
            if score > best_score: best_score, best_b = score, b.clone()
    print(f"[bias-fit] best score={best_score:.4f} bias={best_b.tolist()}")
    return best_b


def print_cls_report(model, loader, device, tag="", tta_fn=None, bias=None):
    y_true, y_pred, _ = _gather_cls(model, loader, device, tta_fn=tta_fn, bias=bias)
    print(f"CM ({tag}):\n", confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))))
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))


# ---- plotting ----

def show_confmat(y_true, y_pred, title: str, normalize: bool = True):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_plot = (cm / cm.sum(axis=1, keepdims=True).clip(min=1)) if normalize else cm.astype(float)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm_plot, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=30, ha="right")
    plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_plot[i, j]
            plt.text(j, i, f"{val:.2f}" if normalize else str(int(cm[i, j])), ha="center", va="center")
    plt.tight_layout(); plt.show()


def show_perclass_f1(y_true, y_pred, title: str):
    rep  = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    f1s  = [rep[c]["f1-score"] for c in CLASS_NAMES]
    plt.figure(figsize=(7, 4))
    plt.bar(CLASS_NAMES, f1s); plt.ylim(0, 1.0); plt.title(title); plt.ylabel("F1")
    plt.tight_layout(); plt.show()


def show_pr_curves_cls(y_true, probs, title: str):
    plt.figure(figsize=(7, 5))
    y_bin = np.eye(NUM_CLASSES)[y_true]
    p_micro, r_micro, _ = precision_recall_curve(y_bin.ravel(), probs.ravel())
    ap_micro = average_precision_score(y_bin, probs, average="micro")
    plt.plot(r_micro, p_micro, label=f"micro-avg (AP={ap_micro:.3f})")
    for c in range(NUM_CLASSES):
        prec, rec, _ = precision_recall_curve((y_true == c).astype(int), probs[:, c])
        ap = average_precision_score((y_true == c).astype(int), probs[:, c])
        plt.plot(rec, prec, label=f"{CLASS_NAMES[c]} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.show()


def show_reliability(y_true, probs, n_bins=10, title="Reliability"):
    preds = probs.argmax(1); confs = probs.max(1); correct = (preds == y_true).astype(np.float32)
    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confs, bins) - 1
    accs, conf_avgs, ece = [], [], 0.0; N = len(y_true)
    for b in range(n_bins):
        idx = (bin_ids == b); nb = idx.sum()
        if nb == 0: accs.append(np.nan); conf_avgs.append((bins[b] + bins[b+1]) / 2); continue
        acc = correct[idx].mean(); conf_avg = confs[idx].mean()
        accs.append(acc); conf_avgs.append(conf_avg); ece += (nb / N) * abs(acc - conf_avg)
    plt.figure(figsize=(7, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.7, label="perfect")
    plt.scatter(conf_avgs, accs, s=30)
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(f"{title} (ECE={ece:.3f})")
    plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.show()
    print(f"ECE = {ece:.4f}")


def plot_learning_curves(log_csv, run="last", smooth=None):
    df = pd.read_csv(log_csv)
    inc = df["epoch"].diff().fillna(1)
    run_id = (inc <= 0).cumsum()
    runs = [g.drop(columns=["_run_id"] if "_run_id" in g.columns else []).reset_index(drop=True)
            for _, g in df.assign(_run_id=run_id.values).groupby(run_id)]
    d  = runs[-1] if run == "last" else runs[int(run)]
    ep = d["epoch"].astype(int).values

    def S(x):
        if smooth is None or smooth <= 1: return x
        return pd.Series(x).rolling(int(smooth), min_periods=1).mean().values

    be = int(d.loc[d["va_macro_f1"].idxmax(), "epoch"]) if "va_macro_f1" in d else int(d["epoch"].iloc[-1])

    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, S(d["tr_loss"]), label="train loss"); plt.plot(ep, S(d["va_loss"]), label="val loss")
    plt.axvline(be, linestyle="--", alpha=0.6, label=f"best epoch={be}")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss vs Epochs"); plt.legend(); plt.grid(alpha=0.2); plt.show()

    if "tr_acc" in d:
        plt.figure(figsize=(7, 4.5))
        plt.plot(ep, S(d["tr_acc"]), label="train acc"); plt.plot(ep, S(d["va_acc"]), label="val acc")
        plt.axvline(be, linestyle="--", alpha=0.6)
        plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy vs Epochs"); plt.legend(); plt.grid(alpha=0.2); plt.show()

    if "va_macro_f1" in d:
        plt.figure(figsize=(7, 4.5))
        plt.plot(ep, d["va_macro_f1"], label="val macro-F1")
        plt.axvline(be, linestyle="--", alpha=0.6)
        plt.xlabel("epoch"); plt.ylabel("macro-F1"); plt.title("Validation Macro-F1"); plt.legend(); plt.grid(alpha=0.2); plt.show()
