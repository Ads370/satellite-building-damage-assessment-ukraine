import os
import csv
import random
import numpy as np
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return device


def init_csv_log(path, seg=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if seg:
            w.writerow(["epoch", "train_loss",
                        "val_IoU@0.5", "val_Dice@0.5", "val_Prec@0.5", "val_Rec@0.5", "val_Acc@0.5",
                        "sweep_t_best", "sweep_IoU_best"])
        else:
            w.writerow(["epoch", "tr_loss", "tr_acc", "va_loss", "va_acc", "va_macro_f1"])


def denorm_batch(imgs_batch: torch.Tensor) -> np.ndarray:
    x = imgs_batch.detach().cpu().float().numpy()
    mean = IMAGENET_MEAN.reshape(1, 3, 1, 1)
    std  = IMAGENET_STD.reshape(1, 3, 1, 1)
    x = np.clip(x * std + mean, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8).transpose(0, 2, 3, 1)


# ---- Segmentation losses / metrics ----

def calculate_dataset_pos_weight(train_loader, max_batches: int = 50) -> torch.Tensor:
    pos = neg = 0
    for i, (_, masks) in enumerate(train_loader):
        if i >= max_batches:
            break
        masks = masks.float()
        pos += masks.sum().item()
        neg += masks.numel() - masks.sum().item()
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(min(neg / pos, 15.0))


def dice_loss_with_logits(logits, targets, smooth: float = 1e-6):
    probs = torch.sigmoid(logits).contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    inter = (probs * targets).sum()
    return 1.0 - (2.0 * inter + smooth) / (probs.sum() + targets.sum() + smooth)


def bce_dice_loss(logits, targets, pos_weight_tensor, dice_w: float = 0.5):
    import torch.nn.functional as F
    bce  = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight_tensor)
    dice = dice_loss_with_logits(logits, targets)
    return (1.0 - dice_w) * bce + dice_w * dice


def compute_batch_metrics(logits, masks, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    tp = (preds * masks).sum().item()
    tn = ((1 - preds) * (1 - masks)).sum().item()
    fp = (preds * (1 - masks)).sum().item()
    fn = ((1 - preds) * masks).sum().item()
    return dict(
        IoU=tp / (tp + fp + fn + 1e-8),
        Dice=2 * tp / (2 * tp + fp + fn + 1e-8),
        Precision=tp / (tp + fp + 1e-8),
        Recall=tp / (tp + fn + 1e-8),
        Acc=(tp + tn) / (tp + tn + fp + fn + 1e-8),
    )


# ---- Classification helpers ----

def effective_alpha(counts: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    counts = counts.astype(np.float64)
    eff = 1.0 - np.power(beta, counts)
    a = (1.0 - beta) / np.maximum(eff, 1e-12)
    return a / a.sum() * len(counts)


def clean_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."): k = k[len("_orig_mod."):]
        if k.startswith("module."):    k = k[len("module."):]
        out[k] = v
    return out
