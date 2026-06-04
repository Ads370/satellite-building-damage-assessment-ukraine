"""
Training loops for segmentation (XBD and Ukraine) and classification (XBD + fine-tune on Ukraine).
"""

import csv
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, confusion_matrix

from utils import (bce_dice_loss, compute_batch_metrics, calculate_dataset_pos_weight,
                   init_csv_log, set_seed, effective_alpha, clean_state_dict)
from models import ResUNet, SiameseResNet, EarlyFusionResNet, FocalLoss, _DummyScaler
from dataset import (SplitConfig, split_manifest, XBDDatasetFixed,
                     SiamesePairsPaired, SiamesePairs,
                     PairAug, EvalAug, PairTrainTF, PairEvalTF,
                     make_val_transform_xbd, make_val_transform_ukr,
                     make_splits, split_by_fold, make_loaders, effective_number_weights)
from eval import best_threshold_on_val


# ============================================================
# Segmentation: shared helper
# ============================================================

def _postproc_batch(preds_bin_np: np.ndarray, min_component: int = 64) -> np.ndarray:
    import cv2
    out = np.zeros_like(preds_bin_np)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i in range(preds_bin_np.shape[0]):
        m = preds_bin_np[i, 0].astype(np.uint8)
        if m.sum() == 0:
            out[i, 0] = m; continue
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m)
        for lab in range(1, num):
            if stats[lab, cv2.CC_STAT_AREA] >= min_component:
                keep[labels == lab] = 1
        out[i, 0] = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel, iterations=1)
    return out


# ============================================================
# Segmentation training (works for both XBD and Ukraine)
# ============================================================

def train_single_manifest(
    manifest_csv: str,
    build_loaders_fn,           # inject build_loaders_xbd or build_loaders_ukr
    device: torch.device,
    run_dir: Path,
    split_cfg: SplitConfig = None,
    backbone: str = "resnet34",
    pretrained: bool = True,
    epochs: int = 40,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 5e-5,
    num_workers: int = 2,
    init_ckpt: str = None,
    freeze_encoder_epochs: int = 0,
    patience: int = 8,
    sweep_tmin: float = 0.75,
    sweep_tmax: float = 0.95,
    sweep_steps: int = 41,
    eval_postproc: bool = False,
    min_component: int = 64,
):
    if split_cfg is None:
        split_cfg = SplitConfig()

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_csv  = run_dir / "train_log.csv"
    ckpt_path = str(run_dir / "best_model.pth")

    df = pd.read_csv(manifest_csv)
    train_df, val_df = split_manifest(df, split_cfg)
    print(f"[data] Train: {len(train_df)} | Val: {len(val_df)}")

    model = ResUNet(n_classes=1, backbone=backbone, pretrained=pretrained).to(device)

    if init_ckpt and Path(init_ckpt).is_file():
        state = torch.load(init_ckpt, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"[init] loaded weights from {init_ckpt}")

    def set_encoder_trainable(flag: bool):
        for m in [model.input_layer, model.layer1, model.layer2, model.layer3, model.layer4]:
            for p in m.parameters():
                p.requires_grad = flag

    if freeze_encoder_epochs > 0:
        set_encoder_trainable(False)
        print(f"[ft] encoder frozen for first {freeze_encoder_epochs} epoch(s)")

    train_loader, val_loader = build_loaders_fn(train_df, val_df, device,
                                                batch_size=batch_size, num_workers=num_workers)
    init_csv_log(log_csv, seg=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    pos_weight = calculate_dataset_pos_weight(train_loader, max_batches=50).to(device)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    best_iou, best_epoch, epochs_no_improve = 0.0, 0, 0

    for epoch in range(epochs):
        if freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs:
            set_encoder_trainable(True)
            print("[ft] encoder unfrozen")

        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}")
        for imgs, masks in pbar:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(autocast_device):
                logits = model(imgs)
                loss   = bce_dice_loss(logits, masks, pos_weight_tensor=pos_weight, dice_w=0.5)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        thr, thr_iou = best_threshold_on_val(model, val_loader, device,
                                             t_min=sweep_tmin, t_max=sweep_tmax, steps=sweep_steps)

        model.eval()
        metrics_sum_05 = {k: 0.0 for k in ["IoU", "Dice", "Precision", "Recall", "Acc"]}
        tp = fp = fn = tn = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True).float()
                logits = model(imgs)
                bm = compute_batch_metrics(logits, masks, threshold=0.5)
                for k, v in bm.items():
                    metrics_sum_05[k] += v * imgs.size(0)
                preds = (torch.sigmoid(logits) > thr).float()
                if eval_postproc:
                    preds_np = preds.detach().cpu().numpy().astype(np.uint8)
                    preds = torch.from_numpy(_postproc_batch(preds_np, min_component)).to(masks.device).float()
                pf = preds.view(preds.size(0), -1); mf = masks.view(masks.size(0), -1)
                tp += (pf * mf).sum().item(); fp += (pf * (1 - mf)).sum().item()
                fn += ((1 - pf) * mf).sum().item(); tn += ((1 - pf) * (1 - mf)).sum().item()

        N = len(val_loader.dataset)
        vm05 = {k: metrics_sum_05[k] / max(1, N) for k in metrics_sum_05}
        eps  = 1e-8
        iou_t = tp / (tp + fp + fn + eps)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val IoU@0.5: {vm05['IoU']:.4f} | "
              f"Dice: {vm05['Dice']:.4f} | Sweep t={thr:.2f} IoU={thr_iou:.4f} | @t IoU={iou_t:.4f}")

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, train_loss, vm05["IoU"], vm05["Dice"],
                                    vm05["Precision"], vm05["Recall"], vm05["Acc"], thr, thr_iou])

        if thr_iou > best_iou:
            best_iou, best_epoch = thr_iou, epoch + 1
            torch.save(model.state_dict(), ckpt_path)
            print(f" → New best IoU: {best_iou:.4f} at epoch {best_epoch} | Saved: {ckpt_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print(f"[early-stop] no improvement for {patience} epochs; stopping.")
                break

    print(f"Training complete. Best IoU: {best_iou:.4f} at epoch {best_epoch}")
    return model, train_loader, val_loader, best_iou, best_epoch


# ============================================================
# Classification training (Phase 1: XBD crops)
# ============================================================

def _run_epoch_cls(model, loader, optimizer, criterion, scaler, device,
                   train=True, accum_steps=1, desc="Train", tta_fn=None, max_steps=None):
    if train:
        model.train(); optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()
    loss_sum = correct = seen = 0
    preds, trues = [], []
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
    for step, (pre, post, y) in enumerate(pbar, 1):
        pre  = pre.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        post = post.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y    = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(pre, post) if (train or tta_fn is None) else tta_fn(model, pre, post)
            loss   = criterion(logits, y)
            if train and accum_steps > 1:
                loss = loss / accum_steps
        if train:
            scaler.scale(loss).backward()
            if step % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
        bs = y.size(0)
        loss_sum += (loss.item() * (accum_steps if (train and accum_steps > 1) else 1)) * bs
        pred = logits.argmax(1); correct += (pred == y).sum().item(); seen += bs
        if not train:
            preds.append(pred.cpu()); trues.append(y.cpu())
        pbar.set_postfix(loss=f"{loss_sum/max(1,seen):.4f}", acc=f"{correct/max(1,seen):.3f}")
        if max_steps and step >= max_steps:
            break
    acc = correct / max(1, seen)
    if train:
        return loss_sum / max(1, seen), acc, None, None, None
    y_true = torch.cat(trues).numpy() if trues else np.array([])
    y_pred = torch.cat(preds).numpy() if preds else np.array([])
    macro_f1 = f1_score(y_true, y_pred, average="macro") if len(y_true) else 0.0
    return loss_sum / max(1, seen), acc, macro_f1, y_true, y_pred


def _save_state(path, epoch, model, best_f1, cfg):
    torch.save({"epoch": epoch, "best_f1": best_f1, "cfg": cfg,
                "model": getattr(model, "_orig_mod", model).state_dict()}, path)


def _load_state(path, model):
    chk = torch.load(path, map_location="cpu")
    model.load_state_dict(chk.get("model", chk), strict=False)
    return chk


def tta_forward_4flip(model, pre, post):
    """4-view TTA: identity + hflip + vflip + hvflip."""
    outs = [model(pre, post)]
    outs.append(model(torch.flip(pre, [-1]), torch.flip(post, [-1])))
    outs.append(model(torch.flip(pre, [-2]), torch.flip(post, [-2])))
    ph = torch.flip(pre, [-1]); qh = torch.flip(post, [-1])
    outs.append(model(torch.flip(ph, [-2]), torch.flip(qh, [-2])))
    return torch.stack(outs, 0).mean(0)


def train_one_run(run_dir: Path, cfg: dict, device: torch.device, splits=None):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    CKPT  = run_dir / "artifacts" / "best.pth"
    STATE = run_dir / "artifacts" / "state.pt"
    LOG   = run_dir / "artifacts" / "log.csv"

    set_seed(cfg["SEED"])

    if splits is not None:
        df_tr, df_va, df_te = splits
    else:
        df_tr, df_va, df_te = make_splits(cfg["MANIFEST"], seed=cfg["SEED"])

    train_pair_tf = PairAug(size=cfg["INPUT_SIZE"])
    eval_tf       = EvalAug(size=cfg["INPUT_SIZE"])
    loader_tr, loader_va, loader_te, _ = make_loaders(
        df_tr, df_va, df_te, train_pair_tf, eval_tf,
        batch=cfg["BATCH_SIZE"], num_workers=cfg["NUM_WORKERS"],
        use_tempered=cfg.get("USE_TEMPERED_SAMPLER", False),
        temper_exp=cfg.get("TEMPER_EXP", 0.5),
        num_classes=cfg["NUM_CLASSES"],
    )
    counts, wts = effective_number_weights(df_tr, cfg["NUM_CLASSES"])
    print(f"Splits — train:{len(df_tr)} val:{len(df_va)} test:{len(df_te)} | "
          f"counts:{counts.tolist()} | eff-wts:{[round(x,3) for x in wts]}")

    model = SiameseResNet(cfg["BACKBONE"], pretrained=True,
                          num_classes=cfg["NUM_CLASSES"], dropout=cfg["DROPOUT"])
    model = model.to(device, memory_format=torch.channels_last)
    try:
        model = torch.compile(model)
    except Exception:
        pass

    for p in model.encoder.parameters():
        p.requires_grad = False

    criterion  = FocalLoss(gamma=cfg["FOCAL_GAMMA"])
    accum_steps = max(1, cfg["TARGET_EFF_BS"] // cfg["BATCH_SIZE"])
    optimizer   = torch.optim.AdamW(
        [{"params": model.head.parameters(), "lr": cfg["LR_HEAD"]}],
        weight_decay=cfg["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["EPOCHS"])
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else _DummyScaler()

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    best_val_f1, no_improve = -1.0, 0
    history = {k: [] for k in ["train_loss","train_acc","val_loss","val_acc","val_f1_macro"]}
    class_names = ["no-damage","minor","major","destroyed"]

    for ep in range(1, cfg["EPOCHS"] + 1):
        if ep == cfg["FREEZE_EPOCHS"] + 1:
            for p in model.encoder.parameters():
                p.requires_grad = False
            if cfg.get("UNFREEZE_POLICY", "layer4") == "layer4":
                enc_children = list(model.encoder.children())
                layer4_mod = next((enc_children[k] for k in range(len(enc_children)-1, -1, -1)
                                   if sum(p.numel() for p in enc_children[k].parameters()) > 0), None)
                if layer4_mod is None:
                    raise RuntimeError("Could not locate a param-bearing block in encoder.")
                for p in layer4_mod.parameters():
                    p.requires_grad = True
            else:
                for p in model.encoder.parameters():
                    p.requires_grad = True
            enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
            if enc_params:
                optimizer = torch.optim.AdamW(
                    [{"params": enc_params, "lr": cfg["LR_HEAD"] * cfg["ENC_MULT"]},
                     {"params": model.head.parameters(), "lr": cfg["LR_HEAD"]}],
                    weight_decay=cfg["WEIGHT_DECAY"])
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=(cfg["EPOCHS"] - ep + 1))
            print(f">> Encoder unfrozen policy={cfg.get('UNFREEZE_POLICY','layer4')}")

        tr_loss, tr_acc, *_ = _run_epoch_cls(
            model, loader_tr, optimizer, criterion, scaler, device, train=True,
            accum_steps=accum_steps, desc=f"Train {ep}/{cfg['EPOCHS']}",
            max_steps=cfg.get("MAX_TRAIN_STEPS"))

        va_loss, va_acc, va_f1, va_true, va_pred = _run_epoch_cls(
            model, loader_va, optimizer=None, criterion=criterion, scaler=scaler,
            device=device, train=False, desc=f"Val {ep}/{cfg['EPOCHS']}",
            tta_fn=tta_forward_4flip if cfg.get("USE_TTA_VAL") else None,
            max_steps=cfg.get("MAX_VAL_STEPS"))
        scheduler.step()

        for k, v in zip(["train_loss","train_acc","val_loss","val_acc","val_f1_macro"],
                        [tr_loss, tr_acc, va_loss, va_acc, va_f1]):
            history[k].append(v)

        row = dict(epoch=ep, tr_loss=tr_loss, tr_acc=tr_acc,
                   va_loss=va_loss, va_acc=va_acc, va_macro_f1=va_f1)
        pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        print(f"Epoch {ep:02d} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f} | F1 {va_f1:.3f}")

        if va_f1 > best_val_f1 + 1e-4:
            best_val_f1 = va_f1; no_improve = 0
            torch.save(model.state_dict(), CKPT)
            print(f"  saved BEST (val macro-F1={va_f1:.4f}) -> {CKPT}")
        else:
            no_improve += 1
            if no_improve >= cfg["PATIENCE"]:
                print(f"Early stopping (no improvement for {cfg['PATIENCE']} epochs).")
                break

        _save_state(STATE, ep, model, best_val_f1, cfg)

    if CKPT.exists():
        model.load_state_dict(torch.load(CKPT, map_location=device))

    te_loss, te_acc, te_f1, te_true, te_pred = _run_epoch_cls(
        model, loader_te, optimizer=None, criterion=criterion, scaler=scaler,
        device=device, train=False, desc="Test",
        tta_fn=tta_forward_4flip if cfg.get("USE_TTA_TEST") else None)
    print(f"[TEST] loss={te_loss:.4f} | acc={te_acc:.3f} | macro-F1={te_f1:.3f}")
    print(classification_report(te_true, te_pred, target_names=class_names, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(te_true, te_pred))

    return dict(best_val_f1=float(best_val_f1), test_f1=float(te_f1),
                test_acc=float(te_acc), ckpt=str(CKPT), config=cfg)


# ============================================================
# Classification fine-tuning (Phase 2: Ukraine / Kolega)
# ============================================================

def tta_forward_8view(model, pre, post):
    """8-view TTA: 4 rotations × hflip."""
    outs = []
    for k in range(4):
        pr = pre.rot90(k, dims=(-2, -1));  po = post.rot90(k, dims=(-2, -1))
        outs.append(model(pr, po))
        outs.append(model(pr.flip(-1), po.flip(-1)))
    return torch.stack(outs, 0).mean(0)


def _run_epoch_ft(model, loader, optimizer, criterion, device,
                  train=True, accum_steps=1, desc="Train", tta=False, bias=None):
    if train:
        model.train(); optimizer.zero_grad(set_to_none=True)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    else:
        model.eval(); scaler = None

    loss_sum = correct = seen = 0
    preds, trues = [], []
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
    for step, (pre, post, y) in enumerate(pbar, 1):
        pre, post, y = pre.to(device), post.to(device), y.to(device)
        if train:
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(pre, post)
                loss   = criterion(logits, y) / accum_steps
            scaler.scale(loss).backward()
            if step % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            with torch.inference_mode(), torch.autocast("cuda", enabled=(device.type == "cuda")):
                logits = tta_forward_8view(model, pre, post) if tta else model(pre, post)
                if bias is not None:
                    logits = logits + bias.to(logits.device)
                loss = criterion(logits, y)
        bs = y.size(0)
        loss_sum += loss.item() * (accum_steps if train else 1) * bs
        pred = logits.argmax(1); correct += (pred == y).sum().item(); seen += bs
        if not train:
            preds.append(pred.cpu()); trues.append(y.cpu())
        pbar.set_postfix(loss=f"{loss_sum/max(1,seen):.4f}", acc=f"{correct/max(1,seen):.3f}")

    acc = correct / max(1, seen)
    if train:
        return loss_sum / max(1, seen), acc, None
    y_true = torch.cat(trues).numpy() if trues else np.array([])
    y_pred = torch.cat(preds).numpy() if preds else np.array([])
    return loss_sum / max(1, seen), acc, f1_score(y_true, y_pred, average="macro") if len(y_true) else 0.0


def _find_layer4(encoder: nn.Module):
    kids = list(encoder.children())
    for k in range(len(kids) - 1, -1, -1):
        if sum(p.numel() for p in kids[k].parameters()) > 0:
            return kids[k]
    return None


def bn_reestimate(model, loader, device, max_batches: int = 200):
    was = model.training; model.train()
    with torch.no_grad():
        for i, (pre, post, _) in enumerate(loader):
            if i >= max_batches: break
            model(pre.to(device), post.to(device))
    model.train(was)


def finetune_on_kolega(
    pretrain_ckpt: Path,
    kolega_root: Path,
    ft_dir: Path,
    device: torch.device,
    cfg,                  # SimpleNamespace from notebook
    build_manifest_fn,    # build_kolega_cls_manifest
):
    """Phase-2 fine-tune of a pre-trained SiameseResNet on the Ukraine dataset."""
    from dataset import SiamesePairs, PairTrainTF, PairEvalTF, split_by_fold
    from torch.utils.data import DataLoader
    import torch.nn as nn

    ft_dir = Path(ft_dir); ft_dir.mkdir(parents=True, exist_ok=True)
    ft_ckpt  = ft_dir / "best_ft.pth"
    log_csv  = ft_dir / "log.csv"

    m = build_manifest_fn(kolega_root)
    df_tr, df_va, df_te = split_by_fold(m, val_fold=cfg.val_fold, test_fold=cfg.test_fold)
    print("TRAIN counts:", np.bincount(df_tr.label_id, minlength=4))
    print("VAL   counts:", np.bincount(df_va.label_id, minlength=4))
    print("TEST  counts:", np.bincount(df_te.label_id, minlength=4))

    tf_train = PairTrainTF(cfg.input_size, scale=cfg.crop_scale,
                           use_jitter=cfg.use_paired_jitter)
    tf_eval  = PairEvalTF(cfg.input_size)

    loader_tr = DataLoader(SiamesePairs(df_tr, tf_train), batch_size=cfg.batch_train,
                           shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    loader_va = DataLoader(SiamesePairs(df_va, tf_eval),  batch_size=cfg.batch_val,
                           shuffle=False, num_workers=0, pin_memory=True)
    loader_te = DataLoader(SiamesePairs(df_te, tf_eval),  batch_size=cfg.batch_test,
                           shuffle=False, num_workers=0, pin_memory=True)

    train_counts = np.bincount(df_tr.label_id.values, minlength=4)
    alpha_vec    = effective_alpha(train_counts)
    print("alpha:", np.round(alpha_vec, 3))

    model = SiameseResNet(cfg.backbone, pretrained=False, num_classes=cfg.num_classes, dropout=0.4)
    model = model.to(device, memory_format=torch.channels_last)
    state = torch.load(pretrain_ckpt, map_location=device)
    state = state.get("model", state.get("state_dict", state))
    try:
        model.load_state_dict(clean_state_dict(state), strict=True)
    except RuntimeError as e:
        print("[warn] strict load failed, trying non-strict:", e)
        model.load_state_dict(clean_state_dict(state), strict=False)
    print("Loaded pretrain:", Path(pretrain_ckpt).name)

    criterion_ce    = nn.CrossEntropyLoss(label_smoothing=0.03)
    criterion_focal = FocalLoss(alpha=alpha_vec, gamma=cfg.focal_gamma)

    for p in model.encoder.parameters(): p.requires_grad = False
    optimizer = torch.optim.AdamW([{"params": model.head.parameters(), "lr": cfg.lr_head}],
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_f1, no_improve = -1.0, 0

    for ep in range(1, cfg.epochs + 1):
        if ep == cfg.warmup_epochs + 1:
            bn_reestimate(model, loader_tr, device)
            for p in model.encoder.parameters(): p.requires_grad = False
            layer4 = _find_layer4(model.encoder)
            assert layer4 is not None, "Could not locate layer4"
            for p in layer4.parameters(): p.requires_grad = True
            optimizer = torch.optim.AdamW(
                [{"params": layer4.parameters(), "lr": cfg.lr_head * cfg.enc_mult},
                 {"params": model.head.parameters(), "lr": cfg.lr_head}],
                weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(cfg.epochs - ep + 1))
            print(">> Unfrozen: layer4 only. BN re-estimated.")

        criterion = criterion_ce if ep <= cfg.warmup_epochs else criterion_focal
        tr_loss, tr_acc, _ = _run_epoch_ft(model, loader_tr, optimizer, criterion, device,
                                           train=True, desc=f"Train {ep}/{cfg.epochs}")
        va_loss, va_acc, va_f1 = _run_epoch_ft(model, loader_va, optimizer=None, criterion=criterion,
                                                device=device, train=False, desc=f"Val {ep}/{cfg.epochs}",
                                                tta=cfg.use_tta_val)
        scheduler.step()

        row = dict(epoch=ep, tr_loss=tr_loss, tr_acc=tr_acc,
                   va_loss=va_loss, va_acc=va_acc, va_macro_f1=va_f1)
        pd.DataFrame([row]).to_csv(log_csv, mode="a", header=not log_csv.exists(), index=False)
        print(f"Epoch {ep:02d} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f} | F1 {va_f1:.3f}")

        if va_f1 > best_f1 + 1e-4:
            best_f1 = va_f1; no_improve = 0
            torch.save(getattr(model, "_orig_mod", model).state_dict(), ft_ckpt)
            print(f"  saved BEST (val macro-F1={va_f1:.4f}) -> {ft_ckpt}")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stopping ({cfg.patience} epochs no improvement)."); break

    if ft_ckpt.exists():
        model.load_state_dict(torch.load(ft_ckpt, map_location=device))

    te_loss, te_acc, te_f1 = _run_epoch_ft(model, loader_te, optimizer=None,
                                            criterion=criterion_focal, device=device,
                                            train=False, desc="Test (TTA)", tta=cfg.use_tta_test)
    print(f"[TEST TTA] loss={te_loss:.4f} | acc={te_acc:.3f} | macro-F1={te_f1:.3f}")
    return model, ft_ckpt, log_csv
