"""
Dataset classes, transforms, data loaders, and manifest builders
for both the segmentation (XBD / Ukraine) and classification pipelines.
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import cv2
import torch
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations import CropNonEmptyMaskIfExists
from albumentations.pytorch import ToTensorV2



# -----Shared split config & manifest splitter-----

@dataclass
class SplitConfig:
    split_col:    str   = "split"
    train_tag:    str   = "train"
    val_tag:      str   = "val"
    val_fraction: float = 0.1


def split_manifest(df: pd.DataFrame, cfg: SplitConfig):
    assert {"pre_path", "mask_path"}.issubset(df.columns), \
        "manifest must contain columns: pre_path, mask_path"
    if cfg.split_col in df.columns:
        train_df = df[df[cfg.split_col].astype(str).str.lower() == cfg.train_tag].copy()
        val_df   = df[df[cfg.split_col].astype(str).str.lower() == cfg.val_tag].copy()
        if len(val_df) > 0:
            return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
        print("[split] No val rows found; falling back to random split.")
    rng   = np.random.RandomState(1337)
    perm  = rng.permutation(len(df))
    n_val = max(1, int(len(df) * cfg.val_fraction))
    val_idx   = set(perm[:n_val])
    train_idx = set(range(len(df))) - val_idx
    return df.iloc[list(train_idx)].reset_index(drop=True), df.iloc[list(val_idx)].reset_index(drop=True)


#------Segmentation dataset------


class XBDDatasetFixed(Dataset):
    """Loads pre-disaster image + binary mask; supports CNEMIE crop strategy."""

    def __init__(self, frame: pd.DataFrame, transform,
                 transform_empty_fallback=None, bg_prob: float = 0.30):
        self.df = frame.reset_index(drop=True)
        self.transform = transform
        self.transform_empty = transform_empty_fallback
        self.bg_prob = float(bg_prob)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["pre_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {row['pre_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        msk = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise FileNotFoundError(f"Mask not found: {row['mask_path']}")
        msk = (msk > 127).astype(np.uint8)

        has_fg = bool(msk.any())
        if self.transform_empty is None:
            tfm = self.transform
        else:
            tfm = self.transform if (has_fg and random.random() >= self.bg_prob) else self.transform_empty

        aug = tfm(image=img, mask=msk)
        img_t, msk_t = aug["image"], aug["mask"]

        if isinstance(msk_t, np.ndarray):
            msk_t = torch.from_numpy(msk_t)
        if msk_t.ndim == 2:
            msk_t = msk_t.unsqueeze(0).float()
        else:
            msk_t = msk_t.float()

        return img_t, msk_t



#-----Segmentation transforms-----


IMG_SIZE_XBD = 512
IMG_SIZE_UKR = 1024

_NORM = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def make_train_transform_xbd():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=15,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.GridDistortion(num_steps=4, distort_limit=0.07, p=0.2),
        A.PadIfNeeded(IMG_SIZE_XBD, IMG_SIZE_XBD, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        CropNonEmptyMaskIfExists(IMG_SIZE_XBD, IMG_SIZE_XBD, p=1.0),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
        A.HueSaturationValue(10, 15, 10, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(**_NORM), ToTensorV2(),
    ])


def make_train_transform_empty_fallback_xbd():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=15,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.GridDistortion(num_steps=4, distort_limit=0.07, p=0.2),
        A.PadIfNeeded(IMG_SIZE_XBD, IMG_SIZE_XBD, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        A.CenterCrop(IMG_SIZE_XBD, IMG_SIZE_XBD),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
        A.HueSaturationValue(10, 15, 10, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(**_NORM), ToTensorV2(),
    ])


def make_val_transform_xbd():
    return A.Compose([
        A.Resize(IMG_SIZE_XBD, IMG_SIZE_XBD),
        A.Normalize(**_NORM), ToTensorV2(),
    ])


def make_train_transform_ukr():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.GridDistortion(num_steps=4, distort_limit=0.07, p=0.1),
        A.PadIfNeeded(IMG_SIZE_UKR, IMG_SIZE_UKR, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        CropNonEmptyMaskIfExists(IMG_SIZE_UKR, IMG_SIZE_UKR, p=1.0),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
        A.HueSaturationValue(8, 12, 8, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(**_NORM), ToTensorV2(),
    ])


def make_train_transform_empty_fallback_ukr():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.GridDistortion(num_steps=4, distort_limit=0.07, p=0.1),
        A.PadIfNeeded(IMG_SIZE_UKR, IMG_SIZE_UKR, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        A.CenterCrop(IMG_SIZE_UKR, IMG_SIZE_UKR),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
        A.HueSaturationValue(8, 12, 8, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(**_NORM), ToTensorV2(),
    ])


def make_val_transform_ukr():
    return A.Compose([
        A.Resize(IMG_SIZE_UKR, IMG_SIZE_UKR),
        A.Normalize(**_NORM), ToTensorV2(),
    ])



# -----Segmentation loaders-----


def build_loaders_xbd(train_df, val_df, device, batch_size: int = 8, num_workers: int = 4):
    train_ds = XBDDatasetFixed(train_df, make_train_transform_xbd(),
                               make_train_transform_empty_fallback_xbd(), bg_prob=0.40)
    val_ds   = XBDDatasetFixed(val_df, make_val_transform_xbd(), bg_prob=0.0)
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin, drop_last=False)
    return train_loader, val_loader


def build_loaders_ukr(train_df, val_df, device, batch_size: int = 4, num_workers: int = 2):
    train_ds = XBDDatasetFixed(train_df, make_train_transform_ukr(),
                               make_train_transform_empty_fallback_ukr(), bg_prob=0.30)
    val_ds   = XBDDatasetFixed(val_df, make_val_transform_ukr(), bg_prob=0.0)
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin, drop_last=False)
    return train_loader, val_loader



# -----Segmentation manifest builders-----


def build_xbd_manifest(images_dir: str, masks_dir: str, out_csv: str, val_frac: float = 0.20):
    """Build manifest.csv pairing pre-disaster images with their binary masks."""
    def which_scene(stem):
        s = stem.lower()
        if "_pre_disaster" in s:  return "pre"
        if "_post_disaster" in s: return "post"
        return None

    def norm_stem(stem):
        for suf in ["_mask", "_target", "_bin", "_binary", "_building", "_buildings"]:
            if stem.endswith(suf):
                stem = stem[:-len(suf)]
        return stem

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    mask_files  = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

    mask_index = {}
    for mp in mask_files:
        s = Path(mp).stem
        mask_index[s] = mp
        mask_index[norm_stem(s)] = mp
        mask_index[s.replace("_pre_disaster", "").replace("_post_disaster", "")] = mp

    pairs_pre = []
    for ip in image_files:
        stem = Path(ip).stem
        if which_scene(stem) != "pre":
            continue
        mp = (mask_index.get(stem)
              or mask_index.get(stem + "_b0_mask")
              or mask_index.get(norm_stem(stem))
              or mask_index.get(stem.replace("_pre_disaster", "")))
        if mp:
            pairs_pre.append((ip, mp))

    rng = random.Random(1337)
    rng.shuffle(pairs_pre)
    n_val   = max(1, int(val_frac * len(pairs_pre)))
    val_p   = pairs_pre[:n_val]
    train_p = pairs_pre[n_val:]

    df_tr = pd.DataFrame(train_p, columns=["pre_path", "mask_path"]); df_tr["split"] = "train"
    df_va = pd.DataFrame(val_p,   columns=["pre_path", "mask_path"]); df_va["split"] = "val"
    df = pd.concat([df_tr, df_va], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"[xBD manifest] train={len(df_tr)} | val={len(df_va)} -> {out_csv}")
    return df


def build_kolega_seg_manifest(root_dir: str, cities, split_map: dict, out_csv: str):
    """Build manifest for Ukraine (Kolega) segmentation dataset (.tif images + .png masks)."""
    rows = []
    for split, folds in split_map.items():
        for city in cities:
            for fold in folds:
                img_dir = os.path.join(root_dir, city, fold, "images")
                msk_dir = os.path.join(root_dir, city, fold, "targets")
                if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
                    print(f"[skip] missing: {img_dir} or {msk_dir}")
                    continue
                for ip in sorted(glob.glob(os.path.join(img_dir, "*.tif"))):
                    stem = Path(ip).stem
                    mp   = os.path.join(msk_dir, stem + ".png")
                    if os.path.exists(mp):
                        rows.append({"pre_path": ip, "mask_path": mp, "split": split,
                                     "city": city, "fold": fold})
    assert rows, f"No pairs found under {root_dir}"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[Kolega seg manifest] total={len(df)} | train={len(df[df.split=='train'])} "
          f"| val={len(df[df.split=='val'])} | test={len(df[df.split=='test'])} -> {out_csv}")
    return df



#-----Classification dataset & augmentations-----

LABEL_MAP = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}
CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]

_CLS_MEAN = [0.485, 0.456, 0.406]
_CLS_STD  = [0.229, 0.224, 0.225]


class CutOutTensor:
    """Random rectangular cutout applied to paired tensors."""

    def __init__(self, p=0.30, n_holes=(1, 3), side_px=(16, 38), max_area_frac=0.20,
                 fill="per_image_mean", paired=True, unpair_prob=0.20, avoid_center_frac=0.0):
        self.p = p; self.nh = n_holes; self.side = side_px
        self.max_area_frac = max_area_frac; self.fill = fill
        self.paired = paired; self.unpair_prob = unpair_prob
        self.avoid_center_frac = avoid_center_frac

    def _fill_value(self, x):
        if self.fill == "random":
            return torch.rand((x.size(0), 1, 1), device=x.device)
        return x.mean(dim=(1, 2), keepdim=True)

    def _make_masks(self, H, W, device, n_holes):
        masks, used_area = [], 0
        max_area = self.max_area_frac * H * W
        cy0 = cx0 = 0; cy1, cx1 = H, W
        if self.avoid_center_frac and 0 < self.avoid_center_frac < 1:
            frac = self.avoid_center_frac
            h_box = int(H * frac); w_box = int(W * frac)
            cy0 = (H - h_box) // 2; cy1 = cy0 + h_box
            cx0 = (W - w_box) // 2; cx1 = cx0 + w_box
        for _ in range(n_holes):
            s = random.randint(self.side[0], self.side[1])
            if used_area + s * s > max_area:
                break
            for attempt in range(11):
                cy = random.randint(0, H - 1); cx = random.randint(0, W - 1)
                if not (self.avoid_center_frac and (cy0 <= cy < cy1 and cx0 <= cx < cx1)):
                    break
            y0 = max(0, cy - s // 2); y1 = min(H, y0 + s)
            x0 = max(0, cx - s // 2); x1 = min(W, x0 + s)
            used_area += (y1 - y0) * (x1 - x0)
            masks.append((y0, y1, x0, x1))
        return masks

    def _apply_masks(self, x, masks, fill_val):
        for (y0, y1, x0, x1) in masks:
            x[:, y0:y1, x0:x1] = fill_val
        return x

    def __call__(self, *imgs):
        if random.random() > self.p:
            return imgs if len(imgs) > 1 else imgs[0]
        H, W = imgs[0].shape[-2:]
        n_holes = random.randint(self.nh[0], self.nh[1])
        if len(imgs) == 2 and self.paired and random.random() > self.unpair_prob:
            masks = self._make_masks(H, W, imgs[0].device, n_holes)
            out = [self._apply_masks(img.clone(), masks, self._fill_value(img)) for img in imgs]
            return tuple(out)
        out = [self._apply_masks(img.clone(), self._make_masks(H, W, img.device, n_holes),
                                 self._fill_value(img)) for img in imgs]
        return tuple(out) if len(out) > 1 else out[0]


class PairAug:
    """Paired training augmentation for classification (shared crop + optional CutOut)."""

    def __init__(self, size, scale=(0.8, 1.0), hflip_p=0.5, vflip_p=0.2, cutout_kwargs=None):
        self.size = size; self.scale = scale; self.hflip_p = hflip_p; self.vflip_p = vflip_p
        if cutout_kwargs is None:
            cutout_kwargs = dict(p=0.30, n_holes=(1, 3),
                                 side_px=(16, 38) if size >= 256 else (12, 32),
                                 max_area_frac=0.20, fill="per_image_mean",
                                 paired=True, unpair_prob=0.20, avoid_center_frac=0.0)
        self.cutout = CutOutTensor(**cutout_kwargs)

    def __call__(self, pre_img, post_img):
        i, j, h, w = T.RandomResizedCrop.get_params(pre_img, scale=self.scale, ratio=(0.75, 1.3333333))
        pre  = TF.resized_crop(pre_img,  i, j, h, w, size=[self.size, self.size])
        post = TF.resized_crop(post_img, i, j, h, w, size=[self.size, self.size])
        if random.random() < self.hflip_p:
            pre = TF.hflip(pre); post = TF.hflip(post)
        if random.random() < self.vflip_p:
            pre = TF.vflip(pre); post = TF.vflip(post)
        pre  = TF.normalize(TF.to_tensor(pre),  _CLS_MEAN, _CLS_STD)
        post = TF.normalize(TF.to_tensor(post), _CLS_MEAN, _CLS_STD)
        pre, post = self.cutout(pre, post)
        return pre, post


class EvalAug:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = TF.resize(img, [self.size, self.size])
        return TF.normalize(TF.to_tensor(img), _CLS_MEAN, _CLS_STD)


class PairedColorJitter:
    def __init__(self, b=0.1, c=0.1, s=0.1, h=0.05):
        self.b, self.c, self.s, self.h = b, c, s, h

    def __call__(self, pre, post):
        fn_idx, b, c, s, h = T.ColorJitter.get_params(
            brightness=[max(0, 1 - self.b), 1 + self.b],
            contrast=[max(0, 1 - self.c), 1 + self.c],
            saturation=[max(0, 1 - self.s), 1 + self.s],
            hue=(-self.h, self.h),
        )
        def apply(img):
            for fn_id in fn_idx:
                if fn_id == 0: img = TF.adjust_brightness(img, b)
                elif fn_id == 1: img = TF.adjust_contrast(img, c)
                elif fn_id == 2: img = TF.adjust_saturation(img, s)
                elif fn_id == 3: img = TF.adjust_hue(img, h)
            return img
        return apply(pre), apply(post)


class PairTrainTF:
    """Paired transform for fine-tuning (Phase 2) with optional color jitter."""

    def __init__(self, size, scale=(0.9, 1.0), use_jitter=False, jb=0.1, jc=0.1, js=0.1, jh=0.05):
        self.size = size; self.scale = scale; self.use_jitter = use_jitter
        self.jitter = PairedColorJitter(jb, jc, js, jh) if use_jitter else None

    def __call__(self, pre_img, post_img):
        i, j, h, w = T.RandomResizedCrop.get_params(pre_img, scale=self.scale, ratio=(0.75, 1.3333))
        pre  = TF.resized_crop(pre_img,  i, j, h, w, size=[self.size, self.size])
        post = TF.resized_crop(post_img, i, j, h, w, size=[self.size, self.size])
        if random.random() < 0.5:
            pre = TF.hflip(pre); post = TF.hflip(post)
        if self.jitter is not None:
            pre, post = self.jitter(pre, post)
        pre  = TF.normalize(TF.to_tensor(pre),  _CLS_MEAN, _CLS_STD)
        post = TF.normalize(TF.to_tensor(post), _CLS_MEAN, _CLS_STD)
        return pre, post


class PairEvalTF:
    def __init__(self, size):
        self.size = size

    def __call__(self, pre_img, post_img):
        pre  = TF.normalize(TF.to_tensor(TF.resize(pre_img,  [self.size, self.size])), _CLS_MEAN, _CLS_STD)
        post = TF.normalize(TF.to_tensor(TF.resize(post_img, [self.size, self.size])), _CLS_MEAN, _CLS_STD)
        return pre, post


class SiamesePairsPaired(Dataset):
    """Classification dataset for XBD crops: returns (pre, post, label_id)."""

    def __init__(self, df: pd.DataFrame, train_pair_tf=None, eval_tf=None, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.train_pair_tf = train_pair_tf
        self.eval_tf = eval_tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        pre_img  = Image.open(r.pre_path).convert("RGB")
        post_img = Image.open(r.post_path).convert("RGB")
        if self.train and self.train_pair_tf is not None:
            pre, post = self.train_pair_tf(pre_img, post_img)
        else:
            pre  = self.eval_tf(pre_img)
            post = self.eval_tf(post_img)
        return pre, post, int(r.label_id)


class SiamesePairs(Dataset):
    """Classification dataset for fine-tuning (Phase 2): always uses a paired transform."""

    def __init__(self, df: pd.DataFrame, tf_pair):
        self.df = df.reset_index(drop=True)
        self.tf_pair = tf_pair

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        pre  = Image.open(r.pre_path).convert("RGB")
        post = Image.open(r.post_path).convert("RGB")
        pre, post = self.tf_pair(pre, post)
        return pre, post, int(r.label_id)



# -----Classification loaders / samplers-----


def make_splits(manifest: str, seed: int = 42):
    df = pd.read_csv(manifest)
    scenes = df.scene_core.unique()
    rng = np.random.default_rng(seed); rng.shuffle(scenes)
    n = len(scenes)
    n_tr = int(0.7 * n); n_val = int(0.15 * n)
    train_s = set(scenes[:n_tr]); val_s = set(scenes[n_tr:n_tr + n_val])
    test_s  = set(scenes[n_tr + n_val:])
    return (df[df.scene_core.isin(train_s)].reset_index(drop=True),
            df[df.scene_core.isin(val_s)].reset_index(drop=True),
            df[df.scene_core.isin(test_s)].reset_index(drop=True))


def split_by_fold(df: pd.DataFrame, val_fold: int = 0, test_fold: int = 1):
    tr = df[(df.fold != val_fold) & (df.fold != test_fold)].reset_index(drop=True)
    va = df[df.fold == val_fold].reset_index(drop=True)
    te = df[df.fold == test_fold].reset_index(drop=True)
    return tr, va, te


def make_tempered_sampler(df_tr: pd.DataFrame, num_classes: int, exponent: float = 0.5):
    counts = np.bincount(df_tr.label_id.values, minlength=num_classes).astype(float)
    w_class = np.power(counts / counts.sum(), -exponent)
    sample_w = w_class[df_tr.label_id.values]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True), counts, w_class


def effective_number_weights(df_tr: pd.DataFrame, num_classes: int):
    counts = np.array([(df_tr.label_id == k).sum() for k in range(num_classes)], dtype=np.float64)
    beta = 0.9999
    eff  = 1.0 - np.power(beta, counts)
    wts  = (1.0 - beta) / np.maximum(eff, 1e-8)
    return counts, wts / wts.sum() * num_classes


def make_loaders(df_tr, df_va, df_te, train_pair_tf, eval_tf, batch, num_workers,
                 use_tempered, temper_exp, num_classes):
    ds_tr = SiamesePairsPaired(df_tr, train_pair_tf=train_pair_tf, eval_tf=None,  train=True)
    ds_va = SiamesePairsPaired(df_va, train_pair_tf=None, eval_tf=eval_tf, train=False)
    ds_te = SiamesePairsPaired(df_te, train_pair_tf=None, eval_tf=eval_tf, train=False)
    kw = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        kw.update(prefetch_factor=2, persistent_workers=True)
    if use_tempered:
        sampler, counts, w = make_tempered_sampler(df_tr, num_classes, exponent=temper_exp)
        print(f"Tempered sampler: exp={temper_exp} | counts={counts.tolist()} | weights~={np.round(w,3).tolist()}")
        loader_tr = DataLoader(ds_tr, batch_size=batch, sampler=sampler, **kw)
    else:
        loader_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, **kw)
    loader_va = DataLoader(ds_va, batch_size=batch, shuffle=False, **kw)
    loader_te = DataLoader(ds_te, batch_size=batch, shuffle=False, **kw)
    return loader_tr, loader_va, loader_te, ds_te



# -----Classification manifest builder (Ukraine / Kolega)------


def build_kolega_cls_manifest(root: Path) -> pd.DataFrame:
    """Build classification manifest from Ukraine crop dataset (fold-based CSV structure)."""
    rows = []
    for city_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        city = city_dir.name
        for fold_dir in sorted([d for d in city_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")]):
            fold_idx = int(fold_dir.name.split("_")[1])
            csvs = sorted(fold_dir.glob("*.csv"))
            if not csvs:
                print(f"[WARN] No CSV in {fold_dir}"); continue
            df = pd.read_csv(csvs[0])
            pre_dir, post_dir, mask_dir = fold_dir / "pre", fold_dir / "post", fold_dir / "target"
            for _, r in df.iterrows():
                pre_name   = os.path.basename(str(r["pre_image"]))
                post_name  = os.path.basename(str(r["post_image"]))
                mask_name  = os.path.basename(str(r.get("mask_image", "")))
                label_txt  = str(r["damage"]).strip().lower().replace("_", "-")
                if label_txt not in LABEL_MAP:
                    continue
                pre_path  = pre_dir / pre_name
                post_path = post_dir / post_name
                if not pre_path.is_file() or not post_path.is_file():
                    continue
                rows.append({
                    "scene_core": city,
                    "fold": int(r.get("fold", fold_idx)),
                    "inst_id": pre_name.rsplit(".", 1)[0],
                    "pre_path":  str(pre_path),
                    "post_path": str(post_path),
                    "mask_path": str(mask_dir / mask_name) if mask_name else "",
                    "label_id":  int(LABEL_MAP[label_txt]),
                })
    if not rows:
        return pd.DataFrame(columns=["scene_core","fold","inst_id","pre_path","post_path","mask_path","label_id"])
    return pd.DataFrame(rows).reset_index(drop=True)
