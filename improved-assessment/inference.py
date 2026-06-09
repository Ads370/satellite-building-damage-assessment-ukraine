"""
Full inference pipeline: sliding-window TTA segmentation (TTASeg) + DenseCRF refinement
+ Siamese classifier per building instance → colour-coded damage overlay.
"""

import re
import gc
import math
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from models import ResUNet, SiameseResNet
from utils import clean_state_dict

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_COLORS = {0: (0, 255, 0), 1: (255, 255, 0), 2: (255, 165, 0), 3: (220, 20, 60)}
UNK_COLOR    = (220, 220, 220)

PRED_INDEX_TO_LABEL = {3: 0, 2: 1, 0: 2, 1: 3}


# ============================================================
# Utilities
# ============================================================

def parse_idx(p):
    nums = re.findall(r"\d+", Path(p).stem)
    return tuple(int(n) for n in nums) if nums else (float("inf"), Path(p).stem.lower())


def read_rgb(p: Path) -> np.ndarray:
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(p)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_to(img: np.ndarray, size_hw) -> np.ndarray:
    return cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)


def ecc_align(ref_rgb: np.ndarray, mov_rgb: np.ndarray) -> np.ndarray:
    ref_f = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mov_f = cv2.cvtColor(mov_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    warp  = np.eye(2, 3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-4)
        _, warp  = cv2.findTransformECC(ref_f, mov_f, warp, cv2.MOTION_EUCLIDEAN, criteria)
        return cv2.warpAffine(mov_rgb, warp, (ref_rgb.shape[1], ref_rgb.shape[0]),
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT)
    except cv2.error:
        return mov_rgb


def to_tensor_norm(rgb: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    t = TF.to_tensor(Image.fromarray(rgb))
    t = TF.resize(t, [size, size])
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)


# ============================================================
# Segmentation: sliding-window TTA
# ============================================================

def _hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h)[:, None]; wx = np.hanning(w)[None, :]
    w2 = (wy * wx).astype(np.float32)
    return w2 / max(1e-8, w2.max())


class TTASeg(nn.Module):
    """Sliding-window segmentation with multi-scale + flip TTA and Hann blending."""

    def __init__(self, ckpt, device, size=1024, stride=512, scales=(1.0, 1.2),
                 use_flips=True, thr=0.35, min_component=64,
                 avg_mode="logit", min_coverage=0.0005):
        super().__init__()
        self.device = device
        self.net = ResUNet(n_classes=1, backbone="resnet34", pretrained=False).to(device).eval()
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("model", state.get("state_dict", state))
        missing = self.net.load_state_dict(clean_state_dict(state), strict=False)
        if getattr(missing, "missing_keys", None) or getattr(missing, "unexpected_keys", None):
            print("[seg] missing/unexpected:", missing.missing_keys, missing.unexpected_keys)
        self.size         = int(size)
        self.stride       = int(stride)
        self.scales       = tuple(scales)
        self.use_flips    = bool(use_flips)
        self.thr          = float(thr)
        self.min_component = int(min_component)
        self.avg_mode     = avg_mode
        self.min_coverage = float(min_coverage)
        self.win = torch.from_numpy(_hann2d(self.size, self.size)).to(device)[None, None]

    @torch.no_grad()
    def probmap(self, rgb: np.ndarray) -> np.ndarray:
        self.net.eval()
        H, W   = rgb.shape[:2]
        tile, stride = self.size, self.stride
        acc  = torch.zeros((1, 1, H, W), device=self.device)
        wsum = torch.zeros((1, 1, H, W), device=self.device)
        base = TF.normalize(TF.to_tensor(Image.fromarray(rgb)).to(self.device),
                            IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0)
        flips = [(0, 0), (1, 0), (0, 1), (1, 1)] if self.use_flips else [(0, 0)]

        for y in range(0, max(1, H - tile + 1), stride):
            for x in range(0, max(1, W - tile + 1), stride):
                patch = base[..., y:y+tile, x:x+tile]
                if patch.shape[-1] < tile or patch.shape[-2] < tile:
                    patch = F.pad(patch, (0, tile - patch.shape[-1], 0, tile - patch.shape[-2]), mode="reflect")
                acc_t = torch.zeros((1, 1, tile, tile), device=self.device)
                n = 0
                for s in self.scales:
                    sz = int(tile * s)
                    p  = F.interpolate(patch, size=(sz, sz), mode="bilinear", align_corners=False)
                    for hf, vf in flips:
                        a = torch.flip(p, [-1]) if hf else p
                        a = torch.flip(a, [-2]) if vf else a
                        out = self.net(a)
                        t = (F.interpolate(out, size=(tile, tile), mode="bilinear", align_corners=False)
                             if self.avg_mode == "logit"
                             else F.interpolate(torch.sigmoid(out), size=(tile, tile), mode="bilinear", align_corners=False))
                        t = torch.flip(t, [-1]) if hf else t
                        t = torch.flip(t, [-2]) if vf else t
                        acc_t += t; n += 1
                acc_t /= float(n)
                if self.avg_mode == "logit":
                    acc_t = torch.sigmoid(acc_t)
                # crop acc_t and win to actual slice size (handles edge tiles smaller than tile)
                y1_actual = min(y + tile, H)
                x1_actual = min(x + tile, W)
                th = y1_actual - y
                tw = x1_actual - x
                acc[..., y:y1_actual, x:x1_actual]  += acc_t[..., :th, :tw] * self.win[..., :th, :tw]
                wsum[..., y:y1_actual, x:x1_actual] += self.win[..., :th, :tw]

        return (acc / torch.clamp_min(wsum, 1e-6)).squeeze().clamp(0, 1).cpu().numpy().astype(np.float32)

    def binarize(self, prob: np.ndarray) -> np.ndarray:
        prob = np.clip(np.asarray(prob, np.float32).squeeze(), 0, 1)
        m    = (prob >= self.thr).astype(np.uint8)
        if float(m.mean()) < self.min_coverage and (prob > 0.02).sum() > 1000:
            thr, _ = cv2.threshold((prob * 255).astype(np.uint8), 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            m = ((prob * 255).astype(np.uint8) >= thr).astype(np.uint8)
        if m.any():
            num, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            keep = np.zeros_like(m, np.uint8)
            for k in range(1, num):
                if stats[k, cv2.CC_STAT_AREA] >= self.min_component:
                    keep[lab == k] = 1
            m = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)
        return m.astype(np.uint8)


# ============================================================
# DenseCRF post-processing
# ============================================================

def refine_with_densecrf(prob: np.ndarray, rgb: np.ndarray,
                         iters=5, sxy_gauss=3, compat_gauss=3,
                         sxy_bilat=40, srgb_bilat=5, compat_bilat=6,
                         rescale=1.0) -> np.ndarray:
    prob = np.clip(np.asarray(prob, np.float32).squeeze(), 0, 1)
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except Exception as e:
        print(f"[densecrf] skipped: {e}"); return prob

    H, W = prob.shape
    if rescale != 1.0:
        h2, w2 = max(1, int(H * rescale)), max(1, int(W * rescale))
        prob = cv2.resize(prob, (w2, h2), interpolation=cv2.INTER_LINEAR)
        rgb  = cv2.resize(rgb,  (w2, h2), interpolation=cv2.INTER_LINEAR)

    h, w = prob.shape
    p_fg = np.clip(prob, 1e-6, 1 - 1e-6)
    softmax = np.stack([1.0 - p_fg, p_fg], axis=0)
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary_from_softmax(softmax))
    d.addPairwiseGaussian(sxy=sxy_gauss, compat=compat_gauss)
    d.addPairwiseBilateral(sxy=sxy_bilat, srgb=srgb_bilat,
                           rgbim=rgb.astype(np.uint8), compat=compat_bilat)
    Q = np.asarray(d.inference(iters), dtype=np.float32)
    q_fg = (Q[1] if Q.shape[0] == 2 else Q[:, 1]).reshape(h, w)
    if rescale != 1.0:
        q_fg = cv2.resize(q_fg, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(q_fg, 0, 1).astype(np.float32)


# ============================================================
# Watershed touching-building splitter
# ============================================================

def split_touching(binary_mask: np.ndarray, prob: np.ndarray,
                   r: int = 11, t_rel: float = 0.35) -> np.ndarray:
    m = (binary_mask > 0).astype(np.uint8)
    if m.sum() == 0: return m
    dist  = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
    dmax  = float(dist.max()) if dist.size else 0.0
    if dmax <= 1e-6: return m
    dist_n = dist / (dmax + 1e-6)
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, r) | 1, max(3, r) | 1))
    dil    = cv2.dilate(dist, k)
    peaks  = ((dist >= dil - 1e-6) & (dist_n >= t_rel)).astype(np.uint8)
    peaks  = cv2.morphologyEx(peaks, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    peaks[m == 0] = 0
    nlab, markers = cv2.connectedComponents(peaks)
    if nlab <= 1: return m
    markers = markers.astype(np.int32) + 1
    markers[m == 0] = 0
    elev_u8 = ((1.0 - np.clip(prob, 0, 1)) * 255).astype(np.uint8)
    cv2.watershed(np.dstack([elev_u8] * 3), markers)
    return (markers > 1).astype(np.uint8)


# ============================================================
# Classifier loading & per-instance prediction
# ============================================================

def load_classifier(ckpt: Path, backbone: str, num_classes: int, device: torch.device) -> nn.Module:
    model = SiameseResNet(backbone=backbone, pretrained=False, num_classes=num_classes).to(device).eval()
    st    = torch.load(ckpt, map_location=device)
    st    = st.get("model", st.get("state_dict", st))
    miss  = model.load_state_dict(clean_state_dict(st), strict=False)
    if getattr(miss, "missing_keys", None) or getattr(miss, "unexpected_keys", None):
        print("[cls] missing/unexpected:", miss.missing_keys, miss.unexpected_keys)
    return model


@torch.no_grad()
def siamese_predict(model, pre_crop: np.ndarray, post_crop: np.ndarray,
                    cls_input: int, device: torch.device, use_tta: bool = True):
    A = to_tensor_norm(pre_crop,  cls_input, device)
    B = to_tensor_norm(post_crop, cls_input, device)
    outs = [model(A, B)]
    if use_tta:
        outs.append(model(torch.flip(A, [-1]), torch.flip(B, [-1])))
        outs.append(model(torch.flip(A, [-2]), torch.flip(B, [-2])))
        Ah = torch.flip(A, [-1]); Bh = torch.flip(B, [-1])
        outs.append(model(torch.flip(Ah, [-2]), torch.flip(Bh, [-2])))
    logits = torch.stack(outs, 0).mean(0)
    probs  = torch.softmax(logits, dim=1).squeeze(0)
    raw    = int(probs.argmax().item())
    return PRED_INDEX_TO_LABEL.get(raw, -1), float(probs.max().item())


@torch.no_grad()
def siamese_predict_multiscale(model, pre: np.ndarray, post: np.ndarray,
                                lab: np.ndarray, k: int, stat,
                                cls_input: int, device: torch.device,
                                pad_ratios=(0.05, 0.15, 0.30),
                                use_tta: bool = True,
                                cls_use_mask: bool = True,
                                mask_dilate: int = 5):
    """
    Crops the same building at multiple padding ratios, runs the model on each,
    and averages logits before decoding. Makes classification robust to zoom/GSD
    variation between training images and inference images.
    """
    all_logits = []
    for pad in pad_ratios:
        if cls_use_mask:
            pre_c  = masked_crop(pre,  lab, k, stat, pad_ratio=pad,
                                 out_size=cls_input, dilate=mask_dilate)
            post_c = masked_crop(post, lab, k, stat, pad_ratio=pad,
                                 out_size=cls_input, dilate=mask_dilate)
        else:
            pre_c  = crop_from_stat(pre,  stat, pad_ratio=pad, out_size=cls_input)
            post_c = crop_from_stat(post, stat, pad_ratio=pad, out_size=cls_input)

        A = to_tensor_norm(pre_c,  cls_input, device)
        B = to_tensor_norm(post_c, cls_input, device)
        outs = [model(A, B)]
        if use_tta:
            outs.append(model(torch.flip(A, [-1]), torch.flip(B, [-1])))
            outs.append(model(torch.flip(A, [-2]), torch.flip(B, [-2])))
            Ah = torch.flip(A, [-1]); Bh = torch.flip(B, [-1])
            outs.append(model(torch.flip(Ah, [-2]), torch.flip(Bh, [-2])))
        all_logits.append(torch.stack(outs, 0).mean(0))

    logits = torch.stack(all_logits, 0).mean(0)
    probs  = torch.softmax(logits, dim=1).squeeze(0)
    raw    = int(probs.argmax().item())
    return PRED_INDEX_TO_LABEL.get(raw, -1), float(probs.max().item())


# ============================================================
# Crop helpers
# ============================================================

def _bbox_from_stat(stat, pad_ratio, H, W):
    x, y, w, h, _ = stat
    py, px = int(pad_ratio * h), int(pad_ratio * w)
    return max(0, y - py), min(H - 1, y + h - 1 + py), max(0, x - px), min(W - 1, x + w - 1 + px)


def masked_crop(img, lab, k, stat, pad_ratio=0.15, out_size=256, dilate=5):
    H, W = img.shape[:2]
    Y0, Y1, X0, X1 = _bbox_from_stat(stat, pad_ratio, H, W)
    crop = img[Y0:Y1+1, X0:X1+1].copy()
    mask = (lab[Y0:Y1+1, X0:X1+1] == k).astype(np.uint8) * 255
    if dilate > 0:
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
    crop[mask == 0] = 0
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def crop_from_stat(img, stat, pad_ratio=0.15, out_size=256):
    H, W = img.shape[:2]
    Y0, Y1, X0, X1 = _bbox_from_stat(stat, pad_ratio, H, W)
    return cv2.resize(img[Y0:Y1+1, X0:X1+1], (out_size, out_size), interpolation=cv2.INTER_LINEAR)


# ============================================================
# Visualisation helpers
# ============================================================

GUTTER_PX    = 8
GUTTER_COLOR = (255, 255, 255)


def draw_perimeters(rgb: np.ndarray, labels: np.ndarray, cls_by_id: dict, thick: int = 2) -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for k in np.unique(labels):
        if k == 0: continue
        comp = (labels == k).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cid   = int(cls_by_id.get(int(k), -1))
        color = CLASS_COLORS.get(cid, UNK_COLOR)
        cv2.drawContours(out, cnts, -1, (color[2], color[1], color[0]), thick, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def stack3(a, b, c, gutter=GUTTER_PX, color=GUTTER_COLOR):
    H = max(a.shape[0], b.shape[0], c.shape[0])
    def pad_h(x):
        if x.shape[0] < H:
            x = np.pad(x, ((0, H - x.shape[0]), (0, 0), (0, 0)), constant_values=255)
        return x
    a, b, c = pad_h(a), pad_h(b), pad_h(c)
    spacer  = np.full((H, gutter, 3), np.array(color, dtype=np.uint8), dtype=np.uint8)
    return np.concatenate([a, spacer, b, spacer, c], axis=1)


# ============================================================
# Main inference loop
# ============================================================

def run_inference(
    pre_dir: Path,
    post_dir: Path,
    seg_ckpt: Path,
    cls_ckpt: Path,
    device: torch.device,
    seg_input: int    = 1380,
    seg_stride: int   = None,
    seg_scales        = (1.0, 1.2),
    seg_thr: float    = 0.25,
    min_component: int = 100,
    split_touching_flag: bool = False,
    use_densecrf: bool = True,
    crf_kwargs: dict  = None,
    cls_backbone: str = "resnet50",
    cls_input: int    = 256,
    use_tta_cls: bool = True,
    cls_use_mask: bool = True,
    mask_dilate: int  = 5,
    crop_pad_ratios: tuple = (0.05, 0.15, 0.30),
    occ_min: float    = 0.05,
    use_ecc_align: bool = True,
    wat_r: int        = 11,
    wat_trel: float   = 0.60,
):
    if seg_stride is None:
        seg_stride = seg_input // 2
    if crf_kwargs is None:
        crf_kwargs = dict(iters=5, sxy_gauss=3, compat_gauss=3,
                          sxy_bilat=40, srgb_bilat=5, compat_bilat=6, rescale=1.0)

    seg = TTASeg(seg_ckpt, device, size=seg_input, stride=seg_stride, scales=seg_scales,
                 thr=seg_thr, min_component=min_component)
    cls = load_classifier(cls_ckpt, cls_backbone, num_classes=4, device=device)

    pre_map  = {parse_idx(p): p for p in Path(pre_dir).glob("*.png")}
    post_map = {parse_idx(p): p for p in Path(post_dir).glob("*.png")}
    common   = sorted(set(pre_map) & set(post_map))
    pairs    = [(pre_map[k], post_map[k]) for k in common]
    if not pairs:
        pairs = list(zip(sorted(Path(pre_dir).glob("*.png")), sorted(Path(post_dir).glob("*.png"))))
    print(f"[pairs] found {len(pairs)}")

    for pre_fp, post_fp in pairs:
        pre  = read_rgb(pre_fp)
        post = read_rgb(post_fp)
        if post.shape[:2] != pre.shape[:2]:
            post = resize_to(post, pre.shape[:2])
        if use_ecc_align:
            post = ecc_align(pre, post)

        prob = seg.probmap(pre)
        assert prob.ndim == 2, f"prob shape unexpected: {prob.shape}"

        if use_densecrf:
            prob = refine_with_densecrf(prob, pre, **crf_kwargs)

        binm = seg.binarize(prob)
        print(f"[seg] prob_mean={prob.mean():.3f} coverage={(binm.mean()*100):.2f}% @ thr={seg_thr}")

        if split_touching_flag:
            binm = split_touching(binm, prob, r=wat_r, t_rel=wat_trel)

        num, lab, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
        cls_by_id = {}; kept = skipped = 0; confs = []

        for k in range(1, num):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < min_component: continue
            x, y, w, h, _ = stats[k]
            if binm[y:y+h, x:x+w].mean() < occ_min:
                skipped += 1; continue
            cid, cf = siamese_predict_multiscale(
                cls, pre, post, lab, k, stats[k],
                cls_input=cls_input, device=device,
                pad_ratios=crop_pad_ratios,
                use_tta=use_tta_cls,
                cls_use_mask=cls_use_mask,
                mask_dilate=mask_dilate,
            )
            cls_by_id[k] = cid; kept += 1; confs.append(cf)

        vis   = draw_perimeters(post, lab, cls_by_id, thick=2)
        panel = stack3(pre, post, vis)
        title = (f"{Path(pre_fp).name}  • comps kept:{kept}/{max(0, num-1)}"
                 f"  skipped:{skipped}  mean_conf:{np.mean(confs) if confs else 0:.2f}")
        plt.figure(figsize=(18, 6))
        plt.imshow(panel); plt.axis("off"); plt.title(title); plt.show()
        gc.collect()
