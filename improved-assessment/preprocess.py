"""
One-time data preprocessing:
  - Rasterise XBD GeoJSON labels → binary PNG masks
  - Normalise and size-match masks to images
  - Crop XBD post-disaster images into per-building chips (for classification)
"""

import os
import re
import csv
import glob
import json
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
from collections import Counter
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon, MultiPolygon

LABEL_NAME_TO_NUM = {
    "no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4, "un-classified": 0
}



# -----Polygon parsing helpers-----


def _scale_if_normalized(coords, W, H):
    xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
    if not xs: return coords
    if (0.0 <= min(xs) and max(xs) <= 1.0 + 1e-6 and
            0.0 <= min(ys) and max(ys) <= 1.0 + 1e-6):
        return [(x * W, y * H) for x, y in coords]
    return coords


def _rings_from_poly(poly, W, H):
    def clip_round(r):
        out = [(int(np.clip(round(x), 0, W - 1)), int(np.clip(round(y), 0, H - 1))) for x, y in r]
        return out if len(out) >= 3 else None

    outer = clip_round(_scale_if_normalized(list(poly.exterior.coords), W, H))
    if not outer: return None, []
    holes = [h for it in poly.interiors
             for h in [clip_round(_scale_if_normalized(list(it.coords), W, H))] if h]
    return outer, holes


def _yield_polys_ms(label_json, W, H):
    feats = ((label_json.get("features") or {}).get("xy")) or []
    for f in feats:
        wkt_str = f.get("wkt")
        if not wkt_str: continue
        try: g = shapely_wkt.loads(wkt_str)
        except Exception: continue
        if g.is_empty: continue
        ps = [g] if isinstance(g, Polygon) else ([p for p in g.geoms] if isinstance(g, MultiPolygon) else [])
        for p in ps:
            outer, holes = _rings_from_poly(p, W, H)
            if outer:
                subtype = (f.get("properties") or {}).get("subtype", "no-damage")
                yield outer, holes, subtype


def _yield_polys_xbd(label_json, W, H):
    feats = label_json.get("features") or []
    for ft in feats:
        if not isinstance(ft, dict): continue
        props = ft.get("properties") or {}
        for key in ("PolygonWKT_Pix", "wkt_im", "polygon_wkt"):
            if props.get(key):
                try: g = shapely_wkt.loads(props[key])
                except Exception: g = None
                if g is None or g.is_empty: continue
                ps = [g] if isinstance(g, Polygon) else ([p for p in g.geoms] if isinstance(g, MultiPolygon) else [])
                for p in ps:
                    outer, holes = _rings_from_poly(p, W, H)
                    if outer: yield outer, holes, props.get("subtype", "no-damage")
                break
        else:
            geom = ft.get("geometry") or {}
            gtype = geom.get("type")
            coords = geom.get("coordinates") or []
            polys = [coords] if gtype == "Polygon" else (coords if gtype == "MultiPolygon" else [])
            for poly in polys:
                if not poly: continue
                outer = [(int(np.clip(round(x), 0, W-1)), int(np.clip(round(y), 0, H-1)))
                         for x, y in _scale_if_normalized(poly[0], W, H)]
                if len(outer) < 3: continue
                holes = [[(int(np.clip(round(x), 0, W-1)), int(np.clip(round(y), 0, H-1)))
                           for x, y in _scale_if_normalized(h, W, H)]
                          for h in poly[1:] if len(h) >= 3]
                yield outer, holes, props.get("subtype", "no-damage")



# -----Rasterisation-----


def _rasterize_tile(label_path, image_path, out_path, binary=True):
    with Image.open(image_path) as im:
        W, H = im.size
    with open(label_path) as f:
        js = json.load(f)

    polys = list(_yield_polys_ms(js, W, H)) or list(_yield_polys_xbd(js, W, H))
    mask  = Image.new("L", (W, H), 0)
    draw  = ImageDraw.Draw(mask, "L")
    for outer, holes, subtype in polys:
        fill = 255 if binary else LABEL_NAME_TO_NUM.get(subtype, 0)
        draw.polygon(outer, outline=fill, fill=fill)
        for h in holes:
            draw.polygon(h, outline=0, fill=0)

    m = np.array(mask, dtype=np.uint8)
    Image.fromarray(m).save(out_path)
    return int(m.any())


def rasterize_xbd_labels(labels_dir: str, images_dir: str, out_dir: str):
    """Rasterise all XBD label JSONs to binary mask PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
    print(f"Label JSONs: {len(label_files)}")
    rebuilt = nonempty = 0
    for jp in tqdm(label_files):
        stem = Path(jp).stem
        imgp = os.path.join(images_dir, stem + ".png")
        if not os.path.exists(imgp):
            for ext in (".jpg", ".tif", ".tiff"):
                alt = imgp[:-4] + ext
                if os.path.exists(alt): imgp = alt; break
            else:
                continue
        outp = os.path.join(out_dir, stem + "_b0.png")
        nz   = _rasterize_tile(jp, imgp, outp, binary=True)
        rebuilt += 1; nonempty += nz
    print(f"Rebuilt: {rebuilt} | non-empty: {nonempty} → {out_dir}")



# -----Mask normalisation-----


def normalise_masks(src_dir: str, images_dir: str, dst_dir: str):
    """
    Convert rasterised masks to strict binary 0/255 PNGs, size-matching images.
    Also heuristically inverts masks where >60% of pixels are foreground.
    """
    os.makedirs(dst_dir, exist_ok=True)
    pngs = sorted(glob.glob(os.path.join(src_dir, "*.png")))
    fixed = resized = inverted = 0
    for p in pngs:
        stem = Path(p).stem.replace("_b2", "").replace("_b0", "")
        ip   = os.path.join(images_dir, stem + ".png")
        if not os.path.exists(ip):
            for ext in (".jpg", ".tif", ".tiff"):
                if os.path.exists(ip[:-4] + ext): ip = ip[:-4] + ext; break
            else:
                ip = None

        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None: continue

        if ip is not None:
            im = cv2.imread(ip, cv2.IMREAD_COLOR)
            if im is not None and m.shape[:2] != im.shape[:2]:
                m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
                resized += 1

        if (m > 0).mean() > 0.60:
            m = cv2.bitwise_not(m); inverted += 1

        out_path = os.path.join(dst_dir, stem + "_mask.png")
        cv2.imwrite(out_path, (m > 0).astype(np.uint8) * 255)
        fixed += 1

    print(f"Saved {fixed} binary masks → {dst_dir}")
    print(f"  Resized: {resized} | Auto-inverted: {inverted}")



# -----XBD classification crop pipeline-----


def _is_stub(name: str) -> bool:
    return name.startswith("._") or name == "__MACOSX"


def _parse_core(fname: str):
    m = re.match(r"^(.+?)_(pre|post)_disaster", fname)
    if m: return m.group(1), m.group(2)
    return None, None


def crop_xbd_to_manifest(
    xbd_root: str,
    out_root: str,
    crop_size: int = 256,
    min_building_px: int = 32,
    min_damage_frac: float = 0.02,
):
    """
    Crop each building bounding box from post-disaster images using per-building
    target masks, build a manifest.csv for classification training.
    """
    images_dir  = os.path.join(xbd_root, "images")
    labels_dir  = os.path.join(xbd_root, "labels")
    targets_dir = os.path.join(xbd_root, "targets")

    out_dir = Path(out_root)
    (out_dir / "pre").mkdir(parents=True, exist_ok=True)
    (out_dir / "post").mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    kept = Counter(); reasons = Counter()

    with open(manifest_path, "w", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=["pre_path", "post_path", "label", "label_id",
                                                 "scene_core", "source"])
        writer.writeheader()

        post_jsons = sorted([
            p for p in glob.glob(os.path.join(labels_dir, "*.json"))
            if re.search(r"_post_disaster\.json$", os.path.basename(p))
        ])

        for jp in tqdm(post_jsons):
            stem      = Path(jp).stem                       # e.g. hurricane-harvey_00000001_post_disaster
            pre_stem  = stem.replace("_post_disaster", "_pre_disaster")
            core, _   = _parse_core(Path(jp).name)
            if core is None or _is_stub(Path(jp).name): continue

            post_img_p = os.path.join(images_dir, stem + ".png")
            pre_img_p  = os.path.join(images_dir, pre_stem + ".png")
            target_p   = os.path.join(targets_dir, stem + "_target.png")

            if not all(os.path.exists(p) for p in [post_img_p, pre_img_p, target_p]):
                reasons["missing_files"] += 1; continue

            post_img = cv2.imread(post_img_p, cv2.IMREAD_COLOR)
            pre_img  = cv2.imread(pre_img_p,  cv2.IMREAD_COLOR)
            target   = cv2.imread(target_p,   cv2.IMREAD_GRAYSCALE)
            if post_img is None or pre_img is None or target is None:
                reasons["load_error"] += 1; continue

            with open(jp) as f: js = json.load(f)
            H, W = post_img.shape[:2]
            polys = list(_yield_polys_ms(js, W, H)) or list(_yield_polys_xbd(js, W, H))

            for i, (outer, holes, subtype) in enumerate(polys):
                label_id = LABEL_NAME_TO_NUM.get(subtype)
                if label_id is None or label_id == 0: reasons["unclassified"] += 1; continue

                xs = [c[0] for c in outer]; ys = [c[1] for c in outer]
                x0, x1 = max(0, min(xs)), min(W, max(xs))
                y0, y1 = max(0, min(ys)), min(H, max(ys))
                if (x1 - x0) < min_building_px or (y1 - y0) < min_building_px:
                    reasons["too_small"] += 1; continue

                tgt_crop = target[y0:y1, x0:x1]
                if tgt_crop.size == 0 or (tgt_crop > 0).mean() < min_damage_frac:
                    reasons["low_damage_frac"] += 1; continue

                pad_y = int((y1 - y0) * 0.15); pad_x = int((x1 - x0) * 0.15)
                cy0, cy1 = max(0, y0 - pad_y), min(H, y1 + pad_y)
                cx0, cx1 = max(0, x0 - pad_x), min(W, x1 + pad_x)

                post_crop = cv2.resize(post_img[cy0:cy1, cx0:cx1], (crop_size, crop_size))
                pre_crop  = cv2.resize(pre_img[cy0:cy1, cx0:cx1],  (crop_size, crop_size))

                fname_base = f"{stem}_{i:04d}"
                post_out   = str(out_dir / "post" / (fname_base + "_post.png"))
                pre_out    = str(out_dir / "pre"  / (fname_base + "_pre.png"))
                cv2.imwrite(post_out, post_crop)
                cv2.imwrite(pre_out,  pre_crop)

                label_txt = subtype
                writer.writerow({"pre_path": pre_out, "post_path": post_out,
                                  "label": label_txt, "label_id": label_id - 1,
                                  "scene_core": core, "source": "xbd"})
                kept[label_txt] += 1

    print(f"Crops kept: {dict(kept)}")
    print(f"Reasons dropped: {dict(reasons)}")
    print(f"Manifest → {manifest_path}")
    return manifest_path
