#!/usr/bin/env python3
"""monuseg_indomain.py — MoNuSeg 2018 in-domain (DATASET THỨ 3, độc lập PanNuke/NuInsSeg).

Vì sao MoNuSeg: đa cơ quan TCGA (breast/kidney/liver/prostate/bladder/colon/stomach), nguồn
KHÁC curation PanNuke → generalization thật, KHÔNG trùng dữ liệu như Lizard/CoNIC.
IN-DOMAIN (train+test CÙNG MoNuSeg) → KHÔNG transfer → không sập kiểu CryoNuSeg (model học
thẳng phân phối + tầm count của MoNuSeg). Patch 1024/256 = 16 tile NATIVE scale
(resize 1000→1024 ~1.02×, KHÔNG co nhân → tránh μ-sập kiểu MoNuSAC 4×).

Count/patch = số nhân có CENTROID rơi trong patch (mỗi nhân đếm 1 lần, không trùng biên).
Split theo ẢNH/WSI (mọi patch của 1 ảnh về cùng train HOẶC test → KHÔNG leak patch).
count-only efflite0 (w_density=0, w_count=1, Poisson-σ, detach_mu) — GIỐNG HỆT headline.
So R² với PanNuke 0.920 / NuInsSeg 0.854.

Chạy Kaggle (GPU):
  !pip -q install tifffile
  !python monuseg_indomain.py --root /kaggle/input/<monuseg-dataset> --seeds 42,43,44 --epochs 40
"""
import argparse, os, glob, sys
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "kaggle", "lib"), _REPO, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import torch
from distill_student_r2 import train
from r2_losses import count_from_density

IMG = 256
GRID = 4
FULL = IMG * GRID   # 1024


def read_tif(p):
    try:
        import tifffile
        return tifffile.imread(p)
    except Exception:
        return np.asarray(Image.open(p))


def parse_centroids(xml_path):
    """XML Aperio (MoNuSeg): mỗi <Region> = 1 nhân; centroid = trung bình đỉnh polygon."""
    root = ET.parse(xml_path).getroot()
    cs = []
    for reg in root.iter("Region"):
        xs, ys = [], []
        for v in reg.iter("Vertex"):
            xs.append(float(v.get("X"))); ys.append(float(v.get("Y")))
        if xs:
            cs.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return cs


def find_pairs(root):
    imgs = {}
    for ext in ("*.tif", "*.tiff"):
        for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
            imgs[os.path.splitext(os.path.basename(p))[0]] = p
    pairs = []
    for x in glob.glob(os.path.join(root, "**", "*.xml"), recursive=True):
        stem = os.path.splitext(os.path.basename(x))[0]
        if stem in imgs:
            pairs.append((imgs[stem], x, stem))
    return pairs


def build_patches(pairs):
    """-> list {img 256 uint8, density zeros, gt count, organ=wsi, wsi}."""
    data = []
    for ip, xp, stem in pairs:
        arr = read_tif(ip)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        arr = np.ascontiguousarray(arr[..., :3]).astype(np.uint8)
        H, W = arr.shape[:2]
        img = np.asarray(Image.fromarray(arr).resize((FULL, FULL), Image.BILINEAR)).astype(np.uint8)
        sx, sy = FULL / W, FULL / H
        cnt = np.zeros((GRID, GRID), int)
        for (cx, cy) in parse_centroids(xp):
            tx = min(GRID - 1, int(cx * sx) // IMG)
            ty = min(GRID - 1, int(cy * sy) // IMG)
            cnt[ty, tx] += 1
        for gy in range(GRID):
            for gx in range(GRID):
                patch = img[gy * IMG:(gy + 1) * IMG, gx * IMG:(gx + 1) * IMG]
                data.append({"img": patch, "density": np.zeros((IMG, IMG), np.float32),
                             "gt": float(cnt[gy, gx]), "organ": stem, "wsi": stem})
    return data


def eval_r2(model, data, idx, dev):
    model.eval(); pr, yt = [], []
    with torch.no_grad():
        for i in idx:
            x = torch.from_numpy(data[i]["img"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
            pr.append(float(count_from_density(model(x)[0])[0])); yt.append(data[i]["gt"])
    pr, yt = np.array(pr), np.array(yt)
    ss = ((yt - pr) ** 2).sum(); st = ((yt - yt.mean()) ** 2).sum()
    return (1 - ss / st if st > 0 else float("nan")), np.abs(pr - yt).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="thư mục MoNuSeg (chứa .tif + .xml, tìm đệ quy)")
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = find_pairs(args.root)
    assert pairs, f"không thấy cặp .tif+.xml trong {args.root}"
    print(f"[data] {len(pairs)} ảnh MoNuSeg có annotation (root={args.root})")
    data = build_patches(pairs)
    wsis = sorted(set(d["wsi"] for d in data))
    gts = np.array([d["gt"] for d in data])
    print(f"[patches] {len(data)} patch 256 từ {len(wsis)} WSI | count/patch "
          f"mean {gts.mean():.1f} min {gts.min():.0f} max {gts.max():.0f}")

    seeds = [int(s) for s in args.seeds.split(",")]
    r2s, maes = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = list(rng.permutation(np.array(wsis)))
        n_te = max(1, len(wsis) // 5)
        te_w = set(perm[:n_te])
        tr = [i for i, d in enumerate(data) if d["wsi"] not in te_w]
        te = [i for i, d in enumerate(data) if d["wsi"] in te_w]
        np.random.seed(seed); torch.manual_seed(seed)
        model = train(data, dev, args.epochs, 32, 1e-3, list(tr),
                      0.0, 1.0, 0.01, 0.5, 16, True, "poisson", args.backbone)
        r2, mae = eval_r2(model, data, te, dev)
        r2s.append(r2); maes.append(mae)
        print(f"  [seed{seed}] test {len(te_w)} WSI / {len(te)} patch  R² {r2:+.4f}  MAE {mae:.3f}")

    print(f"\n=== MoNuSeg IN-DOMAIN count-only efflite0 ({len(seeds)} seed) — dataset thứ 3 ===")
    print(f"  R²  {np.nanmean(r2s):+.4f} ± {np.nanstd(r2s):.4f}")
    print(f"  MAE {np.nanmean(maes):.3f} ± {np.nanstd(maes):.3f}")
    print(f"  -> so PanNuke 0.920 / NuInsSeg 0.854 (in-domain cùng giao thức count-only)")


if __name__ == "__main__":
    main()
