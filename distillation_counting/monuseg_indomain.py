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


def parse_polygons(xml_path):
    """XML Aperio (MoNuSeg): mỗi <Region> = 1 nhân. Trả list (xs, ys) đỉnh polygon."""
    root = ET.parse(xml_path).getroot()
    polys = []
    for reg in root.iter("Region"):
        xs, ys = [], []
        for v in reg.iter("Vertex"):
            xs.append(float(v.get("X"))); ys.append(float(v.get("Y")))
        if len(xs) >= 3:
            polys.append((np.asarray(xs), np.asarray(ys)))
    return polys


def find_pairs(root):
    imgs = {}
    for ext in ("*.tif", "*.tiff"):
        for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
            imgs[os.path.splitext(os.path.basename(p))[0]] = p
    pairs = []
    for x in glob.glob(os.path.join(root, "**", "*.xml"), recursive=True):
        stem = os.path.splitext(os.path.basename(x))[0]
        if stem in imgs:
            ip = imgs[stem]
            is_test = "test" in (ip + x).lower()   # split chính thức MoNuSeg (folder ...TestData)
            pairs.append((ip, x, stem, is_test))
    return pairs


def build_patches(pairs, with_density=False):
    """-> list {img 256 uint8, density, gt count, organ=wsi, wsi, is_test}.
    with_density=True: rasterize polygon -> GT density map (mỗi nhân trải 1/diện-tích) cho baseline mask.
    Count/patch = số nhân có CENTROID trong patch (đếm 1 lần, không trùng biên)."""
    if with_density:
        from skimage.draw import polygon as sk_polygon
    data = []
    for ip, xp, stem, is_test in pairs:
        arr = read_tif(ip)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        arr = np.ascontiguousarray(arr[..., :3]).astype(np.uint8)
        H, W = arr.shape[:2]
        img = np.asarray(Image.fromarray(arr).resize((FULL, FULL), Image.BILINEAR)).astype(np.uint8)
        sx, sy = FULL / W, FULL / H
        cnt = np.zeros((GRID, GRID), int)
        dens_full = np.zeros((FULL, FULL), np.float32) if with_density else None
        for (xs, ys) in parse_polygons(xp):
            xsf, ysf = xs * sx, ys * sy
            cx, cy = float(xsf.mean()), float(ysf.mean())
            tx = min(GRID - 1, int(cx) // IMG); ty = min(GRID - 1, int(cy) // IMG)
            cnt[ty, tx] += 1
            if with_density:
                rr, cc = sk_polygon(ysf, xsf, shape=(FULL, FULL))
                a = len(rr)
                if a > 0:
                    dens_full[rr, cc] += 1.0 / a
        for gy in range(GRID):
            for gx in range(GRID):
                patch = img[gy * IMG:(gy + 1) * IMG, gx * IMG:(gx + 1) * IMG]
                if with_density:
                    dens = dens_full[gy * IMG:(gy + 1) * IMG, gx * IMG:(gx + 1) * IMG].copy()
                else:
                    dens = np.zeros((IMG, IMG), np.float32)
                data.append({"img": patch, "density": dens,
                             "gt": float(cnt[gy, gx]), "organ": stem, "wsi": stem,
                             "is_test": is_test})
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
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--split", default="official", choices=["official", "random"],
                    help="official = train folder Training / test folder TestData (test CỐ ĐỊNH, so literature); "
                         "random = chia WSI ngẫu nhiên theo seed (nhiễu trên dataset nhỏ)")
    ap.add_argument("--parity", action="store_true",
                    help="chạy CẢ count-only VÀ mask (GT-density polygon) -> parity dataset thứ 3 cho Q1")
    ap.add_argument("--w_count_mask", type=float, default=0.1,
                    help="neo count cho config mask (MoNuSeg count~40 -> cần >0.01 để density-MSE không sập; "
                         "PanNuke count nhỏ dùng 0.01 đủ)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = find_pairs(args.root)
    assert pairs, f"không thấy cặp .tif+.xml trong {args.root}"
    n_test_folder = sum(1 for p in pairs if p[3])
    print(f"[data] {len(pairs)} ảnh MoNuSeg có annotation | folder Test={n_test_folder} (root={args.root})")
    data = build_patches(pairs, with_density=args.parity)   # parity cần GT density cho baseline mask
    wsis = sorted(set(d["wsi"] for d in data))
    gts = np.array([d["gt"] for d in data])
    print(f"[patches] {len(data)} patch 256 từ {len(wsis)} WSI | count/patch "
          f"mean {gts.mean():.1f} min {gts.min():.0f} max {gts.max():.0f}")

    if args.split == "official" and n_test_folder == 0:
        print("[cảnh báo] không tìm thấy folder Test -> fallback sang split random")
        args.split = "random"

    def split_idx(seed):
        if args.split == "official":
            tr = [i for i, d in enumerate(data) if not d["is_test"]]
            te = [i for i, d in enumerate(data) if d["is_test"]]
        else:
            rng = np.random.default_rng(seed)
            te_w = set(list(rng.permutation(np.array(wsis)))[:max(1, len(wsis) // 5)])
            tr = [i for i, d in enumerate(data) if d["wsi"] not in te_w]
            te = [i for i, d in enumerate(data) if d["wsi"] in te_w]
        return tr, te

    seeds = [int(s) for s in args.seeds.split(",")]
    # (w_density, w_count): count-only bỏ mask; mask dùng GT-density polygon (chỉ khi --parity)
    configs = [("count-only", 0.0, 1.0)]
    if args.parity:
        configs.append(("mask (GT-density)", 1.0, args.w_count_mask))

    results = {}
    for name, w_d, w_c in configs:
        r2s, maes = [], []
        for seed in seeds:
            tr, te = split_idx(seed)
            np.random.seed(seed); torch.manual_seed(seed)
            model = train(data, dev, args.epochs, 32, 1e-3, list(tr),
                          w_d, w_c, 0.01, 0.5, 16, True, "poisson", args.backbone)
            r2, mae = eval_r2(model, data, te, dev)
            r2s.append(r2); maes.append(mae)
            print(f"  [{name:18s} seed{seed}] split={args.split} test {len(te)} patch  "
                  f"R² {r2:+.4f}  MAE {mae:.3f}")
        results[name] = (np.nanmean(r2s), np.nanstd(r2s), np.nanmean(maes), np.nanstd(maes))

    print(f"\n=== MoNuSeg IN-DOMAIN efflite0 ({len(seeds)} seed, split={args.split}) — dataset thứ 3 ===")
    for name, (r2m, r2s, maem, maes) in results.items():
        print(f"  {name:18s}  R² {r2m:+.4f} ± {r2s:.4f}  |  MAE {maem:.3f} ± {maes:.3f}")
    if args.parity and len(results) == 2:
        dc = results["count-only"][0] - results["mask (GT-density)"][0]
        print(f"\n  Δ(count-only − mask) R² = {dc:+.4f}  -> parity nếu |Δ| < std")
    print(f"  -> so PanNuke 0.920 / NuInsSeg 0.854 (in-domain cùng giao thức)")


if __name__ == "__main__":
    main()
