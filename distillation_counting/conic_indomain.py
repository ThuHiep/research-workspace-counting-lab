#!/usr/bin/env python3
"""conic_indomain.py — count-only vs mask (GT-density) trên CoNIC/Lizard, bộ dữ liệu THỨ TƯ.

Cùng giao thức Bảng 1: efflite0, ch32, 60 epoch, lr 1e-3, bs16, sigma Poisson.
  count-only : w_d=0,      w_c=1
  mask       : w_d=6553.6, w_c=0.1        (w_d = H*W*w_c, xem §3.2)

BA điểm khác CoNIC bắt buộc phải xử lý (README của challenge):
  1. counts.csv chỉ đếm nhân trong vùng TRUNG TÂM 224x224 -> KHÔNG dùng; đếm lại từ kênh
     instance trên TOÀN mảnh 256x256, đúng định nghĩa đang dùng ở PanNuke/MoNuSeg.
  2. Không có split chính thức -> chia 80/20 THEO ẢNH GỐC (patch_info), không theo mảnh:
     các mảnh cùng một ảnh gốc mà rơi hai bên là rò rỉ.
  3. pannuke là 1 trong các nguồn của Lizard -> LOẠI (112 mảnh) để độc lập với bộ chính.

Chạy:
  python conic_indomain.py --root /kaggle/input/conic/data \
      --ckpt_dir /kaggle/working/ckpt_conic --dump_preds /kaggle/working/out/conic_parity.npz
"""
import argparse, csv, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "kaggle", "lib"), _REPO, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
from distill_student_r2 import train
from parity_nuinsseg import per_image_pred, r2mae


def load_conic(root, drop="pannuke"):
    im = np.load(f"{root}/images.npy", mmap_mode="r")
    lb = np.load(f"{root}/labels.npy", mmap_mode="r")
    ids = [r[0] for r in list(csv.reader(open(f"{root}/patch_info.csv")))[1:]]
    assert len(ids) == len(im) == len(lb), f"lệch: {len(ids)}/{len(im)}/{len(lb)}"
    data, src = [], []
    for i, pid in enumerate(ids):
        if pid.startswith(drop):
            continue
        inst = np.asarray(lb[i, :, :, 0], np.int64)
        area = np.bincount(inst.ravel())
        w = np.zeros(area.size, np.float32); w[1:] = 1.0 / np.maximum(area[1:], 1)
        data.append({"img": np.asarray(im[i], np.uint8), "density": w[inst],
                     "gt": float((area[1:] > 0).sum()), "organ": pid.rsplit("-", 1)[0]})
        src.append(pid.rsplit("-", 1)[0])
    return data, np.array(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="thư mục chứa images.npy/labels.npy/patch_info.csv")
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--w_density", type=float, default=6553.6)
    ap.add_argument("--w_count_mask", type=float, default=0.1)
    ap.add_argument("--arms", default="count,mask",
                    help="chạy 1 nhánh/phiên nếu sợ quá 12h Kaggle; ghép npz sau (split theo seed cố định)")
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--dump_preds", default=None)
    ap.add_argument("--delta_frac", type=float, default=0.05)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for d in (args.ckpt_dir, os.path.dirname(args.dump_preds or "")):
        if d:
            os.makedirs(d, exist_ok=True); open(f"{d}/.w", "w").close(); os.remove(f"{d}/.w")
    from scipy.stats import wilcoxon

    data, src = load_conic(args.root)
    wsis = np.unique(src)
    print(f"[data] CoNIC {len(data)} mảnh | {len(wsis)} ảnh gốc | count/mảnh TB "
          f"{np.mean([d['gt'] for d in data]):.1f}")

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [a for a in [("count-only", 0.0, 1.0), ("mask", args.w_density, args.w_count_mask)]
            if a[0].split("-")[0] in args.arms.split(",")]
    rows, err_all = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        te_w = set(rng.permutation(wsis)[:max(1, len(wsis) // 5)])
        te = np.where(np.isin(src, list(te_w)))[0]; tr = np.where(~np.isin(src, list(te_w)))[0]
        gt = np.array([data[i]["gt"] for i in te])
        preds = {}
        for name, wd, wc in arms:
            ck = f"{args.ckpt_dir}/{name}_s{seed}.pt" if args.ckpt_dir else None
            np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
            model = train(data, dev, args.epochs, args.ch, 1e-3, list(tr),
                          wd, wc, 0.01, 0.5, 16, True, "poisson", args.backbone)
            if ck:
                torch.save(model.state_dict(), ck)
            preds[name] = per_image_pred(model, data, te, dev)
        if len(arms) == 1:                       # chạy 1 nhánh: chỉ lưu, ghép ở phiên sau
            n0 = arms[0][0]
            np.savez(args.dump_preds.replace(".npz", f"_{n0}_s{seed}.npz"),
                     gt=gt, pred=preds[n0], te=te)
            r2, mae = r2mae(preds[n0], gt)
            print(f"  [{n0} seed{seed}] test {len(te)} mảnh/{len(te_w)} ảnh | R² {r2:+.4f} | MAE {mae:.3f}")
            continue
        ec = np.abs(gt - preds["count-only"]); em = np.abs(gt - preds["mask"])
        r2c, _ = r2mae(preds["count-only"], gt); r2m, _ = r2mae(preds["mask"], gt)
        try:
            _, p = wilcoxon(ec, em)
        except Exception:
            p = float("nan")
        rows.append((seed, r2c, r2m, ec.mean(), em.mean(), float((ec - em).mean()), p))
        err_all.append((seed, ec, em, gt))
        print(f"  [seed{seed}] test {len(te)} mảnh/{len(te_w)} ảnh | R² count {r2c:+.4f} / mask {r2m:+.4f}"
              f" | MAE {ec.mean():.2f} vs {em.mean():.2f} | p={p:.4g}")

    if not err_all:
        return
    if args.dump_preds:
        np.savez(args.dump_preds, seeds=np.array([r[0] for r in err_all]),
                 ec=np.stack([e for _, e, _, _ in err_all]),
                 em=np.stack([m for _, _, m, _ in err_all]),
                 gt=np.stack([g for _, _, _, g in err_all]))

    r2c = [r[1] for r in rows]; r2m = [r[2] for r in rows]
    print(f"\n=== CoNIC IN-DOMAIN efflite0 ({len(seeds)} seed, split theo ảnh gốc) — bộ thứ 4 ===")
    print(f"  count-only  R² {np.mean(r2c):+.4f} ± {np.std(r2c):.4f} | "
          f"MAE {np.mean([r[3] for r in rows]):.3f} ± {np.std([r[3] for r in rows]):.3f}")
    print(f"  mask        R² {np.mean(r2m):+.4f} ± {np.std(r2m):.4f} | "
          f"MAE {np.mean([r[4] for r in rows]):.3f} ± {np.std([r[4] for r in rows]):.3f}")
    print(f"  ΔR² = {np.mean(r2c) - np.mean(r2m):+.4f}")
    try:
        from equivalence import tost
        ds = []
        for seed, ec, em, gt in err_all:
            r = tost(ec, em, gt=gt, delta_frac=args.delta_frac); ds.append(r["delta_star"])
            print(f"  [seed{seed}] Δ={r['dm']:+.3f} CI90 [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
                  f"δ*={r['delta_star']:.3f}")
        gm = float(np.mean([g.mean() for _, _, _, g in err_all]))
        print(f"  δ* trung vị = {np.median(ds):.3f} nhân = {100 * np.median(ds) / gm:.1f}% số đếm TB")
    except Exception as e:
        print(f"  [TOST lỗi] {e}")


if __name__ == "__main__":
    main()
