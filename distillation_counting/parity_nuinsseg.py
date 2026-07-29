#!/usr/bin/env python3
"""parity_nuinsseg.py — count-only vs mask (GT-density) trên NuInsSeg + PER-IMAGE WILCOXON.

Dataset THỨ 3 cho headline parity Q1 (cùng PanNuke §3.2b + MoNuSeg). Cô lập supervision:
CÙNG efflite0, CÙNG data (GT density từ mask NuInsSeg), CHỈ đổi loss.
  - count-only : w_density=0, w_count=1
  - mask       : w_density=1, w_count=w_count_mask (0.1; 0.01 SẬP trên NuInsSeg dải count rộng)

Split CỐ ĐỊNH (1 seed) -> per-image paired test trên ~130 ảnh test (power CAO hơn so 3-seed R²):
  d(i) = |GT_i - pred_count_i| - |GT_i - pred_mask_i|;  Wilcoxon signed-rank trên {d(i)}.
  p>0.05 => PARITY (không phân biệt); p<0.05 & d<0 => count-only THẮNG có ý nghĩa.

Chạy Kaggle (GPU; cần mask NuInsSeg để build GT density, ~5-10 phút lần đầu):
  !python parity_nuinsseg.py --nuinsseg_root /kaggle/input/datasets/ipateam/nuinsseg --seed 42
  # (tùy chọn) copy cache có sẵn để khỏi build:
  #   cp <backup>/gt_density_nuinsseg.pkl work/gt_density_nuinsseg.pkl
"""
import argparse, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "kaggle", "lib"), _REPO, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
from distill_student_nuinsseg import build_index, find_root
from distill_student_r2 import build_teacher_density, train
from r2_losses import count_from_density


@torch.no_grad()
def per_image_pred(model, data, idx, dev):
    model.eval(); pr = []
    for i in idx:
        x = torch.from_numpy(data[i]["img"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        pr.append(float(count_from_density(model(x)[0])[0]))
    return np.array(pr)


def r2mae(pr, yt):
    ss = ((yt - pr) ** 2).sum(); st = ((yt - yt.mean()) ** 2).sum()
    return (1 - ss / st if st > 0 else float("nan")), float(np.abs(pr - yt).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuinsseg_root", default=None)
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", default="42,43,44,45,46", help="mỗi seed 1 split -> per-image Wilcoxon; xem nhất quán")
    ap.add_argument("--w_count_mask", type=float, default=0.1)
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--deterministic", action="store_true",
                    help="bật cuDNN tất định — CHẬM hơn nhiều với efflite0. NuInsSeg nhỏ+ổn định KHÔNG cần "
                         "-> mặc định TẮT (benchmark=True) cho nhanh.")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:                                       # work_dir có thể là dataset /kaggle/input READ-ONLY
        os.makedirs(args.work_dir, exist_ok=True)
    except OSError as e:
        print(f"[work_dir] không tạo được ({e}) — chỉ ĐỌC cache, ok nếu pkl đã có sẵn")
    from scipy.stats import wilcoxon

    samples = build_index(args.nuinsseg_root or find_root())
    data = build_teacher_density(samples, dev, f"{args.work_dir}/gt_density_nuinsseg.pkl", use_gt=True)
    n = len(data)
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"[data] NuInsSeg {n} ảnh (GT density từ mask) | {len(seeds)} seed")

    r2c_all, r2m_all, rows = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed); idx = rng.permutation(n); n_te = n // 5
        te, tr = idx[:n_te], idx[n_te:]
        gt = np.array([data[i]["gt"] for i in te])
        preds = {}
        for name, wd, wc in [("count-only", 0.0, 1.0), ("mask (GT-density)", 1.0, args.w_count_mask)]:
            np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            if args.deterministic:
                torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True, warn_only=True)
            else:
                torch.backends.cudnn.benchmark = True   # NHANH; NuInsSeg nhỏ+ổn định nên không cần tất định
            model = train(data, dev, args.epochs, args.ch, 1e-3, list(tr),
                          wd, wc, 0.01, 0.5, 16, True, "poisson", args.backbone)
            preds[name] = per_image_pred(model, data, te, dev)
        ec = np.abs(gt - preds["count-only"]); em = np.abs(gt - preds["mask (GT-density)"])
        r2c, _ = r2mae(preds["count-only"], gt); r2m, _ = r2mae(preds["mask (GT-density)"], gt)
        try:
            _, p = wilcoxon(ec, em)
        except Exception as e:
            p = float("nan"); print(f"[wilcoxon lỗi seed{seed}] {e}")
        dmean = float((ec - em).mean())
        r2c_all.append(r2c); r2m_all.append(r2m); rows.append((seed, r2c, r2m, ec.mean(), em.mean(), dmean, p))
        vd = "parity" if p > 0.05 else ("count>mask" if dmean < 0 else "count<mask")
        print(f"  [seed{seed}] R² count {r2c:+.4f} / mask {r2m:+.4f} | |err| {ec.mean():.2f} vs {em.mean():.2f} "
              f"| p={p:.4g} -> {vd}")

    n_sig_win = sum(1 for r in rows if r[6] < 0.05 and r[5] < 0)
    n_win_dir = sum(1 for r in rows if r[5] < 0)
    print(f"\n=== NuInsSeg PARITY + per-image Wilcoxon (efflite0, {len(seeds)} seed) ===")
    print(f"  count-only R² {np.mean(r2c_all):+.4f} ± {np.std(r2c_all):.4f} | "
          f"mask R² {np.mean(r2m_all):+.4f} ± {np.std(r2m_all):.4f}")
    print(f"  count-only lỗi ÍT hơn ở {n_win_dir}/{len(seeds)} seed | p<0.05 ở {n_sig_win}/{len(seeds)} seed")
    print(f"  -> {'count-only ≥ mask nhất quán' if n_win_dir == len(seeds) else 'không nhất quán — xem lại'}"
          f"{' (thắng có ý nghĩa)' if n_sig_win else ' (parity)'}")


if __name__ == "__main__":
    main()
