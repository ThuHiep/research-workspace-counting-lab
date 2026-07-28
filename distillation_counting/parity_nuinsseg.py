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
    ap.add_argument("--seed", type=int, default=42, help="split cố định (per-image pairing)")
    ap.add_argument("--w_count_mask", type=float, default=0.1)
    ap.add_argument("--work_dir", default="work")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.work_dir, exist_ok=True)

    samples = build_index(args.nuinsseg_root or find_root())
    data = build_teacher_density(samples, dev, f"{args.work_dir}/gt_density_nuinsseg.pkl", use_gt=True)
    n = len(data)
    rng = np.random.default_rng(args.seed); idx = rng.permutation(n); n_te = n // 5
    te, tr = idx[:n_te], idx[n_te:]
    gt = np.array([data[i]["gt"] for i in te])
    print(f"[data] NuInsSeg {n} ảnh | train {len(tr)} / test {len(te)} | GT mean {gt.mean():.1f} (GT density từ mask)")

    configs = [("count-only", 0.0, 1.0), ("mask (GT-density)", 1.0, args.w_count_mask)]
    preds = {}
    for name, wd, wc in configs:
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        model = train(data, dev, args.epochs, args.ch, 1e-3, list(tr),
                      wd, wc, 0.01, 0.5, 16, True, "poisson", args.backbone)
        pr = per_image_pred(model, data, te, dev)
        preds[name] = pr
        r2, mae = r2mae(pr, gt)
        print(f"  [{name:18s}] R² {r2:+.4f}  MAE {mae:.3f}")

    err_c = np.abs(gt - preds["count-only"])
    err_m = np.abs(gt - preds["mask (GT-density)"])
    r2c, _ = r2mae(preds["count-only"], gt); r2m, _ = r2mae(preds["mask (GT-density)"], gt)
    dmean = float((err_c - err_m).mean())
    from scipy.stats import wilcoxon
    try:
        stat, p = wilcoxon(err_c, err_m)
    except Exception as e:
        stat, p = float("nan"), float("nan"); print(f"[wilcoxon lỗi] {e}")

    print(f"\n=== NuInsSeg PARITY + per-image Wilcoxon (efflite0, seed{args.seed}, n_test={len(te)}) ===")
    print(f"  count-only  R² {r2c:+.4f} | mask R² {r2m:+.4f} | ΔR² {r2c - r2m:+.4f}")
    print(f"  per-image |err|: count-only {err_c.mean():.2f} vs mask {err_m.mean():.2f} "
          f"(Δ={dmean:+.2f}, âm = count-only lỗi ÍT hơn)")
    print(f"  Wilcoxon signed-rank p = {p:.4g}")
    verdict = ("PARITY (không phân biệt, p>0.05)" if p > 0.05 else
               ("count-only THẮNG có ý nghĩa (p<0.05, lỗi ít hơn)" if dmean < 0 else
                "count-only THUA có ý nghĩa (p<0.05, lỗi nhiều hơn)"))
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
