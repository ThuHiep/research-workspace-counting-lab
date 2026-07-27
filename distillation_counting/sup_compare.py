#!/usr/bin/env python3
"""sup_compare.py — THÍ NGHIỆM CHỐT: count-only vs GT-density (mask), CÙNG kiến trúc efflite0.

Cô lập DUY NHẤT biến supervision: dùng CÙNG một data (build_pannuke_density use_gt=True → có cả
GT-density map lẫn count), CÙNG efflite0+U-Net, CÙNG eval (count=Σdensity trên fold3). Chỉ đổi loss:
  - count-only      : w_density=0, w_count=1  (chỉ học từ 1 số/ảnh, KHÔNG dùng density map)
  - GT-density(mask): w_density=1, w_count=0  (học khớp density map dựng từ instance mask)

Nếu count-only ≥ GT-density → chứng minh: ĐẾM KHÔNG CẦN giám sát pixel, và cái thắng đến từ
SUPERVISION chứ KHÔNG phải kiến trúc (vì kiến trúc y hệt). Đây là finding lõi của paper.

Chạy Kaggle (GPU; cần masks.npy để build GT-density). Lần đầu build cache GT-density hơi lâu:
  !python sup_compare.py --pannuke_root /kaggle/input/datasets/hipinhththu/pannuke --seeds 42,43,44
"""
import argparse, os, sys
# chèn path chứa pannuke_loader (kaggle/lib) — repo root = cha của thư mục script
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "kaggle", "lib"), os.path.join(_REPO, "kaggle", "vast"), _REPO, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
from distill_student_r2 import build_pannuke_density, train
from method_frontier import eval_r2_mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)   # density cần lâu hơn để hội tụ; count-only đã bão hoà ~40
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("work", exist_ok=True)   # nơi lưu cache GT-density (kẻo build xong không save được)
    tr_folds = [int(x) for x in args.train_folds.split(",")]
    fstr = "".join(str(x) for x in sorted(tr_folds))
    train_data = build_pannuke_density(args.pannuke_root, tr_folds, dev,
                                       f"work/gt_density_pannuke_f{fstr}.pkl", use_gt=True)
    test_data = build_pannuke_density(args.pannuke_root, [int(args.test_fold)], dev,
                                      f"work/gt_density_pannuke_f{args.test_fold}.pkl", use_gt=True)
    data = list(train_data) + list(test_data)
    tr = list(range(len(train_data))); te = list(range(len(train_data), len(data)))
    print(f"[data] train fold{tr_folds}={len(tr)} / test fold{args.test_fold}={len(te)} "
          f"(CÙNG data, cùng efflite0 ch{args.ch}; chỉ đổi loss)")

    seeds = [int(s) for s in args.seeds.split(",")]
    # GT-density cần neo count NHỎ (0.01) để train được — density MSE thuần bị background(0) chi phối
    # → collapse về ~0 nếu w_count=0 (đúng recipe density-counting chuẩn; số cũ NuInsSeg 0.881 dùng vậy).
    configs = [("count-only", 0.0, 1.0), ("GT-density (mask)", 1.0, 0.01)]
    rows = []
    for name, wd, wc in configs:
        r2s, maes = [], []
        for sd in seeds:
            np.random.seed(sd); torch.manual_seed(sd)
            # train(data, device, epochs, ch, lr, train_idx, w_density, w_count, w_nll, beta, bs, detach_mu, sigma_mode, backbone)
            model = train(data, dev, args.epochs, args.ch, 1e-3, tr,
                          wd, wc, 0.01, 0.5, 16, True, "poisson", args.backbone)
            r2, mae = eval_r2_mae(model, data, te, dev)
            r2s.append(r2); maes.append(mae)
            print(f"    [{name:16s}] seed {sd}: R² {r2:+.4f}  MAE {mae:.2f}")   # per-seed để bắt collapse
        rows.append((name, np.mean(r2s), np.std(r2s), np.mean(maes), np.std(maes)))
        print(f"  {name:20s} R² {np.mean(r2s):+.4f}±{np.std(r2s):.3f}  MAE {np.mean(maes):.3f}±{np.std(maes):.3f}")

    print(f"\n=== SUPERVISION head-to-head (CÙNG efflite0 ch{args.ch}, {len(seeds)} seed, PanNuke fold{args.test_fold}) ===")
    print(f"  {'supervision':20s} {'R² ↑':>16s} {'MAE ↓':>14s}")
    for name, r2m, r2s, mm, ms in rows:
        print(f"  {name:20s} {r2m:+.4f}±{r2s:.3f}   {mm:6.2f}±{ms:.2f}")
    d = rows[0][1] - rows[1][1]
    print(f"\n  Δ(count-only − GT-density) = {d:+.4f} R²")
    print("  ĐỌC: count-only ≥ GT-density → ĐẾM KHÔNG CẦN pixel; cái thắng đến từ SUPERVISION, không phải kiến trúc.")


if __name__ == "__main__":
    main()
