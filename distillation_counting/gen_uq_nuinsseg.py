#!/usr/bin/env python3
"""gen_uq_nuinsseg.py — sinh pkl preds (mu, sigma) efflite0 count-only trên NuInsSeg (in-domain).

Mục đích: ablation Poisson-σ vs raw-σ TRÊN NuInsSeg — dải count RỘNG (0..500+), nơi raw σ
(1 hằng số exp(log_s)) KHÔNG cover nổi cả nhân thưa lẫn dày, còn Poisson σ=√μ·exp(log_s)
tự co-giãn theo μ. Trên PanNuke (dải hẹp) hai cái gần bằng; ở đây kỳ vọng Poisson thắng RÕ.

Model = DensitySigmaUNet(ch=32, efficientnet_lite0), count-only (w_density=0, w_count=1),
w_nll=0.01, detach_mu — GIỐNG HỆT frontier in-domain NuInsSeg 0.854.
Split GIỐNG method_frontier NuInsSeg: rng(seed), n_te=n//5, te=idx[:n_te]  -> so cùng nền.
Xuất {"preds":[{mu,sigma}], "gts":[[gt]], "organs":[org]} — khớp eval_r2_grouped/baselines_uq.

Chạy Kaggle (GPU):
  # Poisson (headline)
  !python gen_uq_nuinsseg.py --nuinsseg_root /kaggle/input/datasets/ipateam/nuinsseg \\
      --sigma_mode poisson --seed 42 --out work/efflite0_nuins_countonly_uq_poisson_s42.pkl
  # raw (ablation)
  !python gen_uq_nuinsseg.py --nuinsseg_root /kaggle/input/datasets/ipateam/nuinsseg \\
      --sigma_mode raw     --seed 42 --out work/efflite0_nuins_countonly_uq_raw_s42.pkl
Rồi feed cả 2 pkl vào eval_r2_grouped.py để so cov/Winkler/worst-org.
"""
import argparse, os, pickle
import numpy as np
import torch
from PIL import Image
from distill_student_nuinsseg import build_index, find_root, IMG_SIZE
from method_frontier import gt_from_mask
from distill_student_r2 import train, predict_r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuinsseg_root", default=None)
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42, help="split seed (khớp method_frontier NuInsSeg)")
    ap.add_argument("--sigma_mode", default="poisson", choices=["poisson", "raw", "nb"])
    ap.add_argument("--out", default="work/efflite0_nuins_countonly_uq.pkl")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    root = args.nuinsseg_root or find_root()
    samples = build_index(root)
    data = []
    for s in samples:
        im = np.asarray(Image.open(s["image"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        data.append({"img": im.astype(np.uint8),
                     "density": np.zeros((IMG_SIZE, IMG_SIZE), np.float32),
                     "gt": float(gt_from_mask(s["mask"])), "organ": s["organ"]})
    n = len(data)
    print(f"[data] NuInsSeg {n} ảnh (root={root})")

    # split GIỐNG method_frontier: rng(seed), n_te = n//5, te=idx[:n_te]
    rng = np.random.default_rng(args.seed); idx = rng.permutation(n); n_te = n // 5
    te, tr = idx[:n_te], idx[n_te:]
    test_data = [data[i] for i in te]
    gt_mean = np.mean([data[i]["gt"] for i in te])
    print(f"[split] seed{args.seed}: train {len(tr)} / test {len(te)} | GT mean {gt_mean:.1f} "
          f"| σ_mode={args.sigma_mode}")

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    # train(data, device, epochs, ch, lr, train_idx, w_density, w_count, w_nll, beta, bs, detach_mu, sigma_mode, backbone)
    model = train(data, dev, args.epochs, args.ch, 1e-3, list(tr),
                  0.0, 1.0, 0.01, 0.5, 16, True, args.sigma_mode, args.backbone)

    out = predict_r2(model, test_data, dev)
    gt = np.array([g[0] for g in out["gts"]]); pr = np.array([p["mu"] for p in out["preds"]])
    ss = ((gt - pr) ** 2).sum(); st = ((gt - gt.mean()) ** 2).sum()
    print(f"[check] R²={1-ss/st:+.4f}  MAE={np.abs(gt-pr).mean():.3f}  "
          f"σ mean={np.mean([p['sigma'] for p in out['preds']]):.2f}  N={len(gt)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"[saved] {args.out}  -> feed vào eval_r2_grouped.py")


if __name__ == "__main__":
    main()
