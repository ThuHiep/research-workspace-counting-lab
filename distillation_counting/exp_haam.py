#!/usr/bin/env python3
"""exp_haam.py — TEST module D-HAAM (HAAM của user, adapt cho counting) trên PanNuke fold3.

So 3 config, cùng efflite0 count-only, chỉ khác attn trước đầu density:
  none      : baseline (không attn)
  haam      : D-HAAM (CA + SA 1-scale + cổng α), đã sửa lỗi M.max/fusion
  haam_ms   : D-HAAM + SA ĐA-TỈ-LỆ (dilated 1/2/4) -> nhắm saturation vùng dày

ĐỌC: ΔR² > +0.01 & không phá tầng nào -> HAAM transfer sang counting -> thành phần novel của user.
     haam_ms > haam ở TẦNG DÀY (bias 45+ bớt âm) -> đa-scale trị saturation -> ghép cả hai.
     Quanh 0 (±0.005) -> attention không transfer sang đếm -> loại có bằng chứng.

Chạy Kaggle (GPU; hipinhththu/pannuke). ~40 phút/config:
  !python exp_haam.py --pannuke_root /kaggle/input/datasets/hipinhththu/pannuke --seeds 42
"""
import argparse
import numpy as np
import torch
from method_frontier import load_pannuke_data
from distill_student_r2 import train
from r2_losses import count_from_density


def r2_mae(gt, pr):
    gt, pr = np.asarray(gt, float), np.asarray(pr, float)
    ss = ((gt - pr) ** 2).sum(); st = ((gt - gt.mean()) ** 2).sum()
    return (1 - ss / st if st > 0 else float("nan")), np.abs(gt - pr).mean()


@torch.no_grad()
def predict(model, data, idx, dev):
    model.eval(); pr, gt = [], []
    for i in idx:
        x = torch.from_numpy(data[i]["img"].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
        pr.append(float(count_from_density(model(x)[0])[0])); gt.append(data[i]["gt"])
    return np.array(pr), np.array(gt)


def tier_bias(gt, pr, edges):
    tid = np.digitize(gt, edges)
    return [float((pr[tid == t] - gt[tid == t]).mean()) if (tid == t).sum() else float("nan")
            for t in range(len(edges) + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--exclude_tissue", default="")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--backbone", default="efficientnet_lite0",
                    help="encoder; thử 'efficientnet_b0' (sân nhà HAAM, có SE) để đối chiếu")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_pannuke_data(args.pannuke_root, [1, 2], args.exclude_tissue, 0)
    n_tr = len(data)
    data += load_pannuke_data(args.pannuke_root, [3], args.exclude_tissue, 0)
    tr, te = list(range(n_tr)), list(range(n_tr, len(data)))
    print(f"[data] train {len(tr)} / test {len(te)} (full fold3)")

    configs = [("none (baseline)", "none"), ("cbam", "cbam"),
               ("haam", "haam"), ("haam_ms (đa-scale)", "haam_ms")]
    seeds = [int(s) for s in args.seeds.split(",")]
    edges = np.array([10, 20, 30, 45], float)
    tier_names = ["<10", "10-20", "20-30", "30-45", "45+"]

    rows = []
    for name, attn in configs:
        r2s, maes, tiers, npar = [], [], [], 0
        for sd in seeds:
            np.random.seed(sd); torch.manual_seed(sd)
            model = train(data, dev, args.epochs, 32, 1e-3, list(tr), 0.0, 1.0, 0.01, 0.5, 16, True,
                          "poisson", args.backbone, attn=attn)
            pr, gt = predict(model, data, te, dev)
            r2, mae = r2_mae(gt, pr); r2s.append(r2); maes.append(mae); tiers.append(tier_bias(gt, pr, edges))
            npar = sum(p.numel() for p in model.parameters()) / 1e6
        rows.append((name, npar, np.mean(r2s), np.std(r2s), np.mean(maes), np.nanmean(tiers, axis=0)))
        print(f"  {name:18s} {npar:5.2f}M  R² {np.mean(r2s):+.4f}±{np.std(r2s):.3f}  MAE {np.mean(maes):.3f}  "
              + " ".join(f"{t:+.1f}" for t in np.nanmean(tiers, axis=0)))

    print(f"\n=== D-HAAM trên PanNuke fold3 (efflite0 count-only, {len(seeds)} seed) ===")
    print(f"  {'config':18s} {'params':>7s} {'R² ↑':>16s} {'MAE ↓':>8s}   bias tầng: " +
          " ".join(f"{n:>6s}" for n in tier_names))
    base = rows[0]
    for name, npar, r2m, r2s, maem, tb in rows:
        d = f"(ΔR² {r2m-base[2]:+.4f})" if name != base[0] else "(baseline)"
        print(f"  {name:18s} {npar:6.2f}M {r2m:+.4f}±{r2s:.3f} {maem:8.3f}   " +
              " ".join(f"{t:>+6.1f}" for t in tb) + f"  {d}")
    print("\nĐỌC: haam ΔR²>+0.01 -> HAAM transfer sang counting (thành phần của user).")
    print("     haam_ms tầng 45+ bớt âm hơn haam/baseline -> đa-scale trị saturation vùng dày.")
    print("     cả hai quanh 0 -> attention không giúp đếm -> loại có bằng chứng, giữ hướng khác.")


if __name__ == "__main__":
    main()
