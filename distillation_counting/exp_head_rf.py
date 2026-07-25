#!/usr/bin/env python3
"""exp_head_rf.py — TEST: phóng to receptive-field của ĐẦU density có giúp không?

Tiền đề (cơ chế đo được): head 1×1 không gộp không gian lúc readout -> nhiễu điểm lẻ trên ảnh gần trống
-> over-count vùng THƯA (§3.1b). Head to hơn (3×3 / 3×3-dilated) = làm mượt cục bộ -> dập nhiễu lẻ ->
có thể giảm over-count thưa. Test thẳng trên PanNuke fold3 (power cao), đo GLOBAL + LỆCH THEO TẦNG.

So sánh cùng efflite0 count-only, chỉ đổi head:
  1x1 (hiện tại) | 3x3 d1 | 3x3 d2 | 5x5 d1

ĐỌC: config nào giảm over-count tầng thưa MÀ không phá tầng dày -> RF-head là đòn bẩy thật -> giữ.
     Không cái nào hơn 1×1 rõ (ngoài nhiễu ±0.005) -> RF-head KHÔNG phải đòn bẩy -> loại có bằng chứng.

Chạy Kaggle (GPU; đính hipinhththu/pannuke). ~40 phút/config:
  !python exp_head_rf.py --pannuke_root /kaggle/input/datasets/hipinhththu/pannuke --seeds 42,43
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
    tid = np.digitize(gt, edges); out = []
    for t in range(len(edges) + 1):
        m = tid == t
        out.append(float((pr[m] - gt[m]).mean()) if m.sum() else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--exclude_tissue", default="")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", default="42,43")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_pannuke_data(args.pannuke_root, [1, 2], args.exclude_tissue, 0)
    n_tr = len(data)
    data += load_pannuke_data(args.pannuke_root, [3], args.exclude_tissue, 0)
    tr, te = list(range(n_tr)), list(range(n_tr, len(data)))
    print(f"[data] train {len(tr)} / test {len(te)} (full fold3)")

    configs = [("1x1 (hiện tại)", 1, 1), ("3x3 d1", 3, 1), ("3x3 d2", 3, 2), ("5x5 d1", 5, 1)]
    seeds = [int(s) for s in args.seeds.split(",")]
    edges = np.array([10, 20, 30, 45], float)
    tier_names = ["<10", "10-20", "20-30", "30-45", "45+"]

    rows = []
    for name, hk, hd in configs:
        r2s, maes, tiers = [], [], []
        for sd in seeds:
            np.random.seed(sd); torch.manual_seed(sd)
            model = train(data, dev, args.epochs, 32, 1e-3, list(tr), 0.0, 1.0, 0.01, 0.5, 16, True,
                          "poisson", "efficientnet_lite0", head_k=hk, head_d=hd)
            pr, gt = predict(model, data, te, dev)
            r2, mae = r2_mae(gt, pr); r2s.append(r2); maes.append(mae)
            tiers.append(tier_bias(gt, pr, edges))
        npar = sum(p.numel() for p in model.parameters()) / 1e6
        tb = np.nanmean(tiers, axis=0)
        rows.append((name, npar, np.mean(r2s), np.std(r2s), np.mean(maes), tb))
        print(f"  {name:16s} {npar:5.2f}M  R² {np.mean(r2s):+.4f}±{np.std(r2s):.3f}  MAE {np.mean(maes):.3f}  "
              f"bias/tầng " + " ".join(f"{t:+.1f}" for t in tb))

    print(f"\n=== SO HEAD RF (efflite0 count-only, PanNuke fold3, {len(seeds)} seed) ===")
    print(f"  {'config':16s} {'params':>7s} {'R² ↑':>16s} {'MAE ↓':>8s}   bias theo tầng: " +
          " ".join(f"{n:>6s}" for n in tier_names))
    base = rows[0]
    for name, npar, r2m, r2s, maem, tb in rows:
        d = f"(ΔR² {r2m-base[2]:+.4f})" if name != base[0] else "(baseline)"
        print(f"  {name:16s} {npar:6.2f}M {r2m:+.4f}±{r2s:.3f} {maem:8.3f}   " +
              " ".join(f"{t:>+6.1f}" for t in tb) + f"  {d}")
    print("\nĐỌC: cột 'bias theo tầng' <10 nếu ÂM dần khi head to lên -> dập được over-count thưa (cơ chế đúng).")
    print("     ΔR² > +0.01 & không phá tầng dày -> head to là đòn bẩy THẬT -> ghép chính thức + xác nhận thêm seed.")
    print("     ΔR² quanh 0 (±0.005) -> RF-head KHÔNG giúp -> loại CÓ bằng chứng, sang hướng khác (feature/scale).")


if __name__ == "__main__":
    main()
