#!/usr/bin/env python3
"""diagnose_tier_bias.py — ĐO lỗi thật của efflite0 count-only trên PanNuke fold_3.

MỤC ĐÍCH (test premise TRƯỚC khi thiết kế module): lỗi còn lại (R² 0.920) là
  - CÓ CẤU TRÚC (lệch hệ thống theo tầng mật độ / theo tissue) -> sửa được bằng kiến trúc/calibration, hay
  - NHIỄU (ngẫu nhiên per-ảnh, mơ hồ nhãn) -> bất khả sửa bằng accuracy-method -> sang UQ.
Và ĐO HƯỚNG lệch: over-count vùng thưa? under-count vùng dày (saturation)? -> quyết module nhắm cái gì.

Cho 2 con số QUYẾT ĐỊNH:
  - ORACLE-tier ceiling: nếu biết tầng (bin theo GT) và trừ đúng bias trung bình mỗi tầng -> R²/MAE lên tới đâu.
    = TRẦN của mọi method tier-debias. Nhỏ -> accuracy đã cạn (bằng chứng, không phải đoán).
  - Realistic (bin theo PRED, không dùng GT): count-conditional calibration THỰC TẾ đạt được bao nhiêu.

Chạy Kaggle (GPU; đính dataset PanNuke gốc hipinhththu/pannuke):
  !python diagnose_tier_bias.py --pannuke_root /kaggle/input/datasets/hipinhththu/pannuke --seed 42
"""
import argparse
import numpy as np
import torch
from method_frontier import load_pannuke_data, eval_r2_mae
from distill_student_r2 import train
from r2_losses import count_from_density


def r2_mae(gt, pr):
    gt, pr = np.asarray(gt, float), np.asarray(pr, float)
    ss_res = ((gt - pr) ** 2).sum(); ss_tot = ((gt - gt.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, np.abs(gt - pr).mean()


@torch.no_grad()
def predict(model, data, idx, dev):
    model.eval(); pr, gt, org = [], [], []
    for i in idx:
        x = torch.from_numpy(data[i]["img"].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
        pr.append(float(count_from_density(model(x)[0])[0]))
        gt.append(data[i]["gt"]); org.append(data[i].get("organ", "na"))
    return np.array(pr), np.array(gt), np.array(org)


def tier_table(gt, pr, edges):
    """In lệch trung bình + MAE theo tầng mật độ; trả mapping tier->bias để oracle."""
    print(f"\n  {'tầng (GT count)':>16s} {'N':>5s} {'GT tb':>7s} {'Pred tb':>8s} {'bias(P-G)':>10s} {'MAE':>7s}")
    tier_bias = {}
    tid = np.digitize(gt, edges)
    for t in range(len(edges) + 1):
        m = tid == t
        if m.sum() == 0:
            continue
        lo = edges[t - 1] if t > 0 else 0
        hi = edges[t] if t < len(edges) else 999
        bias = float((pr[m] - gt[m]).mean())
        tier_bias[t] = bias
        print(f"  [{lo:>4.0f},{hi:>4.0f})    {m.sum():>5d} {gt[m].mean():>7.1f} {pr[m].mean():>8.1f} "
              f"{bias:>+10.2f} {np.abs(pr[m]-gt[m]).mean():>7.2f}")
    return tier_bias, tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--exclude_tissue", default="")   # "" = full fold3 (khớp 0.920)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_folds = [int(x) for x in args.train_folds.split(",")]
    data = load_pannuke_data(args.pannuke_root, tr_folds, args.exclude_tissue, 0)
    n_tr = len(data)
    data += load_pannuke_data(args.pannuke_root, [int(args.test_fold)], args.exclude_tissue, 0)
    tr, te = list(range(n_tr)), list(range(n_tr, len(data)))
    print(f"[data] train {len(tr)} / test {len(te)} (full fold3, colon giữ)")

    print(f"[train] efflite0 count-only seed {args.seed} ({args.epochs} ep) ...")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = train(data, dev, args.epochs, 32, 1e-3, tr, 0.0, 1.0, 0.01, 0.5, 16, True,
                  "poisson", "efficientnet_lite0")
    pr, gt, org = predict(model, data, te, dev)
    r2, mae = r2_mae(gt, pr)
    print(f"\n=== efflite0 count-only fold3: R² {r2:+.4f} | MAE {mae:.3f} (GT mean {gt.mean():.1f}) ===")

    # 1) LỆCH THEO TẦNG mật độ
    edges = np.array([10, 20, 30, 45], float)
    tier_bias, tid = tier_table(gt, pr, edges)

    # 2) ORACLE-tier: trừ bias trung bình mỗi tầng (bin theo GT = upper bound)
    pr_oracle = pr - np.array([tier_bias.get(t, 0.0) for t in tid])
    r2_o, mae_o = r2_mae(gt, pr_oracle)
    print(f"\n  [ORACLE-tier] trừ bias mỗi tầng (biết GT-tier): R² {r2_o:+.4f} | MAE {mae_o:.3f}")
    print(f"     -> TRẦN tier-debias: ΔR² {r2_o-r2:+.4f}, ΔMAE {mae_o-mae:+.3f}")

    # 3) REALISTIC: calibration bin theo PRED (không dùng GT) — cái đạt được thật
    ped = np.array([10, 20, 30, 45], float)
    pid = np.digitize(pr, ped)
    pr_real = pr.copy()
    for t in np.unique(pid):
        m = pid == t
        # dịch pred về gt trung bình của bin đó (isotonic-ish, dùng thống kê train-like trên chính test = optimistic nhẹ)
        pr_real[m] = pr[m] + (gt[m].mean() - pr[m].mean())
    r2_r, mae_r = r2_mae(gt, pr_real)
    print(f"  [REALISTIC] calibration bin theo PRED: R² {r2_r:+.4f} | MAE {mae_r:.3f}  (ΔR² {r2_r-r2:+.4f})")

    # 4) THEO TISSUE (lệch có tập trung ở loại mô nào?)
    print(f"\n  {'tissue':>14s} {'N':>5s} {'bias(P-G)':>10s} {'MAE':>7s}")
    for o in sorted(set(org)):
        m = org == o
        if m.sum() >= 15:
            print(f"  {o:>14s} {m.sum():>5d} {float((pr[m]-gt[m]).mean()):>+10.2f} {np.abs(pr[m]-gt[m]).mean():>7.2f}")

    print("\n=== ĐỌC KẾT QUẢ ===")
    print("  • bias các tầng đổi dấu (thưa + / dày −) & |bias| lớn -> LỖI CÓ CẤU TRÚC, module/calibration sửa được.")
    print("  • ORACLE-tier ΔR² lớn (vd >+0.03) -> CÓ headroom -> đáng xây đầu de-bias count-conditional.")
    print("  • ORACLE-tier ΔR² tí hon (<+0.01) -> accuracy ĐÃ CẠN (bằng chứng) -> dồn lực UQ, KHÔNG xây accuracy-module.")
    print("  • REALISTIC ~ ORACLE -> calibration đơn giản đủ; REALISTIC ≪ ORACLE -> cần tín hiệu ngoài count (khó, cân nhắc).")
    print("  • bias tập trung 1-2 tissue -> lỗi domain-specific, không phải cơ chế chung.")


if __name__ == "__main__":
    main()
