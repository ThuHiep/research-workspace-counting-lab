#!/usr/bin/env python3
"""exp_loss.py — KHUNG TỔ HỢP (A: loss × B: calibration) trị compression đếm, count-only.

Chẩn đoán: lỗi = compression đổi dấu theo tầng (thưa DƯƠNG, dày ÂM); thay-đổi-đồng-nhất (head-RF) KHÔNG
sửa được. Fix đúng phải PHI-ĐỒNG-NHẤT / trên-trục-scalar-count.

  A (loss, lúc train) — count-DEPENDENT reweighting để model ngừng bỏ rơi đuôi count-cao:
     l1      : |mu-gt| trung bình (baseline)
     lds     : L1 có trọng số nghịch-tần-suất làm mượt (imbalanced regression, Yang 2021)
     blend   : λ·l1 + (1-λ)·lds
  B (calibration, hậu kỳ, trên VAL held-out) — sửa compression trên TRỤC scalar count:
     linear  : pred' = a·pred + b  (least-squares trên val; dốc a>1 mở lại dynamic range)
     -> áp cho MỌI variant; báo CẢ trước-B và sau-B để QUY CÔNG (A làm gì, B làm gì).

KỶ LUẬT chống tự-lừa: (1) ≥3 seed vì nhiễu GPU ~0.017; (2) VAL tách từ train (KHÔNG đụng test fold3);
(3) đọc PER-TIER (cơ chế), không chỉ R²; (4) B fit trên val, eval test.

Chạy Kaggle (GPU; hipinhththu/pannuke). ~40 phút/(config×seed):
  !python exp_loss.py --pannuke_root /kaggle/input/datasets/hipinhththu/pannuke --seeds 42,43,44
"""
import argparse
import numpy as np
import torch
from method_frontier import load_pannuke_data
from distill_student_r2 import DensitySigmaUNet
from r2_losses import count_from_density, gaussian_nll


def r2_mae(gt, pr):
    gt, pr = np.asarray(gt, float), np.asarray(pr, float)
    ss = ((gt - pr) ** 2).sum(); st = ((gt - gt.mean()) ** 2).sum()
    return (1 - ss / st if st > 0 else float("nan")), np.abs(gt - pr).mean()


def tier_bias(gt, pr, edges):
    tid = np.digitize(gt, edges)
    return [float((pr[tid == t] - gt[tid == t]).mean()) if (tid == t).sum() else float("nan")
            for t in range(len(edges) + 1)]


def lds_weights(gts, bin_w=5, ksigma=2.0):
    """Trọng số nghịch-tần-suất làm mượt (LDS): count hiếm (đuôi) -> trọng số lớn. mean=1."""
    gts = np.asarray(gts); bins = (gts // bin_w).astype(int); nb = int(bins.max()) + 1
    hist = np.bincount(bins, minlength=nb).astype(float)
    ks = int(ksigma * 3) | 1
    k = np.exp(-0.5 * (np.arange(-(ks // 2), ks // 2 + 1) / ksigma) ** 2); k /= k.sum()
    sm = np.maximum(np.convolve(hist, k, mode="same"), 1.0)
    wb = 1.0 / np.sqrt(sm)
    w = wb[bins]; return w / w.mean()


def train_variant(data, tr, dev, epochs, mode, lam, seed, w_nll=0.01, beta=0.5, bs=16):
    np.random.seed(seed); torch.manual_seed(seed)
    model = DensitySigmaUNet(32, backbone="efficientnet_lite0").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gts = np.array([data[j]["gt"] for j in tr])
    w_all = {j: float(w) for j, w in zip(tr, lds_weights(gts))} if mode in ("lds", "blend") else None
    tr = list(tr)
    model.train()
    for ep in range(epochs):
        np.random.shuffle(tr)
        for i in range(0, len(tr), bs):
            idxs = tr[i:i + bs]
            imgs = torch.from_numpy(np.stack([data[j]["img"] for j in idxs]).astype(np.float32) / 255.
                                    ).permute(0, 3, 1, 2).to(dev)
            gt = torch.tensor([data[j]["gt"] for j in idxs], device=dev, dtype=torch.float32)
            dens, log_sigma = model(imgs)
            mu = count_from_density(dens)
            ae = (mu - gt).abs()
            if mode == "l1":
                lc = ae.mean()
            else:
                w = torch.tensor([w_all[j] for j in idxs], device=dev, dtype=torch.float32)
                lc_w = (w * ae).mean()
                lc = lc_w if mode == "lds" else lam * ae.mean() + (1 - lam) * lc_w
            nll = gaussian_nll(mu, log_sigma, gt, beta=beta, detach_mu=True)
            loss = lc + w_nll * nll
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model


@torch.no_grad()
def predict(model, data, idx, dev):
    model.eval()
    pr = np.array([float(count_from_density(model(
        torch.from_numpy(data[i]["img"].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev))[0])[0])
        for i in idx])
    gt = np.array([data[i]["gt"] for i in idx])
    return pr, gt


def fit_linear(pr_val, gt_val):
    A = np.vstack([pr_val, np.ones_like(pr_val)]).T
    a, b = np.linalg.lstsq(A, gt_val, rcond=None)[0]
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--exclude_tissue", default="")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--val_frac", type=float, default=0.15)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_pannuke_data(args.pannuke_root, [1, 2], args.exclude_tissue, 0)
    n_tr = len(data)
    data += load_pannuke_data(args.pannuke_root, [3], args.exclude_tissue, 0)
    all_tr, te = list(range(n_tr)), list(range(n_tr, len(data)))
    print(f"[data] train {len(all_tr)} / test {len(te)} (full fold3)")

    edges = np.array([10, 20, 30, 45], float)
    tier_names = ["<10", "10-20", "20-30", "30-45", "45+"]
    configs = [("l1 (baseline)", "l1", 0.0), ("lds", "lds", 0.0), ("blend λ0.5", "blend", 0.5)]
    seeds = [int(s) for s in args.seeds.split(",")]

    print("\n=== A×B (efflite0 count-only, PanNuke fold3) — raw & +B(linear calib trên val) ===")
    for name, mode, lam in configs:
        raw, cal, tiers_raw, tiers_cal = [], [], [], []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            perm = rng.permutation(all_tr); nv = int(len(perm) * args.val_frac)
            val_idx, tr_idx = list(perm[:nv]), list(perm[nv:])
            model = train_variant(data, tr_idx, dev, args.epochs, mode, lam, sd)
            pr_v, gt_v = predict(model, data, val_idx, dev)
            a, b = fit_linear(pr_v, gt_v)
            pr_t, gt_t = predict(model, data, te, dev)
            r2, mae = r2_mae(gt_t, pr_t); raw.append((r2, mae)); tiers_raw.append(tier_bias(gt_t, pr_t, edges))
            prc = np.maximum(a * pr_t + b, 0)
            r2c, maec = r2_mae(gt_t, prc); cal.append((r2c, maec)); tiers_cal.append(tier_bias(gt_t, prc, edges))
        r = np.array(raw); c = np.array(cal)
        tb_r = np.nanmean(tiers_raw, 0); tb_c = np.nanmean(tiers_cal, 0)
        print(f"\n  [{name}]  ({len(seeds)} seed)")
        print(f"    raw : R² {r[:,0].mean():+.4f}±{r[:,0].std():.3f}  MAE {r[:,1].mean():.3f}   "
              f"bias " + " ".join(f"{t:+.1f}" for t in tb_r))
        print(f"    +B  : R² {c[:,0].mean():+.4f}±{c[:,0].std():.3f}  MAE {c[:,1].mean():.3f}   "
              f"bias " + " ".join(f"{t:+.1f}" for t in tb_c))

    print(f"\n  (tầng: {tier_names})")
    print("\nĐỌC (kỷ luật):")
    print("  • so RAW giữa l1/lds/blend -> A (loss) có tự sửa đuôi 45+ không (bias bớt âm) MÀ không phá thưa?")
    print("  • so raw vs +B cùng config -> B (calib scalar) thu thêm bao nhiêu; B nên kéo mọi tầng về ~0.")
    print("  • ΔR² < 0.02 & trong ±0.017 nhiễu -> CHƯA kết luận, cần thêm seed / xem per-tier.")
    print("  • cấu hình tốt nhất phải chọn theo VAL (ở đây in test để tham khảo) rồi xác nhận đa seed.")


if __name__ == "__main__":
    main()
