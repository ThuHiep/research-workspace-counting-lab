#!/usr/bin/env python3
"""method_crowd_gate.py — BƯỚC 2: module cổng-crowding sửa lỗi "boost toàn cục" của Bước 1.

Bước 1 (composite-linearity loss trần): Δsat↓ + MAE-mô-dày↓ ✅ NHƯNG R²-tổng↓ + worst-organ↓ ✗.
Chẩn đoán: loss đẩy count LÊN TOÀN CỤC -> giúp mô dày (undercount) nhưng over-count mô thưa -> R²↓.
= correction KHÔNG có mục tiêu.

Sửa (module b): density-cổng SPATIAL + sparsity-prior.
  D_corr = D_base · (1 + β·g),   g = sigmoid(gate_head(features)) ∈ [0,1] mỗi pixel.
  + phạt L1 mean(g)  -> gate TẮT mặc định, chỉ BẬT nơi composite-loss ĐÒI (vùng crowding).
  => correction LOCALIZED: mô thưa gate≈0 (giữ R²), mô dày gate bật (giảm undercount).

3 nhánh ablation (cùng split, paired, multi-seed):
  baseline   = count-only (không loss, không gate)
  +composite = count + composite-linearity (KHÔNG gate)      [= Bước 1]
  +gate      = count + composite + gate spatial + sparsity   [method đầy đủ]

Gate NGHIÊM (pre-register): +gate vs baseline -> R²-tổng KHÔNG tụt (|Δ|<0.02) VÀ MAE-mô-dày giảm
  -> correction target được -> METHOD THẬT -> ablate/mở rộng. Không giữ nổi R² -> DỪNG.

Chạy Kaggle (GPU; ĐỪNG pip install):  !python method_crowd_gate.py --root /kaggle/input/datasets/ipateam/nuinsseg
"""
import argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from distill_student_nuinsseg import build_index, find_root, _load_mask, IMG_SIZE
from distill_student_r2 import DensitySigmaUNet
from r2_losses import count_from_density


def gt_from_mask(path):
    m = _load_mask(path)
    return int(len(np.unique(m)) - (1 if (m == 0).any() else 0))


def overlay_min(batch_imgs):
    return batch_imgs.min(dim=0)[0]


class CrowdGatedCounter(nn.Module):
    """DensitySigmaUNet + gate spatial: D_corr = D_base·(1+β·g). Trả (count, mean_gate)."""
    def __init__(self, ch=32, backbone="efficientnet_lite0", beta=3.0):
        super().__init__()
        self.base = DensitySigmaUNet(ch, backbone=backbone)
        self.gate = nn.Conv2d(ch, 1, 1)
        self.beta = beta

    def forward(self, x):
        y, _ = self.base._features(x)
        D = F.relu(self.base.dens(y))
        g = torch.sigmoid(self.gate(y))
        Dc = D * (1.0 + self.beta * g)
        return Dc.sum(dim=(1, 2, 3)), g.mean(dim=(1, 2, 3))


def count_of(model, x, gated):
    if gated:
        return model(x)[0]
    return count_from_density(model(x)[0])


def train_arm(imgs, gts, tr, dev, epochs, mode, w_comp, beta, lam_gate,
              bs=16, seed=0, kmax=4, n_comp=4):
    """mode: baseline | composite | gate."""
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gated = mode == "gate"
    model = (CrowdGatedCounter(32, beta=beta) if gated
             else DensitySigmaUNet(32, backbone="efficientnet_lite0")).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr = np.array(tr)
    use_comp = mode in ("composite", "gate")
    for ep in range(epochs):
        order = rng.permutation(tr); model.train()
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            x = imgs[b].to(dev)
            gt = torch.tensor(gts[b], device=dev, dtype=torch.float32)
            if gated:
                cnt, gm = model(x)
                loss = (cnt - gt).abs().mean() + lam_gate * gm.mean()   # sparsity: gate off mặc định
            else:
                loss = (count_of(model, x, False) - gt).abs().mean()
            if use_comp:
                cx, ct = [], []
                for _ in range(n_comp):
                    k = int(rng.integers(2, kmax + 1))
                    pick = rng.choice(tr, size=k, replace=False)
                    cx.append(overlay_min(imgs[pick].to(dev))); ct.append(float(gts[pick].sum()))
                cx = torch.stack(cx); ct = torch.tensor(ct, device=dev, dtype=torch.float32)
                cc = model(cx)[0] if gated else count_of(model, cx, False)
                loss = loss + w_comp * (cc - ct).abs().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model, gated


@torch.no_grad()
def preds(model, imgs, idx, dev, gated):
    model.eval()
    return np.array([float(count_of(model, imgs[i:i + 1].to(dev), gated)[0]) for i in idx])


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum(); ss_tot = ((true - true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def worst_organ_r2(pred, true, organs_te, min_n=8):
    vals = [r2(pred[organs_te == o], true[organs_te == o])
            for o in set(organs_te) if (organs_te == o).sum() >= min_n]
    return min(vals) if vals else float("nan")


@torch.no_grad()
def sat_k4(model, imgs, te, dev, gated, ntup=200, seed=0):
    model.eval(); rng = np.random.default_rng(seed); sp, ov = [], []
    for _ in range(ntup):
        pick = rng.choice(te, size=4, replace=False)
        sp.append(sum(float(count_of(model, imgs[i:i + 1].to(dev), gated)[0]) for i in pick))
        ov.append(float(count_of(model, overlay_min(imgs[pick].to(dev))[None], gated)[0]))
    sp, ov = np.array(sp), np.array(ov)
    return 100 * (sp - ov).mean() / max(sp.mean(), 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--w_comp", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--lam_gate", type=float, default=0.01)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev} | beta {args.beta} w_comp {args.w_comp} lam_gate {args.lam_gate}")

    samples = build_index(args.root or find_root())
    ims, gts, organs = [], [], []
    for s in samples:
        im = np.asarray(Image.open(s["image"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                        np.float32) / 255.
        ims.append(im); gts.append(float(gt_from_mask(s["mask"]))); organs.append(s["organ"])
    ims = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2)
    gts = np.array(gts); organs = np.array(organs)
    org_mean = {o: gts[organs == o].mean() for o in set(organs)}
    dense_orgs = sorted(org_mean, key=org_mean.get, reverse=True)[:4]
    print(f"[data] {len(gts)} ảnh | mô-dày: {[(o, round(org_mean[o])) for o in dense_orgs]}")

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = ["baseline", "composite", "gate"]
    R = {a: {"r2": [], "worst": [], "densemae": [], "sat": []} for a in arms}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(gts)); n_te = len(gts) // 5
        te, tr = idx[:n_te], idx[n_te:]
        dmask = np.array([organs[i] in dense_orgs for i in te])
        for a in arms:
            m, gated = train_arm(ims, gts, tr, dev, args.epochs, a, args.w_comp, args.beta, args.lam_gate, seed=seed)
            pr = preds(m, ims, te, dev, gated); yt = gts[te]
            R[a]["r2"].append(r2(pr, yt))
            R[a]["worst"].append(worst_organ_r2(pr, yt, organs[te]))
            R[a]["densemae"].append(np.abs(pr[dmask] - yt[dmask]).mean() if dmask.any() else np.nan)
            R[a]["sat"].append(sat_k4(m, ims, te, dev, gated, seed=seed))
        print(f"  seed {seed} done")

    print(f"\n=== BƯỚC 2: cổng-crowding ({len(seeds)} seed, paired) ===")
    print(f"  {'metric':24s} {'baseline':>13s} {'+composite':>13s} {'+gate':>13s}")
    for lab, key in [("R² tổng ↑", "r2"), ("worst-organ R² ↑", "worst"),
                     ("MAE mô-dày ↓ [KEY]", "densemae"), ("Δsat% overlay ↓", "sat")]:
        b, c, g = (np.array(R[a][key]) for a in arms)
        print(f"  {lab:24s} {b.mean():>13.3f} {c.mean():>13.3f} {g.mean():>13.3f}")
    dr2 = np.array(R["gate"]["r2"]) - np.array(R["baseline"]["r2"])
    dmae = np.array(R["gate"]["densemae"]) - np.array(R["baseline"]["densemae"])
    print(f"\n  +gate vs baseline (paired): ΔR²-tổng {dr2.mean():+.3f}  ΔMAE-mô-dày {dmae.mean():+.2f}")
    ok_r2 = abs(dr2.mean()) < 0.02
    ok_mae = dmae.mean() < 0
    print(f"  GATE NGHIÊM: R²-tổng giữ (|Δ|<0.02)? {'✅' if ok_r2 else '✗'} ({dr2.mean():+.3f}) | "
          f"MAE-mô-dày giảm? {'✅' if ok_mae else '✗'} ({dmae.mean():+.2f})")
    print("  → cả 2 ✅ = correction TARGET được (giúp dày, không hại thưa) = METHOD THẬT.")
    print("  → R² vẫn tụt = correction không target được dù có gate → DỪNG hướng này (honest).")


if __name__ == "__main__":
    main()
