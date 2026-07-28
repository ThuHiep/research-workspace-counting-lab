#!/usr/bin/env python3
"""localization_emergence.py — TEST PREMISE cho cơ chế Q1 (2026-07-28).

Câu hỏi: model count-only CHỈ học tổng đếm (w_density=0), KHÔNG HỀ được supervise vị trí.
Vậy bản đồ mật độ nó xuất ra có TỰ trùng vị trí nhân GT không? Nếu CÓ ->
"spatial representation emerges without spatial labels" = cơ chế headline giải thích
TẠI SAO count-only đủ để đếm. So với model mask (được supervise vị trí) làm trần.

⚠️ Đây là TEST PREMISE — nhận xét gốc BỊA số (IoU 0.71 vs 0.68). Chạy để BIẾT thật;
có thể ra count-only localize KÉM (khi đó story đổi: "đếm KHÔNG cần localize").

Hai chỉ số trên fold3 test (có GT density từ mask):
  (1) corr: per-image Pearson giữa density DỰ ĐOÁN và density GT (nhân ở đâu).
  (2) mass-on-nuclei: tỉ lệ khối mật độ dự đoán rơi ĐÚNG lên pixel nhân GT.
Cả hai cho: count-only vs mask (cùng kiến trúc efflite0) + sanity count R².

Chạy Kaggle (GPU):
  !python localization_emergence.py --pannuke_root /kaggle/input/datasets/hipinhththu/pannuke \\
      --epochs 40 --seed 42
"""
import argparse, os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "kaggle", "lib"), _REPO, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from distill_student_r2 import build_pannuke_density, train
from r2_losses import count_from_density


@torch.no_grad()
def localize_stats(model, test_data, dev):
    """-> (corr_mean, massfrac_mean, r2_count) trên test."""
    model.eval()
    corrs, massfr, pr, yt = [], [], [], []
    for d in test_data:
        x = torch.from_numpy(d["img"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        dens = model(x)[0]                                  # (1,1,H,W)
        pred = dens.squeeze().cpu().numpy().astype(np.float64)   # (H,W)
        gt = d["density"].astype(np.float64)                # (H,W) GT nhân
        pr.append(float(count_from_density(dens)[0])); yt.append(d["gt"])
        pf, gf = pred.ravel(), gt.ravel()
        if pf.std() > 1e-8 and gf.std() > 1e-8:
            corrs.append(float(np.corrcoef(pf, gf)[0, 1]))
        tot = pred.sum()
        if tot > 1e-8:
            massfr.append(float(pred[gt > 0].sum() / tot))  # khối rơi lên pixel nhân GT
    pr, yt = np.array(pr), np.array(yt)
    ss = ((yt - pr) ** 2).sum(); st = ((yt - yt.mean()) ** 2).sum()
    r2 = 1 - ss / st if st > 0 else float("nan")
    return float(np.mean(corrs)), float(np.mean(massfr)), r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work_dir", default="work")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.work_dir, exist_ok=True)

    tr_folds = [int(x) for x in args.train_folds.split(",")]
    fstr = "".join(str(x) for x in sorted(tr_folds))
    train_data = build_pannuke_density(args.pannuke_root, tr_folds, dev,
                                       f"{args.work_dir}/gt_density_pannuke_f{fstr}.pkl", use_gt=True)
    test_data = build_pannuke_density(args.pannuke_root, [int(args.test_fold)], dev,
                                      f"{args.work_dir}/gt_density_pannuke_f{args.test_fold}.pkl", use_gt=True)
    tr = list(range(len(train_data)))
    print(f"[data] train fold{tr_folds}={len(train_data)} / test fold{args.test_fold}={len(test_data)} (GT density)")

    configs = [("count-only", 0.0, 1.0), ("mask (GT-density)", 1.0, 0.01)]
    rows = []
    for name, w_d, w_c in configs:
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        # train(data, dev, epochs, ch, lr, idx, w_density, w_count, w_nll, beta, bs, detach_mu, sigma_mode, backbone)
        model = train(train_data, dev, args.epochs, args.ch, 1e-3, tr,
                      w_d, w_c, 0.01, 0.5, 16, True, "poisson", args.backbone)
        corr, massfr, r2 = localize_stats(model, test_data, dev)
        rows.append((name, corr, massfr, r2))
        print(f"  [{name:18s}] corr {corr:+.4f} | mass-on-nuclei {massfr:.3f} | count-R² {r2:+.4f}")

    print("\n=== LOCALIZATION EMERGENCE (fold3 test, efflite0) ===")
    print(f"  {'supervision':18s} {'corr↑':>8s} {'mass-on-nuclei↑':>16s} {'count-R²':>10s}")
    for name, corr, massfr, r2 in rows:
        print(f"  {name:18s} {corr:+.4f} {massfr:16.3f} {r2:+10.4f}")
    if len(rows) == 2:
        dc = rows[0][1] - rows[1][1]; dm = rows[0][2] - rows[1][2]
        print(f"\n  Δ(count-only − mask): corr {dc:+.4f} | mass {dm:+.3f}")
        print("  ĐỌC: nếu count-only corr/mass GẦN mask -> spatial localization TỰ NỔI dù không nhãn "
              "không gian (cơ chế headline). Nếu KÉM xa -> đếm không qua localize (story khác).")


if __name__ == "__main__":
    main()
