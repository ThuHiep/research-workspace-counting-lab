#!/usr/bin/env python3
"""count_evidence.py — BẰNG CHỨNG count-only ĐẾM ĐÚNG (không dựa localization đã chết).

① Scatter Pred-vs-GT + Bland-Altman + per-organ  -> "đếm đúng" (mô tả trực quan).
② Dose-response GỠ NHÂN (occlusion nhân-quả): xóa k% nhân khỏi ảnh (inpaint bằng mô nền),
   đo lại count dự đoán. Nếu count tụt theo số nhân bị gỡ (độ dốc ≈ 1, và gỡ HẾT -> count ≈ 0)
   => model THẬT SỰ đáp ứng với nhân, KHÔNG ăn gian bằng prior toàn cục.
   KHÁC localization: không đòi bản đồ density định vị; chỉ hỏi output có nhân-quả giảm khi bớt nhân.

Model đánh giá = count-only efflite0 (headline). Chạy Kaggle GPU (cần masks.npy PanNuke):
  !python count_evidence.py --pannuke_root /kaggle/input/datasets/hipinhththu/pannuke \
      --epochs 60 --seed 42 --n_occ 300
Xuất: evidence_out/{scatter.png, bland_altman.png, doseresp.png, preds.npz, occlusion.npz}
"""
import argparse, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "kaggle", "lib"), os.path.join(_REPO, "kaggle", "vast"), _REPO, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from distill_student_r2 import build_pannuke_density, train
from r2_losses import count_from_density


@torch.no_grad()
def predict_count(model, img_uint8, dev):
    x = torch.from_numpy(img_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
    return float(count_from_density(model(x)[0])[0])


def enumerate_instances(masks):
    """masks: 5 kênh label (mỗi id != 0 = 1 nhân). -> list boolean mask từng nhân."""
    inst = []
    for k in range(5):
        lab = np.asarray(masks[k])
        for iid in np.unique(lab):
            if iid == 0:
                continue
            inst.append(lab == iid)
    return inst


def occlude(img, remove_masks):
    """Gỡ các nhân trong remove_masks: inpaint (Telea) bằng mô xung quanh -> tự nhiên, tránh
    tạo mảng phẳng làm model phản ứng nhầm. Fallback: điền màu nền trung vị."""
    if not remove_masks:
        return img.copy()
    m = np.zeros(img.shape[:2], np.uint8)
    for b in remove_masks:
        m[b] = 1
    try:
        import cv2
        return cv2.inpaint(np.ascontiguousarray(img), m, 3, cv2.INPAINT_TELEA)
    except Exception:
        out = img.copy()
        bg = np.median(img[m == 0].reshape(-1, 3), axis=0) if (m == 0).any() else img.reshape(-1, 3).mean(0)
        out[m == 1] = bg.astype(np.uint8)
        return out


def r2_mae(pred, gt):
    ss = ((gt - pred) ** 2).sum(); st = ((gt - gt.mean()) ** 2).sum()
    return (1 - ss / st if st > 0 else float("nan")), float(np.abs(pred - gt).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_occ", type=int, default=300, help="số ảnh test dùng cho phân tích gỡ-nhân (trải đều)")
    ap.add_argument("--fracs", default="0,0.25,0.5,0.75,1.0", help="tỉ lệ nhân bị gỡ")
    ap.add_argument("--reps", type=int, default=3, help="số tập con ngẫu nhiên mỗi tỉ lệ (giảm nhiễu)")
    ap.add_argument("--out_dir", default="evidence_out")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs("work", exist_ok=True)

    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # ---------- train count-only (headline) ----------
    tr_folds = [int(x) for x in args.train_folds.split(",")]
    fstr = "".join(str(x) for x in sorted(tr_folds))
    train_data = build_pannuke_density(args.pannuke_root, tr_folds, dev,
                                       f"work/gt_density_pannuke_f{fstr}.pkl", use_gt=True)
    test_data = build_pannuke_density(args.pannuke_root, [int(args.test_fold)], dev,
                                      f"work/gt_density_pannuke_f{args.test_fold}.pkl", use_gt=True)
    data = list(train_data) + list(test_data)
    tr = list(range(len(train_data))); te = list(range(len(train_data), len(data)))
    print(f"[data] train fold{tr_folds}={len(tr)} / test fold{args.test_fold}={len(te)}")
    model = train(data, dev, args.epochs, args.ch, args.lr, tr,
                  0.0, 1.0, 0.01, 0.5, 16, True, "poisson", args.backbone)  # count-only
    model.eval()

    # ---------- ① scatter Pred-vs-GT + per-organ ----------
    pred = np.array([predict_count(model, data[i]["img"], dev) for i in te])
    gt = np.array([data[i]["gt"] for i in te])
    organ = np.array([data[i]["organ"] for i in te])
    R2, MAE = r2_mae(pred, gt)
    print(f"\n① Pred-vs-GT (count-only, fold{args.test_fold}, n={len(te)}): R²={R2:+.4f}  MAE={MAE:.3f}")
    print(f"  {'organ':22s} {'n':>4s} {'R²':>8s} {'MAE':>7s}")
    for org in sorted(set(organ)):
        mk = organ == org
        if mk.sum() >= 5:
            r2o, maeo = r2_mae(pred[mk], gt[mk])
            print(f"  {org:22s} {mk.sum():4d} {r2o:+8.3f} {maeo:7.2f}")
    np.savez(f"{args.out_dir}/preds.npz", pred=pred, gt=gt, organ=organ, R2=R2, MAE=MAE)

    organs = sorted(set(organ)); cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(5, 5))
    for j, org in enumerate(organs):
        mk = organ == org
        plt.scatter(gt[mk], pred[mk], s=8, alpha=0.5, color=cmap(j % 20), label=org)
    lim = max(gt.max(), pred.max()) * 1.05
    plt.plot([0, lim], [0, lim], "k--", lw=1)
    plt.xlim(0, lim); plt.ylim(0, lim)
    plt.xlabel("Số đếm thực (GT)"); plt.ylabel("Số đếm dự đoán (count-only)")
    plt.title(f"Pred-vs-GT  R²={R2:.3f}  MAE={MAE:.2f}")
    plt.legend(fontsize=5, ncol=2, markerscale=1.5, loc="upper left")
    plt.tight_layout(); plt.savefig(f"{args.out_dir}/scatter.png", dpi=160); plt.close()

    # Bland-Altman
    mean_c = (pred + gt) / 2; diff = pred - gt
    bias = diff.mean(); sd = diff.std()
    plt.figure(figsize=(5, 4))
    plt.scatter(mean_c, diff, s=8, alpha=0.4, color="steelblue")
    for y, ls in [(bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")]:
        plt.axhline(y, color="crimson", ls=ls, lw=1)
    plt.xlabel("Trung bình (pred, GT)"); plt.ylabel("Hiệu (pred − GT)")
    plt.title(f"Bland–Altman  bias={bias:+.2f}  ±1.96σ={1.96*sd:.1f}")
    plt.tight_layout(); plt.savefig(f"{args.out_dir}/bland_altman.png", dpi=160); plt.close()
    print(f"  Bland-Altman bias={bias:+.3f} (≈0 = không thiên vị hệ thống), 1.96σ={1.96*sd:.2f}")

    # ---------- ② dose-response gỡ nhân ----------
    from pannuke_loader import PanNukeFold
    pf = PanNukeFold(args.pannuke_root, int(args.test_fold))
    idx_occ = np.linspace(0, len(pf) - 1, min(args.n_occ, len(pf))).astype(int)
    fracs = [float(x) for x in args.fracs.split(",")]
    rng = np.random.default_rng(args.seed)
    rows = []   # (frac, N, remaining_true, pred, full_pred)
    for c, i in enumerate(idx_occ):
        s = pf[int(i)]
        if s["masks"] is None:            # masks.npy đã xoá -> không gỡ-nhân được
            continue
        img = np.ascontiguousarray(np.asarray(s["image"])[..., :3]).astype(np.uint8)
        inst = enumerate_instances(s["masks"])
        N = len(inst)
        if N == 0:
            continue
        full_pred = predict_count(model, img, dev)
        for frac in fracs:
            krm = int(round(frac * N))
            for _ in range(args.reps if 0 < frac < 1 else 1):   # frac 0/1 xác định -> khỏi lặp
                sel = rng.choice(N, size=krm, replace=False) if krm > 0 else np.array([], int)
                occ = occlude(img, [inst[j] for j in sel])
                p = predict_count(model, occ, dev)
                rows.append((frac, N, N - krm, p, full_pred))
        if (c + 1) % 50 == 0:
            print(f"  [occ] {c+1}/{len(idx_occ)} ảnh")
    rows = np.array(rows, float)
    frac_a, N_a, rem_a, pred_a, full_a = rows.T

    # độ dốc pred vs remaining_true (gộp mọi ảnh & tỉ lệ); faithful => ~1
    A = np.vstack([rem_a, np.ones_like(rem_a)]).T
    slope, intercept = np.linalg.lstsq(A, pred_a, rcond=None)[0]
    corr = float(np.corrcoef(rem_a, pred_a)[0, 1])
    # đường chuẩn hóa trung bình theo tỉ lệ còn lại
    keep = 1 - frac_a
    norm = np.divide(pred_a, full_a, out=np.zeros_like(pred_a), where=full_a > 1e-6)
    xs = sorted(set(keep)); mean_norm = [norm[keep == k].mean() for k in xs]; sd_norm = [norm[keep == k].std() for k in xs]
    at_zero = norm[frac_a == 1.0].mean() if (frac_a == 1.0).any() else float("nan")
    print(f"\n② Dose-response gỡ nhân (n_ảnh={len(idx_occ)}, {len(rows)} phép đo):")
    print(f"  độ dốc pred vs (số nhân còn lại) = {slope:.3f}  (≈1 = đếm trung thực theo nhân)  |  corr={corr:.3f}")
    print(f"  count chuẩn hóa khi GỠ HẾT nhân (frac=1) = {at_zero:.3f}  (≈0 = không đếm từ cue phi-nhân)")
    np.savez(f"{args.out_dir}/occlusion.npz", frac=frac_a, N=N_a, remaining=rem_a,
             pred=pred_a, full=full_a, slope=slope, intercept=intercept, corr=corr)

    plt.figure(figsize=(5, 4))
    plt.errorbar(xs, mean_norm, yerr=sd_norm, fmt="o-", color="darkorange", capsize=3, label="count-only (chuẩn hóa)")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="đếm trung thực (y=x)")
    plt.xlabel("Tỉ lệ nhân CÒN LẠI (1 − k)"); plt.ylabel("Count dự đoán / count đầy đủ")
    plt.title(f"Gỡ nhân → count tụt theo\nđộ dốc(abs)={slope:.2f}, gỡ-hết={at_zero:.2f}")
    plt.xlim(-0.03, 1.03); plt.ylim(-0.03, 1.1); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{args.out_dir}/doseresp.png", dpi=160); plt.close()

    print(f"\n[xuất] {args.out_dir}/scatter.png, bland_altman.png, doseresp.png, preds.npz, occlusion.npz")
    print("ĐỌC: scatter bám chéo = đếm đúng; độ dốc gỡ-nhân ≈1 & gỡ-hết≈0 = ĐANG ĐẾM NHÂN (nhân quả), "
          "không phải localization mà là đáp ứng nội dung nhân.")


if __name__ == "__main__":
    main()
