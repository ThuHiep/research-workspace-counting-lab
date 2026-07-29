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
from r2_losses import count_from_density


@torch.no_grad()
def _per_image_pred(model, data, idx, dev):
    model.eval(); pr = []
    for i in idx:
        x = torch.from_numpy(data[i]["img"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        pr.append(float(count_from_density(model(x)[0])[0]))
    return np.array(pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)   # density cần lâu hơn để hội tụ; count-only đã bão hoà ~40
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--bs", type=int, default=16,
                    help="batch size. Tăng (48/64) để GPU no hơn (T4 15GB thừa) -> nhanh hơn nhiều/epoch.")
    ap.add_argument("--preload", action="store_true",
                    help="đẩy toàn tập train lên GPU 1 lần (bỏ nghẽn CPU stack/copy mỗi batch) -> nhanh ~5-15x. "
                         "PanNuke fold12 ~2.4GB GPU, vừa T4. TẮT nếu OOM.")
    ap.add_argument("--w_count_mask", type=float, default=0.1,
                    help="neo count cho mask (0.1 ổn định; 0.01 cũ có thể sập theo seed)")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Adam LR; efflite0/CNN dùng 1e-3. Backbone transformer (pvt/swin) có thể cần "
                         "hạ (3e-4/1e-4) nếu 1e-3 phá feature pretrained -> sập hằng số.")
    ap.add_argument("--work_dir", default="work",
                    help="nơi lưu cache GT-density. ĐẶT NGOÀI repo (vd /kaggle/working/pannuke_cache) "
                         "để 'rm -rf <repo>' mỗi lần clone KHÔNG xóa cache -> khỏi build lại (rất chậm).")
    ap.add_argument("--deterministic", action="store_true",
                    help="bật cuDNN tất định — CHẬM hơn nhiều với efflite0. PanNuke lớn+ổn định KHÔNG cần "
                         "-> mặc định TẮT (benchmark=True) cho nhanh.")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.work_dir, exist_ok=True)   # nơi lưu cache GT-density (kẻo build xong không save được)
    tr_folds = [int(x) for x in args.train_folds.split(",")]
    fstr = "".join(str(x) for x in sorted(tr_folds))
    train_data = build_pannuke_density(args.pannuke_root, tr_folds, dev,
                                       f"{args.work_dir}/gt_density_pannuke_f{fstr}.pkl", use_gt=True)
    test_data = build_pannuke_density(args.pannuke_root, [int(args.test_fold)], dev,
                                      f"{args.work_dir}/gt_density_pannuke_f{args.test_fold}.pkl", use_gt=True)
    data = list(train_data) + list(test_data)
    tr = list(range(len(train_data))); te = list(range(len(train_data), len(data)))
    print(f"[data] train fold{tr_folds}={len(tr)} / test fold{args.test_fold}={len(te)} "
          f"(CÙNG data, cùng efflite0 ch{args.ch}; chỉ đổi loss)")

    seeds = [int(s) for s in args.seeds.split(",")]
    # GT-density cần neo count NHỎ (0.01) để train được — density MSE thuần bị background(0) chi phối
    # → collapse về ~0 nếu w_count=0 (đúng recipe density-counting chuẩn; số cũ NuInsSeg 0.881 dùng vậy).
    configs = [("count-only", 0.0, 1.0), ("GT-density (mask)", 1.0, args.w_count_mask)]
    rows = []
    per_img = {name: [] for name, _, _ in configs}   # preds MỖI seed (fold3 cố định) -> Wilcoxon gộp-seed
    gt_te = np.array([data[i]["gt"] for i in te])
    for name, wd, wc in configs:
        r2s, maes = [], []
        for sd in seeds:
            np.random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
            if args.deterministic:
                # cuDNN tất định: mỗi seed tái lập chính xác nhưng CHẬM (efflite0 depthwise). Chỉ bật khi cần.
                torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True, warn_only=True)
            else:
                torch.backends.cudnn.benchmark = True   # NHANH; PanNuke lớn+ổn định nên không cần tất định
            # train(data, device, epochs, ch, lr, train_idx, w_density, w_count, w_nll, beta, bs, detach_mu, sigma_mode, backbone)
            model = train(data, dev, args.epochs, args.ch, args.lr, tr,
                          wd, wc, 0.01, 0.5, args.bs, True, "poisson", args.backbone, preload=args.preload)
            r2, mae = eval_r2_mae(model, data, te, dev)
            r2s.append(r2); maes.append(mae)
            per_img[name].append(_per_image_pred(model, data, te, dev))
            print(f"    [{name:16s}] seed {sd}: R² {r2:+.4f}  MAE {mae:.2f}")   # per-seed để bắt collapse
        rows.append((name, np.mean(r2s), np.std(r2s), np.mean(maes), np.std(maes)))
        print(f"  {name:20s} R² {np.mean(r2s):+.4f}±{np.std(r2s):.3f}  MAE {np.mean(maes):.3f}±{np.std(maes):.3f}")

    print(f"\n=== SUPERVISION head-to-head (CÙNG {args.backbone} ch{args.ch}, lr{args.lr:g}, {len(seeds)} seed, PanNuke fold{args.test_fold}) ===")
    print(f"  {'supervision':20s} {'R² ↑':>16s} {'MAE ↓':>14s}")
    for name, r2m, r2s, mm, ms in rows:
        print(f"  {name:20s} {r2m:+.4f}±{r2s:.3f}   {mm:6.2f}±{ms:.2f}")
    d = rows[0][1] - rows[1][1]
    print(f"\n  Δ(count-only − GT-density) = {d:+.4f} R²")

    # per-image Wilcoxon GỘP-SEED: trung bình |err| từng ảnh qua các seed (fold3 cố định) -> robust
    ec = np.mean([np.abs(gt_te - p) for p in per_img["count-only"]], axis=0)
    em = np.mean([np.abs(gt_te - p) for p in per_img["GT-density (mask)"]], axis=0)
    from scipy.stats import wilcoxon
    try:
        _, p = wilcoxon(ec, em)
    except Exception as e:
        p = float("nan"); print(f"[wilcoxon lỗi] {e}")
    dm = float((ec - em).mean())
    verdict = ("PARITY (p>0.05)" if p > 0.05 else
               ("count-only THẮNG (p<0.05, lỗi ít hơn)" if dm < 0 else "count-only THUA (p<0.05)"))
    print(f"  per-image |err| (gộp {len(seeds)} seed, n={len(te)}): count-only {ec.mean():.2f} vs "
          f"mask {em.mean():.2f} | Wilcoxon p={p:.4g} -> {verdict}")
    print("  ĐỌC: count-only ≥ GT-density → ĐẾM KHÔNG CẦN pixel; cái thắng đến từ SUPERVISION, không phải kiến trúc.")


if __name__ == "__main__":
    main()
