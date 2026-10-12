#!/usr/bin/env python3
"""denuc_count_eval.py — đếm nhân từ output detection của DeNuC (2026) trên PanNuke fold3.

DeNuC (ZijiangY1116/DeNuC) retrain leak-free (train fold1+2 gộp -> test fold3, dataset 'pannuke12_3').
Sau `denuc_eval.py --eval_mode test`, infer_res là dict {image_key: {gt_coords Nx2, pred_coords Mx2}}
với pred_coords ĐÃ qua NMS. Đếm = len(pred_coords) so với GT = len(gt_coords).

Paradigm DETECTION (thứ 3, cạnh density=DualU-Net + instance=StarDist) để so với efflite0 count-only.
Kỳ vọng: recall thấp ở vùng dày (NMS bóp chết TP khi nhân chồng lấn) -> under-count sụp theo mật độ
-> bằng chứng cho động lực CROWDING.

  !python denuc_count_eval.py --infer ./outputs/<exp>/test_results/pannuke12_3/best_checkpoint_infer_res.npy
"""
import argparse
import numpy as np


def r2_mae(gt, pr):
    gt, pr = np.asarray(gt, float), np.asarray(pr, float)
    ss = ((gt - pr) ** 2).sum(); st = ((gt - gt.mean()) ** 2).sum()
    return (1 - ss / st if st > 0 else float("nan")), np.abs(gt - pr).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", required=True, help="path tới *_infer_res.npy (test_results)")
    ap.add_argument("--out_csv", default="", help="xuất gt,pred cho agg_baseline_seeds.py")
    args = ap.parse_args()

    r = np.load(args.infer, allow_pickle=True).item()
    keys = sorted(r)                       # THỨ TỰ CỐ ĐỊNH: không sort thì 2 seed lệch thứ tự ảnh
    gt = np.array([len(r[k]["gt_coords"]) for k in keys], float)
    pr = np.array([len(r[k]["pred_coords"]) for k in keys], float)
    if args.out_csv:
        from pathlib import Path
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w") as f:
            f.write("gt,pred\n")
            for g, p in zip(gt, pr):
                f.write(f"{int(g)},{int(p)}\n")

    r2, mae = r2_mae(gt, pr)
    print(f"DeNuC (detection) fold3  N={len(gt)}")
    print(f"  R2  = {r2:+.4f}")
    print(f"  MAE = {mae:.3f}")
    print(f"  GT mean {gt.mean():.2f} | pred mean {pr.mean():.2f} | bias {(pr - gt).mean():+.2f}")

    edges = np.array([10, 20, 30, 45], float); tid = np.digitize(gt, edges)
    names = ["<10", "10-20", "20-30", "30-45", "45+"]
    for t, nm in enumerate(names):
        m = tid == t
        if m.sum():
            print(f"    {nm:>6s} n={m.sum():4d}  bias {(pr[m] - gt[m]).mean():+.2f}")


if __name__ == "__main__":
    main()
