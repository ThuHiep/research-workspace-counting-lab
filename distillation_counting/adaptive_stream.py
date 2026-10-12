"""Ba chế độ hiệu chuẩn (tĩnh / cửa sổ cố định / cửa sổ thích ứng) trên luồng (mu, sigma, gt).

Cửa sổ thích ứng lấy đúng quy tắc của Paper 1 (kaggle/vast/pathosam_adapt_deadband.py):
dead-band [shrink_thr, grow_thr], co nhanh (shrink_f) khi thiếu phủ, giãn chậm (grow_f) khi dư.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coverage_transfer import winkler_score  # noqa: E402
from prediction_intervals import conformal_q, load_mu_sigma  # noqa: E402


def interval(mu, sigma, q):
    return max(0.0, mu - q * sigma), mu + q * sigma


def run(mu, sigma, gt, alpha, n_warm, window, mode, r_warm=None,
        shrink_thr=0.90, grow_thr=0.93, cov_win=50, shrink_f=0.9, grow_f=1.05, w_min=40):
    if r_warm is None:
        r_warm = np.abs(gt[:n_warm] - mu[:n_warm]) / sigma[:n_warm]
    else:
        n_warm = 0
    scores = list(r_warm[-window:])
    q_static = conformal_q(r_warm, alpha)
    eff, recent = window, []
    cov, wid, wk, eff_hist = [], [], [], []
    for i in range(n_warm, len(mu)):
        if mode == "static":
            q = q_static
        else:
            ref = scores[-eff:] if mode == "adaptive" else scores[-window:]
            q = conformal_q(np.asarray(ref), alpha) if ref else float("inf")
        lo, hi = interval(mu[i], sigma[i], q)
        c = float(lo <= gt[i] <= hi)
        cov.append(c)
        wid.append(hi - lo)
        wk.append(winkler_score(lo, hi, gt[i], alpha))
        eff_hist.append(eff)
        if mode == "adaptive":
            recent = (recent + [c])[-cov_win:]
            rc = float(np.mean(recent))
            if rc < shrink_thr:
                eff = max(w_min, int(eff * shrink_f))
            elif rc > grow_thr:
                eff = min(window, int(eff * grow_f))
        scores = (scores + [abs(gt[i] - mu[i]) / sigma[i]])[-window:]
    return {"cov": 100 * float(np.mean(cov)), "width": float(np.mean(wid)),
            "winkler": float(np.mean(wk)), "eff_med": float(np.median(eff_hist))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--warm_preds", default=None,
                    help="pkl MIỀN NGUỒN làm tập khởi động; có cờ này thì --warm bị bỏ qua và "
                         "toàn bộ --preds là luồng triển khai.")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--warm", type=int, default=200)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--cov_win", type=int, default=50)
    ap.add_argument("--w_min", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    a = ap.parse_args()

    mu, sigma, gt, _ = load_mu_sigma(a.preds)
    if a.shuffle:
        idx = np.random.default_rng(a.seed).permutation(len(mu))
        mu, sigma, gt = mu[idx], sigma[idx], gt[idx]
    r_warm, n_warm = None, a.warm
    if a.warm_preds:
        wmu, wsg, wgt, _ = load_mu_sigma(a.warm_preds)
        r_warm = np.abs(wgt - wmu) / wsg
        n_warm = 0
        print(f"khoi dong tu MIEN NGUON: {a.warm_preds} (N={len(wmu)}, "
              f"r trung vi {np.median(r_warm):.2f})")
    print(f"N={len(mu)}  warm={n_warm}  stream={len(mu) - n_warm}  window={a.window}")
    print(f"{'che do':<10}{'phu %':>9}{'rong':>10}{'winkler':>10}{'e_t med':>9}")
    for mode in ("static", "fixed", "adaptive"):
        r = run(mu, sigma, gt, a.alpha, a.warm, a.window, mode, r_warm=r_warm,
                cov_win=a.cov_win, w_min=a.w_min)
        print(f"{mode:<10}{r['cov']:>9.1f}{r['width']:>10.2f}{r['winkler']:>10.2f}{r['eff_med']:>9.0f}")


if __name__ == "__main__":
    main()
