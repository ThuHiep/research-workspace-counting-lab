#!/usr/bin/env python3
"""make_figs.py — dựng hình cho paper từ các .npz do count_evidence.py xuất. KHÔNG cần GPU/dataset.

  python make_figs.py --preds evidence_out/preds.npz \
      --occ "bgpatch=evidence_bgpatch/occlusion.npz" --occ "telea=evidence_out/occlusion.npz" \
      --out_dir paper2_overleaf/figures

Hình 1 (fig_agreement.pdf): (a) pred-vs-GT + đường y=x, (b) Bland-Altman.
Hình 2 (fig_doseresp.pdf) : count chuẩn hoá theo tỉ lệ nhân bị gỡ, kèm ĐỐI CHỨNG GIẢ (sham) và
                            đường "đếm trung thực" y=x. Mỗi --occ thêm 1 đường gỡ-nhân (so cách gỡ).

Quy ước trình bày: 1 trục (không bao giờ 2 thang y); màu gán theo THỨ TỰ CỐ ĐỊNH của palette đã
kiểm CVD (blue -> orange -> aqua), không quay vòng; chữ dùng màu mực chứ không dùng màu chuỗi;
lưới/trục lùi về sau; có legend + nhãn trực tiếp nên danh tính không phụ thuộc riêng vào màu.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]     # slot 1,2,3 — thứ tự cố định, đã kiểm CVD
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def vn(x, nd=2):
    """số kiểu Việt: dấu phẩy thập phân, cho khớp phần chữ của paper."""
    return f"{x:+.{nd}f}".replace(".", ",") if x < 0 or str(x)[0] == "+" else f"{x:.{nd}f}".replace(".", ",")


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
    })


def fig_agreement(preds_npz, out):
    z = np.load(preds_npz, allow_pickle=True)
    pred, gt = np.asarray(z["pred"], float), np.asarray(z["gt"], float)
    r2 = 1 - ((gt - pred) ** 2).sum() / ((gt - gt.mean()) ** 2).sum()
    mae = np.abs(pred - gt).mean()
    diff = pred - gt; bias, sd = diff.mean(), diff.std()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    lim = max(gt.max(), pred.max()) * 1.04
    a1.plot([0, lim], [0, lim], ls="--", lw=1.0, color=INK2, zorder=1)
    a1.scatter(gt, pred, s=4, alpha=0.22, color=SERIES[0], linewidths=0, zorder=2)
    a1.set_xlim(0, lim); a1.set_ylim(0, lim); a1.set_aspect("equal")
    a1.set_xlabel("Số nhân thực"); a1.set_ylabel("Số nhân dự đoán")
    a1.set_title("(a) Tương hợp dự đoán–thực", loc="left")
    a1.text(0.04, 0.95, f"$R^2$ = {vn(r2, 3)}\nMAE = {vn(mae)}\n$n$ = {len(gt)}",
            transform=a1.transAxes, va="top", ha="left", color=INK, fontsize=7.5)
    a1.annotate("$y = x$", xy=(lim * 0.72, lim * 0.72), xytext=(lim * 0.82, lim * 0.52),
                color=INK2, fontsize=6.8, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))

    a2.axhline(0, lw=0.6, color=GRID)
    a2.scatter((pred + gt) / 2, diff, s=4, alpha=0.22, color=SERIES[0], linewidths=0)
    for y, ls in [(bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")]:
        a2.axhline(y, ls=ls, lw=1.0, color=SERIES[1])
    xr = a2.get_xlim()[1]
    a2.text(xr * 0.99, bias, f"độ chệch {vn(bias)}", va="bottom", ha="right", color=INK2, fontsize=6.8)
    a2.text(xr * 0.99, bias + 1.96 * sd, f"giới hạn tương hợp {vn(bias + 1.96 * sd, 1)}",
            va="bottom", ha="right", color=INK2, fontsize=6.8)
    a2.text(xr * 0.99, bias - 1.96 * sd, f"{vn(bias - 1.96 * sd, 1)}",
            va="top", ha="right", color=INK2, fontsize=6.8)
    a2.set_xlabel("Trung bình của (dự đoán, thực)"); a2.set_ylabel("Dự đoán $-$ thực")
    a2.set_title("(b) Bland–Altman", loc="left")

    fig.tight_layout(w_pad=2.0); fig.savefig(out); plt.close(fig)
    print(f"[hình 1] {out}  (R²={r2:.4f} MAE={mae:.3f} chệch={bias:+.3f} 1.96σ={1.96*sd:.2f})")


def _curve(npz):
    """-> (keep, mean_nhân, sd_nhân, mean_sham) chuẩn hoá theo count trên ảnh nguyên vẹn."""
    o = np.load(npz, allow_pickle=True)
    frac, pred, full = o["frac"], o["pred"], o["full"]
    sham = o["sham"] if "sham" in o.files else np.full_like(pred, np.nan)
    ok = full > 1e-6
    norm = np.where(ok, pred / np.where(ok, full, 1), np.nan)
    norm_s = np.where(ok, sham / np.where(ok, full, 1), np.nan)
    keep = np.array(sorted(set(1 - frac)))
    return (keep,
            np.array([np.nanmean(norm[(1 - frac) == k]) for k in keep]),
            np.array([np.nanstd(norm[(1 - frac) == k]) for k in keep]),
            np.array([np.nanmean(norm_s[(1 - frac) == k]) for k in keep]))


def fig_doseresp(occs, out):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color=INK2, zorder=1)
    ax.annotate("đếm cộng dồn trung thực", xy=(0.62, 0.62), xytext=(0.70, 0.30), color=INK2,
                fontsize=6.5, ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
    sham_done = False
    for i, (label, npz) in enumerate(occs):
        keep, m, sd, ms = _curve(npz)
        c = SERIES[i % len(SERIES)]
        ax.fill_between(keep, m - sd, m + sd, color=c, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(keep, m, "-o", lw=2.0, ms=4.5, color=c, zorder=4,
                label=f"gỡ nhân ({label})" if len(occs) > 1 else "gỡ nhân")
        if not sham_done and np.isfinite(ms).any():
            ax.plot(keep, ms, "-s", lw=2.0, ms=4.0, color=SERIES[2], zorder=3,
                    label="đối chứng giả: gỡ mô không có nhân")
            ax.text(0.05, ms[0] + 0.045, "đối chứng giả", color=INK2, fontsize=6.8, ha="left", va="bottom")
            sham_done = True
        # nhãn trực tiếp neo vào ĐẦU TRÁI (gỡ hết nhân) — nơi các đường tách xa nhau nhất
        ax.text(0.05, m[0] - 0.04, f"gỡ nhân ({label})" if len(occs) > 1 else "gỡ nhân",
                color=INK2, fontsize=6.8, ha="left", va="top")
    from matplotlib.ticker import FuncFormatter
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))   # thống nhất dấu phẩy thập phân
    ax.xaxis.set_major_formatter(comma); ax.yaxis.set_major_formatter(comma)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1.18)
    ax.set_xlabel("Tỉ lệ nhân còn lại trong ảnh"); ax.set_ylabel("Số đếm dự đoán (chuẩn hoá)")
    ax.legend(loc="lower right", handlelength=1.6)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    for label, npz in occs:
        keep, m, sd, ms = _curve(npz)
        gain = [(1 - m[j]) / (1 - keep[j]) for j in range(len(keep)) if keep[j] < 1]
        print(f"[hình 2] {label:8s} count khi gỡ hết={m[0]:.3f}  sham={ms[0]:.3f}  "
              f"hệ số đáp ứng={np.mean(gain):.2f}")
    print(f"[hình 2] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="", help="preds.npz -> hình 1")
    ap.add_argument("--occ", action="append", default=[],
                    help='"nhãn=đường/dẫn/occlusion.npz"; lặp lại để so nhiều cách gỡ (cái ĐẦU = chính)')
    ap.add_argument("--out_dir", default="paper2_overleaf/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    style()
    if args.preds:
        fig_agreement(args.preds, os.path.join(args.out_dir, "fig_agreement.pdf"))
    if args.occ:
        occs = [tuple(x.split("=", 1)) for x in args.occ]
        fig_doseresp(occs, os.path.join(args.out_dir, "fig_doseresp.pdf"))


if __name__ == "__main__":
    main()
