#!/usr/bin/env python3
"""fig_equivalence.py — hình cho §4.1: khoảng tin cậy của Δ|sai số| đặt cạnh các biên tương đương.

Trực quan hoá ĐÚNG phát biểu hai vế của bài:
  - CI không chứa 0  -> khác biệt CÓ thật (Wilcoxon bác bỏ "bằng nhau").
  - CI nằm gọn trong ±δ -> tương đương ở biên δ (TOST bác bỏ "|Δ| ≥ δ").
Một hình duy nhất cho thấy cả hai, nên người đọc không đọc nhầm một vế thành cả kết luận.

Số mặc định = PanNuke fold3 5 hạt giống mỗi nhánh (n=2722, số nhân TB 24,49), khớp Bảng 1.
Chạy:  python fig_equivalence.py --out paper2_overleaf/figures/fig_equivalence.pdf
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, ACCENT = "#1b1b1b", "#8a8a8a", "#2f6f9f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm", type=float, default=0.2100, help="Δ|err| trung bình (nhân)")
    ap.add_argument("--lo", type=float, default=0.1555, help="cận dưới CI 90%%")
    ap.add_argument("--hi", type=float, default=0.2673, help="cận trên CI 90%% (= δ*)")
    ap.add_argument("--gt_mean", type=float, default=24.49, help="số nhân trung bình/ảnh")
    ap.add_argument("--fracs", default="0.01,0.02", help="các biên δ, theo tỉ lệ số nhân TB")
    ap.add_argument("--out", default="paper2_overleaf/figures/fig_equivalence.pdf")
    a = ap.parse_args()

    # Chỉ vẽ các biên ĐÁNG KỂ (1%, 2%). Thêm 5% chỉ làm trục giãn ra 2,5 lần và
    # bóp khoảng tin cậy thành một vạch không đọc được; 5% để trong phần chữ.
    fracs = [float(x) for x in a.fracs.split(",")]
    deltas = [f * a.gt_mean for f in fracs]
    lim = max(deltas) * 1.45

    fig, ax = plt.subplots(figsize=(6.4, 2.0))

    ax.axvspan(-deltas[-1], deltas[-1], color=ACCENT, alpha=0.06, lw=0)
    # Nhãn δ xếp so le theo CHIỀU DỌC và chỉ đặt bên phải -> không cụm nào đè cụm nào.
    for i, (f, d) in enumerate(zip(fracs, deltas)):
        for s in (-1, 1):
            ax.axvline(s * d, color=MUTED, ls=(0, (4, 3)), lw=1.0)
        # '%' KHONG duoc nam trong $...$ (mathtext coi la loi cu phap) -> de ngoai phan toan
        ax.annotate(f"$\\delta$ = {f:.0%} ({d:.2f})", xy=(d + lim * 0.02, 0.72 - 0.30 * i),
                    fontsize=7.5, color=MUTED, ha="left", va="center")
    ax.axvline(0, color=INK, lw=1.2)

    ax.errorbar(a.dm, 0, xerr=[[a.dm - a.lo], [a.hi - a.dm]], fmt="o", color=ACCENT,
                ecolor=ACCENT, elinewidth=2.0, capsize=4, markersize=7, zorder=5)

    # Số liệu (Δ, CI, δ*) nằm trong chú thích hình, không nhồi vào khung vẽ.
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("$\\Delta$ per-image absolute error,  count-only $-$ mask  (nuclei)", fontsize=9)
    ax.annotate("count-only errs less", xy=(-lim * 0.97, -0.78), fontsize=7.5, color=MUTED,
                ha="left", va="center")
    ax.annotate("mask errs less", xy=(lim * 0.97, -0.78), fontsize=7.5, color=MUTED,
                ha="right", va="center")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=INK, labelsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[OUT] {a.out}")
    print(f"  CI {'KHONG chua' if a.lo > 0 else 'chua'} 0 -> khac biet {'CO' if a.lo > 0 else 'KHONG'} y nghia")
    for f, d in zip(fracs, deltas):
        print(f"  delta={f:.0%} ({d:.3f}): {'TUONG DUONG' if a.hi < d else 'chua ket luan duoc'}")


if __name__ == "__main__":
    main()
