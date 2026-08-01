#!/usr/bin/env python3
"""fig_efficiency.py — Hình 1: lập luận hiệu quả chú thích, mã hoá bằng ĐỘ DÀI.

Hình cũ (fig_design.drawio) là sơ đồ quy trình: mọi đại lượng là một con số trong một cái hộp,
nên "rẻ hơn 8 lần" và "kém 1,1%" chiếm diện tích ngang nhau. Mà cả bài nằm ở chỗ hai độ lớn đó
lệch nhau. Ba panel dưới đây để người đọc ĐO bằng mắt:
  (a) trả bao nhiêu   -> cột dài vs cột ngắn
  (b) mất bao nhiêu   -> hai cột trùng nhau
  (c) cần bao nhiêu   -> đường bão hoà sớm
Mọi số khớp main.tex: 5.4.2 (11,5/1,4 s), Bảng 1 (R²), 5.1 (98,6%), 5.4.1 (đường ngân sách).
Chạy: python fig_efficiency.py --out paper2_rineng/figures/fig_efficiency.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1b1b1b", "#8a8a8a", "#d8d8d8"
MASK, CNT = "#EB6834", "#2A78D6"          # cam = nhãn mặt nạ, xanh = nhãn đếm (dùng nhất quán cả bài)

# 5.4.1: điểm 10% và 50% chỉ có 1 lần chạy nên không có độ lệch -> vẽ rỗng, ghi rõ ở chú thích
BUDGET = [(5, 0.453, 0.208), (10, 0.742, None), (25, 0.849, 0.029), (50, 0.859, None), (100, 0.915, 0.010)]


def bars(ax, vals, cols, labels, xmax, fmt, xerr=None):
    ys = [1, 0]
    ax.barh(ys, vals, height=0.52, color=cols, edgecolor="none",
            xerr=xerr, error_kw=dict(ecolor=INK, elinewidth=1.2, capsize=3))
    for y, v, t in zip(ys, vals, fmt):
        ax.text(v + xmax * 0.075, y, t, va="center", ha="left", fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xlim(0, xmax); ax.set_ylim(-0.55, 1.55)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED, length=3)
    ax.tick_params(axis="y", length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper2_rineng/figures/fig_efficiency.pdf")
    a = ap.parse_args()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.5),
                                        gridspec_kw=dict(width_ratios=[1, 1, 1.1], wspace=0.88))

    # (a) TRẢ BAO NHIÊU — tỉ lệ 8,2:1 đọc được bằng thước
    bars(ax1, [11.5, 1.4], [MASK, CNT], ["pixel masks", "image-level\ncount"], 15.5,
         ["11.5 s", "$\\leq$ 1.4 s"])
    ax1.set_xlabel("seconds of expert time per nucleus", fontsize=8.5, color=INK)
    ax1.annotate("", xy=(1.4, 0.34), xytext=(11.5, 0.34),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax1.text(6.45, 0.42, "8.2$\\times$ less", ha="center", va="bottom", fontsize=9.5,
             color=INK, fontweight="bold")
    ax1.set_title("(a) what the labels cost", fontsize=9.5, color=INK, pad=9, loc="left")

    # (b) MẤT BAO NHIÊU — hai cột cùng chiều dài; trục bắt đầu từ 0 nên không phóng đại
    bars(ax2, [0.931, 0.918], [MASK, CNT], ["mask arm", "count-only\narm"], 1.30,
         ["0.931", "0.918"], xerr=[[0.006, 0.011], [0.006, 0.011]])
    ax2.set_xlabel("counting accuracy, $R^2$", fontsize=8.5, color=INK)
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.axvline(0.931, color=MUTED, lw=1.0, ls=(0, (3, 3)), ymin=0.06, ymax=0.94, zorder=4)
    ax2.text(0.465, 0.40, "98.6% of the\nmask $R^2$", ha="center", va="center", fontsize=9.5,
             color=INK, fontweight="bold")
    ax2.set_title("(b) what that buys", fontsize=9.5, color=INK, pad=9, loc="left")

    # (c) CẦN BAO NHIÊU — đường bão hoà; số ở 5.4.1, hiện chỉ nằm trong văn xuôi
    xs = [b[0] for b in BUDGET]; ys = [b[1] for b in BUDGET]
    lo = [b[2] for b in BUDGET]
    ax3.plot(xs, ys, "-", color=CNT, lw=1.8, zorder=2)
    for x, y, e in BUDGET:
        if e is None:
            ax3.plot([x], [y], "o", mfc="white", mec=CNT, mew=1.6, ms=6, zorder=3)
        else:
            ax3.errorbar([x], [y], yerr=[e], fmt="o", color=CNT, ms=6,
                         ecolor=CNT, elinewidth=1.2, capsize=3, zorder=3)
    ax3.axvline(25, color=MUTED, lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax3.annotate("a quarter of the budget\n$\\rightarrow$ 93% of the accuracy",
                 xy=(25, 0.849), xytext=(41, 0.55), fontsize=8.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax3.set_xlim(0, 106); ax3.set_ylim(0.15, 1.03)
    ax3.set_xticks([5, 25, 50, 100]); ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_xlabel("% of images given a count label", fontsize=8.5, color=INK)
    ax3.set_ylabel("$R^2$", fontsize=8.5, color=INK)
    ax3.grid(axis="y", color=GRID, lw=0.6)
    ax3.set_axisbelow(True)
    for s in ("top", "right"): ax3.spines[s].set_visible(False)
    for s in ("bottom", "left"): ax3.spines[s].set_color(MUTED)
    ax3.tick_params(labelsize=8, colors=MUTED, length=3)
    ax3.set_title("(c) how much of it is needed", fontsize=9.5, color=INK, pad=9, loc="left")

    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
