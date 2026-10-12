#!/usr/bin/env python3
"""efficiency_profile.py — đo efficiency THẬT cho model headline efflite0 3.62M count-only.

Thay số CŨ (PACT 1.9M: 10.49 GMACs / 4.91ms) vì model đã đổi -> 3.62M efflite0.
Đo: params, GMACs (thop hoặc ptflops nếu có), latency GPU + CPU (ms/ảnh), peak VRAM.
Input 256x256x3 (đúng native PanNuke/NuInsSeg).

Chạy Kaggle (GPU):
  !python efficiency_profile.py --backbone efficientnet_lite0
"""
import argparse, time
import numpy as np
import torch
from distill_student_r2 import DensitySigmaUNet


def gmacs(model, x):
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return macs / 1e9, "thop"
    except Exception:
        pass
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(model, tuple(x.shape[1:]),
                                            as_strings=False, print_per_layer_stat=False, verbose=False)
        return macs / 1e9, "ptflops"
    except Exception as e:
        return float("nan"), f"n/a ({e})"


@torch.no_grad()
def bench(model, x, n_warm=10, n_run=100):
    dev = x.device
    for _ in range(n_warm):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_run):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_run * 1000.0  # ms/ảnh (bs=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    model = DensitySigmaUNet(32, backbone=args.backbone).eval()
    npar = sum(p.numel() for p in model.parameters())
    print(f"=== EFFICIENCY: DensitySigmaUNet({args.backbone}) count-only ===")
    print(f"  params        : {npar:,} ({npar/1e6:.3f} M)")

    x_cpu = torch.randn(1, 3, args.size, args.size)
    g, src = gmacs(model, x_cpu)
    print(f"  GMACs (bs1)   : {g:.2f}  [{src}]")

    # CPU latency
    ms_cpu = bench(model, x_cpu, n_warm=3, n_run=20)
    print(f"  latency CPU   : {ms_cpu:.2f} ms/ảnh ({1000/ms_cpu:.1f} ảnh/s)")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        model = model.to(dev); x = x_cpu.to(dev)
        torch.cuda.reset_peak_memory_stats()
        ms_gpu = bench(model, x, n_warm=20, n_run=200)
        vram = torch.cuda.max_memory_allocated() / 1e6
        print(f"  latency GPU   : {ms_gpu:.3f} ms/ảnh ({1000/ms_gpu:.1f} ảnh/s)  [{torch.cuda.get_device_name(0)}]")
        print(f"  peak VRAM     : {vram:.1f} MB (bs=1, {args.size}²)")
    else:
        print("  (không có GPU -> bỏ latency/VRAM GPU)")

    print("\nĐiền các số này vào §3.3 (thay số PACT 1.9M cũ).")


if __name__ == "__main__":
    main()
