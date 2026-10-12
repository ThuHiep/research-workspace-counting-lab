#!/usr/bin/env python3
"""fig1_patches.py - 2 anh THAT cho Hinh 1 (fig_design.drawio).

Cung MOT patch PanNuke fold3, hai kieu nhan:
  fig1_mask.png  - phu duong vien that cua tung nhan  -> nhan mat na
  fig1_plain.png - anh tran                           -> nhan dem (chi con so)
So nhan in ra phai KHOP con so trong so do (22).
Ve o 4x roi thu nho -> duong vien min khi in.
  python fig1_patches.py
"""
import json, collections
from PIL import Image, ImageDraw

R = "/Users/thuhiep/Downloads/coco/fold3"
OUT = "paper2_rineng/figures"
IMG, K, W = "im0507.png", 4, 512          # K=he so sieu lay mau, W=be rong xuat

d = json.load(open(f"{R}/annotations.json"))
iid = next(i["id"] for i in d["images"] if i["file_name"] == IMG)
polys = [a["segmentation"] for a in d["annotations"] if a["image_id"] == iid]

im = Image.open(f"{R}/images/{IMG}").convert("RGB")
im.resize((W, W), Image.LANCZOS).save(f"{OUT}/fig1_plain.png")

big = im.resize((im.width * K, im.height * K), Image.LANCZOS)
dr = ImageDraw.Draw(big)
for poly in polys:
    for p in poly:
        pts = [(x * K, y * K) for x, y in zip(p[0::2], p[1::2])]
        dr.line(pts + [pts[0]], fill=(0, 230, 70), width=2 * K, joint="curve")
big.resize((W, W), Image.LANCZOS).save(f"{OUT}/fig1_mask.png")

# --- ban do mat do MUC TIEU: moi nhan gop 1/dien tich -> TONG = dung so nhan ---
# cung cong thuc distill_student_r2.py:189 (dens[mr] += 1.0/a), KHONG phai Gaussian o tam
import numpy as np

dens = np.zeros((im.height, im.width), np.float32)
for poly in polys:
    m = Image.new("1", im.size, 0)
    for p in poly:
        ImageDraw.Draw(m).polygon(list(zip(p[0::2], p[1::2])), fill=1)
    mr = np.asarray(m, bool)
    if mr.sum():
        dens[mr] += 1.0 / int(mr.sum())
assert abs(dens.sum() - len(polys)) < 0.5, f"tong ban do {dens.sum():.2f} != {len(polys)} nhan"

ANCH = np.array([[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]], float)
# CHI de hien thi: chuan hoa theo phan vi 99, KHONG theo max. Anh nay co 1 nhan 9 px cho 1/a
# lon gap 159 lan cac nhan khac -> lay max thi ca ban do den thui. Gia tri that van tong = 22.
u = np.clip(dens / np.percentile(dens[dens > 0], 99), 0, 1) ** 0.7
rgb = np.stack([np.interp(u, np.linspace(0, 1, len(ANCH)), ANCH[:, c]) for c in range(3)], -1)
Image.fromarray(rgb.astype(np.uint8)).resize((W, W), Image.LANCZOS).save(f"{OUT}/fig1_density.png")

print(f"{IMG}: {len(polys)} nhan | tong ban do mat do = {dens.sum():.3f} -> 3 anh trong {OUT}/")
