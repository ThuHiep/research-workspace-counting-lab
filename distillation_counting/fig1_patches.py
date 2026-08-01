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

print(f"{IMG}: {len(polys)} nhan -> {OUT}/fig1_mask.png, fig1_plain.png")
