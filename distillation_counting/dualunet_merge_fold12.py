#!/usr/bin/env python3
"""dualunet_merge_fold12.py — gộp COCO fold1 + fold2 -> fold12 (train), giữ fold3 (test).

DualU-Net train theo COCO annotations.json + images/*.png (do pannuke2coco sinh ra).
Ta train fold_1+2 / eval fold_3 (leak-free, fold_3 SẠCH — xem [[pathosam-training-data]]).
Gộp = nối images[] + annotations[] của 2 fold, RE-INDEX id để không đụng nhau, giữ categories.

Ảnh: symlink (không copy, đỡ tốn disk Kaggle). annotations.json: ghi mới.

Chạy Kaggle (sau khi đã có coco/fold1,fold2,fold3 từ pannuke2coco):
  !python dualunet_merge_fold12.py --coco /kaggle/input/<ten-dataset-coco>/coco --out /kaggle/working/coco
"""
import argparse, json, os
from pathlib import Path


def load(coco, fold):
    p = Path(coco) / f"fold{fold}" / "annotations.json"
    with open(p) as f:
        return json.load(f), Path(coco) / f"fold{fold}" / "images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="thư mục chứa fold1/ fold2/ fold3/")
    ap.add_argument("--out", default="/kaggle/working/coco")
    ap.add_argument("--folds", default="1,2")
    args = ap.parse_args()

    out = Path(args.out) / "fold12"
    (out / "images").mkdir(parents=True, exist_ok=True)

    merged = {"images": [], "annotations": [], "categories": None}
    img_off, ann_off = 0, 0
    for fold in [int(x) for x in args.folds.split(",")]:
        data, imgdir = load(args.coco, fold)
        if merged["categories"] is None:
            merged["categories"] = data["categories"]
        id_map = {}
        for im in data["images"]:
            new_id = im["id"] + img_off
            id_map[im["id"]] = new_id
            im = dict(im); im["id"] = new_id
            # symlink ảnh sang out/images (tên giữ nguyên, các fold không trùng tên PanNuke)
            src = imgdir / im["file_name"]
            dst = out / "images" / im["file_name"]
            if not dst.exists():
                try:
                    os.symlink(src.resolve(), dst)
                except FileExistsError:
                    pass
            merged["images"].append(im)
        for an in data["annotations"]:
            an = dict(an)
            an["id"] = an["id"] + ann_off
            an["image_id"] = id_map[an["image_id"]]
            merged["annotations"].append(an)
        img_off = max(im["id"] for im in merged["images"]) + 1
        ann_off = max(an["id"] for an in merged["annotations"]) + 1
        print(f"[fold{fold}] +{len(data['images'])} ảnh, +{len(data['annotations'])} ann")

    with open(out / "annotations.json", "w") as f:
        json.dump(merged, f)
    print(f"\nfold12: {len(merged['images'])} ảnh, {len(merged['annotations'])} ann, "
          f"{len(merged['categories'])} category -> {out}")


if __name__ == "__main__":
    main()
