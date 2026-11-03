#!/usr/bin/env python3
"""T1.6 停线诊断：探针 px 路径(裸 CLIP processor) vs 原版栈 px 路径(expand2square) 的几何差

只做测量，不生成、不改任何仓库代码。
对 399 条 TYPO 样本各算：
  A. probe 路径复现（裸 CLIP processor：短边 336 + 中心裁剪）→ top6 行，与归档 payload_rows 比对
  B. 原版栈路径（expand2square 均值填充成正方形 → CLIP processor 336）→ top6 行
  C. 两者行集合的 IoU / 重合行数
"""
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

CLIP = "/home/huangqixu01/models/clip-vit-large-patch14-336"
NROW = 24
TOPK = 6

from transformers import AutoImageProcessor  # noqa: E402

proc = AutoImageProcessor.from_pretrained(CLIP)


def top6(px):
    act = [float(px[0, :, r * 14:(r + 1) * 14, :].std()) for r in range(NROW)]
    return sorted(range(NROW), key=lambda r: -act[r])[:6], act


def expand2square(img, bg):
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    out = Image.new("RGB", (side, side), bg)
    # LLaVA 的 mm_utils.expand2square：paste 在 (0,0)
    out.paste(img, (0, (side - h) // 2))
    return out


def main():
    probe = [json.loads(l) for l in open(
        "/home/huangqixu01/experiment/t1_5b_package_2026-10-27/06_probe/t1_5_probe_typo.jsonl")]
    want = {(r["cat"], int(r["qid"])): r["payload_rows"] for r in probe}
    data = "/home/huangqixu01/data/MM-SafetyBench/data"
    rows = []
    for f in sorted(glob.glob(f"{data}/*/TYPO.parquet")):
        cat = f.split("/data/")[-1].split("/")[0]
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            k = (cat, int(r["id"]))
            if k in want:
                rows.append((cat, int(r["id"]), r["image"]))
    print("样本数", len(rows), flush=True)

    bg = tuple(int(x * 255) for x in proc.image_mean)
    sizes, agree_a, ious, shifts, per = {}, 0, [], [], []
    for cat, qid, img_bytes in rows:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        sizes[img.size] = sizes.get(img.size, 0) + 1
        # A. 裸 CLIP processor（探针路径）
        pxA = proc(images=img, return_tensors="pt")["pixel_values"]
        rowsA, _ = top6(pxA)
        # B. expand2square → CLIP processor（原版栈路径）
        pxB = proc(images=expand2square(img, bg), return_tensors="pt")["pixel_values"]
        rowsB, actB = top6(pxB)
        arch = sorted(want[(cat, qid)])
        agree_a += (sorted(rowsA) == arch)
        iou = len(set(arch) & set(rowsB)) / len(set(arch) | set(rowsB))
        ious.append(iou)
        shifts.append(float(np.mean(rowsB) - np.mean(arch)))
        per.append((cat, qid, arch, sorted(rowsB), round(iou, 3), round(shifts[-1], 2)))

    print("图像尺寸分布", sizes)
    print(f"A 裸 CLIP 复现归档行集合: {agree_a}/{len(rows)}")
    print(f"B 原版栈行 vs 归档行: 平均 IoU={np.mean(ious):.3f} 中位={np.median(ious):.3f} "
          f"完全一致={sum(i == 1.0 for i in ious)}")
    print(f"   平均行号位移(原版栈−归档) = {np.mean(shifts):+.2f} 行 (sd {np.std(shifts):.2f})")
    print("前 5 条:", *per[:5], sep="\n  ")
    json.dump({"sizes": {f"{k[0]}x{k[1]}": v for k, v in sizes.items()},
               "agree_probe_path": agree_a, "n": len(rows),
               "mean_iou_orig_vs_archive": float(np.mean(ious)),
               "mean_row_shift": float(np.mean(shifts)),
               "per_sample": per}, open("/tmp/t1_6_geom_diag.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
