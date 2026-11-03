#!/usr/bin/env python3
"""T1.6 诊断(续)：在原版栈几何下，载荷到底占多少行/多少格？
输出：每样本 in-orig 行活动度剖面 → 载荷行=活动度 > 3×整图中位数 的连续行；
      以及 cell 级（24×24，每格 14×14 px）活动度 > 3×中位数的格子数分布。
"""
import glob
import io
import json

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor

CLIP = "/home/huangqixu01/models/clip-vit-large-patch14-336"
N = 24
proc = AutoImageProcessor.from_pretrained(CLIP)
bg = tuple(int(x * 255) for x in proc.image_mean)


def e2s(img):
    w, h = img.size
    if w == h:
        return img
    out = Image.new("RGB", (max(w, h), max(w, h)), bg)
    out.paste(img, (0, 0))
    return out


probe = [json.loads(l) for l in open(
    "/home/huangqixu01/experiment/t1_5b_package_2026-10-27/06_probe/t1_5_probe_typo.jsonl")]
want = {(r["cat"], int(r["qid"])) for r in probe}
rows = []
for f in sorted(glob.glob("/home/huangqixu01/data/MM-SafetyBench/data/*/TYPO.parquet")):
    cat = f.split("/data/")[-1].split("/")[0]
    df = pd.read_parquet(f)
    for _, r in df.iterrows():
        if (cat, int(r["id"])) in want:
            rows.append((cat, int(r["id"]), r["image"]))

band_rows, band_cells, prof0 = [], [], None
for cat, qid, ib in rows:
    img = Image.open(io.BytesIO(ib)).convert("RGB")
    px = proc(images=e2s(img), return_tensors="pt")["pixel_values"][0]
    rowact = np.array([px[:, r * 14:(r + 1) * 14, :].std().item() for r in range(N)])
    thr = 3 * float(np.median(rowact))
    band_rows.append(int((rowact > thr).sum()))
    cells = px.permute(1, 2, 0).reshape(N, 14, N, 14, 3).permute(0, 2, 1, 3, 4).reshape(N, N, -1)
    cact = cells.std(dim=2).numpy()
    band_cells.append(int((cact > thr).sum()))
    if prof0 is None:
        prof0 = (cat, qid, np.round(rowact, 4).tolist())

from collections import Counter
print("in-orig 几何：载荷行数分布", dict(sorted(Counter(band_rows).items())))
print("in-orig 几何：载荷格数分布", dict(sorted(Counter(band_cells).items())))
print("示例行活动度(前 12 行)", prof0[0], prof0[1], prof0[2][:12])
print("示例行活动度(后 12 行)", prof0[2][12:])
json.dump({"band_rows": dict(sorted(Counter(band_rows).items())),
           "band_cells": dict(sorted(Counter(band_cells).items())),
           "example": prof0}, open("/tmp/t1_6_geom_band.json", "w"), ensure_ascii=False, indent=1)
