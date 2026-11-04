#!/usr/bin/env python3
"""T1.8a §1c · FasterVLM 挑格画像（399 × 两图型 × 三种既有标法）

FasterVLM 名单：/tmp/t1_8a_adapter.select_cells（与仓库 f5d5f12 逐行等价）
三种标法（全部复用冻结实现/归档件，不新写）：
  ink     = T1.6 墨迹秩（TYPO 用归档 06_masks；SD 用同函数现算 /tmp/t1_8a_masks_ink_sd_*）
  sam     = T1.7 SAM 前景（SD 用归档 05_masks；TYPO 用同脚本现算 /tmp/t1_8a_masks_typo/）
  center  = T1.7 中心秩（同上文件里的 center 分支）
输出：长表 CSV，一行 = (图型, 样本, 标法)
"""
import base64
import importlib.util
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = "/home/huangqixu01/experiment"
sys.path.insert(0, os.path.join(REPO, "third_party/trim"))
sys.path.insert(0, "/tmp")

from llava.mm_utils import process_images                      # noqa: E402
from transformers import CLIPImageProcessor                    # noqa: E402
from t1_8a_adapter import select_cells                         # noqa: E402

# 用文件路径导入仓库自带的 CLIPVisionTower（避免 llava 包名冲突）
spec = importlib.util.spec_from_file_location(
    "fv_clip_encoder", "/tmp/fastervlm/llava/model/multimodal_encoder/clip_encoder.py")
fv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fv)

CLIP = "/home/huangqixu01/models/clip-vit-large-patch14-336"


def load_masks(path):
    return {(r["cat"], int(r["qid"])): r for r in map(json.loads, open(path))}


def main():
    args = type("A", (), {"mm_vision_select_layer": -2, "mm_vision_select_feature": "patch"})()
    tower = fv.CLIPVisionTower(CLIP, args).to("cuda:0", dtype=torch.float16).eval()  # 与生成一致

    ip = CLIPImageProcessor.from_pretrained(CLIP)
    cfg = type("C", (), {"image_aspect_ratio": "pad"})()

    ARCH = "/home/huangqixu01/archive/01_packages"
    mask_sets = {
        ("TYPO", "ink"): (load_masks(f"{ARCH}/t1_6_package_2026-10-30/06_masks/t1_6_masks_payload.jsonl"),
                          load_masks(f"{ARCH}/t1_6_package_2026-10-30/06_masks/t1_6_masks_blank.jsonl")),
        ("TYPO", "sam"): (load_masks("/tmp/t1_8a_masks_typo/t1_7_masks_payload_sd399.jsonl"),
                          load_masks("/tmp/t1_8a_masks_typo/t1_7_masks_blank_sd399.jsonl")),
        ("SD", "ink"): (load_masks("/tmp/t1_8a_masks_ink_sd_payload.jsonl"),
                        load_masks("/tmp/t1_8a_masks_ink_sd_blank.jsonl")),
        ("SD", "sam"): (load_masks(f"{ARCH}/t1_7_package_2026-11-02/05_masks/t1_7_masks_payload_sd399.jsonl"),
                        load_masks(f"{ARCH}/t1_7_package_2026-11-02/05_masks/t1_7_masks_blank_sd399.jsonl")),
    }

    # center 标法：中心秩（b2_sets 与样本无关，144 近心 / 144 远心，逐样本相同）
    N = 24
    cells = [(r * N + c) for r in range(N) for c in range(N)]
    d = {(r * N + c): ((r - (N - 1) / 2) ** 2 + (c - (N - 1) / 2) ** 2) ** .5
         for r in range(N) for c in range(N)}
    near = sorted(sorted(cells, key=lambda x: (d[x], x))[:144])
    far = sorted(sorted(cells, key=lambda x: (-d[x], x))[:144])
    center_p4 = {"keep": near}
    center_p3 = {"keep": far}

    out = []
    for split, data_path in (("SD", "/tmp/t1_4c_sd399_data.json"),
                             ("TYPO", "/tmp/t1_4c_typo399_data.json")):
        rows = [(r["cat"], int(r["qid"]), base64.b64decode(r["image_b64"]))
                for r in json.load(open(data_path))]
        for i, (cat, qid, ib) in enumerate(rows):
            img = Image.open(io.BytesIO(ib)).convert("RGB")
            px = process_images([img], ip, cfg)
            keep = set(select_cells(tower, px.to("cuda:0", dtype=torch.float16)))  # 与生成同 dtype
            for method in ("ink", "sam", "center"):
                if method == "center":
                    m4, m3 = center_p4, center_p3
                else:
                    p4, p3 = mask_sets[(split, method)]
                    m4 = p4.get((cat, qid)); m3 = p3.get((cat, qid))
                if m4 is None or m3 is None:
                    continue
                s4, s3 = set(m4["keep"]), set(m3["keep"])
                # sam 标法：非 sam 分支的样本其实是 center 回退，单独记 branch
                branch = m4.get("branch", "n/a")
                out.append({
                    "split": split, "cat": cat, "qid": qid, "method": method,
                    "branch": branch, "keep_n": len(keep),
                    "cover4": len(keep & s4) / len(s4),          # 载荷包含率
                    "bg_del": 1 - len(keep & s3) / len(s3),      # 背景删除率
                    "jac4": len(keep & s4) / len(keep | s4),
                    "jac3": len(keep & s3) / len(keep | s3),
                })
            if (i + 1) % 100 == 0:
                print(f"{split} {i+1}/{len(rows)}", flush=True)

    df = pd.DataFrame(out)
    df.to_csv("/tmp/t1_8a_profile.csv", index=False)
    print("\n=== 汇总（按 图型 × 标法）===")
    g = df.groupby(["split", "method"]).agg(
        n=("cover4", "size"), cover4=("cover4", "mean"), bg_del=("bg_del", "mean"),
        jac4=("jac4", "mean"), jac3=("jac3", "mean"), keep_n=("keep_n", "mean"))
    print(g.round(4).to_string())
    print("\n=== TYPO × sam 标法 · 按 branch 拆 ===")
    sub = df[(df.split == "TYPO") & (df.method == "sam")]
    print(sub.groupby("branch").agg(n=("cover4", "size"), cover4=("cover4", "mean"),
                                    bg_del=("bg_del", "mean")).round(4).to_string())
    print("\nCSV -> /tmp/t1_8a_profile.csv")


if __name__ == "__main__":
    main()
