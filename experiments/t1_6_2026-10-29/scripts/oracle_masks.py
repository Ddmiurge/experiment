#!/usr/bin/env python3
"""T1.6 · 掩码预跑审计（修订#2 后：模型几何 cell 级，生成器的同一函数、同一 preprocess 路径）

跑法（服务器，trim-orig venv，纯 CPU，几分钟）：
  /home/huangqixu01/venvs/trim-orig/bin/python oracle_masks.py \
      --data /tmp/t1_4c_typo399_data.json
产出：
  /tmp/t1_6_masks_payload.jsonl / /tmp/t1_6_masks_blank.jsonl   （生成器冒烟时逐条硬比对）
  stdout 审计件：|ink| 分布、pad/trim 计数、kept_rows 位置、fallback 触发数、前 3 条样例
  —— 这些数字就是协议 §2 修订#2 的冻结记录。
"""
import argparse
import base64
import io
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.environ.get("REPO", "/home/huangqixu01/experiment")
sys.path.insert(0, os.path.join(REPO, "third_party/trim"))

from oracle_geom import BUDGET, build_keeps, cell_activity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/t1_4c_typo399_data.json")
    ap.add_argument("--outdir", default="/tmp")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    # 与原版栈逐字同路径：llava.process_images + 真 CLIPImageProcessor + pad 配置
    from PIL import Image
    from transformers import CLIPImageProcessor
    from llava.mm_utils import process_images

    CLIP = os.environ.get("CLIP_PATH", "/home/huangqixu01/models/clip-vit-large-patch14-336")
    ip = CLIPImageProcessor.from_pretrained(CLIP)
    cfg = type("C", (), {"image_aspect_ratio": "pad"})()

    rows = [(r["cat"], int(r["qid"]), base64.b64decode(r["image_b64"]))
            for r in json.load(open(a.data))]
    if a.limit:
        rows = rows[:a.limit]
    fp = open(f"{a.outdir}/t1_6_masks_payload.jsonl", "w", encoding="utf-8")
    fb = open(f"{a.outdir}/t1_6_masks_blank.jsonl", "w", encoding="utf-8")
    ink_n, notes, sample_show = [], Counter(), 0
    for cat, qid, img_bytes in rows:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        px = process_images([img], ip, cfg)
        act = cell_activity(px.numpy())
        kp, kb, au = build_keeps(act)
        assert len(kp) == len(kb) == BUDGET and not (set(kp) & set(kb))
        fp.write(json.dumps({"cat": cat, "qid": qid, "keep": kp, **au}, ensure_ascii=False) + "\n")
        fb.write(json.dumps({"cat": cat, "qid": qid, "keep": kb, **au}, ensure_ascii=False) + "\n")
        ink_n.append((au["ink_n"], au["cov4"], au["leak3"]))
        if sample_show < 3:
            print(f"样例 {cat}#{qid}: ink_n={au['ink_n']} cov4={au['cov4']:.2f} "
                  f"leak3={au['leak3']} kept_rows(④)={sorted({k // 24 for k in kp})} "
                  f"图尺寸={img.size}", flush=True)
            sample_show += 1
    fp.close(); fb.close()
    ink_a = np.array([x[0] for x in ink_n])
    cov = np.array([x[1] for x in ink_n if x[1] == x[1]])          # 去 nan
    leak = sum(1 for x in ink_n if x[2] > 0)
    print(f"\n审计（修订#3 终版口径：纯秩干预，otsu 仅诊断）：n={len(ink_n)}")
    print(f"  |ink| 分布 min/p50/p95/max = {ink_a.min()}/{int(np.percentile(ink_a,50))}/"
          f"{int(np.percentile(ink_a,95))}/{ink_a.max()}（>144 的样本 cov4<1，如实登记）")
    print(f"  ④ 覆盖率 cov4：中位 {np.median(cov):.3f} · <1 的样本 {int((cov<1).sum())} · "
          f"③ 泄漏样本 = {leak}（设计恒 0）")
    stop = leak > 0 or np.median(cov) < 0.9
    print("  停线判定：", "❌ STOP——按 TASKS step0 停报，勿进冒烟" if stop else "PASS → 可进 step2 冒烟")


if __name__ == "__main__":
    main()
