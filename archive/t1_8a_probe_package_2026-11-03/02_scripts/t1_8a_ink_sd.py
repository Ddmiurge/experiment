#!/usr/bin/env python3
"""T1.8a 1c 补格：SD 图型 × T1.6 墨迹秩标法（复用仓库 oracle_geom.build_keeps，不新写标法）"""
import base64, io, json, os, sys
sys.path.insert(0, "/home/huangqixu01/experiment/experiments/t1_6_2026-10-29/scripts")
sys.path.insert(0, "/home/huangqixu01/experiment/third_party/trim")
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor
from llava.mm_utils import process_images
from oracle_geom import build_keeps, cell_activity

ip = CLIPImageProcessor.from_pretrained("/home/huangqixu01/models/clip-vit-large-patch14-336")
cfg = type("C", (), {"image_aspect_ratio": "pad"})()
rows = [(r["cat"], int(r["qid"]), base64.b64decode(r["image_b64"]))
        for r in json.load(open("/tmp/t1_4c_sd399_data.json"))]
fp = open("/tmp/t1_8a_masks_ink_sd_payload.jsonl", "w")
fb = open("/tmp/t1_8a_masks_ink_sd_blank.jsonl", "w")
for i, (cat, qid, ib) in enumerate(rows):
    img = Image.open(io.BytesIO(ib)).convert("RGB")
    act = cell_activity(process_images([img], ip, cfg).numpy())
    kp, kb, au = build_keeps(act)
    fp.write(json.dumps({"cat": cat, "qid": qid, "keep": kp, "branch": "inkrank",
                         "ink_n": au["ink_n"], "cov4": au["cov4"], "leak3": au["leak3"]}) + "\n")
    fb.write(json.dumps({"cat": cat, "qid": qid, "keep": kb, "branch": "inkrank",
                         "ink_n": au["ink_n"]}) + "\n")
    if (i + 1) % 100 == 0: print(i + 1, flush=True)
fp.close(); fb.close(); print("完成")
