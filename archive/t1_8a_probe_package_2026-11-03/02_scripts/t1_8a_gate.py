#!/usr/bin/env python3
"""T1.8a §1b 核对门：适配器名单 vs 仓库原生路径实际保留名单（10 张随机图，Jaccard ≥0.99）

两条原生路径：
  A. 视塔级：仓库自带 CLIPVisionTower.forward → 仓库 llava_arch.encode_images 的同款 3 行
     （topk on mean-over-heads CLS attention）—— 不需 7B 权重
  B. 模型级：完整 LlamaLlavaForCausalLM.encode_images（vanilla 权重就绪后跑）—— 严格门
用法： python t1_8a_gate.py [--full]
"""
import argparse
import base64
import importlib.util
import io
import json
import os
import random
import sys

import torch
from PIL import Image

REPO = "/home/huangqixu01/experiment"
sys.path.insert(0, os.path.join(REPO, "third_party/trim"))
sys.path.insert(0, "/tmp")

from llava.mm_utils import process_images                      # noqa: E402
from transformers import CLIPImageProcessor                    # noqa: E402
from t1_8a_adapter import select_cells                         # noqa: E402

spec = importlib.util.spec_from_file_location(
    "fv_clip_encoder", "/tmp/fastervlm/llava/model/multimodal_encoder/clip_encoder.py")
fv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fv)

CLIP = "/home/huangqixu01/models/clip-vit-large-patch14-336"
VANILLA = "/home/huangqixu01/models/llava-v1.5-7b-vanilla"
KEEP = 144


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="再跑模型级 encode_images 核对")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()

    # 10 张：SD 与 TYPO 各 5，固定种子
    random.seed(42)
    imgs = []
    for split, path in (("SD", "/tmp/t1_4c_sd399_data.json"), ("TYPO", "/tmp/t1_4c_typo399_data.json")):
        rows = [(r["cat"], int(r["qid"]), base64.b64decode(r["image_b64"]))
                for r in json.load(open(path))]
        for cat, qid, ib in random.sample(rows, 5):
            imgs.append((split, cat, qid, ib))

    ip = CLIPImageProcessor.from_pretrained(CLIP)
    cfg = type("C", (), {"image_aspect_ratio": "pad"})()
    args = type("A", (), {"mm_vision_select_layer": -2, "mm_vision_select_feature": "patch"})()
    tower = fv.CLIPVisionTower(CLIP, args).to("cuda:0", dtype=torch.float16).eval()  # 与生成时一致

    # A. 视塔级原生路径（仓库 forward + 仓库 encode_images 同款三行）
    print("=== A · 视塔级原生路径 ===")
    js = []
    for split, cat, qid, ib in imgs:
        img = Image.open(io.BytesIO(ib)).convert("RGB")
        px = process_images([img], ip, cfg)
        keep_ad = select_cells(tower, px)
        with torch.no_grad():
            _, atts = tower(px.to("cuda:0", dtype=torch.float16))   # 仓库 forward
        nat = torch.topk(atts.mean(dim=1), k=KEEP, dim=1)[1][0].tolist()  # 仓库三行
        j = jac(keep_ad, nat)
        js.append(j)
        print(f"  {split}:{cat}#{qid}  Jaccard={j:.4f}  |adapter|={len(keep_ad)} |native|={len(set(nat))}")
    print(f"  最小 Jaccard = {min(js):.4f} → {'PASS' if min(js) >= 0.99 else 'FAIL'}")

    if not a.full:
        return
    # B. 模型级（严格门）
    print("\n=== B · 模型级 encode_images ===")
    sys.path.insert(0, "/tmp/fastervlm")
    from llava.model.builder import load_pretrained_model
    from transformers import AutoConfig
    cfgm = AutoConfig.from_pretrained(VANILLA)
    cfgm.mm_vision_tower = CLIP
    tok, model, ip2, ctx = load_pretrained_model(VANILLA, None, "llava_v1", device="cuda:0",
                                                 use_flash_attn=False, visual_token_num=KEEP,
                                                 config=cfgm)
    model.eval()
    js = []
    for split, cat, qid, ib in imgs:
        img = Image.open(io.BytesIO(ib)).convert("RGB")
        px = process_images([img], ip2, cfgm)
        keep_ad = select_cells(model.get_model().vision_tower, px)
        with torch.no_grad():
            _, masks, _ = model.encode_images(px.to("cuda:0", dtype=torch.float16))
        nat = [i for i, v in enumerate(masks[0].tolist()) if v]
        j = jac(keep_ad, nat)
        js.append(j)
        print(f"  {split}:{cat}#{qid}  Jaccard={j:.4f}  |native|={len(nat)}")
    print(f"  最小 Jaccard = {min(js):.4f} → {'PASS' if min(js) >= 0.99 else 'FAIL'}")


if __name__ == "__main__":
    main()
