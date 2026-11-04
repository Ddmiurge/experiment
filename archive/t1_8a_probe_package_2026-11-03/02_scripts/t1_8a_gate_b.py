#!/usr/bin/env python3
"""T1.8a §1b 严格门 B：只导入 FasterVLM 仓库自带的 llava（不能与我们的栈混用同名包）

做法：独立进程 + 只把 /tmp/fastervlm 放进 sys.path，用仓库自己的
load_pretrained_model / process_images / encode_images 跑 10 张，
与适配器（视觉塔级）名单比 Jaccard。
"""
import base64
import io
import json
import os
import random
import sys

sys.path.insert(0, "/tmp/fastervlm")
sys.path.insert(0, "/tmp")

import torch                                                     # noqa: E402
from PIL import Image                                            # noqa: E402
from transformers import AutoConfig, CLIPImageProcessor          # noqa: E402
from llava.model.builder import load_pretrained_model            # noqa: E402
from llava.mm_utils import process_images                        # noqa: E402
from t1_8a_adapter import select_cells                           # noqa: E402

VANILLA = "/home/huangqixu01/models/llava-v1.5-7b-vanilla"
CLIP = "/home/huangqixu01/models/clip-vit-large-patch14-336"
KEEP = 144


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def main():
    random.seed(42)
    imgs = []
    for split, path in (("SD", "/tmp/t1_4c_sd399_data.json"),
                        ("TYPO", "/tmp/t1_4c_typo399_data.json")):
        rows = [(r["cat"], int(r["qid"]), base64.b64decode(r["image_b64"]))
                for r in json.load(open(path))]
        for cat, qid, ib in random.sample(rows, 5):
            imgs.append((split, cat, qid, ib))

    cfg = AutoConfig.from_pretrained(VANILLA)
    cfg.mm_vision_tower = CLIP
    for _k, _v in (("attention_dropout", 0.0), ("attention_bias", False),
                   ("mm_patch_merge_type", "flat"), ("rope_theta", 10000.0)):
        if getattr(cfg, _k, None) is None:
            setattr(cfg, _k, _v)
    tok, model, ip, ctx = load_pretrained_model(
        VANILLA, None, "llava_v1", device="cuda:0", use_flash_attn=False,
        visual_token_num=KEEP, config=cfg)
    model.eval()
    print("模型类型:", type(model).__name__, "· visual_token_num =", model.get_visual_token_num(),
          "· model.dtype =", model.dtype,
          "· tower.dtype =", model.get_model().vision_tower.dtype)

    js = []
    for split, cat, qid, ib in imgs:
        img = Image.open(io.BytesIO(ib)).convert("RGB")
        px = process_images([img], ip, cfg)
        # 关键：适配器必须收到与模型实际消费同 dtype 的 px（fp16），否则像素精度不同 → 边界格翻转
        px_in = px.to("cuda:0", dtype=model.dtype)
        keep_ad = select_cells(model.get_model().vision_tower, px_in)
        with torch.no_grad():
            # 关键：原生路径必须用模型自身 dtype（否则 fp32/fp16 混算 → 边界格翻转）
            _, masks, atts = model.encode_images(px.to("cuda:0", dtype=model.dtype))
        nat = [i for i, v in enumerate(masks[0].tolist()) if v]
        j = jac(keep_ad, nat)
        js.append(j)
        print(f"  {split}:{cat}#{qid}  Jaccard={j:.4f}  |adapter|={len(keep_ad)} |native|={len(nat)}",
              flush=True)
    print(f"最小 Jaccard = {min(js):.4f} → {'PASS' if min(js) >= 0.99 else 'FAIL'}")


if __name__ == "__main__":
    main()
