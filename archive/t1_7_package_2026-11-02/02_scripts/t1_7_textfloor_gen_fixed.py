#!/usr/bin/env python3
"""T1.7 · E1 文字地板：同批 399 条 SD 问题，**只发文本不发图**（原版栈 TRIM-FT）。
其余解码/模板与 T1.6 完全一致（vicuna_v1, greedy, seed42, 512）。判读见协议 §2。
用法：CUDA_VISIBLE_DEVICES=5 /home/huangqixu01/venvs/trim-orig/bin/python \
        t1_7_textfloor_gen.py --data /tmp/t1_4c_sd399_data.json"""
import argparse, base64, json, os, sys
import torch
REPO = os.environ.get("REPO", "/home/huangqixu01/experiment")
TRIM_REPO = os.path.join(REPO, "third_party/trim")
MODEL_PATH = os.environ.get("TRIM_M", "/home/huangqixu01/models/llava-v1.5-7b-TRIM")
sys.path.insert(0, TRIM_REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp/t1_4c_sd399_data.json")
ap.add_argument("--conv", default="vicuna_v1")
ap.add_argument("--out", default="/tmp/t1_7_answers_textfloor_sd399.json")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX

# [服务器补丁] 检查点里的 mm_vision_tower 是训练机相对路径，与 T1.6 一样按本地 CLIP 覆盖
from transformers import AutoConfig
_cfg = AutoConfig.from_pretrained(MODEL_PATH)
_cfg.mm_vision_tower = os.environ.get("CLIP_PATH",
                                      "/home/huangqixu01/models/clip-vit-large-patch14-336")
tok, model, ip, ctx = load_pretrained_model(MODEL_PATH, None, "llava_v1",
                                            device="cuda:0", use_flash_attn=False, config=_cfg)
try:
    model.get_model().vision_tower.token_reduce_func = None   # 无图，本行仅为状态明确
except Exception as _e:
    print(f"[warn] vision_tower 未加载（文本地板不需要）：{_e}", flush=True)
rows = [(r["cat"], int(r["qid"]), r["question"]) for r in json.load(open(a.data))]
if a.limit: rows = rows[:a.limit]
out = open(a.out, "w", encoding="utf-8")
for i, (cat, qid, q) in enumerate(rows):
    conv = conv_templates[a.conv].copy()
    conv.append_message(conv.roles[0], q)          # 无 <image>
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX,
                                return_tensors="pt").unsqueeze(0).to("cuda:0")
    with torch.inference_mode():
        o = model.generate(ids, do_sample=False, max_new_tokens=512, use_cache=True)
    ans = tok.decode(o[0], skip_special_tokens=True).split(conv.roles[1] + ":")[-1].strip()
    assert IMAGE_TOKEN_INDEX not in o[0].tolist(), "文本地板里混入了 image token"
    out.write(json.dumps({"cat": cat, "qid": qid, "question": q, "answer": ans,
                          "prune": "textfloor"}, ensure_ascii=False) + "\n"); out.flush()
    if (i + 1) % 50 == 0: print(f"[{i+1}/{len(rows)}]", flush=True)
out.close(); print("完成 ->", a.out, flush=True)
