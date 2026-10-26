#!/usr/bin/env python3
"""T1.4 · 无剪枝 vs TRIM(0.25) 生成脚本（临时脚本，/tmp）。

两种条件走同一条手动 embeds 管线（llava-hf），唯一区别是视觉 token 是否剪枝：
  noprun: 576 个视觉特征全部进 LLM
  trim:   CLIP 文本-图像 patch 相似度打分（TRIM 官方口径），保留 top-144（原序）
          + 被丢 432 个的均值聚合 token 拼尾 = 145 个进 LLM
生成参数：eager · fp16 · greedy · seed=42 · max_new_tokens=512（与原文一致）
断点续跑：输出 json 已有的 (cat,qid) 自动跳过。
用法：
  CUDA_VISIBLE_DEVICES=5 python t1_4_generate.py --condition noprun --limit 400
  CUDA_VISIBLE_DEVICES=6 python t1_4_generate.py --condition trim   --limit 400
"""
import argparse
import glob
import io
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

LLAVA = "/home/huangqixu01/models/llava-1.5-7b-hf"
CLIP = "/home/huangqixu01/models/clip-vit-large-patch14-336"
DATA = "/home/huangqixu01/data/MM-SafetyBench/data"
SPLIT = "SD"  # 由 --split 覆盖
KEEP_RATIO = 0.25
SEED = 42

PROMPT_T = "USER: <image>\n{q}\nASSISTANT:"


def stratified_sample(limit, split=None, ids_from=None):
    """ids_from: 上轮回答文件的路径，锁定同一批 (cat,qid)，保证可比"""
    split = split or SPLIT
    want = None
    if ids_from:
        want = {(json.loads(l)["cat"], json.loads(l)["qid"]) for l in open(ids_from)}
    rows = []
    for f in sorted(glob.glob(f"{DATA}/*/{split}.parquet")):
        cat = f.split("/data/")[-1].split("/")[0]
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            if want is not None and (cat, int(r["id"])) not in want:
                continue
            rows.append((cat, int(r["id"]), str(r["question"]), r["image"]))
    if want is not None:
        assert len(rows) == len(want), f"id 对齐失败: {len(rows)} != {len(want)}"
        return rows
    if limit and limit < len(rows):
        rng = random.Random(SEED)
        # 按类目比例分层
        by_cat = {}
        for r in rows:
            by_cat.setdefault(r[0], []).append(r)
        out = []
        for cat, lst in sorted(by_cat.items()):
            k = max(1, round(limit * len(lst) / len(rows)))
            out += rng.sample(lst, min(k, len(lst)))
        rows = out
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["noprun", "trim"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="0=全量 1680")
    ap.add_argument("--split", default="SD")
    ap.add_argument("--ids-from", default=None, help="上轮答案文件，锁定同一批样本 id")
    args = ap.parse_args()

    from PIL import Image
    from transformers import (AutoProcessor, AutoTokenizer, CLIPModel,
                              CLIPTextModelWithProjection, LlavaForConditionalGeneration)

    tag = (args.split if args.split != "SD" else "") + (str(args.limit) if args.limit else "full")
    out_path = f"/tmp/t1_4_answers_{args.condition}_{tag.lower()}.json"
    done = set()
    try:
        for r in json.load(open(out_path)):
            done.add((r["cat"], r["qid"]))
    except Exception:
        pass

    print(f"[{args.condition}] 加载 LLaVA (eager/fp16) ...", flush=True)
    proc = AutoProcessor.from_pretrained(LLAVA)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA, torch_dtype=torch.float16, attn_implementation="eager",
        low_cpu_mem_usage=True).to("cuda:0").eval()
    embed_layer = model.get_input_embeddings()
    proj = model.model.multi_modal_projector  # tf5.x: 在 .model 下
    layer = model.config.vision_feature_layer  # -2
    print(f"[{args.condition}] vision_feature_layer={layer}", flush=True)

    clip = text_tower = text_tok = None
    if args.condition == "trim":
        print(f"[{args.condition}] 加载 CLIP 打分管线 ...", flush=True)
        clip = CLIPModel.from_pretrained(CLIP, torch_dtype=torch.float32).to("cuda:0").eval()
        text_tower = CLIPTextModelWithProjection.from_pretrained(
            CLIP, torch_dtype=torch.float32).to("cuda:0").eval()
        text_tok = AutoTokenizer.from_pretrained(CLIP)

    rows = stratified_sample(args.limit, args.split, args.ids_from)
    rows = [r for r in rows if (r[0], r[1]) not in done]
    print(f"[{args.condition}] 待跑 {len(rows)} 条（已跳过 {len(done)}）", flush=True)

    img_id = 32000  # <image> token id（T0.4 已验证）
    t0 = time.time()
    fout = open(out_path, "a", encoding="utf-8")
    for i, (cat, qid, q, img_bytes) in enumerate(rows):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        prompt = PROMPT_T.format(q=q)
        inputs = proc(text=prompt, images=img, return_tensors="pt").to("cuda:0")
        px = inputs["pixel_values"]

        with torch.no_grad():
            # ── 视觉特征（与 llava-hf forward 同口径：hidden_states[-2] 丢 CLS）──
            vout = model.model.vision_tower(px, output_hidden_states=True)  # tf5.x: 在 .model 下
            feats = vout.hidden_states[layer][:, 1:]          # [1, 576, 1024] fp16

            if args.condition == "trim":
                # TRIM 打分（逐行对应 third_party/trim clip_encoder.py）：
                # texts = prompt 去掉 <image>（model_vqa.py: cur_prompt.replace('<image>',' ')）
                t_in = text_tok(text=prompt.replace("<image>", " "),
                                return_tensors="pt", truncation=True, padding=True).to("cuda:0")
                t_feat = text_tower(**t_in).text_embeds      # [1,768] fp32
                f32 = feats.to(torch.float32)
                nomalized = clip.vision_model.post_layernorm(f32)
                proj_img = clip.visual_projection(nomalized)  # [1,576,768]
                sims = torch.matmul(proj_img, t_feat.unsqueeze(2)).squeeze(2)
                score = F.softmax(-sims, dim=-1)              # [1,576]
                n_keep = int(feats.shape[1] * KEEP_RATIO)     # 144
                _, keep_idx = torch.topk(score, n_keep, dim=1, largest=True, sorted=False)
                keep_idx, _ = torch.sort(keep_idx[0])  # 关键：按原始位置排序（原版 bool-mask 索引语义）
                v_keep = f32[0, keep_idx]                     # 原序（bool-mask 语义）
                all_idx = set(range(feats.shape[1]))
                drop_idx = sorted(all_idx - set(keep_idx.tolist()))
                agg = f32[0, torch.tensor(drop_idx, device=f32.device)].mean(dim=0, keepdim=True)
                feats_sel = torch.cat([v_keep, agg], dim=0).to(torch.float16)  # [145,1024]
            else:
                feats_sel = feats[0]                           # [576,1024]

            # ── 投影 + 手动 scatter 进 embeds（与 llava forward 等价）──
            vis = proj(feats_sel.unsqueeze(0))[0]              # [N,4096] fp16
            ids = inputs["input_ids"][0].clone()
            pos = (ids == img_id).nonzero(as_tuple=True)[0]    # 原始 576 个位置
            s, e = int(pos[0]), int(pos[-1]) + 1               # 连续段
            ids = torch.cat([ids[:s], torch.full((feats_sel.shape[0],), img_id, dtype=ids.dtype, device=ids.device), ids[e:]])
            pos = (ids == img_id).nonzero(as_tuple=True)[0]    # 新 N 个位置
            assert len(pos) == feats_sel.shape[0], f"{len(pos)} != {feats_sel.shape[0]}"
            emb = embed_layer(ids.unsqueeze(0))                # [1,L,4096]
            emb[0, pos] = vis
            out = model.generate(
                inputs_embeds=emb, attention_mask=torch.ones_like(ids).unsqueeze(0),
                max_new_tokens=512, do_sample=False,
                pad_token_id=proc.tokenizer.pad_token_id)
        ans = proc.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        fout.write(json.dumps({"cat": cat, "qid": qid, "question": q, "answer": ans},
                              ensure_ascii=False) + "\n")
        fout.flush()
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[{args.condition}] {i+1}/{len(rows)} · {el/(i+1):.1f}s/条 · ETA {(len(rows)-i-1)*el/(i+1)/60:.0f}min", flush=True)
    fout.close()
    print(f"[{args.condition}] 完成 -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
