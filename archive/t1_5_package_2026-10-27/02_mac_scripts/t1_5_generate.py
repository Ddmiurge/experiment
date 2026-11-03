#!/usr/bin/env python3
"""T1.5 · 四格生成（llava-hf 栈 = vanilla LLaVA-1.5-7B 权重）。

与 T1.4 的 t1_4_generate.py 同一条管线、同一批 399 个 id，只把剪枝方式做成可选：

    --prune none           576 个视觉 token 全进（已有读数，可跳过）
    --prune trim_official  照抄官方 clip_encoder.py：topk(softmax(-s))→保留**低**相似度
    --prune trim_highsim   按 TRIM 原文 §3.2 / SAP 原文对 TRIM 的定义：保留**高**相似度
    --prune random         同预算随机保留（阴性对照，对齐原文 Random 行 56.75）

四种都用同样的「144 保留（原序）+ 1 个被丢 token 的均值聚合 = 145 进 LLM」，
所以唯一变量是**留哪些**，不是**留多少**。

断点续跑：输出 json 已有的 (cat,qid) 自动跳过。
用法：
  CUDA_VISIBLE_DEVICES=5 python t1_5_generate.py --split SD --prune trim_highsim \
      --ids-from /tmp/t1_4_answers_orig_sd_noprun399.json
  CUDA_VISIBLE_DEVICES=6 python t1_5_generate.py --split SD --prune random \
      --ids-from /tmp/t1_4_answers_orig_sd_noprun399.json
"""
import argparse
import glob
import io
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

LLAVA = os.environ.get("LLAVA_PATH", "/home/huangqixu01/models/llava-1.5-7b-hf")
CLIP = os.environ.get("CLIP_PATH", "/home/huangqixu01/models/clip-vit-large-patch14-336")
DATA = os.environ.get("MSB_DATA", "/home/huangqixu01/data/MM-SafetyBench/data")
KEEP_RATIO = 0.25
SEED = 42
PROMPT_T = "USER: <image>\n{q}\nASSISTANT:"


def load_ids(path):
    """兼容两种历史格式：jsonl（回答文件）与 json list（预提取数据文件）。"""
    txt = open(path).read().strip()
    recs = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines()]
    return {(r["cat"], int(r["qid"])) for r in recs}


def stratified_sample(limit, split, ids_from=None):
    """ids_from：历史回答/数据文件，锁定同一批 (cat,qid) 保证跨格子可比。"""
    want = load_ids(ids_from) if ids_from else None
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
        by_cat = {}
        for r in rows:
            by_cat.setdefault(r[0], []).append(r)
        out = []
        for cat, lst in sorted(by_cat.items()):
            k = max(1, round(limit * len(lst) / len(rows)))
            out += rng.sample(lst, min(k, len(lst)))
        return out
    return rows


def read_rows(path):
    """兼容两种历史格式：jsonl 与 json(list)。"""
    txt = open(path).read().strip()
    if txt.startswith("["):
        return [(r["cat"], int(r["qid"])) for r in json.loads(txt)]
    return [(r["cat"], int(r["qid"])) for r in (json.loads(l) for l in txt.splitlines())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", required=True,
                    choices=["none", "trim_official", "trim_highsim", "random"])
    ap.add_argument("--split", default="SD")
    ap.add_argument("--limit", type=int, default=0, help="0=全量 1680")
    ap.add_argument("--ids-from", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="512 与既有格子可比；原文未写，TRIM 官方 model_vqa.py 用 1024")
    args = ap.parse_args()

    tag = f"{args.split.lower()}{'full' if not args.limit else args.limit}"
    out_path = f"/tmp/t1_5_answers_{args.prune}_{tag}.json"
    keep_path = f"/tmp/t1_5_keeps_{args.prune}_{tag}.jsonl"
    done = set()
    try:
        done = {(r["cat"], r["qid"]) for r in (json.loads(l) for l in open(out_path))}
    except Exception:
        pass

    from PIL import Image
    from transformers import (AutoProcessor, AutoTokenizer, CLIPModel,
                              CLIPTextModelWithProjection, LlavaForConditionalGeneration)

    print(f"[{args.prune}] 加载 LLaVA (eager/fp16) ...", flush=True)
    proc = AutoProcessor.from_pretrained(LLAVA)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA, torch_dtype=torch.float16, attn_implementation="eager",
        low_cpu_mem_usage=True).to("cuda:0").eval()
    embed_layer = model.get_input_embeddings()
    proj = model.model.multi_modal_projector
    vlayer = model.config.vision_feature_layer

    need_clip = args.prune in ("trim_official", "trim_highsim")
    clip = text_tower = text_tok = None
    if need_clip:
        print(f"[{args.prune}] 加载 CLIP 打分管线 ...", flush=True)
        clip = CLIPModel.from_pretrained(CLIP, torch_dtype=torch.float32).to("cuda:0").eval()
        text_tower = CLIPTextModelWithProjection.from_pretrained(
            CLIP, torch_dtype=torch.float32).to("cuda:0").eval()
        text_tok = AutoTokenizer.from_pretrained(CLIP)

    rows = stratified_sample(args.limit, args.split, args.ids_from)
    rows = [r for r in rows if (r[0], r[1]) not in done]
    print(f"[{args.prune}] 待跑 {len(rows)} 条（已跳过 {len(done)}）-> {out_path}", flush=True)

    img_id = 32000
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    fout = open(out_path, "a", encoding="utf-8")
    fkeep = open(keep_path, "a", encoding="utf-8")
    for i, (cat, qid, q, img_bytes) in enumerate(rows):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        prompt = PROMPT_T.format(q=q)
        inputs = proc(text=prompt, images=img, return_tensors="pt").to("cuda:0")
        px = inputs["pixel_values"]

        with torch.no_grad():
            vout = model.model.vision_tower(px, output_hidden_states=True)
            feats = vout.hidden_states[vlayer][:, 1:]              # [1,576,1024] fp16
            f32 = feats.to(torch.float32)

            if args.prune == "none":
                feats_sel, keep_idx = feats[0], list(range(feats.shape[1]))
            else:
                n_keep = int(feats.shape[1] * KEEP_RATIO)           # 144
                if args.prune == "random":
                    keep_idx = sorted(rng.choice(feats.shape[1], n_keep,
                                                 replace=False).tolist())
                else:
                    # 与官方 clip_encoder.py 同口径：post_layernorm → visual_projection
                    # → 与 CLIP 文本嵌入点积；文本 = 去掉 <image> 的裸问题
                    t_in = text_tok(text=[prompt.replace("<image>", " ")],
                                    return_tensors="pt", truncation=True,
                                    padding=True).to("cuda:0")
                    t_feat = text_tower(**t_in).text_embeds         # [1,768] fp32
                    proj_img = clip.visual_projection(
                        clip.vision_model.post_layernorm(f32))      # [1,576,768]
                    sims = torch.matmul(proj_img, t_feat.unsqueeze(2)).squeeze(2)
                    if args.prune == "trim_official":
                        score = F.softmax(-sims, dim=-1)            # ← 官方原样（取负）
                    else:
                        score = F.softmax(sims, dim=-1)             # ← 原文 §3.2（不取负）
                    _, k = torch.topk(score, n_keep, dim=1, largest=True, sorted=False)
                    keep_idx = sorted(k[0].tolist())                # 原序（bool-mask 语义）

                ki = torch.tensor(keep_idx, device=f32.device)
                drop = sorted(set(range(feats.shape[1])) - set(keep_idx))
                agg = f32[0, torch.tensor(drop, device=f32.device)].mean(dim=0, keepdim=True)
                feats_sel = torch.cat([f32[0, ki], agg], dim=0).to(torch.float16)  # [145,1024]

            vis = proj(feats_sel.unsqueeze(0))[0]
            ids = inputs["input_ids"][0].clone()
            pos = (ids == img_id).nonzero(as_tuple=True)[0]
            s, e = int(pos[0]), int(pos[-1]) + 1
            ids = torch.cat([ids[:s],
                             torch.full((feats_sel.shape[0],), img_id, dtype=ids.dtype,
                                        device=ids.device), ids[e:]])
            pos = (ids == img_id).nonzero(as_tuple=True)[0]
            assert len(pos) == feats_sel.shape[0], f"{len(pos)} != {feats_sel.shape[0]}"
            emb = embed_layer(ids.unsqueeze(0))
            emb[0, pos] = vis
            out = model.generate(inputs_embeds=emb,
                                 attention_mask=torch.ones_like(ids).unsqueeze(0),
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=proc.tokenizer.pad_token_id)

        ans = proc.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        fout.write(json.dumps({"cat": cat, "qid": qid, "question": q, "answer": ans,
                               "n_vis_tokens": int(feats_sel.shape[0])},
                              ensure_ascii=False) + "\n")
        fout.flush()
        fkeep.write(json.dumps({"cat": cat, "qid": qid, "prune": args.prune,
                                "keep": keep_idx}) + "\n")
        fkeep.flush()
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[{args.prune}] {i+1}/{len(rows)} · {el/(i+1):.1f}s/条 · "
                  f"ETA {(len(rows)-i-1)*el/(i+1)/60:.0f}min", flush=True)
    fout.close()
    fkeep.close()
    print(f"[{args.prune}] 完成 -> {out_path} / {keep_path}", flush=True)


if __name__ == "__main__":
    main()
