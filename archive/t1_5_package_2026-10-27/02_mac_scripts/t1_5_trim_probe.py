#!/usr/bin/env python3
"""T1.5 · 剪枝方向探针（不生成，只算保留集）。约 10–20 分钟，单卡，不需要 LLaVA。

要证的事（诊断 H1）：TRIM 官方代码 `clip_encoder.py` L86–L111 做
    topk(softmax(-sims), largest=True) → 保留的是与文本相似度**最低**的 token，
与 TRIM 原文 §3.2「only the most significant tokens, as determined by their
**high** similarity scores, are retained」相反。

输出四件事：
  1. 每样本 sim(保留集) 与 sim(丢弃集) 的均值差 —— 预测为**负**（官方）/ 正（highsim）
  2. Jaccard(官方保留集, highsim 保留集) —— 两套选择到底分不分得开
  3. Jaccard(官方保留集, 随机子集) —— 「官方 TRIM 是不是约等于随机剪枝」
  4. 换文本后保留集的变化 —— 文本信号有多强
另外对 TYPO 分片报告关键词条带所在行的 token 被保留的比例。

用法：
  CUDA_VISIBLE_DEVICES=6 python t1_5_trim_probe.py --split SD  --ids-from /tmp/t1_4_answers_orig_sd_noprun399.json
  CUDA_VISIBLE_DEVICES=6 python t1_5_trim_probe.py --split TYPO --ids-from /tmp/t1_4c_typo399_data.json
"""
import argparse
import glob
import io
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = os.environ.get("REPO", "/home/huangqixu01/experiment")
CLIP = os.environ.get("CLIP_PATH", "/home/huangqixu01/models/clip-vit-large-patch14-336")
DATA = os.environ.get("MSB_DATA", "/home/huangqixu01/data/MM-SafetyBench/data")
SELECT_LAYER = -2          # 官方 args.mm_vision_select_layer
KEEP = 0.25                # TRIM:0.25 → 保留 144/576
GENERIC_TEXT = "List steps in detail to perform this activity."


def load_rows(split, ids_from):
    """从 HF parquet 取样本；ids_from 锁定与历史格子同一批 (cat,qid)。"""
    want = None
    if ids_from:
        want = set()
        for l in open(ids_from):
            r = json.loads(l)
            if "cat" in r and "qid" in r:
                want.add((r["cat"], int(r["qid"])))
        want = want or None
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


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="SD")
    ap.add_argument("--ids-from", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    ap.add_argument("--text-shuffle-n", type=int, default=30, help="文本敏感度只算前 N 条（慢）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or f"/tmp/t1_5_probe_{args.split.lower()}.jsonl"

    from PIL import Image
    from transformers import (AutoImageProcessor, AutoTokenizer, CLIPModel,
                              CLIPTextModelWithProjection)

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[probe] device={dev}  clip={CLIP}", flush=True)
    clip = CLIPModel.from_pretrained(CLIP, torch_dtype=torch.float32).to(dev).eval()
    text_tower = CLIPTextModelWithProjection.from_pretrained(CLIP, torch_dtype=torch.float32).to(dev).eval()
    text_tok = AutoTokenizer.from_pretrained(CLIP)
    proc = AutoImageProcessor.from_pretrained(CLIP)

    rows = load_rows(args.split, args.ids_from)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[probe] 样本 {len(rows)} 条", flush=True)

    N = clip.vision_model.config.image_size // clip.vision_model.config.patch_size  # 24
    rng = np.random.default_rng(42)

    fout = open(out_path, "w", encoding="utf-8")
    acc = {k: [] for k in ("sim_diff", "jac_off_high", "jac_off_rand", "jac_text_swap",
                           "jac_text_empty", "jac_text_generic", "keep_frac_bottom_strip",
                           "is_low_sim")}
    for i, (cat, qid, q, img_bytes) in enumerate(rows):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        px = proc(images=img, return_tensors="pt")["pixel_values"].to(dev)

        with torch.no_grad():
            # ── 与官方 clip_encoder.forward + token_reduction 完全同口径的打分 ──
            vout = clip.vision_model(px, output_hidden_states=True)
            feats = vout.hidden_states[SELECT_LAYER][:, 1:].float()          # [1,576,1024]
            proj = clip.visual_projection(clip.vision_model.post_layernorm(feats))  # [1,576,768]
            t_in = text_tok(text=[q], return_tensors="pt", truncation=True, padding=True).to(dev)
            sims = torch.matmul(proj, text_tower(**t_in).text_embeds.unsqueeze(2)).squeeze(2)  # [1,576]
            sims = sims[0]

            n_keep = int(sims.shape[0] * KEEP)
            # 官方：softmax(-s) 再 topk largest  ⇔ 按 s 取最小 n_keep 个
            keep_off = set(torch.topk(-sims, n_keep).indices.tolist())
            keep_high = set(torch.topk(sims, n_keep).indices.tolist())
            keep_rand = set(rng.choice(sims.shape[0], n_keep, replace=False).tolist())
            all_idx = set(range(sims.shape[0]))

            s_keep = sims[list(keep_off)].mean().item()
            s_drop = sims[sorted(all_idx - keep_off)].mean().item()

            rec = {"cat": cat, "qid": qid, "n": int(sims.shape[0]), "n_keep": n_keep,
                   "sim_keep_official": s_keep, "sim_dropped_official": s_drop,
                   "sim_diff": s_keep - s_drop,
                   "sim_spread": (sims.max() - sims.min()).item(),
                   "jac_official_vs_highsim": jaccard(keep_off, keep_high),
                   "jac_official_vs_random": jaccard(keep_off, keep_rand)}

            # ── 文本敏感度：同一张图换文本，保留集变不变 ──
            if i < args.text_shuffle_n:
                for tag, txt in (("empty", ""), ("generic", GENERIC_TEXT), ("swap", "")):
                    if tag == "swap":
                        if len(rows) < 2:
                            continue
                        txt = rows[(i + 1) % len(rows)][2]        # 别人的问题
                    ti = text_tok(text=[txt], return_tensors="pt", truncation=True,
                                  padding=True).to(dev)
                    s2 = torch.matmul(proj, text_tower(**ti).text_embeds.unsqueeze(2)).squeeze(2)[0]
                    k2 = set(torch.topk(-s2, n_keep).indices.tolist())
                    rec[f"jac_official_vs_text_{tag}"] = jaccard(keep_off, k2)
                    acc[f"jac_text_{tag}"].append(rec[f"jac_official_vs_text_{tag}"])

            # ── TYPO：底部关键词条带（网格下方 1/4 行）被保留的比例 ──
            if args.split.upper() in ("TYPO", "ORIGINAL", "BLACK"):
                strip = {r * N + c for r in range(int(N * 0.75), N) for c in range(N)}
                rec["frac_strip_kept"] = len(strip & keep_off) / len(strip)
                acc["keep_frac_bottom_strip"].append(rec["frac_strip_kept"])

            acc["sim_diff"].append(rec["sim_diff"])
            acc["jac_off_high"].append(rec["jac_official_vs_highsim"])
            acc["jac_off_rand"].append(rec["jac_official_vs_random"])
            acc["is_low_sim"].append(1.0 if rec["sim_diff"] < 0 else 0.0)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                print(f"[probe] {i+1}/{len(rows)}", flush=True)
    fout.close()

    def m(k):
        v = acc[k]
        return f"{np.mean(v):.4f}" if v else "n/a"

    print("\n═══ T1.5 剪枝方向探针 ═══")
    print(f"split={args.split}  n={len(rows)}  保留={int(576*KEEP)}")
    print(f"1) 官方保留集 vs 丢弃集 相似度均值差 = {m('sim_diff')}"
          f"   ← 预测为负（官方代码在删高相似度 token）")
    print(f"   有多少比例的样本 sim_diff < 0 : {np.mean(acc['is_low_sim']):.3f}"
          f"   ← 该值≈1.0 就证实了方向反了")
    print(f"2) Jaccard(官方保留集, 高相似度保留集) = {m('jac_off_high')}"
          f"   ← 小=两套选择确实不同；≈0.25=等价于随机（文本信号无效）")
    print(f"3) Jaccard(官方保留集, 随机子集)       = {m('jac_off_rand')}"
          f"   ← 期望 0.25；若明显<0.25 说明官方选择是反着内容的，不是随机的")
    print(f"4) 文本敏感度 Jaccard(换文本后保留集不变): "
          f"empty={m('jac_text_empty')} generic={m('jac_text_generic')} swap={m('jac_text_swap')}"
          f"   ← ≈1.0 表示保留集几乎不随问题变")
    if acc["keep_frac_bottom_strip"]:
        print(f"5) TYPO 底部关键词条带被保留的 token 占比 = {m('keep_frac_bottom_strip')}"
              f"   ← <0.25 表示条带被系统性删掉，直接解释 TYPO 上 ASR 显著下降")
    print(f"\n逐样本明细 -> {out_path}")


if __name__ == "__main__":
    main()
