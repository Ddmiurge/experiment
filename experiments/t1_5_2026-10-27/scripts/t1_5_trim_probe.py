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
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msb_queries import SRC_CHOICES, QueryResolver, load_official

REPO = os.environ.get("REPO", "/home/huangqixu01/experiment")
CLIP = os.environ.get("CLIP_PATH", "/home/huangqixu01/models/clip-vit-large-patch14-336")
DATA = os.environ.get("MSB_DATA", "/home/huangqixu01/data/MM-SafetyBench/data")
SELECT_LAYER = -2          # 官方 args.mm_vision_select_layer
KEEP = 0.25                # TRIM:0.25 → 保留 144/576
GENERIC_TEXT = "List steps in detail to perform this activity."


def load_rows(split, ids_from):
    """从 HF parquet 取样本；ids_from 锁定与历史格子同一批 (cat,qid)。

    兼容两种归档格式：jsonl（回答文件）与 JSON 数组（预提取数据文件）。
    T1.5 首轮曾因只支持 jsonl 而静默退回全量 1680 —— 现在解析失败会直接抛错。
    """
    want = None
    if ids_from:
        txt = open(ids_from).read().strip()
        recs = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines()]
        want = {(r["cat"], int(r["qid"])) for r in recs}
        if not want:
            raise SystemExit(f"❌ ids_from 解析出 0 个 id：{ids_from}（拒绝静默退回全量）")
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
    ap.add_argument("--cell-frac", type=float, default=0.15,
                    help="按像素活动取最高的这个比例作为「载荷格」；0=关掉 cell 级读数")
    ap.add_argument("--query-src", default="template", choices=list(SRC_CHOICES),
                    help="TRIM 打分用的查询：template=模板问句（与历史格子可比）/ "
                         "keyphrase=官方 Key Phrase / changed=含关键词的原始问句 / dummy")
    ap.add_argument("--msb-json", default=os.path.join(REPO, "third_party/mm-safetybench/"
                                                      "data/processed_questions"))
    args = ap.parse_args()
    qsfx = "" if args.query_src == "template" else f"_{args.query_src}"
    out_path = args.out or f"/tmp/t1_5_probe_{args.split.lower()}{qsfx}.jsonl"

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

    official = load_official(args.msb_json) if args.query_src not in ("template", "dummy") else None
    resolver = QueryResolver(official, args.query_src)
    if args.query_src not in ("template", "dummy") and not official:
        raise SystemExit(f"❌ 官方 JSON 目录为空或不存在：{args.msb_json}")
    print(f"[probe] 样本 {len(rows)} 条", flush=True)

    N = clip.vision_model.config.image_size // clip.vision_model.config.patch_size  # 24
    rng = np.random.default_rng(42)

    fout = open(out_path, "w", encoding="utf-8")
    acc = {k: [] for k in ("sim_diff", "jac_off_high", "jac_off_rand", "jac_text_swap",
                           "jac_text_empty", "jac_text_generic", "keep_frac_payload",
                           "keep_frac_other", "payload_highsim", "payload_random2",
                           "sim_payload_cells", "sim_blank_cells",
                           "kept_payload_cells_official", "kept_blank_cells_official",
                           "kept_payload_cells_highsim", "kept_payload_cells_random",
                           "is_low_sim")}
    for i, (cat, qid, q, img_bytes) in enumerate(rows):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        px = proc(images=img, return_tensors="pt")["pixel_values"].to(dev)

        with torch.no_grad():
            # ── 与官方 clip_encoder.forward + token_reduction 完全同口径的打分 ──
            vout = clip.vision_model(px, output_hidden_states=True)
            feats = vout.hidden_states[SELECT_LAYER][:, 1:].float()          # [1,576,1024]
            proj = clip.visual_projection(clip.vision_model.post_layernorm(feats))  # [1,576,768]
            query = resolver.resolve(cat, qid, q)      # ← 唯一被 --query-src 控制的地方
            t_in = text_tok(text=[query], return_tensors="pt", truncation=True,
                            padding=True).to(dev)
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

            # ── TYPO/ORIGINAL/BLACK：按「像素行」定位载荷，而不是假设它在底部 ──
            #   重要：原版栈 image_aspect_ratio='pad' → expand2square 会把 1024×180 的
            #   文字条**垂直居中**贴进 1024² 画布，落到网格中部而不是底部。
            #   早期用「下方 1/4 行」当条带区是错的，读数 0.166 不能作为证据。
            #   这里改成三行 24 个网格行的剖面：像素活动(高=有字) / 相似度 / 官方保留率，
            #   直接看「有字的那几行被保留多少」，不需要先验假设位置。
            if args.split.upper() in ("TYPO", "ORIGINAL", "BLACK"):
                prof = {"pix": [], "sim": [], "keep": []}
                for r in range(N):
                    blk = px[0, :, r * 14:(r + 1) * 14, :]           # [3,14,336]
                    prof["pix"].append(float(blk.std().item()))
                    rr = slice(r * N, (r + 1) * N)
                    seg = sims[rr]
                    prof["sim"].append(float(seg.mean().item()))
                    prof["keep"].append(
                        len({i - r * N for i in keep_off} & set(range(N))) / N)
                rec["row_profile"] = prof
                # 载荷行 = 像素活动最高的 6 行
                top = sorted(range(N), key=lambda r: -prof["pix"][r])[:6]
                # 同一套打分下再算 highsim / random 的载荷行保留率，保证三者同源可比
                k_hi = set(torch.topk(sims, n_keep).indices.tolist())
                k_rnd = set(int(x) for x in np.random.default_rng(1000 + i).choice(
                    sims.shape[0], n_keep, replace=False))
                for nm, ss in (("highsim", k_hi), ("random2", k_rnd)):
                    rec[f"frac_payload_kept_{nm}"] = sum(
                        1 for idx in top for c in range(N) if idx * N + c in ss) / (6 * N)
                    rec[f"frac_blank_kept_{nm}"] = sum(
                        1 for r0 in range(N) if r0 not in top for c in range(N)
                        if r0 * N + c in ss) / ((N - 6) * N)
                    acc[f"payload_{nm}"].append(rec[f"frac_payload_kept_{nm}"])
                rec["frac_payload_rows_kept"] = sum(prof["keep"][r] for r in top) / 6
                rec["frac_blank_rows_kept"] = sum(
                    prof["keep"][r] for r in range(N) if r not in top) / (N - 6)
                rec["frac_other_rows_kept"] = (
                    sum(prof["keep"][r] for r in range(N) if r not in top) / (N - 6))
                rec["payload_rows"] = top
                acc["keep_frac_payload"].append(rec["frac_payload_rows_kept"])
                acc["keep_frac_other"].append(rec["frac_other_rows_kept"])

                # ── cell 级（24×24 patch 逐个）载荷定位：行级太粗，
                #    黑字只占行宽的一部分，行保留率会把「留白列」也算成保留 ──
                if args.cell_frac > 0:
                    cells = (px[0].permute(1, 2, 0).reshape(N, 14, N, 14, 3)
                             .permute(0, 2, 1, 3, 4).reshape(N, N, -1))
                    act = cells.std(dim=2).flatten().float()          # [576] 每格像素活动
                    # 载荷格 = 明显比「空白画布基线」更活跃的格。
                    # 不能用 quantile(act, 1-frac)：当载荷占比 < frac 时阈值会落进空白区，
                    # 于是 pmask 会把 576 格全选中（Mac 侧合成图自检抓到过这个 bug）。
                    med = act.median()
                    mad = (act - med).abs().median()
                    thr = med + 6.0 * mad
                    pmask = act > thr
                    n_cell = int(pmask.sum())
                    if n_cell < 8 or n_cell > 200:      # 阈值退化 → 回退到固定 top-k
                        kk = min(60, int(0.12 * act.numel()))
                        pmask = torch.zeros_like(act, dtype=torch.bool)
                        pmask[act.topk(kk).indices] = True
                        rec["cell_thr_fallback"] = True
                    pidx = set(int(x) for x in pmask.nonzero(as_tuple=False).flatten())
                    bidx = set(range(sims.shape[0])) - pidx
                    rec["n_payload_cells"] = len(pidx)
                    rec["sim_payload_cells"] = float(sims[sorted(pidx)].mean())
                    rec["sim_blank_cells"] = float(sims[sorted(bidx)].mean())
                    rec["kept_payload_cells_official"] = len(keep_off & pidx) / len(pidx)
                    rec["kept_blank_cells_official"] = len(keep_off & bidx) / len(bidx)
                    rec["kept_payload_cells_highsim"] = len(k_hi & pidx) / len(pidx)
                    rec["kept_payload_cells_random"] = len(k_rnd & pidx) / len(pidx)
                    for k, v in (("sim_payload_cells", rec["sim_payload_cells"]),
                                 ("sim_blank_cells", rec["sim_blank_cells"]),
                                 ("kept_payload_cells_official", rec["kept_payload_cells_official"]),
                                 ("kept_blank_cells_official", rec["kept_blank_cells_official"]),
                                 ("kept_payload_cells_highsim", rec["kept_payload_cells_highsim"]),
                                 ("kept_payload_cells_random", rec["kept_payload_cells_random"])):
                        acc[k].append(v)

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
    if acc["keep_frac_payload"]:
        print(f"5) 按像素活动定位的「载荷行」（有字的 6 行）保留率："
              f"official={m('keep_frac_payload')} highsim={m('payload_highsim')} "
              f"random={m('payload_random2')}  vs 留白行 official={m('keep_frac_other')}"
              f"   ← 保留率基线 = 0.25；载荷行明显<0.25 即关键词被系统性删掉")
        pr = [r for r in (json.loads(l) for l in open(out_path)) if "row_profile" in r]
        if pr:
            import numpy as _np
            k = pr[0]
            for name in ("pix", "sim", "keep"):
                v = _np.mean([q["row_profile"][name] for q in pr], axis=0)
                print(f"   行剖面 {name:<4}", " ".join(f"{x:7.3f}" for x in v))
    if acc["sim_payload_cells"]:
        ncells = [r["n_payload_cells"] for r in map(json.loads, open(out_path))
                  if "n_payload_cells" in r]
        print(f"\n6) cell 级载荷定位：每样本被判为载荷的格数 ≈ {np.mean(ncells):.0f}"
              f"（{np.mean(ncells)/576:.1%} 的图）" if ncells else "\n6) cell 级读数不可用")
        print(f"   相似度：载荷格 {m('sim_payload_cells')}  vs 留白格 {m('sim_blank_cells')}"
              f"   ← 差值为负 = 载荷格更不像查询（模板查询下 TRIM 反向的证据）")
        print(f"   保留率：official {m('kept_payload_cells_official')} / "
              f"highsim {m('kept_payload_cells_highsim')} / random {m('kept_payload_cells_random')}"
              f"  vs 留白格 official {m('kept_blank_cells_official')}"
              f"   ← 基线 = keep 比例 0.25")
    resolver.report()
    print(f"\n逐样本明细 -> {out_path}")


if __name__ == "__main__":
    main()
