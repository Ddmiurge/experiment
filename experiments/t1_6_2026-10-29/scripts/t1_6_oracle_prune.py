#!/usr/bin/env python3
"""T1.6 · oracle 两格生成（原版栈 · TRIM-FT · TYPO）——不经任何判据，保留集来自真值掩码。

与 t1_5_orig_prune.py 的差别只有一处思想：**剪枝选择不再由相似度决定**，
而是由 oracle_masks.py 从探针归档（冻结口径）构造的逐样本 keep 列表直接给定：

    --prune oracle_payload   ④：载荷行全留（6×24=144，确定性）——"移除背景"的忠实实现（裁决格）
    --prune oracle_blank     ③：只留留白格（144/432 均匀随机）——"移除前景"的净化版（哨兵格）

其余与既有四格逐项相同：聚合 token = 被丢 432 格均值（进 LLM 恒 145、削减 74.8%）、
expand2square(pad)、vicuna_v1、greedy/T=0/seed42/512、同批 399 id、原版栈。

自检三道：
  1) 前 3 条打印 kept-set 行分布 + assert 进 LLM token 数 =145；
  2) 逐样本用**模型实际消费的 px** 复算像素活动 top-6 行，与掩码 payload_rows 比对，
     全程不一致率 >5% 直接抛错（网格错位会让③④静默变成随机格）；
  3) keeps 文件逐条留档（分析脚本按它做保留率核验：④=100%、③=0%，偏差>2pp 查掩码）。

用法：
  # 冒烟（协议 §7.0）：各 3 条
  CUDA_VISIBLE_DEVICES=5 /home/huangqixu01/venvs/trim-orig/bin/python \
    t1_6_oracle_prune.py --prune oracle_payload --limit 3 --out /tmp/t1_6_smoke_p.json
  CUDA_VISIBLE_DEVICES=6 /home/huangqixu01/venvs/trim-orig/bin/python \
    t1_6_oracle_prune.py --prune oracle_blank --limit 3 --out /tmp/t1_6_smoke_b.json
  # 全量（冒烟通过后；两格两卡并跑约 3h/格）
  CUDA_VISIBLE_DEVICES=5 ... --prune oracle_payload
  CUDA_VISIBLE_DEVICES=6 ... --prune oracle_blank
"""
import argparse
import base64
import io
import json
import os
import sys

import numpy as np
import torch

REPO = os.environ.get("REPO", "/home/huangqixu01/experiment")
TRIM_REPO = os.path.join(REPO, "third_party/trim")
MODEL_PATH = os.environ.get("TRIM_M", "/home/huangqixu01/models/llava-v1.5-7b-TRIM")
CLIP_PATH = os.environ.get("CLIP_PATH", "/home/huangqixu01/models/clip-vit-large-patch14-336")
KEEP = os.environ.get("TRIM_KEEP", "0.25")
NROW = NCOL = 24
BUDGET = 144

CURRENT_KEEP = [set()]          # 主循环逐样本注入；token_reduction_oracle 读取
MISMATCH = [0]
TOTAL = [0]


def token_reduction_oracle(self, image_features, all_image_features, text_features=None):
    """结构与官方 token_reduction 同形（144 原序 + 1 聚合 + actual_dims=145），
    唯一差别：选择来自 CURRENT_KEEP（真值掩码），完全不看 text_features。"""
    if not self.token_reduce_func:
        return image_features, [image_features.shape[1]] * image_features.shape[0]
    batch_size, tokens_number, dimension = image_features.shape
    k = float(self.token_reduce_func.replace("TRIM:", ""))
    num_tokens_to_keep = int(tokens_number * k)
    assert num_tokens_to_keep == BUDGET == len(CURRENT_KEEP[0]), \
        f"预算不符: {num_tokens_to_keep} vs {BUDGET} vs keep {len(CURRENT_KEEP[0])}"

    token_mask = torch.zeros(batch_size, tokens_number, device=image_features.device,
                             dtype=torch.bool)
    idx = torch.tensor(sorted(CURRENT_KEEP[0]), device=image_features.device)
    token_mask[0, idx] = True

    if image_features.dtype == torch.float16:
        image_features = image_features.to(torch.float32)
    selected = torch.zeros_like(image_features)
    selected[:, :num_tokens_to_keep] = image_features[token_mask].view(
        batch_size, num_tokens_to_keep, -1)
    remaining = torch.logical_not(token_mask)
    remaining_mean = ((image_features * remaining.unsqueeze(-1)).sum(dim=1)
                      / remaining.sum(dim=1, keepdim=True))
    selected[:, num_tokens_to_keep] = remaining_mean
    actual_dims = [num_tokens_to_keep + 1] * batch_size
    out = selected.to(torch.float16)
    return out, actual_dims


def apply_patch():
    """普通函数直接赋类属性（勿用 MethodType——T1.5b 踩过）。磁盘文件不动。"""
    sys.path.insert(0, TRIM_REPO)
    from llava.model.multimodal_encoder import clip_encoder
    clip_encoder.CLIPVisionTower.token_reduction_orig = \
        clip_encoder.CLIPVisionTower.token_reduction
    clip_encoder.CLIPVisionTower.token_reduction = token_reduction_oracle
    print("[patch] token_reduction → oracle（保留集=真值掩码，无判据）", flush=True)


def activity_rows(px):
    """与探针同法：逐行像素活动度 std，取最高 6 行。px=[1,3,336,336]（模型实际消费的张量）"""
    act = [float(px[0, :, r * 14:(r + 1) * 14, :].std()) for r in range(NROW)]
    return sorted(sorted(range(NROW), key=lambda r: -act[r])[:6])


def load_masks(path):
    d = {}
    for l in open(path):
        r = json.loads(l)
        d[(r["cat"], int(r["qid"]))] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", required=True, choices=["oracle_payload", "oracle_blank"])
    ap.add_argument("--data", default="/tmp/t1_4c_typo399_data.json")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conv", default="vicuna_v1")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    which = "payload" if args.prune == "oracle_payload" else "blank"
    masks_path = args.masks or f"/tmp/t1_6_masks_{which}.jsonl"
    masks = load_masks(masks_path)
    out = args.out or f"/tmp/t1_6_answers_{args.prune}_typo399.json"
    keep_out = out.replace("_answers_", "_keeps_").replace(".json", ".jsonl")
    apply_patch()

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX
    from transformers import AutoConfig, CLIPImageProcessor
    from PIL import Image

    rows = [(r["cat"], int(r["qid"]), r["question"], base64.b64decode(r["image_b64"]))
            for r in json.load(open(args.data))]
    missing = [k for r in rows for k in [(r[0], r[1])] if k not in masks]
    assert not missing, f"掩码缺 id（先跑 oracle_masks.py）：{missing[:3]}…共 {len(missing)}"
    if args.limit:
        rows = rows[:args.limit]
    print(f"样本 {len(rows)} 条 · 掩码 {len(masks)} 条（{masks_path}）", flush=True)

    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    cfg.mm_vision_tower = CLIP_PATH
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, "llava_v1", device="cuda:0", use_flash_attn=False, config=cfg)
    if image_processor is None:
        image_processor = CLIPImageProcessor.from_pretrained(CLIP_PATH)
    vt = model.get_model().vision_tower
    vt.token_reduce_func = f"TRIM:{KEEP}"
    print(f"token_reduce_func={vt.token_reduce_func} "
          f"image_aspect_ratio={getattr(model.config, 'image_aspect_ratio', '(缺)')} "
          f"conv={args.conv}", flush=True)

    fkeep = open(keep_out, "w", encoding="utf-8")
    fout = open(out, "w", encoding="utf-8")
    for i, (cat, qid, q, img_bytes) in enumerate(rows):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = process_images([img], image_processor, model.config)
        mrec = masks[(cat, qid)]
        CURRENT_KEEP[0] = set(mrec["keep"])

        # 自检 2：模型 px 复算载荷行 vs 掩码口径
        mine = activity_rows(img_tensor)
        TOTAL[0] += 1
        same = mine == sorted(mrec["payload_rows"])
        MISMATCH[0] += 0 if same else 1
        if i < 3 or (i + 1) % 50 == 0:
            dist = {}
            for kk in CURRENT_KEEP[0]:
                dist[kk // NCOL] = dist.get(kk // NCOL, 0) + 1
            print(f"[{i}] {cat}#{qid} keep行={sorted(dist)} mask_rows={mrec['payload_rows']} "
                  f"px复算={'一致' if same else '≠' + str(mine)} keep={len(CURRENT_KEEP[0])} "
                  f"累计mismatch={MISMATCH[0]}/{TOTAL[0]}", flush=True)
        if not same and i >= 20 and MISMATCH[0] / TOTAL[0] > 0.05:
            raise SystemExit(f"❌ 掩码与模型网格错位（不一致率 {MISMATCH[0]/TOTAL[0]:.1%} >5%）——"
                             "探针 px 路径与原版栈 px 路径不一致，停下来报回，不要继续生成")

        conv = conv_templates[args.conv].copy()
        conv.append_message(conv.roles[0], f"<image>\n{q}")
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to("cuda:0")
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=img_tensor.to("cuda:0", dtype=torch.float16),
                image_sizes=[img.size], do_sample=False,
                max_new_tokens=args.max_new_tokens, texts=[q])
            # 自检 1：进 LLM 的视觉 token 数
            _, dims = vt(img_tensor.to("cuda:0", dtype=torch.float16), [q])
        assert dims[0] == 145, f"进 LLM 视觉 token {dims[0]} != 145"
        ans = tokenizer.decode(output_ids[0], skip_special_tokens=True).split(
            conv.roles[1] + ":")[-1].strip()
        fout.write(json.dumps({"cat": cat, "qid": qid, "question": q, "answer": ans,
                               "prune": args.prune}, ensure_ascii=False) + "\n")
        fout.flush()
        fkeep.write(json.dumps({"cat": cat, "qid": qid, "prune": args.prune,
                                "keep": sorted(CURRENT_KEEP[0]),
                                "payload_rows": mrec["payload_rows"],
                                "px_rows_match": same}, ensure_ascii=False) + "\n")
        fkeep.flush()
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(rows)}] mismatch累计 {MISMATCH[0]}", flush=True)
    fout.close(); fkeep.close()
    print(f"完成 -> {out} / {keep_out} · 网格不一致 {MISMATCH[0]}/{TOTAL[0]}"
          f"（{MISMATCH[0]/max(1,TOTAL[0]):.2%}）", flush=True)


if __name__ == "__main__":
    main()
