#!/usr/bin/env python3
"""T1.5b · 原版栈（TRIM-FT 权重）统一四模式生成：none / official / highsim / random。

为什么新开一个脚本而不是复用 `t1_5_generate.py`（llava-hf 栈）：
TYPO 图是 1024×180 的横条，llava-hf 的 processor 会 **resize+中心裁剪**（只保留 17.6% 宽度），
而原版栈的 `image_aspect_ratio='pad'` 走 `expand2square`（垂直居中补白，条带完整保留）。
**TYPO 的四格必须同一套预处理**，否则「剪什么」的对比里混进「图被裁成什么样」。
T1.4d 的 TYPO 读数也正是这套预处理下得到的，可直接接续。

四个模式只差「保留哪 144 个」，其余（144 原序 + 1 个被丢集合的均值 = 145 进 LLM）完全一致：

    none      关掉剪枝（官方 token_reduction 在 token_reduce_func=None 时原样返回 576）
    official  不 patch，照官方代码跑：保留与文本相似度**最低**的 144（= T1.4 的 trim）
    highsim   patch 掉 L89 的取负：保留相似度**最高**的 144（= TRIM 原文 §3.2 规范）
    random    随机 144（阴性对照，对齐原文 Random 行）

不改 `third_party/` 磁盘文件：patch 只活在进程里（规则 2）。
venv: /home/huangqixu01/venvs/trim-orig （torch 2.1.2 + transformers 4.37.2）

用法：
  CUDA_VISIBLE_DEVICES=5 python t1_5_orig_prune.py --split TYPO --prune highsim
  CUDA_VISIBLE_DEVICES=6 python t1_5_orig_prune.py --split TYPO --prune random
  # 需要重跑基线时：--prune none / --prune official
"""
import argparse
import base64
import io
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msb_queries import SRC_CHOICES, QueryResolver, load_official

REPO = os.environ.get("REPO", "/home/huangqixu01/experiment")
TRIM_REPO = os.path.join(REPO, "third_party/trim")
MODEL_PATH = os.environ.get("TRIM_M", "/home/huangqixu01/models/llava-v1.5-7b-TRIM")
CLIP_PATH = os.environ.get("CLIP_PATH", "/home/huangqixu01/models/clip-vit-large-patch14-336")
KEEP = os.environ.get("TRIM_KEEP", "0.25")
SEED = 42
MODE = "highsim"          # 由 --prune 设置；patch 进去的函数读它
KEEP_LOG = []             # highsim/random 模式下由 token_reduction_mode 填充
_RNG = None               # random 模式的逐样本掩码；main 里初始化一次（不能每次调用重建）


# ── 与官方 clip_encoder.py L67–L142 同结构，只把「选哪 144 个」换成按 MODE 决定 ──
#    保留：post_layernorm → visual_projection → 与 CLIP 文本嵌入点积 → softmax → topk
#    差别：official 取负（原样）、highsim 不取负（原文规范）、random 用固定种子的随机掩码
def token_reduction_mode(self, image_features, all_image_features, text_features=None):
    import torch.nn.functional as F
    if not self.token_reduce_func:
        return image_features, [image_features.shape[1]] * image_features.shape[0]
    assert "TRIM" in self.token_reduce_func, self.token_reduce_func

    batch_size, tokens_number, dimension = image_features.shape
    k = float(self.token_reduce_func.replace("TRIM:", ""))
    num_tokens_to_keep = int(tokens_number * k)

    token_mask = torch.zeros(batch_size, tokens_number, device=image_features.device,
                             dtype=torch.bool)
    if MODE == "random":
        global _RNG
        if _RNG is None:                       # ← 全进程只建一次，逐样本推进 ⇒ 掩码互不相同
            _RNG = np.random.default_rng(SEED)
        rng = _RNG
        for b in range(batch_size):
            idx = rng.choice(tokens_number, num_tokens_to_keep, replace=False)
            token_mask[b, idx] = True
    else:
        with torch.cuda.amp.autocast(enabled=False):
            if image_features.dtype == torch.float16:
                image_features = image_features.to(torch.float32)
            elif image_features.dtype == torch.float32:
                image_features = image_features.to(
                    self.vision_tower.vision_model.post_layernorm.weight.dtype)
            nomalized = self.vision_tower.vision_model.post_layernorm(image_features)
            proj = self.vision_tower.visual_projection(nomalized)
        similarities = torch.matmul(proj, text_features.unsqueeze(2)).squeeze(2)
        if MODE == "official":                 # 官方 L89 那一行取负
            similarities = -similarities
        similarities = F.softmax(similarities, dim=-1)
        _, topk_indices = torch.topk(similarities, num_tokens_to_keep,
                                      dim=1, largest=True, sorted=False)
        batch_indices = torch.arange(batch_size, device=image_features.device).unsqueeze(1)
        token_mask[batch_indices, topk_indices] = True

    selected_image_features = torch.zeros_like(image_features)
    # bool-mask 索引保持原始行序 ⇒ 保留 token 按视觉位置升序排列（与官方一致）
    selected_image_features[:, :num_tokens_to_keep] = image_features[token_mask].view(
        batch_size, num_tokens_to_keep, -1)

    remaining_mask = torch.logical_not(token_mask)
    if remaining_mask.sum(dim=1).min().item() > 0:
        remaining_mean = ((image_features * remaining_mask.unsqueeze(-1)).sum(dim=1)
                          / remaining_mask.sum(dim=1, keepdim=True))
    else:
        remaining_mean = torch.zeros(batch_size, dimension, device=image_features.device)
    selected_image_features[:, num_tokens_to_keep] = remaining_mean

    if MODE in ("highsim", "random"):        # 留档保留集，供「载荷行命中」等机制读数
        KEEP_LOG.append(sorted(int(x) for x in
                               token_mask[0].nonzero(as_tuple=False).flatten().tolist()))

    actual_dims = [num_tokens_to_keep + 1] * batch_size
    image_features = selected_image_features
    if image_features.dtype == torch.float32:
        image_features = image_features.to(torch.float16)
    return image_features, actual_dims


def apply_patch(mode):
    """把 mode 装进进程内的类方法；third_party 磁盘文件一个字节不动。

    注意：普通函数直接赋给类属性即可成为方法。
    **不要用 types.MethodType(f, cls)** —— 那会把「类」当 self 绑进去（T1.5 首轮踩过）。
    """
    sys.path.insert(0, TRIM_REPO)
    from llava.model.multimodal_encoder import clip_encoder
    if not hasattr(clip_encoder.CLIPVisionTower, "token_reduction_orig"):
        clip_encoder.CLIPVisionTower.token_reduction_orig = \
            clip_encoder.CLIPVisionTower.token_reduction
    if mode in ("highsim", "random"):
        clip_encoder.CLIPVisionTower.token_reduction = token_reduction_mode
    elif mode == "official":
        clip_encoder.CLIPVisionTower.token_reduction = \
            clip_encoder.CLIPVisionTower.token_reduction_orig
    print(f"[patch] MODE={mode}", flush=True)


def load_ids(path):
    txt = open(path).read().strip()
    recs = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines()]
    return {(r["cat"], int(r["qid"])) for r in recs}


def main():
    global MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="TYPO")
    ap.add_argument("--prune", required=True, choices=["none", "official", "highsim", "random"])
    ap.add_argument("--data", default="/tmp/t1_4c_typo399_data.json",
                    help="预提取样本 json（含 base64 图），与历史格子同 id 同序")
    ap.add_argument("--ids-from", default=None, help="若要换批样本，用历史回答文件锁 id")
    ap.add_argument("--conv", default="vicuna_v1")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--query-src", default="template", choices=list(SRC_CHOICES),
                    help="只控制喂给 TRIM 打分的 texts=；模型的 prompt 始终是模板问句，"
                         "否则就同时改了下攻击本体")
    ap.add_argument("--msb-json", default=os.path.join(REPO, "third_party/mm-safetybench/"
                                                      "data/processed_questions"))
    args = ap.parse_args()
    MODE = args.prune

    out = args.out or f"/tmp/t1_5b_answers_{args.prune}_{args.split.lower()}399.json"
    keep_out = f"/tmp/t1_5b_keeps_{args.prune}_{args.split.lower()}399.jsonl"
    apply_patch(args.prune)

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX
    from transformers import AutoConfig, CLIPImageProcessor
    from PIL import Image

    rows = [(r["cat"], r["qid"], r["question"], base64.b64decode(r["image_b64"]))
            for r in json.load(open(args.data))]
    if args.ids_from:
        want = load_ids(args.ids_from)
        rows = [r for r in rows if (r[0], int(r[1])) in want]
    resolver = None
    if args.query_src not in ("template", "dummy"):
        resolver = QueryResolver(load_official(args.msb_json), args.query_src)
    print(f"样本 {len(rows)} 条（{args.data}，split={args.split}，prune={args.prune}，"
          f"query_src={args.query_src}）", flush=True)

    def trim_text(q, cat=None, qid=None):
        """喂给 TRIM 的查询。默认与模型的 prompt 同一个字符串（与历史格子可比）。"""
        if resolver is None:
            return q if args.query_src == "template" else ("a photo." if args.query_src == "dummy" else q)
        return resolver.resolve(cat, qid, q)

    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    cfg.mm_vision_tower = CLIP_PATH
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, "llava_v1", device="cuda:0", use_flash_attn=False, config=cfg)
    if image_processor is None:
        image_processor = CLIPImageProcessor.from_pretrained(CLIP_PATH)
    vt = model.get_model().vision_tower
    # none：把剪枝整个关掉（官方代码在 token_reduce_func 为假值时原样返回 576）
    vt.token_reduce_func = None if args.prune == "none" else f"TRIM:{KEEP}"
    print(f"token_reduce_func={vt.token_reduce_func}  "
          f"image_aspect_ratio={getattr(model.config, 'image_aspect_ratio', '(缺)')}  "
          f"conv={args.conv}", flush=True)

    # ── 自检：前 3 条打印实际进 LLM 的视觉 token 数与保留集统计 ──
    with torch.inference_mode():
        for cat, qid, q, img_bytes in rows[:3]:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            px = process_images([img], image_processor, model.config).to("cuda:0",
                                                                          dtype=torch.float16)
            feats, dims = vt(px, [trim_text(q, cat, qid)])
            print(f"  [自检 {cat}#{qid}] 剪枝后特征 {tuple(feats.shape)} · "
                  f"actual_dims={dims[:2]} · 图像原尺寸 {img.size}", flush=True)
            expect = 576 if args.prune == "none" else int(576 * float(KEEP)) + 1
            assert dims[0] == expect, f"进 LLM 的视觉 token {dims[0]} != 预期 {expect}"

    KEEP_LOG.clear()
    fkeep = None
    if args.prune != "none":
        fkeep = open(keep_out, "w", encoding="utf-8")
    fout = open(out, "w", encoding="utf-8")
    for i, (cat, qid, q, img_bytes) in enumerate(rows):
        conv = conv_templates[args.conv].copy()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = process_images([img], image_processor, model.config)
        conv.append_message(conv.roles[0], f"<image>\n{q}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to("cuda:0")
        KEEP_LOG.clear()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=img_tensor.to("cuda:0", dtype=torch.float16),
                image_sizes=[img.size], do_sample=False,
                max_new_tokens=args.max_new_tokens,
                texts=[trim_text(q, cat, qid)])
        ans = tokenizer.decode(output_ids[0], skip_special_tokens=True).split(
            conv.roles[1] + ":")[-1].strip()
        fout.write(json.dumps({"cat": cat, "qid": qid, "question": q, "answer": ans,
                               "trim_query": trim_text(q, cat, qid)},
                              ensure_ascii=False) + "\n")
        fout.flush()
        if fkeep is not None:
            fkeep.write(json.dumps({"cat": cat, "qid": qid, "prune": args.prune,
                                    "keep": KEEP_LOG[0] if KEEP_LOG else []},
                                   ensure_ascii=False) + "\n")
            fkeep.flush()
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(rows)}]", flush=True)
    if resolver is not None:
        resolver.report()
    fout.close()
    if fkeep is not None:
        fkeep.close()
    if fkeep is not None:
        uniq = len({frozenset(json.loads(l)["keep"]) for l in open(keep_out)})
        print(f"[自检] {keep_out} 里不同掩码种类 = {uniq}（应≈样本数；=1 说明掩码没逐样本独立）",
              flush=True)
    print(f"完成 -> {out}", flush=True)


if __name__ == "__main__":
    main()
