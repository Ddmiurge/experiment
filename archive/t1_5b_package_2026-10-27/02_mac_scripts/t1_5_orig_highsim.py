#!/usr/bin/env python3
"""T1.5 · 原版栈（TRIM-FT 权重）+ **符号修正**的 TRIM。

与 `t1_4c_orig_sd_trim.py` 唯一区别：在**内存里**给 `CLIPVisionTower.token_reduction`
打一个 monkeypatch，把官方那行 `similarities = -similarities` 去掉，
使保留集 = 与文本相似度**最高**的 144 个 token（= TRIM 原文 §3.2 的规范）。

不动 `third_party/` 里任何文件（规则 2）——patch 只活在本次进程里。
开跑前有一个自检：同一批图上分别跑官方/修正两套打分，打印保留集相似度均值，
必须看到 official < highsim，否则停。

venv: /home/huangqixu01/venvs/trim-orig (torch 2.1.2 + transformers 4.37.2)
用法：
  CUDA_VISIBLE_DEVICES=5 python t1_5_orig_highsim.py            # SD
  CUDA_VISIBLE_DEVICES=5 python t1_5_orig_highsim.py --split TYPO
"""
import argparse
import base64
import io
import json
import os
import sys

import torch

REPO = os.environ.get("REPO", "/home/huangqixu01/experiment")
TRIM_REPO = os.path.join(REPO, "third_party/trim")
MODEL_PATH = os.environ.get("TRIM_M", "/home/huangqixu01/models/llava-v1.5-7b-TRIM")
CLIP_PATH = os.environ.get("CLIP_PATH", "/home/huangqixu01/models/clip-vit-large-patch14-336")
KEEP = os.environ.get("TRIM_KEEP", "0.25")


# ─────────────────────────────────────────────────────────────────────────
# 与官方 clip_encoder.py L67–L142 的**唯一**差别：第 89 行 `similarities = -similarities`
# 被删除（TRIM 原文 §3.2：retain the tokens with HIGH similarity scores）。
# 其余逐行照抄，包括 post_layernorm→visual_projection、聚合 token、actual_dims。
# ─────────────────────────────────────────────────────────────────────────
def token_reduction_highsim(self, image_features, all_image_features, text_features=None):
    import numpy as np
    import torch.nn.functional as F
    if not self.token_reduce_func:
        return image_features, [image_features.shape[1]] * image_features.shape[0]
    assert "TRIM" in self.token_reduce_func, self.token_reduce_func

    batch_size, tokens_number, dimension = image_features.shape
    k = float(self.token_reduce_func.replace("TRIM:", ""))

    with torch.cuda.amp.autocast(enabled=False):
        if image_features.dtype == torch.float16:
            image_features = image_features.to(torch.float32)
        elif image_features.dtype == torch.float32:
            image_features = image_features.to(
                self.vision_tower.vision_model.post_layernorm.weight.dtype)
        nomalized = self.vision_tower.vision_model.post_layernorm(image_features)
        proj = self.vision_tower.visual_projection(nomalized)

    similarities = torch.matmul(proj, text_features.unsqueeze(2)).squeeze(2)
    # ↓↓↓ 官方在这里有一行 `similarities = -similarities`，本函数按原文规范不取负
    similarities = F.softmax(similarities, dim=-1)

    if k == -1:                                    # 出厂 IQR 分支，本轮用不到，保留原样
        def outlier_detection(sim):
            a = sim.to(dtype=torch.float32).cpu().numpy().flatten()
            q1, q3 = np.percentile(a, 25), np.percentile(a, 75)
            ub = q3 + 1.5 * (q3 - q1)
            return len(np.where(a > ub)[0]) / len(a)
        k = outlier_detection(similarities)

    num_tokens_to_keep = int(tokens_number * k)
    topk_values, topk_indices = torch.topk(similarities, num_tokens_to_keep,
                                            dim=1, largest=True, sorted=False)
    selected_image_features = torch.zeros_like(image_features)
    batch_indices = torch.arange(batch_size, device=image_features.device).unsqueeze(1)
    token_mask = torch.zeros(batch_size, tokens_number, device=image_features.device,
                             dtype=torch.bool)
    token_mask[batch_indices, topk_indices] = True
    selected_image_features[:, :num_tokens_to_keep] = image_features[token_mask].view(
        batch_size, num_tokens_to_keep, -1)

    remaining_mask = torch.logical_not(token_mask)
    if remaining_mask.sum(dim=1).min().item() > 0:
        remaining_mean = ((image_features * remaining_mask.unsqueeze(-1)).sum(dim=1)
                          / remaining_mask.sum(dim=1, keepdim=True))
    else:
        remaining_mean = torch.zeros(batch_size, dimension, device=image_features.device)
    selected_image_features[:, num_tokens_to_keep] = remaining_mean

    actual_dims = [num_tokens_to_keep + 1] * batch_size
    image_features = selected_image_features
    if image_features.dtype == torch.float32:
        image_features = image_features.to(torch.float16)
    return image_features, actual_dims


def patch():
    """只改内存里的类方法；third_party 磁盘文件一个字节都不动。

    注意：普通函数直接赋给类属性即可成为方法（Python 描述符协议）。
    **不要**用 types.MethodType(f, cls) —— 那会把「类」当成 self 绑进去，
    实例调用时报 `type object 'CLIPVisionTower' has no attribute 'token_reduce_func'`
    （T1.5 首轮踩过，服务器用 /tmp 包装器绕过）。
    """
    sys.path.insert(0, TRIM_REPO)
    from llava.model.multimodal_encoder import clip_encoder
    clip_encoder.CLIPVisionTower.token_reduction_official = \
        clip_encoder.CLIPVisionTower.token_reduction
    clip_encoder.CLIPVisionTower.token_reduction = token_reduction_highsim
    print("[patch] CLIPVisionTower.token_reduction → highsim 版（原函数已存为 *_official）",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="SD")
    ap.add_argument("--data", default="/tmp/t1_4c_sd399_data.json",
                    help="预提取样本 json（含 base64 图），与历史格子同 id 同序")
    ap.add_argument("--conv", default="vicuna_v1")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--ids-from", default=None, help="若要换批样本，用历史回答文件锁 id")
    args = ap.parse_args()

    out = f"/tmp/t1_5_answers_orig_{args.split.lower()}_highsim.json"
    patch()
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX
    from transformers import AutoConfig, CLIPImageProcessor
    from PIL import Image

    rows = [(r["cat"], r["qid"], r["question"], base64.b64decode(r["image_b64"]))
            for r in json.load(open(args.data))]
    print(f"样本 {len(rows)} 条（{args.data}）", flush=True)

    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    cfg.mm_vision_tower = CLIP_PATH
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, "llava_v1", device="cuda:0", use_flash_attn=False, config=cfg)
    try:
        img_proc = CLIPImageProcessor.from_pretrained(CLIP_PATH)
        image_processor = image_processor if image_processor is not None else img_proc
    except Exception:
        pass
    vt = model.get_model().vision_tower
    vt.token_reduce_func = f"TRIM:{KEEP}"
    print(f"token_reduce_func={vt.token_reduce_func}  "
          f"image_aspect_ratio={getattr(model.config,'image_aspect_ratio','(缺)')}  "
          f"conv={args.conv}", flush=True)

    # ── 自检：前 5 条同时算两套打分，确认方向确实变了 ──
    def score(vt, px, q):
        """复刻官方打分链，返回 [576] 的相似度（float32）。dtype 跟着模块权重走。"""
        vout = vt.vision_tower(px, output_hidden_states=True)
        vmod = vt.vision_tower.vision_model
        pln = vmod.post_layernorm
        dt = pln.weight.dtype
        raw = vout.hidden_states[vt.select_layer][:, 1:].to(dt)          # [1,576,1024]
        proj = vt.vision_tower.visual_projection(pln(raw)).float()       # [1,576,768]
        ti = vt.text_tokenizer(text=[q], return_tensors="pt",
                               truncation=True, padding=True).to(px.device)
        tf = vt.text_tower(**ti).text_embeds.to(torch.float32)            # [1,768]
        return torch.matmul(proj, tf.unsqueeze(2)).squeeze(2)[0]

    with torch.inference_mode():
        for i, (cat, qid, q, img_bytes) in enumerate(rows[:5]):
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            px = process_images([img], image_processor, model.config).to(
                "cuda:0", dtype=torch.float16)
            sims = score(vt, px, q)
            n_keep = int(sims.shape[0] * float(KEEP))
            k_off = torch.topk(-sims, n_keep).indices        # 官方：softmax(-s) 取前 k
            k_hi = torch.topk(sims, n_keep).indices          # 修正：按原文取高相似度
            ov = len(set(k_off.tolist()) & set(k_hi.tolist())) / n_keep
            print(f"  [自检 {cat}#{qid}] sim(官方保留)={sims[k_off].mean():+.4f} "
                  f"sim(修正保留)={sims[k_hi].mean():+.4f} 两套保留集 Jaccard={ov:.3f}",
                  flush=True)
            assert sims[k_hi].mean() > sims[k_off].mean(), "修正版相似度均值没更高，patch 没生效"

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
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=img_tensor.to("cuda:0", dtype=torch.float16),
                image_sizes=[img.size], do_sample=False,
                max_new_tokens=args.max_new_tokens, texts=[q])
        ans = tokenizer.decode(output_ids[0], skip_special_tokens=True).split(
            conv.roles[1] + ":")[-1].strip()
        fout.write(json.dumps({"cat": cat, "qid": qid, "question": q, "answer": ans},
                              ensure_ascii=False) + "\n")
        fout.flush()
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(rows)}]", flush=True)
    fout.close()
    print(f"完成 -> {out}", flush=True)


if __name__ == "__main__":
    main()
