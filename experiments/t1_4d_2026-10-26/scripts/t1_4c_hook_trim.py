#!/usr/bin/env python3
"""T1.4c · A/B 对比：forward 内部替换（hook 版）vs 手工拼接（/tmp，临时脚本）。

方案 A：给 llava-hf 的 LlavaModel.get_image_features 挂包装，返回 TRIM 剪枝后的
145 个投影特征；input_ids 占位缩到 145；其余（scatter、position、attention）全走库标准路径。
对照：/tmp/t1_4_answers_trim_typo399.json（手工拼接版，同 399 条 TYPO）
输出：/tmp/t1_4_answers_trimhook_typo399.json + 逐条 diff 统计
"""
import glob
import io
import json
import sys

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/huangqixu01/experiment/scripts")

LLAVA = "/home/huangqixu01/models/llava-1.5-7b-hf"
CLIP = "/home/huangqixu01/models/clip-vit-large-patch14-336"
DATA = "/home/huangqixu01/data/MM-SafetyBench/data"
IDS_FROM = "/tmp/t1_4_answers_noprun_typo399.json"
MANUAL = "/tmp/t1_4_answers_trim_typo399.json"
OUT = "/tmp/t1_4_answers_trimhook_typo399.json"
KEEP_RATIO = 0.25
PROMPT_T = "USER: <image>\n{q}\nASSISTANT:"
IMG_ID = 32000


def main():
    from PIL import Image
    from transformers import (AutoProcessor, AutoTokenizer, CLIPModel,
                              CLIPTextModelWithProjection,
                              LlavaForConditionalGeneration)
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    # ── 数据：同 399 条（ids_from 锁定，顺序与手工版一致）──
    want = [(json.loads(l)["cat"], json.loads(l)["qid"]) for l in open(IDS_FROM)]
    wset = set(want)
    rows = []
    for f in sorted(glob.glob(f"{DATA}/*/TYPO.parquet")):
        cat = f.split("/data/")[-1].split("/")[0]
        for _, r in pd.read_parquet(f).iterrows():
            if (cat, int(r["id"])) in wset:
                rows.append((cat, int(r["id"]), str(r["question"]), r["image"]))
    rows.sort(key=lambda x: want.index((x[0], x[1])))
    assert [(c, q) for c, q, _, _ in rows] == want, "样本顺序与手工版不一致"
    print(f"样本 {len(rows)} 条（与手工版同 id 同序）", flush=True)

    # ── 模型 ──
    proc = AutoProcessor.from_pretrained(LLAVA)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA, torch_dtype=torch.float16, attn_implementation="eager",
        low_cpu_mem_usage=True).to("cuda:0").eval()
    clip = CLIPModel.from_pretrained(CLIP, torch_dtype=torch.float32).to("cuda:0").eval()
    text_tower = CLIPTextModelWithProjection.from_pretrained(
        CLIP, torch_dtype=torch.float32).to("cuda:0").eval()
    text_tok = AutoTokenizer.from_pretrained(CLIP)

    def make_trim_hook(texts):
        """签名与原版 get_image_features 一致（generate 会按签名匹配参数）"""
        def hook(pixel_values, vision_feature_layer=None,
                 vision_feature_select_strategy=None, image_sizes=None, **kw):
            return hooked_get_image_features(pixel_values, _texts=texts)
        return hook

    def hooked_get_image_features(pixel_values, *args, _texts=None, **kwargs):
        """TRIM 剪枝在 get_image_features 内完成（与原版 clip_encoder 同构位置）。"""
        with torch.no_grad():
            vout = model.model.vision_tower(pixel_values, output_hidden_states=True)
            feats = vout.hidden_states[-2][:, 1:]              # [1,576,1024] fp16
            t_in = text_tok(text=_texts.replace("<image>", " "),
                            return_tensors="pt", truncation=True, padding=True).to("cuda:0")
            t_feat = text_tower(**t_in).text_embeds            # [1,768] fp32
            f32 = feats.to(torch.float32)
            proj_img = clip.visual_projection(clip.vision_model.post_layernorm(f32))
            sims = torch.matmul(proj_img, t_feat.unsqueeze(2)).squeeze(2)
            score = F.softmax(-sims, dim=-1)
            n_keep = int(feats.shape[1] * KEEP_RATIO)
            _, keep_idx = torch.topk(score, n_keep, dim=1, largest=True, sorted=False)
            keep_idx, _ = torch.sort(keep_idx[0])               # 原始位置序（上轮修复口径）
            v_keep = f32[0, keep_idx]
            drop_idx = sorted(set(range(feats.shape[1])) - set(keep_idx.tolist()))
            agg = f32[0, torch.tensor(drop_idx, device=f32.device)].mean(dim=0, keepdim=True)
            feats_sel = torch.cat([v_keep, agg], dim=0).to(torch.float16)   # [145,1024]
            out = model.model.multi_modal_projector(feats_sel)  # [145,4096]
        return BaseModelOutputWithPooling(pooler_output=[out])

    out_rows = []
    for i, (cat, qid, q, img_bytes) in enumerate(rows):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        prompt = PROMPT_T.format(q=q)
        inputs = proc(text=prompt, images=img, return_tensors="pt").to("cuda:0")
        ids = inputs["input_ids"][0]
        pos = (ids == IMG_ID).nonzero(as_tuple=True)[0]
        s, e = int(pos[0]), int(pos[-1]) + 1
        n = int(145)  # 144 保留 + 1 聚合
        new_ids = torch.cat([ids[:s], torch.full((n,), IMG_ID, dtype=ids.dtype, device=ids.device), ids[e:]])
        model.model.get_image_features = make_trim_hook(prompt)
        with torch.no_grad():
            o = model.generate(
                input_ids=new_ids.unsqueeze(0),
                attention_mask=torch.ones_like(new_ids).unsqueeze(0),
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512, do_sample=False,
                pad_token_id=proc.tokenizer.pad_token_id)
        ans = proc.tokenizer.decode(o[0][new_ids.shape[0]:], skip_special_tokens=True).strip()  # 走 input_ids 路径需刪掉 prompt 段
        out_rows.append({"cat": cat, "qid": qid, "question": q, "answer": ans})
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(rows)}]", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── A/B diff ──
    manual = {(json.loads(l)["cat"], json.loads(l)["qid"]): json.loads(l)["answer"]
              for l in open(MANUAL)}
    same = diff = 0
    first_diffs = []
    for r in out_rows:
        m = manual.get((r["cat"], r["qid"]))
        if m == r["answer"]:
            same += 1
        else:
            diff += 1
            if len(first_diffs) < 3:
                first_diffs.append((r["cat"], r["qid"], (m or "")[:60], r["answer"][:60]))
    print(f"\n═══ A/B 对比（hook 内部替换 vs 手工拼接）═══")
    print(f"完全一致: {same}/{len(out_rows)}   不同: {diff}/{len(out_rows)}")
    for c, q, m, h in first_diffs:
        print(f"  diff 例: {c}#{q}\n    手工: {m}\n    hook: {h}")


if __name__ == "__main__":
    main()
