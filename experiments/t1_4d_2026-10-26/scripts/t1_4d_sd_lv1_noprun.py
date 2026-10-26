#!/usr/bin/env python3
"""T1.4c · 方案B：原版栈（third_party/trim 原始代码 + 官方合并权重）跑 399 条 TYPO。

- venv: /home/huangqixu01/venvs/trim-orig (torch 2.1.2 + transformers 4.37.2)
- 权重: /home/huangqixu01/models/llava-v1.5-7b-TRIM（出厂 TRIM:-1，覆盖为 TRIM:0.25）
- 样本: 与手工版完全相同的 399 条 TYPO（ids_from 锁定，同序）
- 输出: /tmp/t1_4_answers_orig_sd_noprun_llavav1_399.json
"""
import base64
import io
import json
import os
import sys

import torch

TRIM_REPO = "/home/huangqixu01/experiment/third_party/trim"
MODEL_PATH = "/home/huangqixu01/models/llava-v1.5-7b-TRIM"
DATA = "/home/huangqixu01/data/MM-SafetyBench/data"
IDS_FROM = "/tmp/t1_4_answers_noprun_typo399.json"
OUT = "/tmp/t1_4_answers_orig_sd_noprun_llavav1_399.json"
KEEP = os.environ.get("TRIM_KEEP", "0.25")  # 出厂 -1，按论文口径覆盖


def main():
    sys.path.insert(0, TRIM_REPO)
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.conversation import conv_templates

    # ── 样本：读预提取 JSON（同 399 条，同 id 同序）──
    rows = [(r["cat"], r["qid"], r["question"], base64.b64decode(r["image_b64"]))
            for r in json.load(open("/tmp/t1_4c_sd399_data.json"))]
    print(f"样本 {len(rows)} 条（预提取 JSON，与手工版同 id 同序）", flush=True)

    # ── 原版加载 ──
    from transformers import AutoConfig, AutoProcessor
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    # 权重的 config 里 vision_tower 是作者机器上的相对快照路径，指向本地 CLIP
    cfg.mm_vision_tower = "/home/huangqixu01/models/clip-vit-large-patch14-336"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, "llava_v1", device="cuda:0",
        use_flash_attn=False, config=cfg)
    # 合并权重目录无 preprocessor_config.json，图像处理器用 CLIP 目录的
    from transformers import CLIPImageProcessor
    img_proc = CLIPImageProcessor.from_pretrained("/home/huangqixu01/models/clip-vit-large-patch14-336")
    img_proc = image_processor if image_processor is not None else img_proc
    # 覆盖保留率为 0.25（出厂 TRIM:-1 自动）
    model.get_model().vision_tower.token_reduce_func = None  # 无剪枝对照
    print(f"token_reduce_func = {model.get_model().vision_tower.token_reduce_func}", flush=True)
    out_rows = []
    for i, (cat, qid, q, img_bytes) in enumerate(rows):
        conv = conv_templates["llava_v1"].copy()  # 每样本新建，避免多轮累积
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # img_bytes 已是 bytes
        img_tensor = process_images([img], img_proc, model.config)
        conv.append_message(conv.roles[0], f"<image>\n{q}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        from llava.constants import IMAGE_TOKEN_INDEX
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda:0")
        img_batch = img_tensor.to("cuda:0", dtype=torch.float16)  # process_images 已返回 [1,3,336,336]
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=img_batch, image_sizes=None,
                do_sample=False, max_new_tokens=512,
                texts=[q],  # TRIM 需要文本侧特征
            )
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # 去掉 conv 模板回显的 prompt 部分，只留回答
        ans = output.split(conv.roles[1] + ":")[-1].strip()
        out_rows.append({"cat": cat, "qid": qid, "question": q, "answer": ans})
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(rows)}]", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
