#!/usr/bin/env python3
"""T1.7 · E1' 三格生成：保留集**来自外部真值掩码文件**（与 t1_6 的关键区别：
不在运行时重算活动度秩——掩码由 t1_7_fg_masks.py 一次性产出并登记 branch）。
栈/权重/解码/aggregation 与 T1.6 全同；进 LLM 恒 145（assert）。
用法：... --prune payload|blank|random --masks /tmp/t1_7_masks_<which>_sd399.jsonl"""
import argparse, base64, io, json, os, sys
import torch
REPO = os.environ.get("REPO", "/home/huangqixu01/experiment")
TRIM_REPO = os.path.join(REPO, "third_party/trim")
MODEL_PATH = os.environ.get("TRIM_M", "/home/huangqixu01/models/llava-v1.5-7b-TRIM")
CLIP_PATH = os.environ.get("CLIP_PATH", "/home/huangqixu01/models/clip-vit-large-patch14-336")
BUDGET = 144
CUR = [set()]

def token_reduction_extmask(self, image_features, all_image_features, text_features=None):
    if not self.token_reduce_func:
        return image_features, [image_features.shape[1]] * image_features.shape[0]
    B, N, D = image_features.shape
    k = float(self.token_reduce_func.replace("TRIM:", ""))
    n_keep = int(N * k)
    assert n_keep == BUDGET == len(CUR[0])
    m = torch.zeros(B, N, device=image_features.device, dtype=torch.bool)
    m[0, torch.tensor(sorted(CUR[0]), device=image_features.device)] = True
    if image_features.dtype == torch.float16:
        image_features = image_features.to(torch.float32)
    sel = torch.zeros_like(image_features)
    sel[:, :n_keep] = image_features[m].view(B, n_keep, -1)
    rem = torch.logical_not(m)
    sel[:, n_keep] = ((image_features * rem.unsqueeze(-1)).sum(dim=1)
                      / rem.sum(dim=1, keepdim=True))
    out = sel.to(torch.float16)
    return out, [n_keep + 1] * B

def apply():
    sys.path.insert(0, TRIM_REPO)
    from llava.model.multimodal_encoder import clip_encoder
    clip_encoder.CLIPVisionTower.token_reduction_orig = clip_encoder.CLIPVisionTower.token_reduction
    clip_encoder.CLIPVisionTower.token_reduction = token_reduction_extmask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", required=True, choices=["payload", "blank", "random"])
    ap.add_argument("--data", default="/tmp/t1_4c_sd399_data.json")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--conv", default="vicuna_v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mpath = a.masks or f"/tmp/t1_7_masks_{a.prune}_sd399.jsonl"
    masks = {}
    for l in open(mpath):
        r = json.loads(l); masks[(r["cat"], int(r["qid"]))] = r
    out = a.out or f"/tmp/t1_7_answers_oracle_{a.prune}_sd399.json"
    keep_out = os.path.splitext(out)[0] + "_keeps.jsonl"
    apply()
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX
    from transformers import AutoConfig
    from PIL import Image
    rows = [(r["cat"], int(r["qid"]), r["question"], base64.b64decode(r["image_b64"]))
            for r in json.load(open(a.data))]
    missing = [(r[0], r[1]) for r in rows if (r[0], r[1]) not in masks]
    assert not missing, f"掩码缺 id：{missing[:3]}（先跑 t1_7_fg_masks.py）"
    if a.limit: rows = rows[:a.limit]
    cfg = AutoConfig.from_pretrained(MODEL_PATH); cfg.mm_vision_tower = CLIP_PATH
    tok, model, ip, ctx = load_pretrained_model(MODEL_PATH, None, "llava_v1",
                                                device="cuda:0", use_flash_attn=False, config=cfg)
    vt = model.get_model().vision_tower
    vt.token_reduce_func = "TRIM:0.25"
    fo = open(out, "w", encoding="utf-8"); fk = open(keep_out, "w", encoding="utf-8")
    leak = 0
    for i, (cat, qid, q, img_bytes) in enumerate(rows):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        mrec = masks[(cat, qid)]
        CUR[0] = set(mrec["keep"])
        if a.prune == "payload":                       # ④ 侧核验：fg 覆盖登记
            leak = 0
        px = process_images([img], ip, model.config).to("cuda:0", dtype=torch.float16)
        conv = conv_templates[a.conv].copy()
        conv.append_message(conv.roles[0], f"<image>\n{q}")
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX,
                                    return_tensors="pt").unsqueeze(0).to("cuda:0")
        with torch.inference_mode():
            o = model.generate(ids, images=px, image_sizes=[img.size], do_sample=False,
                               max_new_tokens=512, texts=[q])
            _, dims = vt(px, [q])
        assert dims[0] == 145, f"进 LLM 视觉 token {dims[0]} != 145"
        ans = tok.decode(o[0], skip_special_tokens=True).split(conv.roles[1] + ":")[-1].strip()
        fo.write(json.dumps({"cat": cat, "qid": qid, "question": q, "answer": ans,
                             "prune": f"oracle_{a.prune}"}, ensure_ascii=False) + "\n"); fo.flush()
        fk.write(json.dumps({"cat": cat, "qid": qid, "prune": a.prune,
                             "keep": sorted(CUR[0]),
                             "branch": mrec.get("branch"), "fg_n": mrec.get("fg_n"),
                             "pad": mrec.get("pad", 0)}, ensure_ascii=False) + "\n"); fk.flush()
        if i < 3 or (i + 1) % 50 == 0:
            print(f"[{i}] {cat}#{qid} branch={mrec.get('branch')} fg_n={mrec.get('fg_n')} "
                  f"keep={len(CUR[0])}", flush=True)
    fo.close(); fk.close()
    print(f"完成 -> {out} / {keep_out}", flush=True)

if __name__ == "__main__":
    main()
