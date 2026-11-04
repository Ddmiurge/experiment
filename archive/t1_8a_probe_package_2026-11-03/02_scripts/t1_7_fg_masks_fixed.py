#!/usr/bin/env python3
"""T1.7 · E1' 场景介质真值掩码（协议 §3.1：B1 Grounded-SAM，质量门不过→B2 中心秩；
GD/SAM 装不上→ --force-b2 全量 B2-only，headline 标注"中心秩操作化"）。
产出三份 masks（payload/blank/random）+ 类目×branch 审计表（协议 §3.2）。纯秩、无手调参数。
用法（服务器 exp 或独立 venv，需 transformers≥4.37 + GD/SAM 权重）：
  python t1_7_fg_masks.py --data /tmp/t1_4c_sd399_data.json [--force-b2]"""
import argparse, base64, io, json, os, sys
from collections import Counter
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/huangqixu01/experiment/experiments/t1_5_2026-10-27/scripts")  # [patch] /tmp 下相对路径失效
from msb_queries import load_official            # noqa: E402
N = 24; BUDGET = 144; SEED = 42
LO, HI = 29, 346                                  # 质量门：|fg| ∈ [5%,60%]（协议 §3.1）

def rank_sets(fg_cells, cx, cy):
    """fg/背景按到 (cx,cy) 距离成秩：④=fg 近心优先+非fg 补；③=非fg 远心优先。"""
    allc = set(range(N * N))
    fg = sorted(fg_cells)
    bg = sorted(allc - set(fg))
    d = {c: ((c // N - cx) ** 2 + (c % N - cy) ** 2) ** .5 for c in allc}
    if len(fg) >= BUDGET:
        kp = sorted(sorted(fg, key=lambda c: (d[c], c))[:BUDGET]); pad = 0
    else:
        pad = BUDGET - len(fg)
        kp = sorted(fg + sorted(bg, key=lambda c: (d[c], c))[:pad])
    kb = sorted(sorted(bg, key=lambda c: (-d[c], c))[:BUDGET])
    return kp, kb, pad

def b2_sets():
    kp, kb, pad = rank_sets([], (N - 1) / 2, (N - 1) / 2)
    return kp, kb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/t1_4c_sd399_data.json")
    ap.add_argument("--msb-json", default=os.environ.get("REPO", "/home/huangqixu01/experiment")
                    + "/third_party/mm-safetybench/data/processed_questions")
    ap.add_argument("--force-b2", action="store_true")
    ap.add_argument("--gd", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--sam", default="facebook/sam-vit-huge")
    ap.add_argument("--threshold", type=float, default=0.30)   # GD 置信阈（冻结值）
    ap.add_argument("--outdir", default="/tmp")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = [(r["cat"], int(r["qid"]), base64.b64decode(r["image_b64"]))
            for r in json.load(open(a.data))]
    if a.limit:
        rows = rows[:a.limit]
    official = load_official(a.msb_json)
    gd_pipe = None
    if not a.force_b2:
        try:
            import torch
            from transformers import pipeline, SamModel, SamProcessor
            det = pipeline("zero-shot-object-detection", model=a.gd, device=0)   # [patch] device=0 → CUDA_VISIBLE_DEVICES 指定
            sam = SamModel.from_pretrained(a.sam).to("cuda:0").eval()
            samp = SamProcessor.from_pretrained(a.sam)
            gd_pipe = (det, sam, samp)
            print("[masks] B1 Grounded-SAM 就绪", flush=True)
        except Exception as e:
            print(f"⚠ B1 不可用（{e}）→ 全量 B2-only（协议允许，headline 标注中心秩操作化）",
                  flush=True)

    fp = open(f"{a.outdir}/t1_7_masks_payload_sd399.jsonl", "w", encoding="utf-8")
    fb = open(f"{a.outdir}/t1_7_masks_blank_sd399.jsonl", "w", encoding="utf-8")
    fr = open(f"{a.outdir}/t1_7_masks_random_sd399.jsonl", "w", encoding="utf-8")
    audit = Counter(); per_cat = {}
    for i, (cat, qid, img_bytes) in enumerate(rows):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        kp = kb = None
        if gd_pipe is not None:
            kp, kb, fg_n, branch = _b1(gd_pipe, img, cat, qid, official, a.threshold)
            if kp is None:
                branch = None
        if kp is None:
            kp, kb = b2_sets(); fg_n = -1; branch = "center" + ("_forced" if a.force_b2 else "")
        rnd = sorted(int(x) for x in np.random.default_rng(SEED + i).choice(N * N, BUDGET,
                                                                            replace=False))
        for f, keep in ((fp, kp), (fb, kb), (fr, rnd)):
            f.write(json.dumps({"cat": cat, "qid": qid, "keep": keep, "branch": branch,
                                "fg_n": fg_n}) + "\n")
        audit[branch] += 1
        per_cat.setdefault(cat, Counter())[branch] += 1
        if (i + 1) % 50 == 0: print(f"[{i+1}/{len(rows)}] {dict(audit)}", flush=True)
    for f in (fp, fb, fr): f.close()
    print("\n═══ 类目 × branch 审计（协议 §3.2，只报告不硬掰）═══")
    for cat in sorted(per_cat):
        print(f"  {cat:<22}", dict(per_cat[cat]))
    b1 = audit.get("sam", 0)
    print(f"B1 率 = {b1/len(rows):.1%}（<20% 且未声明 B2-only ⇒ 停线，见协议 §5）")
    print(f"keeps 三件 -> {a.outdir}/t1_7_masks_(payload|blank|random)_sd399.jsonl")

def _b1(pipe, img, cat, qid, official, thr):
    det, sam, samp = pipe
    key = (official.get(cat, {}).get(str(qid), {}) or {}).get("Key Phrase", "") or "object"
    import torch
    boxes = det(img, candidate_labels=[key], threshold=thr)   # [patch] key 必须当 prompt
    if not boxes:
        return None, None, 0, "center"
    bx = []
    for b in boxes[:5]:
        bb = b["box"]
        bx.append([bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]] if isinstance(bb, dict)
                  else list(bb))                                          # [patch] 新版 pipeline 返回 dict
    inp = samp(images=img, return_tensors="pt", input_boxes=[bx]).to(sam.device)
    with torch.no_grad():
        pm = sam(**inp, multimask_output=False).pred_masks[0]         # (k,1,256,256) logits
    # [patch] 关键修正：原写法 .any(dim=0) 把 logits 当 bool（非零即真）⇒ 掩码≈全图
    # （fg≈576 全被质量门拒掉，B1 率假 0%）。改为 logits>0 再跨框取并集。
    m = (pm.float()[:, 0] > 0).any(dim=0)                            # [H,W] bool
    m2 = m.float().reshape(1, 1, *m.shape[-2:])
    g = torch.nn.functional.interpolate(m2, size=(N, N), mode="area") > 0.5
    fg = [int(r * N + c) for r in range(N) for c in range(N) if bool(g[0, 0, r, c])]
    if not (29 <= len(fg) <= 346):
        return None, None, len(fg), "center"
    cy = sum(c // N for c in fg) / len(fg); cx = sum(c % N for c in fg) / len(fg)
    kp, kb, _ = rank_sets(fg, cx, cy)
    return kp, kb, len(fg), "sam"

if __name__ == "__main__":
    main()
