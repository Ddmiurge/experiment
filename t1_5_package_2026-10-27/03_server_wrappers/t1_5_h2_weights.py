"""T1.5 步骤2 补充：按改名映射做张量级比对（H2：TRIM-FT 是否微调过）。"""
import json, re
from collections import defaultdict
import torch
from safetensors.torch import load_file

A = "/home/huangqixu01/models/llava-1.5-7b-hf"
B = "/home/huangqixu01/models/llava-v1.5-7b-TRIM"

def idx(d):
    return json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]

def map_name(k):
    if k == "language_model.model.embed_tokens.weight": return "model.embed_tokens.weight"
    if k == "language_model.lm_head.weight": return "lm_head.weight"
    m = re.match(r"language_model\.model\.(layers\.\d+\..*)$", k)
    if m: return "model." + m.group(1)
    m = re.match(r"multi_modal_projector\.linear_1\.(.*)$", k)
    if m: return f"model.mm_projector.0.{m.group(1)}"
    m = re.match(r"multi_modal_projector\.linear_2\.(.*)$", k)
    if m: return f"model.mm_projector.2.{m.group(1)}"
    m = re.match(r"vision_tower\.(.*)$", k)
    if m: return "model.vision_tower.vision_tower." + m.group(1)
    return None

ia, ib = idx(A), idx(B)
pairs = []
for ka in ia:
    kb = map_name(ka)
    if kb and kb in ib:
        pairs.append((ka, kb))
print(f"可映射同名张量: {len(pairs)} / A={len(ia)} B={len(ib)}")

# 按 shard 批量加载
def load_shard(d, name):
    return load_file(f"{d}/{name}")

cache = {}
def get(d, wm, key):
    shard = wm[key]
    if (d, shard) not in cache:
        cache[(d, shard)] = load_shard(d, shard)
    return cache[(d, shard)][key]

grp = defaultdict(lambda: [0, 0, 0.0])  # 组 -> [总, 完全相等, 最大绝对差]
worst = []
for ka, kb in pairs:
    a = get(A, ia, ka); b = get(B, ib, kb)
    if a.shape != b.shape:
        g = "形状不一致"; grp[g][0] += 1; continue
    same = torch.equal(a, b)
    md = (a.float() - b.float()).abs().max().item() if not same else 0.0
    g = ka.split(".")[0] if not ka.startswith("language_model.model.layers") else "language_model.model.layers"
    if ka.startswith("vision_tower"): g = "vision_tower"
    elif ka.startswith("multi_modal_projector"): g = "multi_modal_projector"
    elif ka.startswith("language_model.model.layers"): g = "language_model.model.layers"
    grp[g][0] += 1; grp[g][1] += int(same); grp[g][2] = max(grp[g][2], md)
    if not same: worst.append((md, ka))

print(f"\n{'组':<34}{'张量':>6}{'完全相等':>9}{'比例':>9}{'最大绝对差':>14}")
for g,(n,s,md) in sorted(grp.items()):
    print(f"{g:<34}{n:>6}{s:>9}{s/n*100:>8.1f}%{md:>14.3e}")
tot = sum(v[0] for v in grp.values()); eq = sum(v[1] for v in grp.values())
print(f"\n合计: {eq}/{tot} 张量完全相等 ({eq/tot*100:.1f}%)")
print("不等最多的 5 个:")
for md, k in sorted(worst, reverse=True)[:5]:
    print(f"  {k}  maxdiff={md:.3e}")
