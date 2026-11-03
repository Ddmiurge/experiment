#!/usr/bin/env python3
"""T1.5 · 权重与配置比对（CPU，几分钟）。定 H2 / 顺带定 H6。

问题：T1.4c/d 的「原版栈」两格都跑在 `FreedomIntelligence/llava-v1.5-7b-TRIM` 上，
而 TRIM 仓库 README 明写那是 **fine-tuned with TRIM** 的 checkpoint
（不是 vanilla LLaVA 重新打包）。若两个 checkpoint 权重确实不同，
我们测的「无剪枝基线」就不是原文 Original 行那个模型，58.40≈59.37 的对齐要重新解释。

做三件事：
  A. config.json 关键字段对照（顺带回答 H6：image_aspect_ratio 到底是 'pad' 还是 'square'，
     以及 mm_vision_token_reduce_func 出厂值）
  B. 张量级签名比对：同名张量是否逐元素相等；不同名但同 (shape,mean,std) 签名的算「等价」
  C. 结论一句话

用法：
  python t1_5_weights_diff.py \
    --a /home/huangqixu01/models/llava-1.5-7b-hf \
    --b /home/huangqixu01/models/llava-v1.5-7b-TRIM
"""
import argparse
import glob
import json
import os

import torch
from safetensors import safe_open

KEYS_OF_INTEREST = ["image_aspect_ratio", "mm_vision_token_reduce_func", "mm_projector_type",
                    "mm_vision_select_layer", "mm_vision_tower", "vision_feature_layer",
                    "mm_patch_merge_type", "architectures", "_name_or_path", "model_type"]


def tensor_index(d):
    """返回 {name: file} —— 优先用 index.json，否则单文件 model.safetensors。"""
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        m = json.load(open(idx))["weight_map"]
        return {k: os.path.join(d, v) for k, v in m.items()}
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    out = {}
    for f in files:
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                out[k] = f
    return out


def sig(t):
    """与键名无关的签名：形状 + 均值 + 标准差（float32 下算，抗 fp16 舍入）。"""
    v = t.detach().float()
    return (tuple(v.shape), round(v.mean().item(), 4), round(v.std().item(), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="通常是 vanilla（llava-1.5-7b-hf）")
    ap.add_argument("--b", required=True, help="通常是 TRIM-FT（llava-v1.5-7b-TRIM）")
    ap.add_argument("--sample", type=int, default=0, help="只比对前 N 个同名张量（0=全部）")
    ap.add_argument("--sig-filter", default="",
                    help="签名匹配只比对名字含该子串的张量，如 mm_projector / vision_tower / lm_head；空=全部")
    args = ap.parse_args()

    for p in (args.a, args.b):
        assert os.path.isdir(p), f"目录不存在：{p}"

    print("═══ A · config.json 关键字段 ═══")
    ca = json.load(open(os.path.join(args.a, "config.json")))
    cb = json.load(open(os.path.join(args.b, "config.json")))
    flat = lambda c: {k: c.get(k) for k in KEYS_OF_INTEREST}
    fa, fb = flat(ca), flat(cb)
    for k in KEYS_OF_INTEREST:
        mark = "  " if fa.get(k) == fb.get(k) else "≠≠"
        print(f"{mark} {k:<32} A={fa.get(k)}   B={fb.get(k)}")

    print("\n═══ B · 张量级比对 ═══")
    ia, ib = tensor_index(args.a), tensor_index(args.b)
    print(f"A 张量 {len(ia)} 个 / B 张量 {len(ib)} 个")
    common = sorted(set(ia) & set(ib))
    print(f"同名张量 {len(common)} 个；仅 A {len(set(ia)-set(ib))} 个；仅 B {len(set(ib)-set(ia))} 个")

    eq, neq = [], []
    names = common[:args.sample] if args.sample else common
    for n in names:
        with safe_open(ia[n], framework="pt") as h:
            ta = h.get_tensor(n)
        with safe_open(ib[n], framework="pt") as h:
            tb = h.get_tensor(n)
        if ta.shape != tb.shape:
            neq.append((n, "shape")); continue
        if torch.equal(ta.float(), tb.float()):
            eq.append(n)
        else:
            neq.append((n, f"max|Δ|={(ta.float()-tb.float()).abs().max().item():.3e}"))
    print(f"同名张量中：完全相等 {len(eq)} / 不相等 {len(neq)}")
    for n, why in neq[:15]:
        print(f"   ≠ {n:<70} {why}")
    if len(neq) > 15:
        print(f"   … 其余 {len(neq)-15} 个略")

    # 不同名但同签名的（HF 转换只改名不改值）——按签名多重集统计
    print("\n按 (shape,mean,std) 签名做与改名无关的匹配：")
    sa, sb = {}, {}
    for d, idx, store in ((args.a, ia, sa), (args.b, ib, sb)):
        for f in sorted(set(idx.values())):
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    if args.sig_filter and args.sig_filter not in k:
                        continue
                    s = sig(h.get_tensor(k))
                    store[s] = store.get(s, 0) + 1
    na = sum(sa.values())
    shared = sum(min(v, sb.get(k, 0)) for k, v in sa.items())
    print(f"   过滤词={args.sig_filter or '(不过滤)'}  A 有 {len(sa)} 种签名/{na} 个张量，"
          f"B 有 {len(sb)} 种/{sum(sb.values())} 个")
    print(f"   签名重合 {shared} 个 → 比例 {shared/max(1,na):.3f}")

    print("\n═══ C · 怎么读 ═══")
    print("· 同名张量几乎全等 / 签名重合比例≈1 → 两 checkpoint 是同一套权重（改名而已），H2 排除")
    print("· 大量不相等或签名重合比例低 → TRIM-FT 确实微调过，")
    print("  则 T1.4c/d 的『无剪枝 58.40』不是原文 Original 行的模型，必须补 vanilla 无剪枝格")
    print("· A 段的 image_aspect_ratio 若是 'square' → 原版栈的 process_images 没走 expand2square，")
    print("  TYPO 图仍被中心裁剪，T1.4 报告里『预处理已修』这条要重新核")


if __name__ == "__main__":
    main()
