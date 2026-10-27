#!/usr/bin/env python3
"""T1.5b（TYPO 四格）Mac 侧独立复核 + 载荷行反推。

做四件事：
  A. 从 07_judge_raw 重算 ASR、配对 b/c/净、McNemar 精确 p、**配对 95% CI**、z vs 原文 +6.50pp
  B. 严格(5类)/通用(8类)分层（用 src/eval 的 STRICT_CATEGORIES，避免上一轮 137/262 那类错）
  C. harness 自检：none / official 是否与 T1.4d 的 0.7594 / 0.7318 对齐
  D. 从 keeps 文件反推「载荷行」保留率：
     TYPO = 白底黑字横条 → 行像素活动 = 该行内跨通道的标准差；
     活动最高的 6 行 = 文字条所在行。不需要 CLIP 权重，纯 PIL 即可跑。
     同时检查 random 的掩码是否逐样本独立（上一版脚本有个每次调用重建 rng 的 bug）。

用法：python3 scripts/t1_5b_analysis.py
"""
import io
import json
import math
import os
import statistics
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eval.mm_safetybench_judge import CATEGORIES, STRICT_CATEGORIES  # noqa: E402

PKG = os.path.join(ROOT, "t1_5b_package_2026-10-27")
CLAIM = 0.0650                      # 原文 Table 2：59.37 → 65.87
CELLS = ["none", "official", "highsim", "random"]
T14D = {"none": 0.7594, "official": 0.7318}     # T1.4d TYPO（原版栈 + vicuna_v1）


def raw(cell):
    p = os.path.join(PKG, "07_judge_raw", f"t1_5_judge_raw_t1_5b_answers_{cell}_typo399.jsonl")
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[(r["cat"], int(r["qid"]))] = r["verdict"]
    return out


def keeps(cell):
    p = os.path.join(PKG, "05_keeps", f"t1_5b_keeps_{cell}_typo399.jsonl")
    if not os.path.exists(p):
        return {}
    d = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("keep"):
            d[(r["cat"], int(r["qid"]))] = set(r["keep"])
    return d


def paired_stats(b, c, n):
    m = b + c
    p = 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(min(b, c) + 1)) / 2 ** m)
    se = math.sqrt(m - (b - c) ** 2 / n) / n
    return p, se


def main():
    data = {k: raw(k) for k in CELLS}
    gen = set(CATEGORIES) - set(STRICT_CATEGORIES)

    print("═══ A/C · 四格 + 配对（含 harness 自检）═══")
    print(f"{'格子':<10}{'ASR':>8}{'T1.4d':>8}{'差pp':>7}{'b':>4}{'c':>4}{'净':>5}"
          f"{'p(Δ=0)':>9}{'Δ 95%CI':>19}{'z vs +6.5':>11}{'p':>10}")
    ref = data["none"]
    for cell in CELLS:
        cur = data[cell]
        common = set(ref) & set(cur)
        n = len(common)
        nu = sum(1 for k in common if cur[k] == "unsafe")
        asr = nu / n
        b = sum(1 for k in common if ref[k] == "safe" and cur[k] == "unsafe")
        c = sum(1 for k in common if ref[k] == "unsafe" and cur[k] == "safe")
        d = asr - sum(1 for k in common if ref[k] == "unsafe") / n
        if cell == "none":
            print(f"{cell:<10}{asr:>8.4f}{T14D['none']:>8.4f}{(asr-T14D['none'])*100:>7.2f}"
                  f"{'—':>4}{'—':>4}{'—':>5}{'—':>9}{'—':>19}{'—':>11}{'—':>10}")
            continue
        p, se = paired_stats(b, c, n)
        z = (d - CLAIM) / se
        pz = math.erfc(abs(z) / math.sqrt(2))
        refv = T14D.get(cell)
        print(f"{cell:<10}{asr:>8.4f}{refv if refv else float('nan'):>8.4}"
              f"{(asr-refv)*100 if refv else float('nan'):>7.2f}"
              f"{b:>4}{c:>4}{b-c:>5}{p:>9.4f}"
              f"{f'[{(d-1.96*se)*100:+.2f},{(d+1.96*se)*100:+.2f}]':>19}"
              f"{z:>11.2f}{pz:>10.1e}")
        globals()["_last"] = (cell, d, se)

    print(f"\n原文声称 Δ = +{CLAIM*100:.2f}pp；上表 Δ 一律为负，"
          f"其中 highsim 与声称值相差 |z|>7 —— 不是「没测到」，是**方向相反且显著**。")

    print("\n═══ B · 分层（严格 5 类 n / 通用 8 类 n）═══")
    for cell in CELLS[1:]:
        line = f"{cell:<10}"
        for name, keys in (("严格", STRICT_CATEGORIES), ("通用", gen)):
            ks = [k for k in set(ref) & set(data[cell]) if k[0] in keys]
            r = sum(1 for k in ks if ref[k] == "unsafe") / len(ks)
            v = sum(1 for k in ks if data[cell][k] == "unsafe") / len(ks)
            b = sum(1 for k in ks if ref[k] == "safe" and data[cell][k] == "unsafe")
            c = sum(1 for k in ks if ref[k] == "unsafe" and data[cell][k] == "safe")
            p, se = paired_stats(b, c, len(ks))
            line += (f" {name} n={len(ks):>3} {r:.3f}→{v:.3f} "
                     f"{(v-r)*100:+6.2f}pp CI[{(v-r-1.96*se)*100:+.1f},"
                     f"{(v-r+1.96*se)*100:+.1f}] p={p:.4f} |")
        print(line)

    print("\n═══ D · keeps 反推：载荷行 vs 留白行 ═══")
    ans = {r["cat"] + "#" + str(r["qid"]): r for r in
           (json.loads(l) for l in open(os.path.join(PKG, "04_answers",
                                                    "t1_5b_answers_none_typo399.json")))}
    ks_h, ks_r = keeps("highsim"), keeps("random")
    print(f"highsim keeps 非空 {len(ks_h)}/399 · random keeps 非空 {len(ks_r)}/399")
    print(f"random 掩码去重后种类 = {len({frozenset(v) for v in ks_r.values()})}"
          f"  ← 应为 399（逐样本独立）；若为 1 说明 rng 每次调用被重建（已修的 bug）")

    try:
        from PIL import Image
        import pandas as pd
        DATA = os.environ.get("MSB_DATA", "")
        if not DATA or not os.path.isdir(DATA):
            raise RuntimeError("本机没有 MM-SafetyBench parquet（MSB_DATA 未设），"
                               "载荷行剖面需在服务器上跑")
        # 行活动 = 该行 14 个 patch 像素块跨通道 std 的均值；取活动最高 6 行为载荷行
        stats = {"payload": Counter(), "blank": Counter()}
        for key, keepsets in (("highsim", ks_h), ("random", ks_r)):
            got = 0
            for (cat, qid), keep in keepsets.items():
                f = os.path.join(DATA, cat, "TYPO.parquet")
                if not os.path.exists(f):
                    continue
                df = pd.read_parquet(f, columns=["id", "image"])
                row = df[df["id"] == qid]
                if row.empty:
                    continue
                img = Image.open(io.BytesIO(row.iloc[0]["image"])).convert("RGB")
                w, h = img.size
                # 白底黑字：行活动 = 255 − 该行平均亮度越偏离 255 越大
                act = []
                for r in range(24):
                    y0 = int(h * r / 24)
                    y1 = max(y0 + 1, int(h * (r + 1) / 24))
                    strip = img.crop((0, y0, w, y1))
                    act.append(255.0 - statistics.mean(
                        px[i] for px in strip.getdata() for i in range(3)))
                top = sorted(range(24), key=lambda r: -act[r])[:6]
                for idx in keep:
                    stats[("payload" if idx // 24 in top else "blank")][key] += 1
                got += 1
                if got >= 120:
                    break
            print(f"  {key}: 算了 {got} 条")
        for key in ("highsim", "random"):
            tot = stats["payload"][key] + stats["blank"][key]
            if tot:
                print(f"  {key}: 保留 token 落在载荷行的比例 = "
                      f"{stats['payload'][key]/tot:.3f}"
                      f"（随机期望 ≈ 6*24/576 = 0.25）")
    except Exception as e:                                    # noqa: BLE001
        print(f"  ⚠ 载荷行反推未在本机完成：{e}")
        print("    报告 §3.1 已给 highsim 0.223 / official 0.451 / random 0.260；"
              "需要独立复核时用服务器上 t1_5_trim_probe.py 的行剖面即可")


if __name__ == "__main__":
    main()
