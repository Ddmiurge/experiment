#!/usr/bin/env python3
"""T1.5b 步骤3 补充分析（/tmp 临时脚本）：配对 CI、z vs +6.50pp、5/8 类分层、回答长度变化。"""
import json
import math
import sys
from math import comb

sys.path.insert(0, "/home/huangqixu01/experiment")
sys.path.insert(0, "/home/huangqixu01/experiment/src")
from eval.mm_safetybench_judge import STRICT_CATEGORIES

RAW = {
    "none": "/tmp/t1_5_judge_raw_t1_5b_answers_none_typo399.jsonl",
    "official": "/tmp/t1_5_judge_raw_t1_5b_answers_official_typo399.jsonl",
    "highsim": "/tmp/t1_5_judge_raw_t1_5b_answers_highsim_typo399.jsonl",
    "random": "/tmp/t1_5_judge_raw_t1_5b_answers_random_typo399.jsonl",
}
ANS = {
    "none": "/tmp/t1_5b_answers_none_typo399.json",
    "official": "/tmp/t1_5b_answers_official_typo399.json",
    "highsim": "/tmp/t1_5b_answers_highsim_typo399.json",
    "random": "/tmp/t1_5b_answers_random_typo399.json",
}
TARGET = 0.065


def load_raw(p):
    d = {}
    for line in open(p):
        r = json.loads(line)
        d[(r["cat"], int(r["qid"]))] = r
    return d


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def stats(ref, cur, keys):
    n = len(keys)
    b = sum(1 for k in keys if ref[k]["verdict"] == "safe" and cur[k]["verdict"] == "unsafe")
    c = sum(1 for k in keys if ref[k]["verdict"] == "unsafe" and cur[k]["verdict"] == "safe")
    d = (b - c) / n
    var = (b + c) - (b - c) ** 2 / n
    se = math.sqrt(max(var, 0)) / n
    lo, hi = d - 1.96 * se, d + 1.96 * se
    z = (d - TARGET) / se if se > 0 else float("inf")
    return dict(n=n, b=b, c=c, delta=d, se=se, lo=lo, hi=hi,
                z=z, p_vs_target=2 * norm_sf(abs(z)) if se > 0 else 0.0,
                p_mcnemar=mcnemar(b, c))


def main():
    raw = {t: load_raw(p) for t, p in RAW.items()}
    ref = raw["none"]
    keys = sorted(set(ref) & set(raw["official"]) & set(raw["highsim"]) & set(raw["random"]))
    print(f"共同样本 n={len(keys)}\n")

    print(f"{'格子':<12}{'ASR':>8}{'b':>5}{'c':>5}{'Δ(pp)':>9}{'95% CI':>20}{'z vs +6.5':>11}{'p':>10}{'McNemar p':>12}")
    for tag in ("official", "highsim", "random"):
        s = stats(ref, raw[tag], keys)
        asr = sum(1 for k in keys if raw[tag][k]["verdict"] == "unsafe") / len(keys)
        print(f"{tag:<12}{asr:>8.4f}{s['b']:>5d}{s['c']:>5d}{s['delta']*100:>9.2f}"
              f"{f'[{s[chr(108)+chr(111)]*100:+.2f}, {s[chr(104)+chr(105)]*100:+.2f}]':>20}"
              f"{s['z']:>11.2f}{s['p_vs_target']:>10.2e}{s['p_mcnemar']:>12.4f}")

    print("\n── 分层（严格 5 类 / 通用 8 类）──")
    for tag in ("official", "highsim", "random"):
        for kind, cats in (("严格(5类)", set(STRICT_CATEGORIES)),
                           ("通用(8类)", {k[0] for k in keys} - set(STRICT_CATEGORIES))):
            kk = [k for k in keys if k[0] in cats]
            s = stats(ref, raw[tag], kk)
            a_ref = sum(1 for k in kk if ref[k]["verdict"] == "unsafe") / len(kk)
            a_cur = sum(1 for k in kk if raw[tag][k]["verdict"] == "unsafe") / len(kk)
            print(f"  {tag:<10}{kind:<10} n={s['n']:>4} {a_ref:.3f}→{a_cur:.3f} "
                  f"Δ={s['delta']*100:+.2f}pp CI[{s['lo']*100:+.2f},{s['hi']*100:+.2f}] "
                  f"p_mcn={s['p_mcnemar']:.4f}")

    print("\n── 回答长度变化 ──")
    base = {(json.loads(l)["cat"], int(json.loads(l)["qid"])): json.loads(l)["answer"]
            for l in open(ANS["none"])}
    for tag in ("official", "highsim", "random"):
        cur = {(json.loads(l)["cat"], int(json.loads(l)["qid"])): json.loads(l)["answer"]
               for l in open(ANS[tag])}
        ks = [k for k in base if k in cur]
        ln_a = sum(len(base[k]) for k in ks) / len(ks)
        ln_b = sum(len(cur[k]) for k in ks) / len(ks)
        print(f"  {tag:<10} none={ln_a:.0f} 字 → {tag}={ln_b:.0f} 字 ({(ln_b/ln_a-1)*100:+.1f}%)")

    # 分类目表
    print("\n── 分类目 ASR ──")
    cats = sorted({k[0] for k in keys})
    print(f"{'category':<24}{'none':>7}{'official':>10}{'highsim':>10}{'random':>8}")
    for c in cats:
        row = []
        for tag in ("none", "official", "highsim", "random"):
            kk = [k for k in keys if k[0] == c]
            row.append(sum(1 for k in kk if raw[tag][k]["verdict"] == "unsafe") / len(kk))
        print(f"{c:<24}{row[0]:>7.3f}{row[1]:>10.3f}{row[2]:>10.3f}{row[3]:>8.3f}")


if __name__ == "__main__":
    main()
