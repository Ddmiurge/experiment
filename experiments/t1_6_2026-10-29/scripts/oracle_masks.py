#!/usr/bin/env python3
"""T1.6 · oracle 掩码构造（CPU，服务器跑，几分钟；不需要模型/GPU）

口径冻结（§2.1 的"沿用探针脚本原样，不得新写"在此取**字面**解释）：
  载荷行 = **直接读取 T1.5b 探针归档里每样本的 payload_rows**
  （探针算法：逐行像素活动度 std，取最高的 6 行，逐样本集合）。
  归档实测：每样本恒 6 行、不同行集合 33 种、众数 {11..16} 出现 171/399——
  即 §2.1 写的"范围约 5–7"不成立，6×24=144 恒等于预算，
  **§2.3 的溢出/补足规则在本数据上退化为 no-op**（代码仍实现两支，触发即报警）。
  固定行 10/11/15/16 只进敏感性分析，不进本模块（§2.1 末行）。

输出（均带 (cat,qid) 键，与 399 条同批 id 对齐；id 序已与 answers/probe 双侧核对一致）：
  /tmp/t1_6_masks_payload.jsonl  ④ O-载荷：保留载荷行全部 144 格（确定性）
  /tmp/t1_6_masks_blank.jsonl    ③ O-留白：留白行 432 格中均匀随机 144
                                  （default_rng(42) 逐样本重建 = 与既有 random 臂同实现；
                                    池子逐样本不同 ⇒ 掩码逐样本不同）
  /tmp/t1_6_masks_audit.txt      冻结核验记录（协议 §7.0 要求的存档件）
用法：
  python oracle_masks.py --probe code/t1_5b_package_2026-10-27/06_probe/t1_5_probe_typo.jsonl --show 3
"""
import argparse
import json
from collections import Counter

import numpy as np

SEED = 42          # 与全部历史条件同种子
NROW = NCOL = 24
BUDGET = 144


def build(probe_path):
    rows = [json.loads(l) for l in open(probe_path)]
    out_p, out_b, audit = [], [], []
    ov_ct, pad_ct = 0, 0
    for r in rows:
        cat, qid, prow = r["cat"], int(r["qid"]), sorted(r["payload_rows"])
        assert len(prow) == 6, (cat, qid, len(prow))      # 冻结核验：恒 6
        pool_p = sorted(c for c in range(NROW * NCOL) if c // NCOL in set(prow))
        pool_b = sorted(c for c in range(NROW * NCOL) if c not in set(pool_p))
        # ④ O-载荷
        if len(pool_p) > BUDGET:                          # 溢出支（本数据不触发）
            ov_ct += 1
            keep_p = pool_p[:BUDGET]                      # 无逐格活动度可用→取格序；触发即报警
        elif len(pool_p) < BUDGET:                        # 补足支（本数据不触发）
            pad_ct += 1
            extra = np.random.default_rng(SEED).choice(
                [c for c in pool_b if c not in set(pool_p)], BUDGET - len(pool_p),
                replace=False).tolist()
            keep_p = sorted(pool_p + [int(x) for x in extra])
        else:
            keep_p = pool_p
        # ③ O-留白
        assert len(pool_b) >= BUDGET
        keep_b = sorted(int(x) for x in
                        np.random.default_rng(SEED).choice(pool_b, BUDGET, replace=False))
        out_p.append({"cat": cat, "qid": qid, "payload_rows": prow, "keep": keep_p})
        out_b.append({"cat": cat, "qid": qid, "payload_rows": prow, "keep": keep_b})
    sets = Counter(frozenset(r["payload_rows"]) for r in rows)
    header = [
        f"样本数 = {len(rows)}；载荷行数/样本 分布 = {dict(Counter(len(r['payload_rows']) for r in rows))}",
        f"不同载荷行集合种类 = {len(sets)}；众数 = {sorted(max(sets, key=sets.__getitem__))}"
        f"（出现 {max(sets.values())}/{len(rows)}）",
        f"④ 溢出样本 = {ov_ct}，补足样本 = {pad_ct}（应恒 0：6行×24列=144=预算）",
        f"③ 留白池恒 {NROW*NCOL-144} 格，均匀采样 144，rng=default_rng({SEED}) 每样本重建"
        f"（与既有 random 臂同实现）",
    ]
    return out_p, out_b, header


def rowmap(keep):
    """kept-set 的行分布文本图（协议 §7.0 的 smoke 打印件）"""
    c = Counter(k // NCOL for k in keep)
    return " ".join(f"{r}:{c.get(r,0)}" for r in range(NROW))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True,
                    help="t1_5b_package_2026-10-27/06_probe/t1_5_probe_typo.jsonl")
    ap.add_argument("--show", type=int, default=0, help="打印前 N 条 kept-set 行分布（smoke）")
    ap.add_argument("--outdir", default="/tmp")
    a = ap.parse_args()
    out_p, out_b, header = build(a.probe)
    for path, data in ((f"{a.outdir}/t1_6_masks_payload.jsonl", out_p),
                       (f"{a.outdir}/t1_6_masks_blank.jsonl", out_b)):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                assert len(r["keep"]) == BUDGET
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"完成 {path}（{len(data)} 条）", flush=True)
    print("\n".join(header))
    with open(f"{a.outdir}/t1_6_masks_audit.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for r in out_p[:max(a.show, 3)]:
            f.write(f"payload {r['cat']}#{r['qid']} rows={r['payload_rows']} "
                    f"行分布[{rowmap(r['keep'])}]\n")
            b = next(x for x in out_b if x["cat"] == r["cat"] and x["qid"] == r["qid"])
            f.write(f"blank   {r['cat']}#{r['qid']} 载荷行命中={sum(1 for k in b['keep'] if k//NCOL in set(r['payload_rows']))}（③应为0）\n")
    if a.show:
        print(f"=== 前 {a.show} 条 smoke ===")
        for r in out_p[:a.show]:
            print(f"④ {r['cat']}#{r['qid']} 行分布[{rowmap(r['keep'])}]")
        for r in out_b[:a.show]:
            hit = sum(1 for k in r["keep"] if k // NCOL in set(r["payload_rows"]))
            print(f"③ {r['cat']}#{r['qid']} 行分布[{rowmap(r['keep'])}] 载荷行命中={hit}")


if __name__ == "__main__":
    main()
