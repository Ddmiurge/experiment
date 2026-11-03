#!/usr/bin/env python3
"""T1.6 · 四档分布（协议 §4.4 指标；0 生成，只调判官 API）

对每个回答文件：官方二分判定（若已有 raw 判定则直接复用，不重判）
+ 拒绝检测（`REFUSAL_PROMPT`，模块里那份"排除贴警告假拒绝"的），
经 `judge_binary_to_level` 映射成 0拒绝/1误解/2部分合规；unsafe 细分到 3 需要
额外一次判定（模块 docstring 已注明），本轮**不细分 2/3**，与全部历史格子同形。
`is_refusal_conflict` 的样本单列（模块实测会被装饰性免责声明骗出假拒绝）。

判读（协议 §5 的消歧轴）：highsim/③ 相对 ① 少掉的 unsafe 若主要转成 **1误解**
→ 致盲；转成 **0拒绝** → 真拒绝。

用法（服务器，~/venvs/exp）：
  python t1_6_level4.py \
    --cell none=/tmp/t1_5b_answers_none_typo399.json:/tmp/t1_5_judge_raw_t1_5b_answers_none_typo399.jsonl \
    --cell blank=/tmp/t1_6_answers_oracle_blank_typo399.json:/tmp/t1_5_judge_raw_t1_6_answers_oracle_blank_typo399.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request
from collections import Counter

REPO = os.environ.get("REPO", "/home/huangqixu01/experiment")
sys.path.insert(0, os.path.join(REPO, "src"))
from eval.mm_safetybench_judge import (build_prompt, build_refusal_prompt,  # noqa: E402
                                       is_refusal_conflict, judge_binary_to_level,
                                       parse_refusal, parse_verdict)

BASE = "http://127.0.0.1:8003/v1/chat/completions"
MODEL = "bindguard-qwen3-8b-nonthinking"


def get_api_key():
    out = subprocess.run(
        ["sudo", "-n", "docker", "inspect", "bindguard-qwen3-8b", "--format",
         "{{range .Config.Env}}{{println .}}{{end}}"],
        capture_output=True, text=True).stdout
    return [l.split("=", 1)[1].strip() for l in out.splitlines()
            if l.startswith("VLLM_API_KEY=")][0]


def call_llm(prompt, key, max_tokens=60):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "seed": 42, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(BASE, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def load_verdicts(raw_path):
    d = {}
    if raw_path and os.path.exists(raw_path):
        for l in open(raw_path):
            r = json.loads(l)
            d[(r["cat"], int(r["qid"]))] = r["verdict"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True,
                    help="tag=answers.jsonl[:judge_raw.jsonl]；不给 raw 就现场重判二分")
    args = ap.parse_args()
    key = get_api_key()

    print(f"{'cell':<12}{'n':>5}{'0拒绝':>7}{'1误解':>7}{'2部分合规':>9}"
          f"{'矛盾':>6}   档位占比")
    for spec in args.cell:
        tag, rest = spec.split("=", 1)
        parts = rest.split(":")
        rows = [json.loads(l) for l in open(parts[0])]
        verdicts = load_verdicts(parts[1] if len(parts) > 1 else None)
        dist, conflicts = Counter(), []
        for r in rows:
            k = (r["cat"], int(r["qid"]))
            v = verdicts.get(k)
            if v is None:
                v = parse_verdict(call_llm(build_prompt(r["cat"], r["answer"]), key)) or "safe"
            ref_raw = call_llm(build_refusal_prompt(r["answer"]), key)
            refused = parse_refusal(ref_raw)
            refused = False if refused is None else refused
            if is_refusal_conflict(v, refused):
                conflicts.append(k)
                refused = False        # 模块口径：以官方二分为准，矛盾单列复核
            dist[judge_binary_to_level(v, refused)] += 1
        n = sum(dist.values())
        print(f"{tag:<12}{n:>5}{dist.get(0,0):>7}{dist.get(1,0):>7}{dist.get(2,0):>9}"
              f"{len(conflicts):>6}   " +
              " ".join(f"{k}:{v/max(n,1):.1%}" for k, v in sorted(dist.items())))
        with open(f"/tmp/t1_6_levels_{tag}.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"tag": tag, "dist": dict(dist),
                                "conflicts": [[c, q] for c, q in conflicts]},
                               ensure_ascii=False) + "\n")
    print("读法：③/④ 相对 ① 掉下去的 unsafe，落在 0 还是 1——0 多=真拒绝，1 多=致盲。")


if __name__ == "__main__":
    main()
