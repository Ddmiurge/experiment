#!/usr/bin/env python3
"""T1.5 步骤4 重跑（/tmp 临时脚本）。

原因：仓库版 t1_5_judge.py 两处问题
  1) raw jsonl 用 tag 命名，两行实验 tag 相同 → 第二行覆盖第一行；
  2) result_block.emit 的 cfg 里 prune_sign=n/a 含 "/"，生成文件名非法 → FileNotFoundError。
本脚本用带行前缀的 tag、并按合法 cfg 自行 emit，其余（判官协议、prompt、温度/seed/max_tokens）与仓库版一致。
"""
import json
import subprocess
import sys
from math import comb

sys.path.insert(0, "/home/huangqixu01/experiment")
sys.path.insert(0, "/home/huangqixu01/experiment/src")
sys.path.insert(0, "/home/huangqixu01/experiment/scripts")
from eval.mm_safetybench_judge import build_prompt, parse_verdict, CATEGORIES
import result_block

BASE = "http://127.0.0.1:8003/v1/chat/completions"
MODEL = "bindguard-qwen3-8b-nonthinking"
STRICT = {"Political_Lobbying", "Financial_Advice", "Health_Consultation", "Gov_Decision"}

ROWS = {
    "vanilla": {
        "model": "llava-1.5-7b-hf",
        "files": {
            "none": "/tmp/t1_4_answers_noprun_400.json",
            "trim_official": "/tmp/t1_4_answers_trim_400.json",
            "trim_highsim": "/tmp/t1_5_answers_trim_highsim_sdfull.json",
            "random": "/tmp/t1_5_answers_random_sdfull.json",
        },
    },
    "trimft": {
        "model": "llava-v1.5-7b-TRIM",
        "files": {
            "none": "/tmp/t1_4_answers_orig_sd_noprun399.json",
            "trim_official": "/tmp/t1_4_answers_orig_sd_trim399.json",
            "trim_highsim": "/tmp/t1_5_answers_orig_sd_highsim.json",
        },
    },
}
SIGN = {"none": "na", "trim_official": "low_sim", "trim_highsim": "high_sim", "random": "random"}


def get_key():
    out = subprocess.run(
        ["sudo", "-n", "docker", "inspect", "bindguard-qwen3-8b", "--format",
         "{{range .Config.Env}}{{println .}}{{end}}"], capture_output=True, text=True).stdout
    return [l.split("=", 1)[1].strip() for l in out.splitlines()
            if l.startswith("VLLM_API_KEY=")][0]


def call(prompt, key):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "seed": 42, "max_tokens": 60}).encode()
    req = urllib.request.Request(BASE, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"].strip(), d["usage"]["completion_tokens"]


def judge(row, tag, path, key):
    out = {}
    levels = []
    n_tok = []
    with open(f"/tmp/t1_5_raw_{row}_{tag}.jsonl", "w", encoding="utf-8") as f:
        for line in open(path):
            r = json.loads(line)
            raw, ntok = call(build_prompt(r["cat"], r["answer"]), key)
            v = parse_verdict(raw)
            v = v if v in ("safe", "unsafe") else "PARSE_FAIL"
            out[(r["cat"], int(r["qid"]))] = v
            levels.append(2 if v == "unsafe" else 0)
            n_tok.append(ntok)
            f.write(json.dumps({"row": row, "tag": tag, "cat": r["cat"], "qid": int(r["qid"]),
                                "verdict": v, "raw_output": raw, "n_tokens": ntok},
                               ensure_ascii=False) + "\n")
    return out, levels, n_tok


def mcn(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


import urllib.request  # noqa: E402


def main():
    key = get_key()
    res = {}
    for row, spec in ROWS.items():
        res[row] = {}
        for tag, path in spec["files"].items():
            v, lv, ntok = judge(row, tag, path, key)
            res[row][tag] = (v, lv, ntok)
            nu = sum(1 for x in v.values() if x == "unsafe")
            print(f"[{row}/{tag}] n={len(v)} unsafe={nu} ASR={nu/len(v):.4f} "
                  f"解析失败={sum(1 for x in v.values() if x=='PARSE_FAIL')} "
                  f"token均值={sum(ntok)/len(ntok):.1f}", flush=True)

        ref = res[row]["none"][0]
        print(f"\n--- {row} 配对（vs none）---")
        for tag, (v, lv, ntok) in res[row].items():
            if tag == "none":
                continue
            b = sum(1 for k in ref if ref[k] == "safe" and v[k] == "unsafe")
            c = sum(1 for k in ref if ref[k] == "unsafe" and v[k] == "safe")
            a = sum(1 for k in ref if ref[k] == "safe" and v[k] == "safe")
            d = sum(1 for k in ref if ref[k] == "unsafe" and v[k] == "unsafe")
            p = mcn(b, c)
            print(f"  {tag:<15} a={a} b={b} c={c} d={d}  净={b-c:+d}  p={p:.4f}")
            for kind, cats in (("严格", STRICT), ("通用", set(CATEGORIES) - STRICT)):
                kk = [k for k in ref if k[0] in cats]
                b2 = sum(1 for k in kk if ref[k] == "safe" and v[k] == "unsafe")
                c2 = sum(1 for k in kk if ref[k] == "unsafe" and v[k] == "safe")
                nu2 = sum(1 for k in kk if v[k] == "unsafe")
                nr = sum(1 for k in kk if ref[k] == "unsafe")
                print(f"      {kind}: n={len(kk)} none={nr/len(kk):.3f}→{nu2/len(kk):.3f} "
                      f"净={b2-c2:+d} p={mcn(b2,c2):.4f}")

        # emit blocks
        for tag, (v, lv, ntok) in res[row].items():
            asr = sum(1 for x in v.values() if x == "unsafe") / len(v)
            per = {c: {"safe": 0, "unsafe": 0} for c in CATEGORIES}
            for (c, _), verdict in v.items():
                per[c][verdict if verdict in ("safe", "unsafe") else "safe"] += 1
            result_block.emit(
                exp=f"t1_5_{row}_sd_{tag.replace('_','-')}",
                levels=lv,
                cfg={"model": spec["model"],
                     "pruner": "none" if tag == "none" else ("random" if tag == "random" else "TRIM"),
                     "keep_ratio": "1.0" if tag == "none" else "0.25",
                     "prune_sign": SIGN[tag], "attn": "eager", "decode": "greedy",
                     "seed": "42", "split": "SD", "n": str(len(v)),
                     "ids": "same-as-SD-round"},
                env={**result_block.fingerprint(),
                     "judge": "Qwen3-8B(non-thinking) official-protocol"},
                extra={"asr": f"{asr:.4f}",
                       "n_unsafe": str(sum(1 for x in v.values() if x == "unsafe")),
                       "prune_sign": SIGN[tag],
                       **{f"asr_{c.lower()}": f"{(p['unsafe']/max(1,p['safe']+p['unsafe'])):.4f}"
                          for c, p in per.items()}},
                out_dir="/home/huangqixu01/experiment/results",
                kind="asr",
            )


if __name__ == "__main__":
    main()
