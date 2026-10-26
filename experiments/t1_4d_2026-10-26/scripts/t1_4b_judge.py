#!/usr/bin/env python3
"""T1.4b · TYPO split 判官 + SD/TYPO 并排对照（临时脚本，/tmp）。"""
import json
import subprocess
import sys
import urllib.request

sys.path.insert(0, "/home/huangqixu01/experiment")
sys.path.insert(0, "/home/huangqixu01/experiment/src")
from eval.mm_safetybench_judge import build_prompt, parse_verdict, CATEGORIES

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
    body = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "seed": 42, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(BASE, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def judge_file(path, key, tag):
    rows = [json.loads(l) for l in open(path)]
    per_cat = {c: {"safe": 0, "unsafe": 0} for c in CATEGORIES}
    levels, fails = [], []
    for i, r in enumerate(rows):
        v = parse_verdict(call_llm(build_prompt(r["cat"], r["answer"]), key))
        if v is None:
            fails.append(f"{r['cat']}#{r['qid']}")
            v = "safe"
        per_cat[r["cat"]][v] += 1
        levels.append(2 if v == "unsafe" else 0)
        if (i + 1) % 50 == 0:
            print(f"[{tag}] {i+1}/{len(rows)}", flush=True)
    return rows, levels, per_cat, fails


def main():
    sys.path.insert(0, "/home/huangqixu01/experiment/scripts")
    import result_block

    key = get_api_key()
    out = {}
    for cond in ("noprun", "trim"):
        path = f"/tmp/t1_4_answers_{cond}_typo399.json"
        rows, levels, per_cat, fails = judge_file(path, key, cond)
        n = len(rows)
        n_unsafe = sum(c["unsafe"] for c in per_cat.values())
        asr = n_unsafe / n
        out[cond] = dict(n=n, per_cat=per_cat, asr=asr, fails=fails, levels=levels)
        print(f"[TYPO/{cond}] n={n} unsafe={n_unsafe} ASR={asr:.4f} 解析失败={len(fails)}")

    # SD 上轮数字（从 t1_4_summary.json 读）
    sd = json.load(open("/tmp/t1_4_summary.json"))
    sd_asr = {c: sd[c]["asr"] for c in ("noprun", "trim")}
    sd_cat = {c: sd[c]["per_cat"] for c in ("noprun", "trim")}

    for cond in ("noprun", "trim"):
        s = out[cond]
        result_block.emit(
            exp=f"t1_4b_typo_mm_safetybench_{cond}",
            levels=s["levels"],
            cfg={"model": "llava-1.5-7b-hf", "pruner": "none" if cond == "noprun" else "TRIM",
                 "keep_ratio": "1.0" if cond == "noprun" else "0.25",
                 "attn": "eager", "decode": "greedy", "seed": "42",
                 "split": "TYPO", "n": str(s["n"]),
                 "ids": "same-as-SD-round"},
            env={**result_block.fingerprint(),
                 "judge": "Qwen3-8B(non-thinking) official-protocol"},
            extra={"asr": f"{s['asr']:.4f}",
                   "n_unsafe": str(sum(c['unsafe'] for c in s['per_cat'].values())),
                   **{f"asr_{c.lower()}": f"{(v['unsafe']/(v['safe']+v['unsafe'])):.4f}"
                      for c, v in s["per_cat"].items()}},
            out_dir="/home/huangqixu01/experiment/results",
            kind="asr",
        )

    d_ty = out["trim"]["asr"] - out["noprun"]["asr"]
    d_sd = sd_asr["trim"] - sd_asr["noprun"]
    print("\n═══ T1.4b 方向性检查（TYPO）═══")
    print(f"{'':14}{'SD(上轮)':>10}{'TYPO(本轮)':>10}")
    print(f"{'无剪枝 ASR':<14}{sd_asr['noprun']:>10.4f}{out['noprun']['asr']:>10.4f}")
    print(f"{'TRIM ASR':<14}{sd_asr['trim']:>10.4f}{out['trim']['asr']:>10.4f}")
    print(f"{'差值(pp)':<14}{d_sd*100:>10.2f}{d_ty*100:>10.2f}   （原文 +6.50pp）")
    print(f"\nTYPO 分类目 ASR：")
    print(f"{'category':<22} {'SD-nop':>7} {'SD-trim':>8} {'TY-nop':>7} {'TY-trim':>8}")
    for c in CATEGORIES:
        a = sd_cat["noprun"][c]; b = sd_cat["trim"][c]
        ta = out["noprun"]["per_cat"][c]; tb = out["trim"]["per_cat"][c]
        na = a["safe"] + a["unsafe"]; nb = b["safe"] + b["unsafe"]
        nta = ta["safe"] + ta["unsafe"]; ntb = tb["safe"] + tb["unsafe"]
        print(f"{c:<22} {a['unsafe']/na:>7.3f} {b['unsafe']/nb:>8.3f} "
              f"{ta['unsafe']/nta:>7.3f} {tb['unsafe']/ntb:>8.3f}")

    json.dump({k: {"asr": v["asr"], "per_cat": v["per_cat"], "n": v["n"],
                   "fails": v["fails"]} for k, v in out.items()},
              open("/tmp/t1_4b_summary.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
