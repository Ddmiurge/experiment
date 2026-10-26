#!/usr/bin/env python3
"""T1.4c · 原版栈结果判官 + 三方对比（临时脚本 /tmp）。"""
import json
import subprocess
import sys
import urllib.request

sys.path.insert(0, "/home/huangqixu01/experiment")
sys.path.insert(0, "/home/huangqixu01/experiment/src")
sys.path.insert(0, "/home/huangqixu01/experiment/scripts")
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


def judge(path, key, tag):
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
    import result_block
    key = get_api_key()

    # 原版栈判分
    rows, levels, per_cat, fails = judge("/tmp/t1_4_answers_trimorig_typo399.json", key, "orig")
    n = len(rows)
    n_unsafe = sum(c["unsafe"] for c in per_cat.values())
    asr = n_unsafe / n
    print(f"[原版栈/trim] n={n} unsafe={n_unsafe} ASR={asr:.4f} 解析失败={len(fails)}")

    result_block.emit(
        exp="t1_4c_orig_stack_typo_mm_safetybench_trim",
        levels=levels,
        cfg={"model": "llava-v1.5-7b-TRIM(merged)", "pruner": "TRIM(official-code)",
             "keep_ratio": "0.25", "stack": "haotian-liu+transformers4.37.2",
             "decode": "greedy", "seed": "42", "split": "TYPO", "n": str(n),
             "ids": "same-as-SD-round"},
        env={**result_block.fingerprint(),
             "judge": "Qwen3-8B(non-thinking) official-protocol"},
        extra={"asr": f"{asr:.4f}", "n_unsafe": str(n_unsafe),
               **{f"asr_{c.lower()}": f"{(v['unsafe']/(v['safe']+v['unsafe'])):.4f}"
                  for c, v in per_cat.items()}},
        out_dir="/home/huangqixu01/experiment/results",
        kind="asr",
    )

    # 三方对比：手工拼接 / hook / 原版栈
    manual = {(json.loads(l)["cat"], json.loads(l)["qid"]): json.loads(l)["answer"]
              for l in open("/tmp/t1_4_answers_trim_typo399.json")}
    orig = {(r["cat"], r["qid"]): r["answer"] for r in rows}
    same = diff = 0
    ex = []
    for k, a in orig.items():
        if manual.get(k) == a:
            same += 1
        else:
            diff += 1
            if len(ex) < 3:
                ex.append((k, (manual.get(k) or "")[:70], a[:70]))

    print("\n═══ 三方对比（TYPO / TRIM 0.25）═══")
    print(f"{'':28}{'ASR':>8}")
    print(f"{'手工拼接(llava-hf)':<28}{0.5388:>8.4f}")
    print(f"{'hook内部替换(llava-hf)':<28}{0.5388:>8.4f}  （回答与手工版 399/399 一致）")
    print(f"{'原版栈(TRIM官方代码+合并权重)':<28}{asr:>8.4f}")
    print(f"\n原版栈 vs 手工版：回答一致 {same}/{n}，不同 {diff}/{n}")
    for k, m, o in ex:
        print(f"  diff 例: {k}\n    手工: {m}\n    原版: {o}")

    json.dump({"asr": asr, "n": n, "n_unsafe": n_unsafe, "fails": fails,
               "per_cat": per_cat, "agree_with_manual": same},
              open("/tmp/t1_4c_orig_summary.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
