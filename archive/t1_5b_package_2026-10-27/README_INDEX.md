# T1.5b 实验包 · TYPO 四格（图承载载荷的分片）· 2026-10-27

对照 TASKS：`01_reports/TASKS_T1.5b.md`（commit 5971107）
Mac 复核：`01_reports/11_MAC_RECHECK_T1.5_2026-10-27.md`
主报告：`01_reports/12_REPORT_T1.5b_TYPO_2026-10-27.md`

---

## 一、结论

**原文主张（TRIM 75% 剪枝 → ASR +6.50pp）在 TYPO 分片上被反向证伪：按原文规范实施，ASR 显著下降 9.77pp。**

| 格子 | ASR | vs none | 95% CI（配对） | z vs +6.50pp | McNemar p |
|---|---|---|---|---|---|
| none | 0.7594 | — | — | — | — |
| official（低相似度，官方符号） | 0.7293 | −3.01pp | [−6.18, +0.16] | −5.88 | 0.088 |
| **highsim（高相似度，原文规范）** | **0.6617** | **−9.77pp** | **[−14.20, −5.35]** | **−7.21** | **<0.0001** |
| random（阴性对照） | 0.7293 | −3.01pp | [−6.54, +0.52] | −5.28 | 0.126 |

**Harness 自检**：none 0.7594（T1.4d 0.7594，差 0.00pp）、official 0.7293（T1.4d 0.7318，差 −0.25pp）→ 无漂移。

**机制**：`highsim` 把承载有害短语的 patch 删掉了（载荷行保留率 22.3% < 随机基线 25%），
`official` 反向规则反倒偏向保留载荷（45.1%）。剪枝在本分片的净效应是"图像信息损失 → 遵从率下降"。

---

## 二、目录结构

```
01_reports/
  ├── 12_REPORT_T1.5b_TYPO_2026-10-27.md   ← 主报告（结论/自检/探针/四格/分层/分类目/长度/bug）
  ├── 11_MAC_RECHECK_T1.5_2026-10-27.md    ← Mac 复核 T1.5 的意见（本轮任务的来源）
  └── TASKS_T1.5b.md                       ← 本轮指令快照

02_mac_scripts/           Mac 提供的 6 个脚本（镜像仓库当前状态，未改动）
  t1_5_trim_probe.py / t1_5_weights_diff.py / t1_5_generate.py
  t1_5_orig_highsim.py（bug1 已由 Mac 修）/ t1_5_orig_prune.py（bug3 仍在）/ t1_5_judge.py（bug2 已由 Mac 修）

03_server_wrappers/       服务器在 /tmp 写的绕过脚本（仓库零改动）
  ├── t1_5b_prune_fixed.py    修 KEEP_LOG 1-D 索引后调用原脚本 main()
  ├── t1_5b_analysis.py       配对 CI + z vs +6.50pp + 5/8 类分层 + 长度变化
  └── （附）T1.5 轮 SD 的 3 个回答文件，便于对照

04_answers/              本轮四格回答（各 399 条，同批 id）
  t1_5b_answers_{none,official,highsim,random}_typo399.json

05_keeps/                剪枝实际保留的 token 索引
  t1_5b_keeps_{official,highsim,random}_typo399.jsonl
  （official 的 keep 全为空——走未 patch 的上游函数，不写 KEEP_LOG，预期行为）

06_probe/                TYPO 探针逐样本明细（n=399）
07_judge_raw/            判官原始输出（4 格，含 raw_output 与 n_tokens）
08_result_blocks/        4 块 kind=asr（exp 前缀 t1_5b，prune_sign 无斜杠）
09_analysis/             逐行保留率剖面（载荷行 vs 留白行）
```

---

## 三、五类脚本问题（3 个崩溃型 + 2 个口径型）

| # | 位置 | 症状 | 修法 | 仓库现状 |
|---|---|---|---|---|
| 1 | `t1_5_orig_highsim.py` `patch()` | `AttributeError: type object 'CLIPVisionTower' has no attribute 'token_reduce_func'` | 普通函数直接赋给类属性，**不要** `types.MethodType(f, cls)` | ✅ Mac 已修 |
| 2 | `t1_5_judge.py` | ① raw jsonl 同名覆盖（两行实验互冲）② `cfg` 含 `n/a` → 斜杠进文件名 → emit 崩溃 | ① 文件名加行前缀 ② `prune_sign` 用 `na/low_sim/high_sim/random` | ✅ Mac 已修 |
| 3 | **`t1_5_orig_prune.py:98`** | `IndexError: tuple index out of range`；**只影响 highsim/random**（none/official 照常跑，易误判为无 bug） | `token_mask[0].nonzero(as_tuple=False).flatten().tolist()` | ❌ 仍在 |
| 4 | `t1_5_orig_prune.py` random 分支 | `rng` 每次调用重建 → 399 条共用**同一个随机掩码**（keeps 只有 1 种） | rng 提到模块级只初始化一次 | ⚠ 口径问题 |
| 5 | `t1_5_trim_probe.py` `--ids-from` | 传 JSON **数组**文件时按 jsonl 解析失败 → 静默退回全量 1680 | 兼容 `[` 开头的数组（`t1_5_orig_prune.py` 已兼容，探针未） | ⚠ 本轮用 jsonl 绕过 |

**建议冒烟自检**（一次覆盖 #3/#4）：
```bash
python experiments/t1_5_2026-10-27/scripts/t1_5_orig_prune.py --split TYPO --prune random --limit 3
```

---

## 四、本轮同时更正的两条历史结论

1. **"TYPO 条带保留 16.58%" 作废**（T1.5 报告）：那测的是**留白**——`pad` 把 1024×180 文字条垂直居中，载荷在第 8–17 行，旧脚本假设 18–23 行。真实载荷行保留率 **45.07%**，关键词并未被系统性删除。
2. **分层口径更正**：`STRICT_CATEGORIES` 是 **5 类**（含 `Legal_Opinion`），n=168；通用 8 类 n=231。此前写的 137/262 与"通用 ASR 0.36"均错误。

---

## 五、待 Mac 决定

1. **交付物 1 写法**：带功效的负结果 + 官方实现与原文矛盾（报告 §八 给了可直接用的表述）
2. **阶段 2–4 走向**：换 pruner（FasterVLM 72.02 / PruMerge 70.04 / HiPrune 69.44）或换基准
3. **是否扩到 TYPO 全量 1680**：当前 −9.77pp 远大于 ±3.1pp CI，我判断不改变结论
4. **修 #3/#4，并给三个脚本加冒烟自检**
