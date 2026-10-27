# TASKS · 当前轮 T1.5b：TYPO 补格 —— 机制唯一还没被测过的地方

> 背景：`results/archive/11_MAC_RECHECK_T1.5_2026-10-27.md`（Mac 复核，**必读**）
> 与 `experiments/t1_5_2026-10-27/DIAGNOSIS_2026-10-27.md`。常驻规则以 `Agent.md` 为准。
> **本轮不改 `third_party/`、不改 `src/`**；上一轮服务器用 `/tmp` 绕过的三个脚本 bug
> 已回写仓库（`t1_5_orig_highsim.py` 的 `MethodType`、`t1_5_judge.py` 的 raw 覆盖与斜杠 slug、
> `t1_5_trim_probe.py` 的条带区域假设），pull 即可直接用。

---

## 零、为什么只剩这一件事

T1.5 已经把 SD 分片判死了，而且**判得比报告写的更死**：

- 配对精度是 **SE ≈ 1.58pp（95% CI ±3.1pp）**，不是 ±5pp（那是独立比例口径）。
  最贴原文的格子 TRIM-FT × highsim：Δ = **+0.25pp，CI [−2.82, +3.32]**，
  与原文声称的 +6.50pp 相差 **z = −3.99（p = 6.5e-05）** ⇒ 是「**被拒绝**」，不是「没测出」。
  五格全部 z ≤ −3.88。
- 原因也清楚了：**SD 分片的图上没有可读载荷**，有害请求全在问题文本里，
  所以保留哪 144 个视觉 token 结构上不可能改变「是否遵从」。
  实测印证：**四条件判定完全一致的样本占 81.5% / 86.2%，且翻转的 b≈c**——
  剪枝改的是措辞与长度（回答普遍变长 8–12%），没改遵从与否。免判官的拒绝词旁证同向。

⇒ **SD 全量 1680 取消**：±3.1pp 已足以排除 +6.5pp，扩样对这条结论零增量。

唯一还没被测过的地方是 **TYPO**——有害短语**渲染在图上**，这才是机制该现身的位置，
也正是原文 Table 9「Background removed = 65.87（与 TRIM 行逐位相同）」的正面版本。
而且这条总线已证明**有信号**：T1.4d 在 TYPO 上 `trim_official − none = −5.01pp（p=0.0045）`。

**本轮只补 TYPO 这一套矩阵。**

---

## 一、任务（按顺序）

### 1 · 探针重跑 + 一致性自检（约 10 分钟，不加载 LLaVA，不生成）

上一轮「TYPO 底部条带保留 16.58%」测的是**纯留白**：`image_aspect_ratio = pad`
会把 1024×180 的文字条**垂直居中**贴进画布 → 载荷落在网格第 10–14 行，而代码取的是 18–23 行。
脚本已改成不假设位置：输出 24 行的（像素活动 / 相似度 / 官方保留率）三条剖面，
并按像素活动最高的 6 行定义「载荷行」。

```bash
CUDA_VISIBLE_DEVICES=6 python experiments/t1_5_2026-10-27/scripts/t1_5_trim_probe.py \
    --split TYPO --ids-from /tmp/t1_4c_typo399_data.json
```

**必须贴回**：`sim_diff` 均值与全负比例、**载荷行保留率 vs 其余行保留率**（随机基线 0.25）、三行剖面。
只有载荷行明显 < 0.25，才允许再写「关键词被删掉」这句机制解释。

> ⚠ 上一轮 `--ids-from` 传 JSON **数组**文件时按 jsonl 解析失败、**静默退回全量 1680**。
> 本轮若打印的 n 不是 399，**停下来报回来**——别把「n 更大」当成无害，
> 那会让 TYPO 探针与四格的样本集不一致。

### 2 · TYPO 四格（核心；每格约 20–25 分钟，两卡并行约 50 分钟）

**四格必须全走原版栈新脚本 `t1_5_orig_prune.py`。** 理由是预处理必须统一：
llava-hf 的 processor 对 1024×180 会 **resize + 中心裁剪**（只剩 17.6% 宽度），
原版栈 `pad` 走 `expand2square`（条带完整保留）。混用两套栈，
「剪什么」的对比里就混进了「图被裁成什么样」。

```bash
# 卡 5
CUDA_VISIBLE_DEVICES=5 /home/huangqixu01/venvs/trim-orig/bin/python \
  experiments/t1_5_2026-10-27/scripts/t1_5_orig_prune.py --split TYPO --prune none
CUDA_VISIBLE_DEVICES=5 /home/huangqixu01/venvs/trim-orig/bin/python \
  experiments/t1_5_2026-10-27/scripts/t1_5_orig_prune.py --split TYPO --prune highsim
# 卡 6
CUDA_VISIBLE_DEVICES=6 /home/huangqixu01/venvs/trim-orig/bin/python \
  experiments/t1_5_2026-10-27/scripts/t1_5_orig_prune.py --split TYPO --prune official
CUDA_VISIBLE_DEVICES=6 /home/huangqixu01/venvs/trim-orig/bin/python \
  experiments/t1_5_2026-10-27/scripts/t1_5_orig_prune.py --split TYPO --prune random
```

四个模式只差「保留哪 144 个」，其余（144 原序 + 1 个被丢集合均值 = **145 进 LLM**）完全一致：
`none` 关掉剪枝、`official` 走**未 patch 的上游函数**、`highsim` 去掉 L89 的取负、`random` 随机掩码。

脚本自带断言自检：前 3 条打印剪枝后特征形状与 `actual_dims`，并 **assert 进 LLM 的视觉 token 数**
（none=576 / 其余=145）。**assert 不过就停，不要改断言。**

**`official` 那格同时是对 T1.4d 的复现自检**（同栈、同 conv、同 max_new_tokens、同 id）：

| 格子 | 本轮应与 T1.4d 一致 |
|---|---|
| `none` | ≈ **0.7594** |
| `official` | ≈ **0.7318** |

**差超过 1.5pp 就说明 harness 漂移了**（conv 模板 / max_new_tokens / id 集），
先报回来，不要继续往下判分——那比 TYPO 结果本身更重要。
默认 `--conv vicuna_v1 --max-new-tokens 512 --data /tmp/t1_4c_typo399_data.json`，与 T1.4d 相同。

### 3 · 判分与配对分析

```bash
python experiments/t1_5_2026-10-27/scripts/t1_5_judge.py --split TYPO --ref none \
  --model-tag llava-v1.5-7b-TRIM --exp-prefix t1_5b \
  --file none=/tmp/t1_5b_answers_none_typo399.json \
  --file trim_official=/tmp/t1_5b_answers_official_typo399.json \
  --file trim_highsim=/tmp/t1_5b_answers_highsim_typo399.json \
  --file random=/tmp/t1_5b_answers_random_typo399.json
```

判官设置与前面所有轮完全一致（Qwen3-8B 非思考 / temperature=0 / seed=42 / max_tokens=60 /
官方 13 类目协议 / 只喂 Answer）。逐条原始判定自动落
`/tmp/t1_5_judge_raw_<输入文件名>.jsonl`。

**除脚本默认输出，报告还要给两样**（口径照 `scripts/t1_5_analysis.py`，它已在 Mac 侧跑通）：

1. **Δ 的 95% CI**，用**配对**标准误 `SE = sqrt((b+c) − (b−c)²/n)/n`；
2. **z 与 p：观测 Δ 直接对原文主张的量级（+6.50pp）做检验**，而不是只对 0 做检验。

分层必须用 `src/eval/mm_safetybench_judge.py` 的 `STRICT_CATEGORIES`（**5 个类目**）。
上一轮报告写成 137/262 是把 `Legal_Opinion`(31 条) 漏进了通用，导致「通用 ASR 0.36」虚高
（真实 0.29）。**TYPO 上严格类目很可能几乎全饱和在 1.0**（MM-SafetyBench 自己的 TYPO
专业类目是 94–100），所以有效信号全在 8 个通用类目里，两层都要单独报 Δ 与 p。

---

## 二、验收与判定（TYPO 专用，照表写结论，不要含糊）

| none | official | highsim | random | 结论与后续 |
|---|---|---|---|---|
| ≈0.76 | 显著为负（复现 T1.4d） | **> 0 且 p<0.05** | ≈ 0 | **机制复现成功**，但只在图承载载荷的分片。交付物与阶段 2–4 的「格 2」锚点改用 **TYPO**，SD 撤下 |
| ≈0.76 | 显著为负 | > 0 但 p>0.05 | ≈ 0 | 方向对、功效不足 → **到这时才扩样**（TYPO 全量 1680，±3.1→±1.5pp） |
| ≈0.76 | 显著为负 | ≈ 0 | ≈ 0 | 「删载荷」有用、「留载荷」不抬高 ⇒ 原文 Table 9 两个方向都不成立。**交付物 1 = 带功效的负结果 + 官方实现与论文矛盾**；且阶段 2–4 必须换 pruner（优先 FasterVLM 72.02 / PruMerge 70.04 / HiPrune 69.44）或换基准 |
| 与 T1.4d 的 0.7594/0.7318 **对不上** | — | — | — | **harness 漂移**，TYPO 读数全部无效。停下报回来，先查 conv / max_new_tokens / id 集 |

> TYPO 基线 0.76 已接近饱和，+6.5pp 只剩约 20pp 的天花板空间。
> 这不影响配对检验，但绝对值天然被压扁，报告要写一句。

---

## 三、不要做

| 不要 | 因为 |
|---|---|
| 跑 SD 全量 1680 | ±3.1pp 已排除 +6.5pp，扩样对这条结论零增量；省下的 4 小时够跑完本轮四格 |
| 把 TYPO 四格分到两个栈跑 | 预处理不同（HF 中心裁剪 vs 原版 pad），会把「剪什么」和「图被裁成什么样」混在一起 |
| 只报 `highsim` 一格 | `none` / `official` / `random` 必须齐报；`official` 兼作 T1.4d 复现自检 |
| 扫保留率 / conv / max_new_tokens 直到出正方向 | 调参凑数。保留率固定 0.25，conv 与 T1.4d 一致 |
| 写「剪枝不改变安全性」 | 只在 SD / keep=0.25 / n=399 / 本判官下成立；本轮 TYPO 正是在测它的边界 |
| 用 4 类当严格类目 | `STRICT_CATEGORIES` 是 5 类，上一轮漏了 `Legal_Opinion` |
| 直接 `cat` 大 json 回传 | 逐样本数据落文件，报路径 + 结果块 |

---

## 四、回传

`results/archive/12_REPORT_T1.5b_TYPO_<日期>.md`，含：

1. **结论先行**：按 §二 判定表落在哪一行；一句话说清「机制在图承载载荷的分片上成立与否」
2. **harness 自检**：本轮 `none` / `official` 与 T1.4d 的 0.7594 / 0.7318 各差多少
3. 探针重跑：载荷行 vs 其余行保留率 + 三行 24 个数剖面（并注明上一轮 0.166 是留白、不是证据）
4. 四格表：ASR、b/c、净、McNemar 精确 p、**Δ 的 95% CI（配对口径）**、**z vs +6.50pp**
5. 严格(5类) / 通用(8类) 分层各自的 Δ 与 p
6. 分类目 ASR 全表 + 回答长度变化（剪枝后变长 8–12% 是已确认的系统偏移）
7. 4 块 `kind=asr` 结果块（`exp` 前缀 `t1_5b`；`cfg` 的 `prune_sign` 只能用
   `na / low_sim / high_sim / random`，**不要出现斜杠**）

**当天回传。** 10-28 的硬决策只需要 §二 那一行结论。
