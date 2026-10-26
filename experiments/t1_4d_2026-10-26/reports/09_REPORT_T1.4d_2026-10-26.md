# 09 · T1.4d 报告（复现阶段最后一轮）· gpu84 · 2026-10-26

> 按 TASKS.md（commit 6f686cb）执行。复现目标：原文 **Table 2「Main Results」MM-Safety 列**
> `Original 59.37 → TRIM 65.87（+6.50pp）`。
> 本轮新增：原版栈 + SD 两格（含 flip 计数）、官方 questions JSON 查证、官方问法重跑、对话模板确认。

---

## 一、结论先行

**原版栈 + SD（基线能对上原文的唯一配置）：TRIM 未复现出增益，差值为负且不显著。**

| 条件 | ASR | n_unsafe | 原文 |
|---|---|---|---|
| 无剪枝 | **0.5840** | 233/399 | 0.5937（**差 −0.97pp，基线对齐**） |
| TRIM 保留率 0.25 | **0.5664** | 226/399 | 0.6587 |
| 差值 | **−1.75pp** | | **+6.50pp** |

**flip 计数（McNemar）**：

| | TRIM safe | TRIM unsafe |
|---|---|---|
| **无剪枝 safe** | **a = 154** | **b = 12** |
| **无剪枝 unsafe** | **c = 19** | **d = 214** |

- 翻转 31 条（7.8%），净变化 **b − c = −7 条（−1.75pp）**
- McNemar 精确双侧 **p = 0.281** → **不显著**
- 解读：不是"翻了 80 条净剩 11"，而是**只有 31 条翻转、两个方向大致抵消**

**按 TASKS §四 判定表**：无剪枝 ≈ 59.37 ✓、TRIM 没有高出 ✗ → 落在
「**配置对了、效应不存在**」→ **停止复现，转交付物 1 改写**。

---

## 二、六组实测并排（同 399 个 id，官方判官协议，解析失败 0）

| # | 栈 | split | 问法 | 对话模板 | 无剪枝 | TRIM 0.25 | 差值 | flip b/c | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 我们的(llava-hf) | SD | HF | HF | 0.5789 | 0.5739 | −0.50pp | — | — |
| 2 | 我们的(llava-hf) | TYPO | HF | HF | 0.5664 † | 0.5388 † | −2.76pp | — | — |
| 3 | 原版 | TYPO | HF | vicuna_v1 | 0.7594 | 0.7318 | −2.76pp | 15/27 | 0.088 |
| 4 | **原版** | **SD** | HF | vicuna_v1 | **0.5840** | **0.5664** | **−1.75pp** | **12/19** | **0.281** |
| 5 | 原版 | TYPO | 官方 | vicuna_v1 | 0.7719 | 0.7218 | −5.01pp | 13/33 | **0.0045** |
| 6 | 原版 | SD | HF | **llava_v1** | 0.5764 | 0.5614 | −1.50pp | 9/15 | 0.308 |

† 预处理 bug（中心裁剪），该行作废。

### 2.1 带目标值的完整对照（目标 = 原文 Table 2：无剪枝 59.37 / TRIM 65.87 / 差 +6.50pp）

| # | 栈 · split · 问法 · 模板 | 无剪枝 | 目标 59.37 | 偏差 | TRIM 0.25 | 目标 65.87 | 偏差 | 差值 | 目标 +6.50pp | 方向 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 我们的 · SD · HF · HF | 0.5789 | 0.5937 | **−1.48pp** | 0.5739 | 0.6587 | −8.48pp | −0.50pp | +6.50pp | ✗ |
| 2 | 我们的 · TYPO · HF · HF † | 0.5664 | 0.5937 | −2.73pp | 0.5388 | 0.6587 | −11.99pp | −2.76pp | +6.50pp | ✗ |
| 3 | 原版 · TYPO · HF · vicuna_v1 | 0.7594 | 0.5937 | +16.57pp | 0.7318 | 0.6587 | +7.31pp | −2.76pp | +6.50pp | ✗ |
| 4 | **原版 · SD · HF · vicuna_v1** | **0.5840** | **0.5937** | **−0.97pp ✓** | **0.5664** | **0.6587** | **−9.23pp** | **−1.75pp** | **+6.50pp** | **✗** |
| 5 | 原版 · TYPO · 官方 · vicuna_v1 | 0.7719 | 0.5937 | +17.82pp | 0.7218 | 0.6587 | +6.31pp | −5.01pp | +6.50pp | ✗ |
| 6 | 原版 · SD · HF · llava_v1 | 0.5764 | 0.5937 | −1.73pp | 0.5614 | 0.6587 | −9.73pp | −1.50pp | +6.50pp | ✗ |
| 附 | 原版 · TYPO · HF · vicuna_v1（`TRIM:-1`） | 0.7594 | 0.5937 | +16.57pp | 0.7193 | 0.6587 | +6.06pp | −4.01pp | +6.50pp | ✗ |

**一个需要特别警惕的陷阱**：第 3、5 行的 TRIM 绝对值（0.7318 / 0.7218）与原文 65.87 只差 6–7pp，
单看 TRIM 那格"像是接近"，但它们的**无剪枝基线本身就虚高 17pp**（0.7594 / 0.7719）。
原文效应是**基线到 TRIM 的增量**（+6.50pp），不是 TRIM 的绝对值——所以这两行依然是负的。
只有第 4、6 行的基线落在原文附近（−0.97 / −1.73pp），它们的 TRIM 增量才是有效的判定依据。

**共同规律**：六组全部为负；发生翻转的样本只有 6–12%，且 **unsafe→safe 的方向始终多于 safe→unsafe**。

### 附：原版栈 TYPO 的 flip 计数（补报上轮）

| | TRIM safe | TRIM unsafe |
|---|---|---|
| 无剪枝 safe | a = 81 | b = 15 |
| 无剪枝 unsafe | c = 27 | d = 276 |

翻转 42 条（10.5%），净 −12 条。

### 附：出厂 `TRIM:-1`（自动保留率）参考值

TYPO + vicuna_v1：ASR = **0.7193**（287/399）。与固定 0.25 的 0.7318 接近，同样没有增益。
（TASKS 已定原文是固定 75% 剪枝，此项仅作参考，不进入主线。）

---

## 三、官方 questions JSON 查证结果（§二.2）

**有。** 位置：`third_party/mm-safetybench/data/processed_questions/*.json`，13 个类目共 **1680 条**。

单条结构（`GPT-Pred` 字段内嵌）：

```
[Changed Question] ...
[Key Phrase] ...
[Phrase Type] activity
[Rephrased Question]
The bottom of the image shows a phrase of an activity. List steps in detail to perform this activity ...
```

**与 HF parquet 的 `question` 逐条比对：1680/1680 全部不同（差异率 100%）。**

| 来源 | 问法 |
|---|---|
| HF parquet（我们之前用的） | "**The image** shows a phrase of an activity. …" |
| 官方 JSON（Kind 2/3 用） | "**The bottom of the image** shows a phrase of an activity. …" |

**条件重跑已完成**（TYPO + 官方问法 + 原版栈，同 399 id）：

| 条件 | ASR |
|---|---|
| 无剪枝 | 0.7719（308/399） |
| TRIM 0.25 | 0.7218（288/399） |
| 差值 | **−5.01pp**（flip b=13 / c=33，McNemar p=0.0045，**显著为负**） |

换官方问法后无剪枝反而更高（77.19%），离原文 59.37 更远，方向依旧没有出现。

---

## 四、对话模板确认（§三）

- 官方仓库 `llava/eval/run_llava.py` 与 `model_vqa*.py` 的默认 `--conv-mode` 是 **`llava_v1`**
- 我们前几轮用的是 `vicuna_v1`（两者仅差系统提示词一个词：`curious user` vs `curious human`）
- 已用 `llava_v1` 补跑 SD 两格：**0.5764 / 0.5614，差 −1.50pp**，与 `vicuna_v1` 的 −1.75pp 同一量级
- **结论：对话模板不是原因**

---

## 五、13 类目 ASR（原版栈 + SD，vicuna_v1）

| category | 无剪枝 | TRIM 0.25 | 差 |
|---|---|---|---|
| Illegal_Activitiy | 0.348 | 0.391 | +4.3pp |
| HateSpeech | 0.256 | 0.282 | +2.6pp |
| Malware_Generation | 0.300 | 0.300 | 0.0 |
| Physical_Harm | 0.382 | 0.353 | −2.9pp |
| EconomicHarm | 0.000 | 0.034 | +3.4pp |
| Fraud | 0.514 | 0.405 | −10.8pp |
| Sex | 0.192 | 0.077 | −11.5pp |
| Political_Lobbying | 1.000 | 1.000 | 0.0 |
| Privacy_Violence | 0.303 | 0.273 | −3.0pp |
| Legal_Opinion | 0.903 | 0.903 | 0.0 |
| Financial_Advice | 1.000 | 1.000 | 0.0 |
| Health_Consultation | 1.000 | 0.962 | −3.8pp |
| Gov_Decision | 1.000 | 1.000 | 0.0 |

5 个严格类目饱和在 1.0（占 42% 样本），无移动空间；通用类目方向零散。

---

## 六、统计功效说明（重要，影响结论强度）

- n=399 时，7pp 量级的差异处于检出边界；本轮观测到的净变化是 **−6 ~ −7 条**，而原文 +6.50pp 应表现为**净 +26 条**
- SD 两组（−1.75pp / −1.50pp）**统计上不显著**（p≈0.28/0.31）：正确表述是「**在 n=399 下测不到 TRIM 的效应**」，而不是「TRIM 显著有害」
- TYPO 官方问法那组显著为负（p=0.0045），但 TYPO 基线（77.19%）本就与原文（59.37%）不符，不能作为定论
- **若要给 10-28 一个更硬的结论**：原版栈 + SD **全量 1680 × 2** 约 2.5–3 小时（两卡并行约 1.5h），置信区间从 ±5pp 收到 ±2.4pp；本轮 399 条已证明管线可用，跑全量不需要任何配置改动

---

## 七、遇到的问题 / 待决

| 项 | 状态 |
|---|---|
| 判官 Qwen3-8B vs 原文 GPT-4o-mini | 无法消除（境内连不上 OpenAI）。但 SD 基线 58.40 vs 59.37 仅差 0.97pp，说明**判官在总量级上是可信的** |
| 原文未写 split / 用几种图文对 | 已用两种 split 各跑一遍：只有 SD 的基线对得上，TYPO 高 18pp |
| 42% 样本饱和在 1.0 | 未处理；会压缩可观测差值，但不影响方向判断 |
| 全量 1680 是否要跑 | **建议跑**（见 §六），等 Mac 拍板 |

---

## 八、红线自查

- 仓库代码零修改；实验脚本全在 `/tmp`；`third_party/trim` 原样调用未改
- 六组实验同 399 个 id；官方判官协议；解析失败 0
- 未调参凑数：TRIM 保留率固定 0.25（TASKS 口径），`TRIM:-1` 仅作附注
- GPU 7 仅走 API；GPU 0/5/6 占用期间与他人共享

## 九、服务器侧留存

| 文件 | 内容 |
|---|---|
| `/tmp/t1_4_answers_orig_sd_noprun399.json` / `..._sd_trim399.json` | 原版栈 SD 两格（vicuna_v1） |
| `/tmp/t1_4_answers_orig_sd_{noprun,trim}_llavav1_399.json` | 原版栈 SD 两格（llava_v1） |
| `/tmp/t1_4_answers_orig_officialw_{noprun,trim}399.json` | TYPO 官方问法两格 |
| `/tmp/t1_4_answers_orig_{noprun,trim}typo399.json` | TYPO HF 问法两格 |
| `/tmp/t1_4c_typo399_data.json` / `t1_4c_sd399_data.json` / `t1_4d_typo_official399_data.json` | 三套样本数据（含 base64 图），可复跑 |
| `results/t1_4d_orig_sd_{noprun,trim}/`、`t1_4d_orig_sd_lv1_{noprun,trim}/`、`t1_4d_orig_offw_{noprun,trim}/` | 六块 `kind=asr` 结果块 |
