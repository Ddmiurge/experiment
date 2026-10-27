# TASKS · 当前轮 T1.5c：查询口径开关 —— 复现的最后一格 + 致盲判别

> 背景：`results/archive/12_...` 与 `t1_5b_package_2026-10-27/01_reports/12_REPORT_T1.5b_TYPO_2026-10-27.md`
> （**Mac 侧已独立复核，四格数字、分层、CI、z 全部对得上**，见 `scripts/t1_5b_analysis.py`）。
> 本轮只做两件便宜的事，都是**已归档数据的延伸**，不改攻击本体。
> 常驻规则见 `Agent.md`；本轮仍不改 `third_party/`、不改 `src/`。

---

## 零、这轮要回答的问题（一句话）

TRIM 的评分对象是「图像 patch 与**文本查询**的相似度」。MM-SafetyBench 为了构成攻击，
**故意把关键词从问题里删掉了**（"The image shows a phrase of an activity…"）。
T1.5b 的探针实测：用这个模板问句当查询时，**白底留白的相似度（28.6–29.3）高于黑字条带（24.9–26.9）**
⇒「保留高相似度 token」恰好等于「删掉载荷」（实测 ASR −9.77pp）。

**也就是说：TRIM 原文 §3.2 的机制（保留前景→恶意语义被提纯）在这个基准上从未被触发过。**
唯一能让它触发的办法，是把查询换成**真的说出载荷**的文本（官方 JSON 里有 `Key Phrase`
与 `Changed Question` 两个字段）。这就是本轮唯一的自变量。

> 这一格**不是为了凑正方向**：无论正负都是结论。
> 正 ⇒ 复现成功，且我们说清了原文依赖的那个未声明口径；
> 负 ⇒ 原文机制在三种查询定义下都不成立，负结果彻底封死。

---

## 一、任务

### 1 · 探针：查询换成关键词，选择会不会翻转（几分钟，0 生成）

```bash
# 先跑 40 条看方向，别一上来跑全量
CUDA_VISIBLE_DEVICES=6 python experiments/t1_5_2026-10-27/scripts/t1_5_trim_probe.py \
    --split TYPO --query-src keyphrase --limit 40
CUDA_VISIBLE_DEVICES=6 python experiments/t1_5_2026-10-27/scripts/t1_5_trim_probe.py \
    --split TYPO --query-src changed  --limit 40
CUDA_VISIBLE_DEVICES=6 python experiments/t1_5_2026-10-27/scripts/t1_5_trim_probe.py \
    --split TYPO --query-src dummy    --limit 40
```

脚本会先打印 5 条**人工核对**（`prompt → 查询`）。**核对不上就停，别往下跑**
（官方 JSON 的 id 有洞，`EconomicHarm` 键跳到 132 只有 122 条，所以对齐只能用
parquet 的 `id`；Mac 侧已用你们的 798 条验证过 798/798 命中且问句逐字相等）。

**关键读数**：`载荷行保留率` 三档对比（chance = 0.25）

| query-src | highsim 载荷行保留率 | 含义 |
|---|---|---|
| template | **0.223**（T1.5b 实测） | 查询无语义 → 高相似度=留白 → 删载荷 |
| keyphrase | **?** ← 本轮要测 | 若升到 ≥0.45，机制方向被恢复，值得跑生成 |
| changed | **?** | 同上，且更贴近原文的「文本指令」 |
| dummy | **?** | 下界对照：查询完全无信息时选到什么 |

**门槛**：只有 `keyphrase` 或 `changed` 让 highsim 的载荷行保留率**明显高于 0.25**（理想 >0.40），
才进入第 2 步。若仍然 ≈0.2 或更低，说明 CLIP 相似度在这个图上是像素纹理驱动的，
与语义无关 —— 那本身就是结论，直接跳到第 3 步写报告。

### 2 · 生成两格（约 50 分钟，只在第 1 步过门槛时跑）

**只换 `texts=`（TRIM 的查询），模型的 prompt 仍是模板问句** ——
`--query-src` 只作用于剪枝打分，脚本会把实际用的查询写进答案文件的 `trim_query` 字段留档。

```bash
CUDA_VISIBLE_DEVICES=5 /home/huangqixu01/venvs/trim-orig/bin/python \
  experiments/t1_5_2026-10-27/scripts/t1_5_orig_prune.py --split TYPO --prune highsim \
  --query-src keyphrase
CUDA_VISIBLE_DEVICES=6 /home/huangqixu01/venvs/trim-orig/bin/python \
  experiments/t1_5_2026-10-27/scripts/t1_5_orig_prune.py --split TYPO --prune official \
  --query-src keyphrase
```

判分同上（`--model-tag llava-v1.5-7b-TRIM --exp-prefix t1_5c`），
必报：ASR、vs `none`(0.7594) 与 vs `highsim`模板查询(0.6617) 两个配对口径、
**Δ 的 95% CI（配对 SE = `sqrt((b+c)−(b−c)²/n)/n`）**、严格5/通用8 分层。

**预测（写在前头，事后不许改口径）**
- 若原文机制成立：`highsim_keyphrase` **> none(0.7594)**，且 `official_keyphrase` 更低
  （因为此时 official 反过来删载荷）。→ 复现出「保留语义相关前景 → 攻击面打开」
- 若仍在 none 以下：机制在三种查询下都不成立 → 负结果封死，直接进交付物

### 3 · 致盲 vs 拒绝判别（0 生成，只判已归档答案；**这条比第 2 步更重要**）

交付物一定会被问：TYPO 上 `highsim` 的 −9.77pp 到底是「模型开始拒绝」还是「模型读不到字了」。
前者是安全增益，后者是**致盲**，两者含义完全相反。
`实验口径与判据` §五早就为这一问准备了四档（拒绝/误解/部分合规/合规），现在用它：

对已归档的 4×399 TYPO 回答加判「档位」（脚本里 `judge_binary_to_level` / `build_refusal_prompt`
已在 `src/eval/mm_safetybench_judge.py`，可直接用）：

| 条件 | 应报 |
|---|---|
| none / official / highsim / random | 四档占比，尤其 **误解(档1) 占比** |

**判据**：若 `highsim` 相对 `none` 少掉的 unsafe 主要转成**档 1（误解/答非所问）**
而非**档 0（拒绝）**，那么「剪枝降低 ASR」是**致盲**，不是安全 ——
这句话必须写进交付物，否则负结果会被读成"剪枝是一种防御"。

---

## 二、不要做

| 不要 | 因为 |
|---|---|
| 用 `--query-src` 同时改模型 prompt | 那改了下攻击本体，两个变量混在一起，全部格子不可比 |
| 第 1 步不核对 5 条映射就跑 | id 有洞，错位对齐会静默产出"看起来合理"的垃圾 |
| 跑到 SD 全量 / 换保留率凑正方向 | 调参凑数。keep 固定 0.25 |
| 把 `random` 当逐样本随机引用 | T1.5b 已确认那 399 条**共用一张掩码**（本轮脚本已修，重跑才可信） |
| 再补 T1.4/T1.5/T1.5b 的重复格子 | 三套数据已相互对齐（harness 差 0.00pp / −0.25pp），够了 |

---

## 三、回传

`results/archive/13_REPORT_T1.5c_query_<日期>.md`：

1. 第 1 步三档 query-src 的**载荷行保留率表**（+ 5 条人工核对原文）
2. 是否过门槛；过则给第 2 步两格 ASR/CI/分层与两个配对口径
3. **第 3 步四档分布表**（这一条无论第 2 步结果如何都必须做，成本只是判官调用）
4. 结论二选一，写死：
   - 「原文机制需要查询含载荷，我们在 X 下复现出 +Ypp」，或
   - 「三种查询定义下 +6.50pp 都不存在；且 −9.77pp 是致盲/拒绝中的【哪一种】」

**当天回传。** 这是复现阶段的最后一格，之后进入交付物改写。
